from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.config import settings
from app.services.gemini_service import GeminiService
from app.services.qdrant_service import QdrantService
from app.services.document_ingestion import DocumentIngestionService
from app.services.chain import build_chain

router = APIRouter()

# Use configured conversation collection instead of legacy 'chat_messages'
COLLECTION = settings.conversation_collection


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


def get_services():
    gemini = GeminiService(api_key=settings.api_key)
    qdrant = QdrantService(host=settings.qdrant_host, port=settings.qdrant_port)
    return gemini, qdrant


@router.post("/chat")
def chat(req: ChatRequest, services=Depends(get_services)):
    try:
        chain, memory = build_chain()
        answer = chain.invoke(req.message)
        memory.add_turn(user="default", message=req.message, response=answer, session_id=req.session_id or "default")
        return {"response": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")


@router.post("/search")
def semantic_search(req: ChatRequest, services=Depends(get_services)):
    try:
        gemini, qdrant = services
        # Ensure collection exists (lazy create) using vector size probe
        probe_vec = gemini.embed("dimension probe")
        qdrant.ensure_collection(collection_name=COLLECTION, vector_size=len(probe_vec))
        vector = gemini.embed(req.message)
        results = qdrant.semantic_search(COLLECTION, query_vector=vector, limit=5, session_id=req.session_id)
        return {
            "matches": [
                {"score": r.score, "payload": r.payload} for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name missing")
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        suffix = file.filename.lower().split('.')[-1]
        if suffix not in {"pdf", "docx", "csv"}:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        ingestor = DocumentIngestionService()
        chunks = ingestor.ingest(file.filename, content)
        return {"status": "ok", "chunks_indexed": chunks}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


class TopicsRequest(BaseModel):
    session_id: Optional[str] = "default"


@router.post("/topics")
def topics(req: TopicsRequest, services=Depends(get_services)):
    try:
        _, qdrant = services
        points = qdrant.list_points(COLLECTION, session_id=req.session_id or "default", with_vectors=False, limit=1000)
        clusters: Dict[int, Dict[str, Any]] = {}
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            meta = payload.get("metadata") or {}
            if meta.get("session_id") != (req.session_id or "default"):
                continue
            cid = meta.get("cluster_id")
            if cid is None:
                continue
            ts = meta.get("timestamp")
            msg = meta.get("user_message")
            cid = int(cid)
            info = clusters.setdefault(cid, {"cluster_id": cid, "turn_count": 0, "last_timestamp": "", "example_message": None})
            info["turn_count"] += 1
            if ts and ts > info["last_timestamp"]:
                info["last_timestamp"] = ts
            if not info["example_message"] and msg:
                info["example_message"] = msg[:120]
        result = sorted(clusters.values(), key=lambda x: x["turn_count"], reverse=True)
        return {"topics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topics listing failed: {e}")


@router.get("/health")
def health():
    return {"status": "ok"}
