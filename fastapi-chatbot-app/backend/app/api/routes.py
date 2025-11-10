from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.config import settings
from app.services.gemini_service import GeminiService
from app.services.qdrant_service import QdrantService
from app.services.document_ingestion import DocumentIngestionService
from app.services.chain import build_chain
import uuid

router = APIRouter()

# Use configured conversation collection instead of legacy 'chat_messages'
COLLECTION = settings.conversation_collection


class ChatRequest(BaseModel):
    message: str


def get_services():
    gemini = GeminiService(api_key=settings.api_key)
    qdrant = QdrantService(host=settings.qdrant_host, port=settings.qdrant_port)
    return gemini, qdrant


@router.post("/chat")
def chat(req: ChatRequest, services=Depends(get_services)):
    try:
        chain, memory = build_chain()
        answer = chain.invoke(req.message)
        memory.add_turn(user="default", message=req.message, response=answer)
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
        results = qdrant.semantic_search(COLLECTION, query_vector=vector, limit=5)
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

@router.get("/health")
def health():
    return {"status": "ok"}
