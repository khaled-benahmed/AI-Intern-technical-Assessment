from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel
from app.config import settings
from app.services.gemini_service import GeminiService
from app.services.qdrant_service import QdrantService
from app.services.document_ingestion import DocumentIngestionService
from app.services.chain import build_chain
import uuid

router = APIRouter()

COLLECTION = "chat_messages"


class ChatRequest(BaseModel):
    message: str


def get_services():
    gemini = GeminiService(api_key=settings.api_key)
    qdrant = QdrantService(host=settings.qdrant_host, port=settings.qdrant_port)
    return gemini, qdrant


@router.post("/chat")
def chat(req: ChatRequest, services=Depends(get_services)):
    # Use LangChain RAG + memory chain
    chain, memory = build_chain()
    answer = chain.invoke(req.message)
    # also store the turn in memory store
    memory.add_turn(user="default", message=req.message, response=answer)
    return {"response": answer}


@router.post("/search")
def semantic_search(req: ChatRequest, services=Depends(get_services)):
    gemini, qdrant = services
    vector = gemini.embed(req.message)
    results = qdrant.semantic_search(COLLECTION, query_vector=vector, limit=5)
    return {
        "matches": [
            {"score": r.score, "payload": r.payload} for r in results
        ]
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document (CSV, PDF, DOCX)."""
    content = await file.read()
    ingestor = DocumentIngestionService()
    chunks = ingestor.ingest(file.filename, content)
    return {"status": "ok", "chunks_indexed": chunks}
