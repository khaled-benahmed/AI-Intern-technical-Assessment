from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.config import settings
from app.services.embeddings import GeminiEmbeddings
from app.services.qdrant_service import QdrantService
from qdrant_client.models import VectorParams, Distance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Chatbot!"}


@app.on_event("startup")
def ensure_qdrant_collections_on_startup():
    """Ensure required Qdrant collections exist at app launch.

    Creates both documents and conversation_history collections with the correct vector size.
    Falls back to a reasonable default if embedding dimension detection fails.
    """
    try:
        emb = GeminiEmbeddings(api_key=settings.api_key, model=settings.gemini_embedding_model)
        dim = len(emb.embed_query("__dimension_probe__")) or 768
    except Exception:
        dim = 768

    service = QdrantService(host=settings.qdrant_host, port=settings.qdrant_port)
    for name in [settings.documents_collection, settings.conversation_collection, settings.topics_collection]:
        try:
            service.ensure_collection(collection_name=name, vector_size=dim)
        except Exception:
            # Do not block app startup if Qdrant is not reachable yet
            pass