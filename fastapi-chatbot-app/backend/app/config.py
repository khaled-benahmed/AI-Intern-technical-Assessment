from pydantic import BaseSettings


class Settings(BaseSettings):
    # Gemini / Google GenAI
    api_key: str  # GEMINI / Google API key
    gemini_model: str = "gemini-2.5-flash"  # default text model
    gemini_embedding_model: str = "text-embedding-004"

    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Collections
    documents_collection: str = "documents"
    conversation_collection: str = "conversation_history"

    # Chunking parameters
    chunk_size: int = 800
    chunk_overlap: int = 120

    class Config:
        env_file = ".env"


settings = Settings()