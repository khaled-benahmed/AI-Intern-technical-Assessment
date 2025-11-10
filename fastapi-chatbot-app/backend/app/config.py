from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    # Gemini / Google GenAI
    api_key: str = Field(..., validation_alias=AliasChoices("API_KEY", "api_key"))
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"),
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        validation_alias=AliasChoices("GEMINI_EMBEDDING_MODEL", "gemini_embedding_model"),
    )

    # Qdrant configuration
    qdrant_host: str = Field(default="localhost", validation_alias=AliasChoices("QDRANT_HOST", "qdrant_host"))
    qdrant_port: int = Field(default=6333, validation_alias=AliasChoices("QDRANT_PORT", "qdrant_port"))

    # Collections
    documents_collection: str = "documents"
    conversation_collection: str = "conversation_history"

    # Chunking parameters
    chunk_size: int = 800
    chunk_overlap: int = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()