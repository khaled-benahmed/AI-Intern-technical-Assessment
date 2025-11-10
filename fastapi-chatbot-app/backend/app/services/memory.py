"""Conversation memory utilities using LangChain.

Stores conversation turns in Qdrant (vector) and provides helper to fetch recent context.
"""
from typing import List, Dict
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from app.services.embeddings import GeminiEmbeddings
from langchain_core.documents import Document
from app.config import settings


class ConversationMemoryService:
    def __init__(self):
        self.embedding = GeminiEmbeddings(
            api_key=settings.api_key,
            model=settings.gemini_embedding_model,
        )
        self.qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        self.collection = settings.conversation_collection

    def add_turn(self, user: str, message: str, response: str) -> None:
        content = f"User({user}): {message}\nAssistant: {response}"
        doc = Document(page_content=content, metadata={"timestamp": datetime.utcnow().isoformat()})
        Qdrant.from_documents(
            [doc],
            self.embedding,
            url=self.qdrant_url,
            collection_name=self.collection,
        )

    def get_context_for_message(self, message: str, k_similar: int = 5, k_recent: int = 3) -> List[str]:
        """Return similar past turns to the current message and a few recent ones."""
        store = Qdrant.from_existing_collection(
            embedding=self.embedding,
            collection_name=self.collection,
            url=self.qdrant_url,
        )
        sim_results = store.similarity_search(message, k=k_similar)
        # Qdrant vectorstore doesn't support list-all; approximate recency by reusing similarity with a generic token
        recent_results = store.similarity_search("most recent conversation turns", k=k_recent)
        combined = sim_results + [r for r in recent_results if r not in sim_results]
        return [r.page_content for r in combined]

    def build_context_block(self, message: str, k_similar: int = 5, k_recent: int = 3) -> str:
        turns = self.get_context_for_message(message, k_similar=k_similar, k_recent=k_recent)
        return "\n\n".join(turns)
