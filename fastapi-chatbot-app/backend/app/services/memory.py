"""Conversation memory utilities using LangChain.

Stores conversation turns in Qdrant (vector) and provides helper to fetch recent context.
"""
from typing import List, Dict
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from app.config import settings


class ConversationMemoryService:
    def __init__(self):
        self.embedding = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.api_key,
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

    def get_recent_context(self, k: int = 5) -> List[str]:
        # naive approach: perform a similarity search against a blank/neutral query; fallback to listing not available
        # We'll approximate by searching using a generic token
        store = Qdrant.from_existing_collection(
            embedding=self.embedding,
            collection_name=self.collection,
            url=self.qdrant_url,
        )
        results = store.similarity_search("conversation", k=k)
        return [r.page_content for r in results]

    def build_context_block(self, k: int = 5) -> str:
        turns = self.get_recent_context(k=k)
        return "\n\n".join(turns)
