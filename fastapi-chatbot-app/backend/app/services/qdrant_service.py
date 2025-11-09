"""Qdrant service helper for storing and retrieving chat messages by embeddings."""

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams


class QdrantService:
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create collection if missing with cosine similarity."""
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance="Cosine"),
            )

    def upsert_messages(self, collection_name: str, items: List[Dict[str, Any]]) -> None:
        """Upsert chat messages with embeddings.

        Each item must include id (int/uuid), vector (List[float]), and payload (dict).
        """
        points = [
            PointStruct(id=i["id"], vector=i["vector"], payload=i["payload"])
            for i in items
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def semantic_search(self, collection_name: str, query_vector: List[float], limit: int = 10):
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)
