"""Qdrant service helper for storing and retrieving chat messages by embeddings.

Adds optional session-aware semantic search and generic scrolling helper.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue


class QdrantService:
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create collection if missing with cosine similarity."""
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
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

    def semantic_search(self, collection_name: str, query_vector: List[float], limit: int = 10, session_id: Optional[str] = None):
        """Similarity search optionally filtered by session_id stored in payload.metadata.session_id."""
        payload_filter = None
        if session_id:
            payload_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))])
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=payload_filter,
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)

    def list_points(self, collection_name: str, session_id: Optional[str] = None, with_vectors: bool = False, limit: int = 1000):
        """Scroll points optionally filtered by session_id; returns list of PointStruct."""
        filt = None
        if session_id:
            filt = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))])
        points: list = []
        next_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filt,
                with_vectors=with_vectors,
                limit=min(256, limit - len(points)),
                offset=next_offset,
            )
            points.extend(batch)
            if not next_offset or len(points) >= limit:
                break
        return points[:limit]
