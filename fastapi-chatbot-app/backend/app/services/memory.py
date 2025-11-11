"""Conversation memory utilities using LangChain.

Stores conversation turns in Qdrant (vector) and provides helper to fetch recent context.
Also assigns a lightweight topic cluster_id to each turn and stores it in metadata.
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from langchain_qdrant import Qdrant
from app.services.embeddings import GeminiEmbeddings
from langchain_core.documents import Document
from app.config import settings
import numpy as np
from qdrant_client import QdrantClient


class ConversationMemoryService:
    def __init__(self):
        self.embedding = GeminiEmbeddings(
            api_key=settings.api_key,
            model=settings.gemini_embedding_model,
        )
        self.qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        self.collection = settings.conversation_collection
        self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    def _list_points(self, with_vectors: bool = True, limit: int = 1000):
        points = []
        next_offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection,
                with_vectors=with_vectors,
                limit=min(256, limit - len(points)),
                offset=next_offset,
            )
            points.extend(batch)
            if not next_offset or len(points) >= limit:
                break
        return points[:limit]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _assign_cluster_id(self, user_vec: List[float]) -> int:
        """Assign a cluster id based on nearest centroid among existing clusters.

        If no clusters exist, returns 0. Uses a fixed threshold to join else creates a new cluster.
        """
        points = self._list_points(with_vectors=True, limit=1000)
        # Build existing clusters from points' metadata.cluster_id, else treat each as singleton
        cluster_to_vecs: Dict[int, List[List[float]]] = {}
        max_existing = -1
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            meta = payload.get("metadata") or {}
            cid = meta.get("cluster_id")
            vec = getattr(p, "vector", None)
            if isinstance(vec, dict):
                # named vector - take first
                if vec:
                    vec = list(vec.values())[0]
            if cid is None:
                # treat as its own implicit cluster if vector exists
                if isinstance(vec, list) and vec:
                    cid = (max_existing := max(max_existing, 0))  # ensure non-negative base
                    cluster_to_vecs.setdefault(cid, []).append(vec)
            else:
                max_existing = max(max_existing, int(cid))
                if isinstance(vec, list) and vec:
                    cluster_to_vecs.setdefault(int(cid), []).append(vec)

        if not cluster_to_vecs:
            return 0

        # Compute centroids
        centroids: List[Tuple[int, List[float]]] = []
        for cid, vecs in cluster_to_vecs.items():
            arr = np.asarray(vecs, dtype=np.float32)
            centroids.append((cid, arr.mean(axis=0).tolist()))

        # Find best centroid
        best_cid, best_sim = None, -1.0
        for cid, cen in centroids:
            sim = self._cosine(user_vec, cen)
            if sim > best_sim:
                best_cid, best_sim = cid, sim
        if best_cid is not None and best_sim >= 0.85:
            return int(best_cid)
        return (max_existing + 1) if max_existing >= 0 else 0

    def add_turn(self, user: str, message: str, response: str, session_id: str = "default") -> None:
        # Compute cluster assignment from user message embedding
        user_vec = self.embedding.embed_query(message)
        cluster_id = self._assign_cluster_id(user_vec) if user_vec else 0
        content = f"User({user}): {message}\nAssistant: {response}"
        doc = Document(
            page_content=content,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "user": user,
                "user_message": message,
                "cluster_id": cluster_id,
                "session_id": session_id,
            },
        )
        Qdrant.from_documents(
            [doc],
            self.embedding,
            url=self.qdrant_url,
            collection_name=self.collection,
        )

    def get_context_for_message(self, message: str, session_id: str = "default", k_similar: int = 5, k_recent: int = 3) -> List[str]:
        """Return similar past turns to the current message and a few recent ones, scoped to session."""
        store = Qdrant.from_existing_collection(
            embedding=self.embedding,
            collection_name=self.collection,
            url=self.qdrant_url,
        )
        # Filter by session_id
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        q_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))])
        sim_results = store.similarity_search(message, k=k_similar, filter=q_filter)
        # Qdrant vectorstore doesn't support list-all; approximate recency by reusing similarity with a generic token
        recent_results = store.similarity_search("most recent conversation turns", k=k_recent, filter=q_filter)
        combined = sim_results + [r for r in recent_results if r not in sim_results]
        return [r.page_content for r in combined]

    def build_context_block(self, message: str, session_id: str = "default", k_similar: int = 5, k_recent: int = 3) -> str:
        turns = self.get_context_for_message(message, session_id=session_id, k_similar=k_similar, k_recent=k_recent)
        return "\n\n".join(turns)
