"""Gemini service wrapper using the official Google GenAI SDK.

Provides text generation and embedding functionality.
"""

from typing import Optional, List
from google import genai


class GeminiService:
    def __init__(self, api_key: str):
        # Official client from google-genai package
        self.client = genai.Client(api_key=api_key)

    def generate_text(self, prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate a text response for a given prompt."""
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )
        # .text is a convenience property (may be empty if response has parts)
        return getattr(response, "text", "")

    def embed(self, text: str, model: str = "text-embedding-004") -> List[float]:
        """Create an embedding vector for the provided text."""
        resp = self.client.models.embed_content(
            model=model,
            contents=text,
        )
        # embeddings returns a list; take first vector values
        return resp.embeddings[0].values if resp.embeddings else []

    def batch_embed(self, texts: List[str], model: str = "text-embedding-004") -> List[List[float]]:
        resp = self.client.models.embed_content(
            model=model,
            contents=texts,
        )
        return [e.values for e in resp.embeddings]
