"""Custom LangChain-compatible embeddings using the official google-genai client.

Avoids protobuf marshalling issues seen with langchain_google_genai. 
"""
from typing import List
from langchain_core.embeddings import Embeddings
from google import genai


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.models.embed_content(model=self.model, contents=text)
        return resp.embeddings[0].values if resp.embeddings else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # google-genai supports batching in a single call with contents=list[str]
        resp = self.client.models.embed_content(model=self.model, contents=texts)
        return [e.values for e in resp.embeddings]
