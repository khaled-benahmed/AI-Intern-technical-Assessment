"""Document ingestion service: parses CSV, PDF, DOCX, chunks text, embeds and stores in Qdrant via LangChain."""

from typing import List
from pathlib import Path
import io
import pandas as pd
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from app.config import settings


class DocumentIngestionService:
    def __init__(self):
        self.embedding = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _load_pdf(self, file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)

    def _load_docx(self, file_bytes: bytes) -> str:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def _load_csv(self, file_bytes: bytes) -> str:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_csv(index=False)

    def parse_file(self, filename: str, file_bytes: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_bytes)
        if suffix == ".docx":
            return self._load_docx(file_bytes)
        if suffix == ".csv":
            return self._load_csv(file_bytes)
        raise ValueError(f"Unsupported file type: {suffix}")

    def chunk(self, raw_text: str) -> List[Document]:
        return [Document(page_content=chunk) for chunk in self.splitter.split_text(raw_text)]

    def ingest(self, filename: str, file_bytes: bytes) -> int:
        """Parse, chunk, embed and upsert into Qdrant collection. Returns number of chunks."""
        raw_text = self.parse_file(filename, file_bytes)
        docs = self.chunk(raw_text)
        # Initialize or extend vector store
        qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        Qdrant.from_documents(
            docs,
            self.embedding,
            url=qdrant_url,
            collection_name=settings.documents_collection,
        )
        return len(docs)
