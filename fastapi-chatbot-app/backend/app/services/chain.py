"""RAG chain using LangChain, Gemini model and Qdrant retriever.

- Retrieves relevant chunks from document and conversation collections
- Builds a prompt with context + memory
- Generates an answer using Google Gemini via LangChain LLM wrapper
"""
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.embeddings import GeminiEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_qdrant import Qdrant
from app.config import settings
from app.services.memory import ConversationMemoryService


def build_vectorstores():
    embedding = GeminiEmbeddings(
        api_key=settings.api_key,
        model=settings.gemini_embedding_model,
    )
    qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"

    # Ensure collections exist using raw Qdrant client to avoid lazy initialization issues
    client = QdrantClient(url=qdrant_url)
    dim = len(embedding.embed_query("__init_dim__")) or 768

    def ensure_collection(name: str):
        try:
            existing = client.get_collections().collections
            names = {c.name for c in existing}
            if name not in names:
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception:
            # If any error occurs, attempt recreate
            try:
                client.recreate_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            except Exception:
                pass

    ensure_collection(settings.documents_collection)
    ensure_collection(settings.conversation_collection)

    docs_store = Qdrant.from_existing_collection(
        embedding=embedding,
        url=qdrant_url,
        collection_name=settings.documents_collection,
    )
    convo_store = Qdrant.from_existing_collection(
        embedding=embedding,
        url=qdrant_url,
        collection_name=settings.conversation_collection,
    )
    return docs_store, convo_store


def build_chain():
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        api_key=settings.api_key,
        temperature=0.3,
    )

    docs_store, convo_store = build_vectorstores()
    memory = ConversationMemoryService()

    # Retrievers
    docs_retriever = docs_store.as_retriever(search_kwargs={"k": 5})
    convo_retriever = convo_store.as_retriever(search_kwargs={"k": 5})

    # Compose inputs
    def format_context(docs: List, convos: List, recent: str):
        doc_block = "\n\n".join([d.page_content for d in docs]) if docs else ""
        convo_block = "\n\n".join([c.page_content for c in convos]) if convos else ""
        blocks = [
            ("Documents:", doc_block),
            ("Conversation memory:", convo_block if convo_block else recent),
        ]
        return "\n\n".join([f"{title}\n{content}" for title, content in blocks if content])

    template = (
        "You are a helpful assistant. Use the provided context to answer the user's question.\n"
        "If the answer is not in the context, say you don't know. Be concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    )

    # Build a small LCEL pipeline
    def memory_block(q: str) -> str:
        return memory.build_context_block(message=q)

    gather = RunnableParallel(
        question=lambda x: x["question"],
        docs=lambda x: docs_retriever.invoke(x["question"]),
        convos=lambda x: convo_retriever.invoke(x["question"]),
        recent=lambda x: memory_block(x["question"]),
    )

    chain = (
        {"question": (lambda x: x)}  # input is the question string
        | gather
        | (lambda x: {"question": x["question"], "context": format_context(x["docs"], x["convos"], x["recent"])})
        | (lambda x: template.format(**x))
        | llm
        | StrOutputParser()
    )

    return chain, memory
