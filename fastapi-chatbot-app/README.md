## FastAPI + LangChain + Gemini (Qdrant) Chatbot — Short Docs

This project provides a small RAG chatbot that: ingests documents (PDF/DOCX/CSV),
indexes text and images (OCR), stores vectors in Qdrant, and answers
questions using Google Gemini with conversation memory.

Quick layout
- `backend/app/main.py` — FastAPI app; startup hooks
- `backend/app/api/routes.py` — endpoints: `/chat`, `/search`, `/upload`, `/topics`, `/health`
- `backend/app/services/` — core logic: embeddings, qdrant helper, ingestion, memory, chain

Environment (minimum)
- `API_KEY` — Google Gemini API key (required)
- `QDRANT_HOST` (default: `localhost`)
- `QDRANT_PORT` (default: `6333`)
- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `GEMINI_EMBEDDING_MODEL` (default: `text-embedding-004`)

How to run (dev)
From the `backend` folder:
```pwsh
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API reference (concise)
- POST /api/chat
	- Body: { "message": "...", "session_id": "optional" }
	- Response: { "response": "<assistant text>" }
	- Notes: Runs RAG (documents + conversation memory); saves the turn to `conversation_history`.

- POST /api/search
	- Body: { "message": "...", "session_id": "optional" }
	- Response: { "matches": [{ "score": <float>, "payload": <object> }, ...] }
	- Notes: Performs semantic search (Qdrant) scoped by `session_id` when provided.

- POST /api/upload
	- Body: multipart/form-data `file` (pdf, docx, csv)
	- Response: { "status": "ok", "chunks_indexed": <int> }
	- Notes: Parses, (OCR images if available), chunks, embeds, and indexes into `documents` collection.

- POST /api/topics
	- Body: { "session_id": "optional" }
	- Response: { "topics": [ {"cluster_id": <int>, "turn_count": <int>, "last_timestamp": "...", "example_message": "..."}, ... ] }
	- Notes: Aggregates lightweight cluster/topic info from conversation history.

- GET /api/health
	- Response: { "status": "ok" }
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend (optional)

```pwsh
cd frontend
uv venv
uv sync
uv run streamlit run streamlit_app.py
```

## Docker (optional)

```pwsh
docker-compose up --build
```

## Development Notes

- LangChain LCEL pipeline in `chain.py` combines document + memory retrieval.
- `memory.py` stores each turn as an embedded document in Qdrant.
- Pattern recognition (recurring topic detection) pending future implementation.

## Challenges & Key actions & Trade-offs

Key actions
- Config: centralized settings via `config.py` (Pydantic).
- Embeddings: added `services/embeddings.py` wrapper for Gemini.
- Ingestion: parse PDF/DOCX/CSV, optional OCR, chunk, embed, and index into Qdrant.
- Memory: store each conversation turn with metadata (timestamp, session_id, cluster_id).
- Chain: small LangChain pipeline combining document + conversation retrievers and Gemini LLM.

Challenges and how they were handled
- Qdrant unavailable at startup — startup probes are non-blocking so the API can start.
- Embedding dimension detection — probe an embedding call and fallback to 768 if needed.
- Optional OCR dependencies — OCR runs only if Pillow/pytesseract/PyMuPDF are present; otherwise skip.
- Topic clustering cost — used a cheap online centroid assignment to avoid heavy offline clustering.
- Latency/blocking — endpoints are synchronous; recommend background tasks or async for production.

Trade-offs (short)
- Simplicity vs. scalability: sync endpoints and inline ingestion are simple but not production-ready.
- Online vs offline clustering: online centroid is cheap but can fragment topics; periodic reclustering will help.
- Single Qdrant for docs+memory: session_id scoping is used; separate collections or access controls are better for strict isolation.

## Roadmap

1. Align Streamlit endpoint payload -> `/chat` {"message": ...}
2. Add pattern recognition endpoint (/patterns)
3. Add unit tests for upload + chat + search
