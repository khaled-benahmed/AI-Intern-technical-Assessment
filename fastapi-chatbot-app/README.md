# FastAPI + LangChain + Gemini (Qdrant) Chatbot

AI-powered document chatbot with conversation memory. Backend: FastAPI. Orchestration: LangChain (Gemini via google-genai). Vector DB: Qdrant. .

## Features

- Upload CSV, PDF, DOCX documents
- Chunk + embed with Google Gemini embeddings; store in Qdrant
- Retrieval-Augmented Generation (RAG) using document chunks + conversation memory
- Conversation memory stored as embedded turns to recall recurring topics

## Project Structure

```
fastapi-chatbot-app/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app bootstrap + CORS
│   │   ├── config.py                # Settings (API key, Qdrant, models)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py            # /upload, /chat, /search endpoints
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── gemini_service.py    # google-genai client wrapper
│   │       ├── qdrant_service.py    # Raw Qdrant helper (non-LangChain paths)
│   │       ├── document_ingestion.py# Parse/chunk/embed documents via LangChain
│   │       ├── memory.py            # Conversation memory (vector-based)
│   │       └── chain.py             # LangChain RAG + memory pipeline
│   ├── pyproject.toml               # Dependencies (uv managed)
│   └── requirements.txt             # Optional list
├── docker-compose.yml               # Qdrant + backend (Dockerfile expected)
├── .env.example                     # Example environment variables
└── README.md
```

## Environment Variables

Create `.env` based on `.env.example`:

| Variable | Purpose | Default |
|----------|---------|---------|
| API_KEY | Google Gemini API key | (none) |
| QDRANT_HOST | Qdrant host | localhost |
| QDRANT_PORT | Qdrant port | 6333 |
| GEMINI_MODEL | Text model | gemini-2.5-flash |
| GEMINI_EMBEDDING_MODEL | Embedding model | text-embedding-004 |

## API Endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | /upload | multipart file | Ingest & index document chunks |
| POST | /chat | {"message": "..."} | Answer using RAG + memory |
| POST | /search | {"message": "..."} | Raw semantic search (messages) |

## Backend (uv)

```pwsh
cd backend
uv venv
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

## Roadmap

1. Align Streamlit endpoint payload -> `/chat` {"message": ...}
2. Add pattern recognition endpoint (/patterns)
3. Add unit tests for upload + chat + search
4. CI pipeline (GitHub Actions) & deployment docs

## License

MIT License (see LICENSE file).