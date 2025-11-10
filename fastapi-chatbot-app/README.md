<div align="center">

# Document Chatbot (FastAPI · LangChain · Gemini · Qdrant)

Talk to your documents. Ask follow‑ups. The app remembers what you discussed. 📄💬

</div>

## 1. What this project does

You can upload CSV, PDF, or DOCX files. The backend parses them, breaks the content into manageable chunks, creates vector embeddings with Google Gemini, and stores everything in Qdrant so it can quickly pull back the most relevant pieces when you ask a question.

It also keeps a lightweight memory of the conversation by embedding past turns. That helps provide continuity instead of treating every question in isolation.

### Enhanced ingestion
PDFs and Word documents aren’t just text: they can have images of charts or scanned pages. The ingestion pipeline now tries to OCR images (when Pillow + Tesseract + PyMuPDF are available) and merges the extracted text with the regular page or paragraph content. If OCR isn’t available, it just skips that part—no crashes.

## 2. Current feature set

| Area | Status |
|------|--------|
| Upload CSV / PDF / DOCX | ✅ |
| Text chunking & embeddings | ✅ Gemini embeddings via official SDK |
| Vector store | ✅ Qdrant (collections auto‑created) |
| RAG answer endpoint | ✅ Combines document + conversation memory |
| Conversation memory | ✅ Stores turns as embedded docs |
| Image/graph OCR | ✅ Best effort (optional dependencies) |
| Semantic search endpoint | ✅ `/search` over conversation vectors |
| Streamlit frontend | ✅ Upload, Chat, Search tabs |
| Pattern/topic clustering | ⏳ Planned |
| Tests & CI | ⏳ Planned |
| Docker setup | ⏳ Planned |

## 3. Project layout (quick tour)

```
backend/app/main.py            # FastAPI bootstrap + CORS + startup collection checks
backend/app/api/routes.py      # /upload, /chat, /search, /health
backend/app/services/          # Embeddings, ingestion (with OCR), memory, RAG chain, Qdrant helper
frontend/streamlit_app.py      # Simple UI (Upload / Chat / Search)
```

## 4. Configuration

Provide an `.env` file (see `.env.example`):

| Variable | What it is | Example |
|----------|------------|---------|
| API_KEY | Gemini API key | `AIza...` |
| QDRANT_HOST | Qdrant host | `localhost` |
| QDRANT_PORT | Qdrant port | `6333` |
| GEMINI_MODEL | Chat model | `gemini-2.5-flash` |
| GEMINI_EMBEDDING_MODEL | Embedding model | `text-embedding-004` |

## 5. API overview (manual summary)

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| POST | `/api/upload` | multipart file | Parse + chunk + embed + store document |
| POST | `/api/chat` | `{ "message": "..." }` | Answer using RAG + memory |
| POST | `/api/search` | `{ "message": "..." }` | Similarity search over conversation turns |
| GET | `/api/health` | — | Simple health check |

Responses are JSON. Errors return an HTTP error code with a human‑readable message.

## 6. Running locally

Backend:
```pwsh
cd backend
uv venv
uv sync
# Ensure .env exists with API_KEY
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend (optional):
```pwsh
cd frontend
uv venv
uv sync
uv run streamlit run streamlit_app.py
```

Open the docs: `http://localhost:8000/docs` and the UI: `http://localhost:8501` (default Streamlit port).

## 7. How the answer pipeline works

1. User question hits `/api/chat`.
2. We build a LangChain mini graph: fetch top document chunks + similar past turns.
3. Construct a prompt that blends: document excerpts + conversation snippets.
4. Call Gemini via the official client wrapper (`gemini_service.py`).
5. Return the trimmed answer and store the new turn in vector memory.

Edge cases handled:
* Empty files → rejected early.
* Unsupported extensions → 400 error.
* OCR unavailability → silently skipped; text only still works.
* Qdrant not ready at startup → collections creation errors are swallowed to avoid blocking.

## 8. Limits and next steps

Planned improvements:
* Topic/pattern detection across sessions.
* Proper session IDs (currently default user only).
* Docker + docker-compose for reproducible spin‑up.
* CI (GitHub Actions) with test suite (upload + chat + search + failure cases).
* Deployment guide (e.g., Render, Hugging Face Spaces).

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Backend exits immediately | Missing `API_KEY` | Add to `.env` or shell env |
| Embedding errors | Invalid Gemini key | Regenerate key or check quotas |
| OCR missing text from images | No Tesseract installed | Install Tesseract or ignore |
| Qdrant connection errors | Service not running | Start local Qdrant container |

## 10. Manual authorship note

All narrative documentation here was written manually for clarity. The code integrates AI services, but this README content is human‑crafted.

## License

MIT License – see `LICENSE`.