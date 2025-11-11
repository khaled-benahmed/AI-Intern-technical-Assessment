# Backend Dockerfile: FastAPI + Qdrant client + Gemini integration
# Multi-stage optional; kept single-stage for simplicity.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on

# Install system deps (curl for healthcheck, build tools for some libs, tesseract optional OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first for layer caching
COPY backend/pyproject.toml backend/requirements.txt ./

# Install dependencies (prefer pip for simplicity; could switch to uv later)
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy actual source
COPY backend/app ./app

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

# Environment variables expected at runtime:
# API_KEY=your_gemini_key
# QDRANT_HOST=qdrant
# QDRANT_PORT=6333
# (Optional) GEMINI_MODEL, GEMINI_EMBEDDING_MODEL

ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
