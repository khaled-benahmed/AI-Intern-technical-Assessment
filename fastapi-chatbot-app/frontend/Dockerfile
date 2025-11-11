# Frontend Dockerfile: Streamlit UI

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on

WORKDIR /app

COPY frontend/pyproject.toml frontend/ ./

RUN pip install --upgrade pip && \
    pip install streamlit requests

COPY frontend/streamlit_app.py ./

EXPOSE 8501

ENV API_BASE=http://backend:8000/api

ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
