import os
import streamlit as st
import requests
from typing import Optional

st.set_page_config(page_title="Doc Chatbot", page_icon="💬", layout="centered")

# Prefer environment variable; fall back to local backend. Avoid st.secrets to prevent crash when no secrets.toml exists.
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

st.title("📄 Document Chatbot (FastAPI + LangChain + Qdrant + Gemini)")

# Sidebar for backend URL config
with st.sidebar:
    st.subheader("Backend Settings")
    api_base = st.text_input("API base URL", API_BASE)
    if api_base != API_BASE:
        API_BASE = api_base

# Tabs for Upload and Chat
upload_tab, chat_tab, search_tab = st.tabs(["Upload", "Chat", "Search"]) 

with upload_tab:
    st.header("Upload a document")
    file = st.file_uploader("Choose a CSV, PDF, or DOCX", type=["csv", "pdf", "docx"])
    if st.button("Upload", type="primary"):
        if not file:
            st.warning("Please select a file first.")
        else:
            try:
                files = {"file": (file.name, file.getvalue(), file.type)}
                resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
                if resp.ok:
                    data = resp.json()
                    st.success(f"Uploaded and indexed {data.get('chunks_indexed', 0)} chunks.")
                else:
                    st.error(f"Upload failed: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

with chat_tab:
    st.header("Chat")
    user_msg = st.text_input("Ask a question about your documents")
    if st.button("Send"):
        if not user_msg:
            st.warning("Please enter a message.")
        else:
            try:
                resp = requests.post(f"{API_BASE}/chat", json={"message": user_msg}, timeout=60)
                if resp.ok:
                    st.write(resp.json().get("response", ""))
                else:
                    st.error(f"Chat failed: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

with search_tab:
    st.header("Semantic Search (messages)")
    query = st.text_input("Enter a search query")
    if st.button("Search"):
        if not query:
            st.warning("Please enter a query.")
        else:
            try:
                resp = requests.post(f"{API_BASE}/search", json={"message": query}, timeout=60)
                if resp.ok:
                    matches = resp.json().get("matches", [])
                    for m in matches:
                        st.write(f"Score: {m.get('score')}\nPayload: {m.get('payload')}")
                else:
                    st.error(f"Search failed: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")
