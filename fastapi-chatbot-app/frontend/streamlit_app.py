import os
import streamlit as st
import requests
from typing import Optional
import uuid

st.set_page_config(page_title="Doc Chatbot", page_icon="💬", layout="centered")

# Prefer environment variable; fall back to local backend. Avoid st.secrets to prevent crash when no secrets.toml exists.
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

st.title("📄 Document Chatbot (FastAPI + LangChain + Qdrant + Gemini)")

# Session state: persistent session_id and chat messages
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role: 'user'|'assistant', 'content': str}

# Sidebar for backend URL config
with st.sidebar:
    st.subheader("Backend Settings")
    api_base = st.text_input("API base URL", API_BASE)
    if api_base != API_BASE:
        API_BASE = api_base
    st.markdown("---")
    st.subheader("Session")
    col1, col2 = st.columns([3,1])
    with col1:
        st.text_input("Session ID", value=st.session_state.session_id, key="_sid_edit")
    with col2:
        if st.button("New", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state._sid_edit = st.session_state.session_id
            st.session_state.messages = []

# Tabs for Upload and Chat
upload_tab, chat_tab, search_tab, topics_tab = st.tabs(["Upload", "Chat", "Search", "Topics"]) 

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
    # Display existing messages as a messenger-like conversation
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Type your message"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            payload = {"message": prompt, "session_id": st.session_state.session_id}
            resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=90)
            if resp.ok:
                answer = resp.json().get("response", "")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
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
                payload = {"message": query, "session_id": st.session_state.session_id}
                resp = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
                if resp.ok:
                    matches = resp.json().get("matches", [])
                    for m in matches:
                        st.write(f"Score: {m.get('score')}\nPayload: {m.get('payload')}")
                else:
                    st.error(f"Search failed: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

with topics_tab:
    st.header("Topics (per session)")
    if st.button("Refresh topics"):
        try:
            resp = requests.post(f"{API_BASE}/topics", json={"session_id": st.session_state.session_id}, timeout=30)
            if resp.ok:
                topics = resp.json().get("topics", [])
                if not topics:
                    st.info("No topics yet. Chat a bit first!")
                else:
                    for t in topics:
                        st.write(f"• Topic {t.get('cluster_id')}: {t.get('turn_count')} turns | last: {t.get('last_timestamp')}\nExample: {t.get('example_message')}")
            else:
                st.error(f"Failed to load topics: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")
