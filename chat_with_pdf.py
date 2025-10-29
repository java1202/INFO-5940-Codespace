import os
import streamlit as st
from typing import List
from io import BytesIO

# OpenAI client for generation (keeps your existing pattern)
from openai import OpenAI

# Try safe imports for LangChain components (match Codespace template)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        raise

# Embeddings + Chroma - try notebook-style imports first (match provided template)
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
except Exception:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
    except Exception:
        raise

# For PDF parsing
try:
    from pypdf import PdfReader
except Exception:
    # fallback import name in some installs
    from PyPDF2 import PdfReader

# Basic Document wrapper used with Chroma.from_documents
from langchain.schema import Document

# --- Configuration / Client setup ---
OPENAI_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("Missing OpenAI API key. Set API_KEY or OPENAI_API_KEY in environment.")
    st.stop()

# If other parts of stack expect OPENAI_API_KEY, set it
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# If you rely on a custom base url (example in your file), set below or rely on env:
OPENAI_BASE = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE")  # optional

client_kwargs = {"api_key": OPENAI_KEY}
if OPENAI_BASE:
    client_kwargs["base_url"] = OPENAI_BASE

client = OpenAI(**client_kwargs)

st.set_page_config(page_title="RAG File Q&A", layout="wide")
st.title("📝 Multi-file RAG — Upload files and chat")

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (chars)", value=1000, min_value=200, max_value=5000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=200, min_value=0, max_value=chunk_size // 2, step=50)
    top_k = st.slider("Retrieval top-k", 1, 10, 4)

    # Model selection: dropdown from common proxy-model ids, with a Custom fallback
    model_options = [
        "openai.gpt-4o",
        "openai.gpt-4.1",
        "openai.gpt-4o-mini",
        "openai.gpt-5-mini",
        "openai.o1-mini.2024-12-17",
        "Custom"
    ]
    model_choice = st.selectbox("Generation model", options=model_options, index=0)
    if model_choice == "Custom":
        model_name = st.text_input("Custom model id", value=os.environ.get("GEN_MODEL", "openai.gpt-4o"))
    else:
        model_name = model_choice

    # Debug / safety controls
    show_retrieved = st.checkbox("Show retrieved chunks with answers (debug)", value=False)

    
# --- File upload (multiple files) ---
uploaded_files = st.file_uploader("Upload one or more files (.txt, .md, .pdf)", type=["txt", "md", "pdf"], accept_multiple_files=True)

# Initialize session state containers
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload files and ask questions about them."}]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []  # raw documents and metadata

def extract_text_from_pdf(file_obj: BytesIO) -> str:
    reader = PdfReader(file_obj)
    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""
        if txt:
            texts.append(txt)
    return "\n".join(texts)

def load_and_chunk_files(files) -> List[Document]:
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for f in files:
        name = f.name
        try:
            raw = None
            if name.lower().endswith(".pdf"):
                # streamlit UploadedFile is file-like; ensure pointer at start
                raw_bytes = f.read()
                text = extract_text_from_pdf(BytesIO(raw_bytes))
            else:
                # txt / md
                raw_bytes = f.read()
                try:
                    text = raw_bytes.decode("utf-8")
                except Exception:
                    text = raw_bytes.decode("latin-1")
            # split into chunks
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            for i, c in enumerate(chunks):
                meta = {"source": name, "chunk": i}
                docs.append(Document(page_content=c, metadata=meta))
        finally:
            try:
                f.seek(0)
            except Exception:
                pass
    return docs

def build_vectorstore(docs: List[Document]):
    if not docs:
        return None
    # Use full embedding model id from env or fallback to 'openai.text-embedding-3-large'
    emb_model = os.environ.get("EMBEDDING_MODEL", "openai.text-embedding-3-large")
    embeddings = OpenAIEmbeddings(model=emb_model)
    # create an in-memory Chroma vectorstore
    vect = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vect

# When files uploaded, process and build/update vector store
if uploaded_files:
    with st.spinner("Ingesting files and building vector store..."):
        docs = load_and_chunk_files(uploaded_files)
        if not docs:
            st.warning("No text extracted from uploaded files.")
        else:
            vs = build_vectorstore(docs)
            st.session_state.vectorstore = vs
            st.session_state.documents = docs
            st.success(f"Ingested {len(uploaded_files)} file(s), created {len(docs)} chunks.")

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input (disabled when no vectorstore)
query = st.chat_input("Ask a question about the uploaded documents", disabled=(st.session_state.vectorstore is None))

def retrieve_and_answer(question: str):
    vs = st.session_state.vectorstore
    if not vs:
        return {"content": "No documents indexed.", "sources": [], "retrieved": []}
    candidate_k = max(top_k * 5, 10)
    try:
        raw_hits = vs.similarity_search(question, k=candidate_k)
    except Exception:
        raw_hits = vs.similarity_search(question, k=top_k)

    # No explicit source restriction — use top candidates and dedupe
    filtered = raw_hits

    seen = set()
    hits = []
    for h in filtered:
        key = (h.metadata.get("source"), str(h.metadata.get("chunk")), h.page_content[:200])
        if key in seen:
            continue
        seen.add(key)
        hits.append(h)
        if len(hits) >= top_k:
            break

    if not hits:
        seen = set()
        for h in raw_hits:
            key = (h.metadata.get("source"), str(h.metadata.get("chunk")), h.page_content[:200])
            if key in seen:
                continue
            seen.add(key)
            hits.append(h)
            if len(hits) >= top_k:
                break

    context_parts = []
    sources = []
    retrieved = []
    for h in hits:
        src = h.metadata.get("source", "(unknown)")
        chunk_id = h.metadata.get("chunk", "")
        context_parts.append(f"[{src} | chunk {chunk_id}]\n{h.page_content}")
        sources.append(src)
        snippet = h.page_content.replace("\n", " ")
        retrieved.append({"source": src, "chunk": chunk_id, "snippet": snippet[:800], "full": h.page_content})

    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a factual assistant. Use ONLY the provided context to answer. "
        "If the answer is not present in the context, respond exactly: 'I don't know.' "
        "Always include source citations in brackets like [source | chunk]. "
        "Keep answers concise (<=3 sentences)."
    )

    messages = [
        {"role": "system", "content": system + "\n\nContext:\n" + context},
        {"role": "user", "content": question}
    ]

    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0, max_tokens=500)

    content = ""
    try:
        content = resp.choices[0].message.get("content")
    except Exception:
        try:
            content = resp.choices[0].message.content
        except Exception:
            try:
                content = resp.choices[0].text
            except Exception:
                content = str(resp)

    unique_sources = list(dict.fromkeys(sources))

    return {"content": content.strip() if isinstance(content, str) else str(content),
            "sources": unique_sources,
            "retrieved": retrieved}

if query:
    # append user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            reply = {"content": "No documents indexed. Upload files first.", "sources": [], "retrieved": []}
        else:
            with st.spinner("Retrieving context and generating answer..."):
                reply = retrieve_and_answer(query)

        # Render response in a clean message format
        # Main assistant text
        st.markdown(reply["content"])

        # Sources (compact)
        if reply["sources"]:
            with st.expander("Sources"):
                for s in reply["sources"]:
                    st.write(f"- {s}")

        # Retrieved chunks (debug / transparency)
        if show_retrieved and reply["retrieved"]:
            with st.expander("Retrieved chunks (click to expand)"):
                for i, r in enumerate(reply["retrieved"], 1):
                    st.markdown(f"**[{i}] {r['source']} | chunk {r['chunk']}**")
                    st.write(r["snippet"])
                    st.markdown("---")

    # store assistant visible text in message history
    st.session_state.messages.append({"role": "assistant", "content": reply["content"]})