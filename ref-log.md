# Reference Log (ref-log.md)

Date: 2025-10-29

Tools / libraries used
- Streamlit — UI and chat interface
- LangChain — text splitting, Document wrapper, and LangChain helpers
- chromadb / LangChain Chroma integration — vector store
- openai Python SDK — chat completions & embeddings client
- pypdf / PyPDF2 — PDF text extraction
- (optional) sentence-transformers / HuggingFace — local embedding fallback (not required by default)

External documentation consulted
- LangChain docs — text splitters, embeddings, vectorstores
- Chroma / chromadb docs — runtime usage & from_documents API
- OpenAI API docs — models and embedding/chat endpoints
- pypdf documentation — PDF extraction APIs

Code changes (summary)
- chat_with_pdf.py: single-file app implementing ingestion, chunking, embeddings, Chroma vectorstore, retrieval, and chat UI.
  - Added model selection dropdown with Custom option (GEN_MODEL env var supported).
  - Made embedding model configurable via EMBEDDING_MODEL env var (default `openai.text-embedding-3-large`).
  - Removed source-restriction multiselect and automatic filename detection.
  - Implemented deduplication of retrieved chunks, stricter system prompt, and deterministic generation (temperature=0.0).
  - Added "Show retrieved chunks" debug expander for transparency.

GenAI usage
- GitHub Copilot (assistant) was used to:
  - Scaffold and iterate on the single-file Streamlit app.
  - Propose fixes for embedding model selection and error handling for different endpoints.
  - Implement UI improvements (model dropdown, debug retrieval view) and retrieval deduplication.
  - Rationale: accelerate development, generate robust code patterns (error handling, fallback), and help tune prompts to reduce hallucination.
- Human review: all generated code was reviewed and manually adjusted to meet project requirements and security constraints (no API keys committed).

Notes for graders
- The app runs in the provided Codespace devcontainer. Ensure the post-create script `.devcontainer/setup.sh` ran or run it manually.
- If your environment uses a proxy/gateway (LiteLLM, custom endpoint), set OPENAI_BASE or adjust EMBEDDING_MODEL/GEN_MODEL to model ids returned by your endpoint's models.list().

Contact / troubleshooting
- For import errors (chromadb/langchain), run:
  ```bash
  pip install -r requirements.txt chromadb langchain_chroma
  ```
- For embedding model errors, run the small script to list models for your API key and set EMBEDDING_MODEL accordingly.
