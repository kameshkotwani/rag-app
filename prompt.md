You are an AI scaffolding agent that creates a Retrieval-Augmented Generation (RAG) web app for PDF question-answering. Your output should be a complete project skeleton (folders, files, contents) plus instructions so that a developer can clone it, create a virtual environment, install dependencies, and run the app.

## 1. Environment & Package Management
- Target Python 3.10+.
- Create and activate an isolated virtual environment in project-root/venv:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Use pip; generate a top-level requirements.txt with **pinned versions** (e.g. `langchain==0.0.XX`, `chromadb==0.3.34`, `ollama==0.1.5`, `streamlit==1.20.0`, `python-dotenv==0.21.0`).
- Create a `.gitignore` in project root to exclude:
  ```
  venv/
  __pycache__/
  .env
  *.pyc
  ```

## 2. Directory Structure
Project root should look like:
```
project-root/
├── venv/                   # python -m venv venv
├── .gitignore
├── requirements.txt        # pinned deps
├── src/
│   ├── main.py             # Streamlit entrypoint
│   └── utils/
│       ├── pdf_parser.py   # LangChain PDF loader, splitter
│       ├── embeddings.py   # Ollama embeddings wrapper
│       └── vectorstore.py  # ChromaDB ingestion & query
├── tests/                  # pytest placeholder
│   └── test_pdf_parser.py
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions to run pytest
└── README.md               # Setup, usage, run instructions, screenshots
```

## 3. Dependencies
In `requirements.txt` include pinned versions:
```
langchain
chromadb
ollama
streamlit
python-dotenv
pytest
```

## 4. Module Responsibilities

### utils/pdf_parser.py
- Use `langchain.document_loaders.PyPDFLoader` or `UnstructuredPDFLoader`.
- Implement `load_and_split(pdf_path: str) -> List[Document]`:
  1. Load the PDF.
  2. Chunk it using a `RecursiveCharacterTextSplitter` with defaults:
     ```python
     from langchain.text_splitter import RecursiveCharacterTextSplitter

     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
     docs = splitter.split_documents(raw_docs)
     ```
  3. Return list of `Document` objects.

### utils/embeddings.py
- Wrap Ollama’s local model in a LangChain embeddings class:
  ```python
  from langchain.embeddings.base import Embeddings

  class OllamaEmbeddings(Embeddings):
      def __init__(self, model: str, host: str = "http://localhost:11434"):
          ...
      def embed_documents(self, texts: List[str]) -> List[List[float]]:
          ...
      def embed_query(self, text: str) -> List[float]:
          ...
  ```

### utils/vectorstore.py
- Initialize a Chroma vectorstore:
  ```python
  from chromadb import Client

  def get_chroma_collection(name: str, persist_dir: str):
      client = Client(persist_directory=persist_dir)
      return client.get_or_create_collection(name)
  ```
- Provide:
  - `upsert(documents: List[Document], embeddings: List[List[float]])`  
  - `query(embedding: List[float], k: int = 5) -> List[Document]`

## 5. Streamlit App (src/main.py)
- Load `.env` for configuration with `python-dotenv`:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  import os
  OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ggml-ollama-7b")
  CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
  ```
- Display a `st.file_uploader("Upload PDF")`.
- On upload:
  1. Save file to a temp folder.
  2. `docs = load_and_split(uploaded_path)`.
  3. `embs = OllamaEmbeddings(model=OLLAMA_MODEL).embed_documents([d.page_content for d in docs])`.
  4. `col = get_chroma_collection("pdf_qa", CHROMA_PERSIST_DIR); col.upsert(docs, embs)`.
- Chat interface:
  - `query = st.text_input("Ask a question")`.
  - On submit:
    1. Validate non-empty.
    2. `q_emb = OllamaEmbeddings(model=OLLAMA_MODEL).embed_query(query)`.
    3. `results = col.query(q_emb, k=5)`.
    4. Use `LLMChain` with an Ollama LLM to generate an answer over `results`.
    5. Display with `st.chat_message()` streaming; handle exceptions with `st.error()`.

## 6. Error Handling & Edge Cases
- Use `st.warning()` or `st.error()` for missing upload/query.
- Wrap calls to Ollama and Chroma in `try/except`, log errors via Python `logging`.

## 7. Testing & CI
- Create `tests/test_pdf_parser.py` with a basic pytest that asserts `load_and_split` returns non-zero chunks for a sample PDF.
- Add `.github/workflows/ci.yml` to run `pytest` on each push.

## 8. README.md
- Setup instructions:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- `.env` variables and defaults.
- How to run:
  ```bash
  streamlit run src/main.py
  ```
- Usage examples and sample queries.
- (Optional) Include screenshots or GIFs of the app flow.

## 9. Code Quality
- Use type hints and docstrings.
- Python logging for debug/info.
- Modular, importable functions.
- Clear directory and file comments.