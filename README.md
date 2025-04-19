# PDF Question Answering with RAG

This application is completely made using Agent mode in VSCode, with Claude 3.7 model. The prompt which I used is stored in prompt.md which I generated using ChatGPT ofcourse.

A Retrieval-Augmented Generation (RAG) web application for answering questions about PDF documents using Streamlit, LangChain, ChromaDB, and Ollama.

## Features

- 📁 **PDF Upload**: Upload any PDF document for analysis
- 🔍 **Semantic Search**: Retrieve relevant information using vector similarity
- 💬 **Question Answering**: Ask questions about your documents in natural language
- 🧠 **RAG Architecture**: Combines retrieval with generative AI for accurate answers
- 🔄 **Conversation History**: Keep track of your previous questions and answers

## Technical Stack

- **Frontend**: Streamlit
- **LLM Integration**: LangChain + Ollama
- **Vector Database**: ChromaDB
- **PDF Processing**: PyPDF + Unstructured
- **Embeddings**: Ollama embedding model (gemma3)

## Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/kameshkotwani/rag-app.git
cd rag-app

# Create and activate a virtual environment (recommended)
source .venv/bin/activate  # On Windows: venv\Scripts\activate
uv sync
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```bash
OLLAMA_MODEL=gemma3
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

## Running the Application

1. Start the Ollama server (make sure you have Ollama installed):

   ```bash
   ollama serve
   ```

2. Pull the Llama2 model:

   ```bash
   ollama pull llama2
   ```

3. Start the Streamlit application:

   ```bash
   streamlit run src/main.py
   ```

4. Open your browser and go to [http://localhost:8501](http://localhost:8501)

## Usage

1. Upload a PDF document using the file uploader in the sidebar
2. Wait for the document to be processed and indexed
3. Type a question in the chat input at the bottom of the page
4. View the answer generated based on the content of your document

## Testing

Run the tests with pytest:

```bash
pytest tests/
```

## Project Structure

```bash
.
├── .github/            # GitHub Actions workflows
├── data/               # Data storage
│   └── chroma/         # ChromaDB persistent storage
├── src/                # Source code
│   ├── main.py         # Main Streamlit application
│   └── utils/          # Utility modules
│       ├── embeddings.py  # Ollama embeddings wrapper
│       ├── pdf_parser.py  # PDF loading and splitting
│       └── vectorstore.py # ChromaDB integration
├── tests/              # Test files
├── .env                # Environment variables (create this file)
├── .gitignore          # Git ignore file
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
