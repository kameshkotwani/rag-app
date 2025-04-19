"""
Main Streamlit application for PDF question-answering with RAG.
"""
import os
import logging
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")

# Import utilities
from utils.pdf_parser import load_and_split
from utils.embeddings import OllamaEmbeddings
# Rename imported query function to avoid naming conflict
from utils.vectorstore import get_chroma_collection, upsert, query as vector_query

# Import LangChain components for the QA chain
from langchain_ollama import OllamaLLM  # Updated import
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="wide"
)

# Helper function to format context for the LLM
def format_context(docs):
    # Enhanced context format with document source information
    formatted_content = []
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        formatted_content.append(f"Document {i+1} [Page {page_num}]:\n{doc.page_content}")
    
    return "\n\n" + "\n\n".join(formatted_content)

# Create LLMChain for question answering
def get_qa_chain():
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
    
    template = """
    You are a helpful AI assistant answering questions about documents.
    
    CONVERSATION HISTORY:
    {chat_history}
    
    CONTEXT FROM DOCUMENTS:
    {context}
    
    CURRENT QUESTION: {question}
    
    IMPORTANT INSTRUCTIONS:
    1. Base your answer on the provided context and conversation history
    2. If the question refers to previous questions or answers in the conversation, make sure to address those references
    4. If asked to generate questions from an exercise, look for content labeled as "Exercise" in the context
    5. If you're asked to answer questions you previously generated, refer back to exactly those questions and provide answers
    6. If the context doesn't contain the specific exercise mentioned, state that clearly
    7. DO NOT HALLUCINATE CONTENT. If you can't find information about a specific exercise, say so directly
    8. Be concise but thorough in your answers
    
    YOUR ANSWER:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    
    return LLMChain(llm=llm, prompt=prompt)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = get_chroma_collection(
        "pdf_qa", 
        CHROMA_PERSIST_DIR
    )
    
# Track processed files to avoid reprocessing
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Store generated questions for better context and consistency
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []

# Streamlit UI
st.title("📚 PDF Question Answering with RAG")

# Sidebar with PDF upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload a PDF document to ask questions about."
    )
    
    if uploaded_file is not None:
        # Create a unique identifier for the file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if this file has already been processed
        if file_id not in st.session_state.processed_files:
            with st.spinner("Processing PDF..."):
                try:
                    # Save the uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name
                    
                    # Process the PDF
                    docs = load_and_split(temp_path)
                    
                    if not docs:
                        st.error("No text could be extracted from the PDF.")
                    else:
                        # Generate embeddings
                        embeddings_model = OllamaEmbeddings(model=OLLAMA_MODEL)
                        embeddings = embeddings_model.embed_documents([doc.page_content for doc in docs])
                        
                        # Store in ChromaDB
                        upsert(st.session_state.collection, docs, embeddings)
                        
                        # Mark this file as processed
                        st.session_state.processed_files.add(file_id)
                        
                        st.success(f"PDF processed and indexed: {len(docs)} chunks extracted")
                        
                        # Clean up the temp file
                        os.unlink(temp_path)
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        else:
            st.success("This PDF has already been processed and indexed.")

# Chat interface
st.header("Ask Questions")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        
        try:
            # Generate embedding for the query
            embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)
            query_embedding = embedding_model.embed_query(user_query)
            
            # Retrieve relevant documents - using renamed vector_query function
            with st.spinner("Searching documents..."):
                # Check if the query is specifically about exercises
                if "exercise" in user_query.lower():
                    # Use a larger k for exercise-related queries to ensure we get the relevant sections
                    k_value = 8
                else:
                    k_value = 5
                    
                relevant_docs = vector_query(st.session_state.collection, query_embedding, query_text=user_query, k=k_value)
            
            if not relevant_docs:
                response = "I couldn't find any relevant information in the uploaded documents. Please try a different question or upload a document first."
            else:
                # Format context
                context = format_context(relevant_docs)
                
                # Get QA chain
                qa_chain = get_qa_chain()
                
                # Generate response with streaming
                response_placeholder = st.empty()
                
                with st.spinner("Generating answer..."):
                    # Format chat history for the model - improved format
                    chat_history = ""
                    
                    # Only include message pairs before the current question
                    # This ensures we don't have an incomplete pair
                    message_pairs = len(st.session_state.messages) - 1
                    
                    for i in range(0, message_pairs, 2):
                        if i+1 < len(st.session_state.messages):
                            user_msg = st.session_state.messages[i]['content']
                            ai_msg = st.session_state.messages[i+1]['content']
                            chat_history += f"USER: {user_msg}\n\nASSISTANT: {ai_msg}\n\n---\n\n"
                    
                    # Updated: Using invoke method instead of run for LangChain 0.3.x
                    chain_response = qa_chain.invoke({
                        "context": context, 
                        "question": user_query, 
                        "chat_history": chat_history or "No previous conversation."
                    })
                    
                    # Extract the text response from the chain output
                    if isinstance(chain_response, dict):
                        response = chain_response.get("text", str(chain_response))
                    else:
                        response = str(chain_response)
                    
                    # Track questions if the response appears to be generating questions
                    if any(phrase in user_query.lower() for phrase in ["give me questions", "generate questions", "list questions", "create questions"]):
                        # Look for questions in the response
                        question_lines = []
                        for line in response.split('\n'):
                            # Look for numbered questions (1., 2., etc.) or question marks
                            if (any(f"{i}." in line for i in range(1, 10)) or '?' in line) and len(line.strip()) > 10:
                                question_lines.append(line.strip())
                        
                        if question_lines:
                            # Store the generated questions
                            st.session_state.generated_questions = question_lines
                            logger.info(f"Stored {len(question_lines)} generated questions")
                    
                    # Check if this is a request for answers to previous questions
                    if any(phrase in user_query.lower() for phrase in ["give me answers", "provide answers", "answer these", "solve these", "solution"]):
                        # If we have stored questions, explicitly include them in the response
                        if st.session_state.generated_questions:
                            questions_text = "\n\n".join(f"Question {i+1}: {q}" for i, q in enumerate(st.session_state.generated_questions))
                            response = f"I'll answer the questions I previously generated:\n\n{questions_text}\n\n**Answers:**\n\n{response}"
                        
                    response_container.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            response_container.error(error_msg)
            logger.error(error_msg, exc_info=True)

# Empty storage notice
if not st.session_state.messages:
    st.info("👋 Upload a PDF document and ask questions about its content!")

if __name__ == "__main__":
    logger.info("Application started")