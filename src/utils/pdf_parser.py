"""
PDF Parser module using LangChain for loading and splitting PDF documents.
"""
import logging
from typing import List

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def load_and_split(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and split it into chunks.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of Document objects containing chunks of the PDF.
    """
    try:
        logger.info(f"Loading PDF from {pdf_path}")
        # Use PyPDFLoader as the primary loader
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        
        logger.info(f"Loaded {len(raw_docs)} pages from PDF")
        
        # Split the documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        docs = splitter.split_documents(raw_docs)
        
        logger.info(f"Split PDF into {len(docs)} chunks")
        return docs
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise