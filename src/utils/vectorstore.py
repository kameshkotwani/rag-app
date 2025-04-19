"""
ChromaDB vectorstore integration for document storage and retrieval.
"""
import logging
from typing import List, Dict, Any
import os
import uuid

import chromadb
from chromadb.config import Settings
from langchain.schema import Document

logger = logging.getLogger(__name__)

def get_chroma_collection(name: str, persist_dir: str):
    """
    Get or create a ChromaDB collection.
    
    Args:
        name: Name of the collection
        persist_dir: Directory to persist the collection
        
    Returns:
        ChromaDB collection
    """
    try:
        logger.info(f"Initializing ChromaDB collection '{name}' in {persist_dir}")
        
        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        collection = client.get_or_create_collection(name)
        logger.info(f"ChromaDB collection '{name}' initialized")
        return collection
    
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        raise

def upsert(collection, documents: List[Document], embeddings: List[List[float]]):
    """
    Insert or update documents in the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        documents: List of Document objects
        embeddings: List of embedding vectors
    """
    try:
        if not documents:
            logger.warning("No documents to upsert")
            return
        
        logger.info(f"Upserting {len(documents)} documents to ChromaDB")
        
        # Prepare data for upsert
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        metadatas = [
            {
                "source": getattr(doc, "metadata", {}).get("source", "unknown"),
                "page": getattr(doc, "metadata", {}).get("page", 0)
            } 
            for doc in documents
        ]
        texts = [doc.page_content for doc in documents]
        
        # Upsert to ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            
        )
        
        logger.info(f"Successfully upserted {len(documents)} documents")
    
    except Exception as e:
        logger.error(f"Error upserting to ChromaDB: {str(e)}")
        raise

def query(collection, embedding: List[float], query_text: str = "", k: int = 5) -> List[Document]:
    """
    Query ChromaDB for similar documents.
    
    Args:
        collection: ChromaDB collection
        embedding: Query embedding vector
        query_text: Original query text (for logging)
        k: Number of documents to return
        
    Returns:
        List of Document objects
    """
    try:
        logger.info(f"Querying ChromaDB for {k} similar documents with query: {query_text[:50]}...")
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )
        
        # Convert results to Document objects
        documents = []
        for i, doc_text in enumerate(results.get("documents", [[]])[0]):

            metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
            documents.append(Document(page_content=doc_text, metadata=metadata))
        
        logger.info(f"Retrieved {len(documents)} documents from ChromaDB")
        return documents
    
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {str(e)}")
        raise