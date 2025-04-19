"""
Ollama embeddings wrapper for LangChain.
"""
import logging
import requests
from typing import List
import os

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class OllamaEmbeddings(Embeddings):
    """
    Wrapper for Ollama embeddings model.
    """
    
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama embeddings model.
        
        Args:
            model: Name of the Ollama model to use
            host: Host URL for the Ollama API
        """
        self.model = model
        self.host = host
        self.embed_url = f"{host}/api/embeddings"
        logger.info(f"Initialized OllamaEmbeddings with model: {model}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} documents")
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        logger.info(f"Generating embedding for query: {text[:50]}{'...' if len(text) > 50 else ''}")
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings from Ollama API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                self.embed_url,
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {str(e)}")
            raise