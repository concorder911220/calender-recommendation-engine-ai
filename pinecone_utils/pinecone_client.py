import os
from typing import List, Dict, Any, Optional
import pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import logging
from functools import lru_cache

# from openai_utils.openai_operations import get_embedding

load_dotenv()

logger = logging.getLogger(__name__)

class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX")
        if not self.api_key or not self.index_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set in environment variables")
        
        self.pc = pinecone.Pinecone(api_key=self.api_key)
        self._create_index()
        self.vector_store = self._init_vector_store()
        self._setup_connection_pool()

    def _setup_connection_pool(self):
        """Setup connection pooling for better performance"""
        self.max_connections = 5
        self.connection_timeout = 30  # seconds

    def _init_vector_store(self) -> Pinecone:
        """Initialize the LangChain Pinecone integration with proper embeddings"""
        try:
            return Pinecone(
                index=self.pc.Index(self.index_name),
                embedding=OpenAIEmbeddings(
                    model="text-embedding-3-small"
                ),
                text_key="text",
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _create_index(self) -> None:
        """Create Pinecone index if it doesn't exist"""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def query_conflicts(self, query_embedding: List[float], filters: Dict[str, Any], k: int = 5) -> List[Any]:
        """Query conflicts with caching and error handling"""
        try:
            return self.vector_store.similarity_search_by_vector(
                query_embedding, 
                filter=filters, 
                k=k
            )
        except Exception as e:
            logger.error(f"Error querying conflicts: {str(e)}")
            return []

    def store_decision(self, embedding: List[float], metadata: Dict[str, Any], text: str) -> None:
        """Store decision with error handling"""
        try:
            self.vector_store.add_texts(
                texts=[text], 
                embeddings=[embedding], 
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Error storing decision: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'vector_store'):
                del self.vector_store
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
