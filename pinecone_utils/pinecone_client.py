import os
from typing import List, Dict, Any, Optional
import pinecone
from dotenv import load_dotenv
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
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._setup_connection_pool()

    def _setup_connection_pool(self):
        """Setup connection pooling for better performance"""
        self.max_connections = 5
        self.connection_timeout = 30  # seconds

    def _create_index(self) -> None:
        """Create Pinecone index if it doesn't exist"""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    cloud="aws",
                    region="us-east-1"
                )
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def query_conflicts(self, query_embedding: List[float], filters: Dict[str, Any], k: int = 5) -> List[Any]:
        """Query conflicts with caching and error handling"""
        try:
            index = self.pc.Index(self.index_name)
            results = index.query(
                vector=query_embedding,
                filter=filters,
                top_k=k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Error querying conflicts: {str(e)}")
            return []

    def store_decision(self, embedding: List[float], metadata: Dict[str, Any], text: str) -> None:
        """Store decision with error handling"""
        try:
            index = self.pc.Index(self.index_name)
            index.upsert(
                vectors=[{
                    "id": str(hash(text)),  # Generate a unique ID for the vector
                    "values": embedding,
                    "metadata": {**metadata, "text": text}
                }]
            )
        except Exception as e:
            logger.error(f"Error storing decision: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'embeddings'):
                del self.embeddings
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
