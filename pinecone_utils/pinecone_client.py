import os
from typing import List, Dict, Any, Optional
import pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import logging
from functools import lru_cache
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)


class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX")
        if not self.api_key or not self.index_name:
            raise ValueError(
                "PINECONE_API_KEY and PINECONE_INDEX must be set in environment variables"
            )

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
                    region="us-east-1",
                )
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def query_conflicts(
        self, query_embedding: List[float], filters: Dict[str, Any], k: int = 5
    ) -> List[Any]:
        """Query conflicts with caching and error handling"""
        try:
            index = self.pc.Index(self.index_name)

            # Build Pinecone filter expression
            filter_expressions = []
            for key, value in filters.items():
                if isinstance(value, (str, int, float, bool)):
                    filter_expressions.append(f"{key} == '{value}'")
                elif isinstance(value, list):
                    filter_expressions.append(f"{key} == '{str(value)}'")
                else:
                    filter_expressions.append(f"{key} == '{str(value)}'")

            # Combine filter expressions with AND
            filter_expression = " && ".join(filter_expressions)

            logger.info(f"Querying with filter: {filter_expression}")

            results = index.query(
                vector=query_embedding,
                filter=filter_expression,
                top_k=k,
                include_metadata=True,
            )

            logger.info(f"Found {len(results.matches)} matches")
            return results.matches
        except Exception as e:
            logger.error(f"Error querying conflicts: {str(e)}")
            return []

    def store_decision(
        self, embedding: List[float], metadata: Dict[str, Any], text: str
    ) -> None:
        """Store decision with error handling"""
        try:
            index = self.pc.Index(self.index_name)

            # Ensure all metadata values are strings
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    processed_metadata[key] = str(value)
                elif isinstance(value, list):
                    processed_metadata[key] = str(value)
                else:
                    processed_metadata[key] = str(value)

            # Add text to metadata
            processed_metadata["text"] = text

            # Generate a unique ID
            vector_id = str(hash(f"{text}_{datetime.now().timestamp()}"))

            logger.info(f"Storing vector with metadata: {processed_metadata}")

            index.upsert(
                vectors=[
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": processed_metadata,
                    }
                ]
            )
            logger.info("Successfully stored vector")
        except Exception as e:
            logger.error(f"Error storing decision: {str(e)}")
            raise

    def store_calendar_event(
        self, event_data: Dict[str, Any], embedding: List[float]
    ) -> None:
        """Store a calendar event with its embedding"""
        try:
            index = self.pc.Index(self.index_name)

            # Prepare metadata
            metadata = {
                "user_id": str(event_data.get("user_id", "")),
                "title": str(event_data.get("title", "")),
                "description": str(event_data.get("description", "")),
                "time": str(event_data.get("time", "")),
                "day": str(event_data.get("day", "")),
                "recurring": str(event_data.get("recurring", False)),
                "event_type": str(event_data.get("event_type", "")),
                "action_taken": str(event_data.get("action_taken", "")),
                "timestamp": str(datetime.now().timestamp()),
            }

            # Generate unique ID
            vector_id = f"{metadata['user_id']}_{metadata['timestamp']}"

            # Store in Pinecone
            index.upsert(
                vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}]
            )
            logger.info(
                f"Stored event: {metadata['title']} for user {metadata['user_id']}"
            )
        except Exception as e:
            logger.error(f"Error storing calendar event: {str(e)}")
            raise

    def find_similar_events(
        self,
        query_embedding: List[float],
        user_id: str,
        time: Optional[str] = None,
        day: Optional[str] = None,
        event_type: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar calendar events using vector similarity search"""
        try:
            index = self.pc.Index(self.index_name)

            # Build filter using Pinecone's metadata filter syntax
            # Only filter by user_id and event_type for similarity search
            filter_dict = {"user_id": {"$eq": user_id}}

            if event_type:
                filter_dict["event_type"] = {"$eq": event_type}

            logger.info(f"Searching with filter: {filter_dict}")

            # Query Pinecone with increased top_k to get more candidates
            # We'll filter and sort them after
            results = index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=k * 2,  # Get more candidates than needed
                include_metadata=True,
            )

            # Process and filter results
            similar_events = []
            for match in results.matches:
                # Calculate additional similarity score based on time and day
                time_score = (
                    1.0 if not time or match.metadata.get("time") == time else 0.3
                )
                day_score = 1.0 if not day or match.metadata.get("day") == day else 0.3

                # Combine vector similarity with time/day scores
                combined_score = match.score * (
                    0.7 + 0.15 * time_score + 0.15 * day_score
                )

                similar_events.append(
                    {
                        "score": combined_score,
                        "metadata": match.metadata,
                        "vector_score": match.score,
                        "time_match": time_score,
                        "day_match": day_score,
                    }
                )

            # Sort by combined score and take top k
            similar_events.sort(key=lambda x: x["score"], reverse=True)
            similar_events = similar_events[:k]

            logger.info(f"Found {len(similar_events)} similar events")
            for event in similar_events:
                logger.info(
                    f"Event: {event['metadata'].get('title')}, "
                    f"Score: {event['score']:.2f}, "
                    f"Vector: {event['vector_score']:.2f}, "
                    f"Time: {event['time_match']}, "
                    f"Day: {event['day_match']}"
                )

            return similar_events

        except Exception as e:
            logger.error(f"Error finding similar events: {str(e)}")
            return []

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "embeddings"):
                del self.embeddings
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
