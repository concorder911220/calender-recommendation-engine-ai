import os
from datetime import datetime
from typing import Dict, List, Optional
from functools import lru_cache
import logging
from uuid import uuid4
from dotenv import load_dotenv
from pinecone_utils.pinecone_client import PineconeClient
from openai_utils.openai_operations import OpenAIOps, get_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class ConflictResolver:
    def __init__(self, user_id: str = "test_user_001"):
        self.user_id = user_id
        self.pinecone = PineconeClient()
        self.openai = OpenAIOps()
        self._setup_error_handling()

    def _setup_error_handling(self):
        """Setup error handling decorators and retry mechanisms"""
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    @lru_cache(maxsize=100)
    def _generate_mock_conflict(self) -> Dict:
        """Generate a mock conflict with caching"""
        return {
            "title": "Team Sync",
            "description": "Weekly project update",
            "time": "16:00",
            "day": "Tuesday",
            "recurring": True,
        }

    def resolve_conflict(self, new_event_title: str, new_event_desc: str) -> str:
        try:
            # Generate mock conflict
            new_event = {
                "title": new_event_title,
                "description": new_event_desc,
                "time": "16:00",
                "day": "Tuesday",
            }
            conflict_event = self._generate_mock_conflict()

            # Categorize events with error handling
            try:
                new_type = self.openai.categorize_event(
                    new_event["title"], new_event["description"]
                )
                conflict_type = self.openai.categorize_event(
                    conflict_event["title"], conflict_event["description"]
                )
            except Exception as e:
                logger.error(f"Error categorizing events: {str(e)}")
                new_type = "unknown"
                conflict_type = "unknown"

            # Generate embeddings with caching
            new_event_text = f"{new_event['title']}: {new_event['description']}"
            embedding = get_embedding(new_event_text)

            # Query historical decisions with error handling
            try:
                results = self.pinecone.query_conflicts(
                    embedding,
                    filters={
                        "user_id": self.user_id,
                        "new_event_type": new_type,
                        "conflict_type": conflict_type,
                    },
                )
            except Exception as e:
                logger.error(f"Error querying conflicts: {str(e)}")
                results = []

            # Process history
            history = [
                f"{m.metadata['action']} at {m.metadata['timestamp']}" for m in results
            ]
            action_counts = {"reschedule_new": 0, "delete_conflict": 0}
            for doc in results:
                action = doc.metadata.get("action")
                if action in action_counts:
                    action_counts[action] += 1

            total = len(results) or 1
            stats = {
                "reschedule_new": round(action_counts["reschedule_new"] / total * 100, 1),
                "delete_conflict": round(action_counts["delete_conflict"] / total * 100, 1),
            }

            # Generate recommendation with error handling
            try:
                recommendation = self.openai.generate_recommendation(
                    {
                        "user_id": self.user_id,
                        "new_event": f"{new_event['title']} ({new_type})",
                        "conflict_event": f"{conflict_event['title']} ({conflict_type})",
                        "history": "\n- ".join(history),
                        **stats,
                    }
                )
            except Exception as e:
                logger.error(f"Error generating recommendation: {str(e)}")
                recommendation = "Unable to generate recommendation at this time."

            # Store decision with error handling
            try:
                self.pinecone.store_decision(
                    embedding,
                    metadata={
                        "user_id": self.user_id,
                        "new_event_type": new_type,
                        "conflict_type": conflict_type,
                        "action": "delete_conflict",  # Simulate user choice
                        "timestamp": datetime.now().isoformat(),
                    },
                    text=new_event_text,
                )
            except Exception as e:
                logger.error(f"Error storing decision: {str(e)}")

            return recommendation

        except Exception as e:
            logger.error(f"Unexpected error in resolve_conflict: {str(e)}")
            return "An error occurred while processing your request. Please try again later."


if __name__ == "__main__":
    resolver = ConflictResolver()

    print("First run:")
    print(resolver.resolve_conflict("Client Meeting", "Quarterly review with ABC Corp"))

    print("\nSecond run:")
    print(resolver.resolve_conflict("Client Follow-up", "Post-meeting action items"))
