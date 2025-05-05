from typing import Dict
from functools import lru_cache
import logging
from dotenv import load_dotenv
from pinecone_utils.pinecone_client import PineconeClient
from openai_utils.openai_operations import OpenAIOps, get_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
            "title": "Client intro call",
            "description": "Meeting with Wilm",
            "time": "12:00",
            "day": "Monday",
            "recurring": True,
        }

    def resolve_conflict(self) -> str:
        try:
            conflict_event = self._generate_mock_conflict()

            # Categorize events with error handling
            try:
                conflict_type = self.openai.categorize_event(
                    conflict_event["title"], conflict_event["description"]
                )
                print(f"Conflict Type: {conflict_type}")
            except Exception as e:
                logger.error(f"Error categorizing events: {str(e)}")
                conflict_type = "unknown"

            # Create event data for embedding
            event_data = {
                "user_id": self.user_id,
                "title": conflict_event["title"],
                "description": conflict_event["description"],
                "time": conflict_event["time"],
                "day": conflict_event["day"],
                "recurring": conflict_event["recurring"],
                "event_type": conflict_type,
            }

            # Generate weighted embedding text
            # Repeat important fields multiple times to increase their weight
            embedding_text = (
                # Event type (high weight - repeated 5 times)
                f"Type: {event_data['event_type']}\n" * 5
                +
                # Title and description (medium weight - repeated 3 times)
                f"Title: {event_data['title']}\n" * 3
                + f"Description: {event_data['description']}\n" * 3
                +
                # Time and day (low weight - repeated once)
                f"Time: {event_data['time']}\n"
                + f"Day: {event_data['day']}\n"
                + f"Recurring: {event_data['recurring']}"
            )

            embedding = get_embedding(embedding_text)

            # Find similar historical events with relaxed filters
            # Only filter by user_id and event_type for similarity search
            similar_events = self.pinecone.find_similar_events(
                query_embedding=embedding,
                user_id=self.user_id,
                event_type=conflict_type,  # Keep event_type filter for exact match
                k=5,
            )

            # Process history and statistics
            history = []
            action_counts = {"reschedule_new": 0, "delete_conflict": 0}

            for event in similar_events:
                action = event["metadata"].get("action_taken", "")
                if action in action_counts:
                    action_counts[action] += 1
                history.append(
                    f"{action} for {event['metadata']['title']} (score: {event['score']:.2f})"
                )

            total = len(similar_events) or 1
            stats = {
                "reschedule_new": round(
                    action_counts["reschedule_new"] / total * 100, 1
                ),
                "delete_conflict": round(
                    action_counts["delete_conflict"] / total * 100, 1
                ),
            }

            # Generate recommendation
            try:
                recommendation = self.openai.generate_recommendation(
                    {
                        "user_id": self.user_id,
                        "new_event": f"{conflict_event['title']} ({conflict_type})",
                        "conflict_event": f"{conflict_event['title']} ({conflict_type})",
                        "history": "\n- ".join(history),
                        **stats,
                    }
                )
            except Exception as e:
                logger.error(f"Error generating recommendation: {str(e)}")
                recommendation = "Unable to generate recommendation at this time."

            # Store the current event and decision
            try:
                event_data["action_taken"] = (
                    "delete_conflict"  # This would be the actual decision
                )
                self.pinecone.store_calendar_event(event_data, embedding)
            except Exception as e:
                logger.error(f"Error storing decision: {str(e)}")

            return recommendation

        except Exception as e:
            logger.error(f"Unexpected error in resolve_conflict: {str(e)}")
            return "An error occurred while processing your request. Please try again later."


if __name__ == "__main__":
    resolver = ConflictResolver()

    print("First run:")
    print(resolver.resolve_conflict())

    # print("\nSecond run:")
    # print(resolver.resolve_conflict())
