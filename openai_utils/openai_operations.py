from typing import Dict, List, Optional
from openai import OpenAI, RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
import logging
from functools import lru_cache
import time
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenAI client with rate limiting
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Rate limiting configuration
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """Get embedding with caching and retry logic"""
    try:
        return embeddings.embed_query(text)
    except RateLimitError as e:
        logger.warning(f"Rate limit hit, waiting {RATE_LIMIT_DELAY} seconds: {str(e)}")
        time.sleep(RATE_LIMIT_DELAY)
        raise
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

class OpenAIOps:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        self.prompt = ChatPromptTemplate.from_template(
            """
            Analyze this calendar conflict for {user_id}:
            
            New Event: {new_event}
            Conflict Event: {conflict_event}
            
            Historical Decisions:
            {history}
            
            Statistics:
            - Rescheduled new: {reschedule_new}%
            - Deleted conflict: {delete_conflict}%
            
            Consider these factors:
            1. Event priority patterns
            2. Time of day preferences
            3. Recurrence status
            4. Recent similar decisions
            
            Generate a recommendation with 1-2 options.
        """
        )
        self._setup_rate_limiting()

    def _setup_rate_limiting(self):
        """Setup rate limiting configuration"""
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_recommendation(self, context: Dict) -> str:
        """Generate recommendation with rate limiting and retry logic"""
        try:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            chain = self.prompt | self.llm
            result = chain.invoke(context).content
            self.last_request_time = time.time()
            return result
        except RateLimitError as e:
            logger.warning(f"Rate limit hit, waiting {RATE_LIMIT_DELAY} seconds: {str(e)}")
            time.sleep(RATE_LIMIT_DELAY)
            raise
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    @lru_cache(maxsize=1000)
    def categorize_event(self, title: str, description: str) -> str:
        """Categorize event with caching and retry logic"""
        try:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Classify into 'client', 'internal', or 'personal':\nTitle: {title}\nDescription: {description}",
                    }
                ],
            )
            self.last_request_time = time.time()
            return response.choices[0].message.content.strip().lower()
        except RateLimitError as e:
            logger.warning(f"Rate limit hit, waiting {RATE_LIMIT_DELAY} seconds: {str(e)}")
            time.sleep(RATE_LIMIT_DELAY)
            raise
        except Exception as e:
            logger.error(f"Error categorizing event: {str(e)}")
            raise
