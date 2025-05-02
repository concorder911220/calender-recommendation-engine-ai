# setup_pinecone.py
from pinecone_utils.pinecone_client import PineconeClient

if __name__ == "__main__":
    PineconeClient()  # Creates the index if it doesn't exist
    print("Done! Check your Pinecone dashboard.")
