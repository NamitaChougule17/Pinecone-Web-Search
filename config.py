import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Ensure API keys are loaded
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError(" Missing API keys. Check your .env file!")
