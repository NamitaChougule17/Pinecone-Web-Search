import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]

# Pinecone Setup (New API)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ✅ Ensure API key is loaded
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Missing Pinecone API key or index name in .env file")

# ✅ Initialize Pinecone Client Correctly
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# OpenAI Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
