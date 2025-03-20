import os
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if PINECONE_INDEX not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX}")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,  # Ensure this matches your embedding model's dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )
else:
    print(f"Pinecone index '{PINECONE_INDEX}' already exists.")

# Connect to the index
index = pc.Index(PINECONE_INDEX)

print("Pinecone is set up successfully!")
