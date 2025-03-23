import openai
from pinecone import Pinecone
from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text: str, model="text-embedding-3-small"):
    try:
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None

async def store_document_embeddings(doc_id: str, text: str):
    try:
        vector = get_embedding(text)
        if vector:
            index.upsert([
                {
                    "id": doc_id,
                    "values": vector,
                    "metadata": {"text": text}
                }
            ])
            print(f"✅ Document {doc_id} stored successfully in Pinecone")
        else:
            print("⚠️ Error storing document embeddings: Failed to generate embeddings")
    except Exception as e:
        print(f"⚠️ Error storing document embeddings: {e}")

def get_all_documents():
    try:
        query_results = index.query(
            vector=[0.0] * 1536,
            top_k=100,
            include_metadata=True
        )
        if not query_results.get("matches"):
            return []
        return [{"id": match["id"], "metadata": match["metadata"]} for match in query_results["matches"]]
    except Exception as e:
        print(f"⚠️ Error fetching documents: {e}")
        return []