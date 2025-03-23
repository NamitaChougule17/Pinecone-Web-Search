import openai
import requests
from pinecone import Pinecone
from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, SERPER_API_KEY
from app.services.openai_chat import generate_chat_response

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

def search_web(query: str):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": 3}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "answerBox" in data:
                return data["answerBox"].get("answer", "No answer found.")
            elif "knowledgeGraph" in data:
                return data["knowledgeGraph"].get("description", "No relevant details available.")
            elif "organic" in data and data["organic"]:
                web_results = [entry.get("snippet", "") for entry in data["organic"][:3] if "snippet" in entry]
                return " ".join(web_results) if web_results else "No relevant search results found."
            return "No relevant web search results found."
        else:
            print(f"⚠️ Web search failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return None

async def search_documents(query: str, top_k: int = 3):
    try:
        query_vector = get_embedding(query)
        if query_vector:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            relevant_docs = [match['metadata']['text'] for match in results['matches'] if 'metadata' in match and 'text' in match['metadata']]
            if relevant_docs:
                return relevant_docs
        print("⚠️ No relevant document found. Performing web search...")
        web_results = search_web(query)
        return [web_results] if web_results else []
    except Exception as e:
        print(f"⚠️ Error searching documents: {e}")
        return []

async def get_answer(query: str):
    retrieved_data = await search_documents(query)
    if retrieved_data and "No relevant" not in retrieved_data[0]:
        summary = await generate_chat_response(query, retrieved_data)
        return {
            "answer": summary,
            "source": "documents"
        }
    else:
        summary = await generate_chat_response(query, retrieved_data)
        return {
            "answer": summary,
            "source": "web"
        }
