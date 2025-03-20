import openai
import requests
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, SERPER_API_KEY

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Function to generate query embeddings
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# Function to search for similar text in Pinecone
def search_pinecone(query, top_k=5, score_threshold=0.2):
    query_vector = get_embedding(query)

    if query_vector:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        if results.get("matches"):
            filtered_texts = [
                match["metadata"]["text"] for match in results["matches"]
                if "text" in match["metadata"] and match["score"] >= score_threshold
            ]
            
            if filtered_texts:
                return " ".join(filtered_texts)  # Combine all highly relevant texts
    return None  # No relevant data found

# Function to generate a summary using OpenAI
def summarize_data(data, query):
    try:
        if not data:
            return "No relevant text found in Pinecone."

        prompt = f"Here is relevant information:\n{data}\n\nAnswer concisely: {query}"

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides concise answers based on given data."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# Function to perform web search using Serper.dev API
def search_web(query):
    url = "https://google.serper.dev/search"  # ✅ Correct API URL
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": 3}


    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Extract relevant information
            if "answerBox" in data:
                return data["answerBox"].get("answer", "No answer found.")
            
            elif "knowledgeGraph" in data:
                return data["knowledgeGraph"].get("description", "No relevant details available.")
            
            elif "organic" in data and data["organic"]:
                web_results = [entry.get("snippet", "") for entry in data["organic"][:3] if "snippet" in entry]
                return " ".join(web_results) if web_results else "No relevant search results found."

            return "No relevant web search results found."

        else:
            print(f" Web search failed: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f" Web search error: {e}")
        return None

# Main function to process user query
def get_answer(query):

    # First, check Pinecone DB
    retrieved_data = search_pinecone(query)

    if retrieved_data:
        return summarize_data(retrieved_data, query)
    else:
        web_data = search_web(query)

        if web_data:
            return summarize_data(web_data, query)
        else:
            return "No relevant data found in Pinecone or on the web."

# Run search
if __name__ == "__main__":
    query_text = input("\nEnter your search query: ")
    response = get_answer(query_text)
    print("\nResponse:\n", response)
