import openai
from app.config import OPENAI_API_KEY

async def generate_chat_response(question, documents):
    """Generate a response using OpenAI based on retrieved documents or web search results."""
    context = "\n".join(documents) if documents else "No relevant documents found."
    prompt = f"Context: {context}\nUser: {question}\nAI:"

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides concise answers."},
            {"role": "user", "content": prompt}
        ],
        api_key=OPENAI_API_KEY
    )

    return response["choices"][0]["message"]["content"]
