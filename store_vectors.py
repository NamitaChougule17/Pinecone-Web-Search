import os
import openai
import pandas as pd
import PyPDF2
import tiktoken  # For token counting
import openpyxl
from docx import Document
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Function to count tokens
def count_tokens(text, model="text-embedding-3-small"):
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

# Function to extract text from different file types
def extract_text_from_file(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as txt_file:
                text = txt_file.read()

        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
            text = " ".join(df.astype(str).stack().tolist())

    except Exception as e:
        print(f" Error extracting text from {file_path}: {e}")

    return text.strip()

# Function to generate embeddings (truncates if too long)
def get_embedding(text, model="text-embedding-3-small"):
    try:
        token_count = count_tokens(text)
        
        # Limit to 8192 tokens (max for OpenAI embeddings)
        if token_count > 8192:
            print(f" Text exceeds 8192 tokens ({token_count} tokens). Truncating...")
            words = text.split()
            text = " ".join(words[:8000])  # Trim to approx. 8000 words

        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f" Error generating embedding: {e}")
        return None

# Process all files in a directory
def process_files(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if file_path.endswith((".pdf", ".docx", ".txt", ".xlsx")):
            print(f" Processing: {file_name}")
            text_content = extract_text_from_file(file_path)

            if text_content:
                vector = get_embedding(text_content)
                if vector:
                    index.upsert(vectors=[{
                        "id": file_name,
                        "values": vector,
                        "metadata": {"text": text_content[:500]}  # Store small snippet of text
                    }])
                    print(f"Stored vector for: {file_name}")
                else:
                    print(f" Failed to store vector for: {file_name}")
            else:
                print(f" No text extracted from: {file_name}")

# Directory containing files to process
directory_path = r"H:\Pinecone_OpenAI\DataTest"

# Run the processing function
if __name__ == "__main__":
    process_files(directory_path)
    print("All files processed and stored in Pinecone!")
