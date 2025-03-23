from fastapi import FastAPI
from app.routes import chatbot, documents
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5001"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routes
app.include_router(chatbot.router, prefix="/api/chat", tags=["Chatbot"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": "FastAPI Backend is Running!"}