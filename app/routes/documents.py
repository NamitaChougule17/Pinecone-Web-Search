from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.store_vector import store_document_embeddings, get_all_documents
from app.models.document_model import Document
from app.config import db
import requests
from io import BytesIO

NODEJS_SERVER = "http://localhost:5001"  # Node.js backend URL
router = APIRouter()

def extract_text(file: UploadFile, content: bytes) -> str:
    """Extract text based on the file extension."""
    ext = file.filename.split('.')[-1].lower()
    
    if ext in ['txt', 'md', 'html']:
        return content.decode("utf-8", errors="ignore")
    
    elif ext == 'pdf':
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")
    
    elif ext in ['doc', 'docx']:
        try:
            import docx
            document = docx.Document(BytesIO(content))
            text = "\n".join(para.text for para in document.paragraphs)
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from DOCX: {e}")
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receive document, extract text, store embeddings in Pinecone directly."""
    try:
        content = await file.read()
        text_data = extract_text(file, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store embeddings directly in Pinecone
    doc_id = str(file.filename)
    await store_document_embeddings(doc_id, text_data)

    # ✅ Notify Node.js to refresh document list
    try:
        requests.post(f"{NODEJS_SERVER}/api/refresh-documents", json={"doc_id": doc_id})
    except Exception as e:
        print(f"⚠️ Failed to notify Node.js: {e}")

    return {"message": "Document stored successfully", "doc_id": doc_id}

# ✅ Fetch all stored documents
@router.get("/")
async def get_documents():
    """Fetch all stored documents from Pinecone."""
    try:
        documents = get_all_documents()
        return {"documents": documents or []}  # ✅ Default empty list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))