from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.search_vector import get_answer

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    return await get_answer(request.question)