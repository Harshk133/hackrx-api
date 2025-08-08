import time
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from .auth import verify_token
from .pdf_utils import extract_text_from_pdf_url
from .pinecone_utils import store_pdf_in_pinecone_cached
from .retrieval import retrieve_clauses_from_pinecone_async
from .answer_generator import answer_question_from_clause_async
from .config import pc, INDEX_NAME, TOP_K

router = APIRouter()

class RequestData(BaseModel):
    documents: str
    questions: List[str]

class ResponseData(BaseModel):
    answers: List[str]

@router.get("/debug/index_stats")
def debug_index_stats():
    return pc.describe_index_stats(index_name=INDEX_NAME)

@router.post("/api/v1/hackrx/run", response_model=ResponseData, dependencies=[Depends(verify_token)])
async def process_pdf_and_questions(data: RequestData):
    try:
        pdf_text = await extract_text_from_pdf_url(data.documents)
        namespace = await store_pdf_in_pinecone_cached(pdf_text)

        async def handle_question(q: str):
            clauses = await retrieve_clauses_from_pinecone_async(q, namespace, TOP_K)
            return await answer_question_from_clause_async(clauses, q)

        answers = await asyncio.gather(*[handle_question(q) for q in data.questions])
        return ResponseData(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
