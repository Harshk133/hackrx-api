import asyncio
import google.generativeai as genai
from .config import TOP_K

def _generate_answer_sync(context: str, question: str) -> str:
    prompt = f"""
You are an assistant that answers questions based ONLY on the provided clauses from a policy document.
If the answer is not present in the clauses, reply exactly with: "Data is not present".

Clauses:
\"\"\"{context}\"\"\"

Question: {question}
Answer:
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

async def answer_question_from_clause_async(clauses, question):
    if not clauses:
        return "Data is not present"
    context = "\n".join(clauses[:TOP_K])
    return await asyncio.to_thread(_generate_answer_sync, context, question)
