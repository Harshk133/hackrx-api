import asyncio
import google.generativeai as genai
from .config import EMBED_CONCURRENCY

def _get_gemini_embedding_sync(text: str):
    model = "models/text-embedding-004"
    res = genai.embed_content(model=model, content=text)
    return res["embedding"]

async def get_gemini_embedding_async(text: str, sem: asyncio.Semaphore):
    async with sem:
        return await asyncio.to_thread(_get_gemini_embedding_sync, text)
