import asyncio
from .config import index, TOP_K, MIN_SCORE_THRESHOLD, EMBED_CONCURRENCY
from .embeddings import get_gemini_embedding_async

async def retrieve_clauses_from_pinecone_async(query: str, namespace: str, top_k: int = TOP_K):
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    q_emb = await get_gemini_embedding_async(query, sem)

    def _query_sync():
        return index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)

    results = await asyncio.to_thread(_query_sync)
    matches_raw = results.get("matches", [])

    matches = [
        {"score": m.get("score", 0.0), "text": (m.get("metadata") or {}).get("text", "")}
        for m in matches_raw
    ]
    if not matches:
        return []

    max_score = max(m["score"] for m in matches)
    if max_score < MIN_SCORE_THRESHOLD:
        return []

    cutoff = max(MIN_SCORE_THRESHOLD, 0.55 * max_score)
    return [m["text"] for m in matches if m["score"] >= cutoff]
