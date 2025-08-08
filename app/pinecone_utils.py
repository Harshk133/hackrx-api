import asyncio
import uuid
from .config import index, UPSERT_BATCH_SIZE, EMBED_CONCURRENCY
from .cache import DOC_CACHE, save_cache
from .text_splitter import chunk_text
from .embeddings import get_gemini_embedding_async
import hashlib
import time

def _pinecone_upsert_sync(vectors, namespace=None):
    if namespace:
        return index.upsert(vectors=vectors, namespace=namespace)
    return index.upsert(vectors=vectors)

async def pinecone_upsert_async(vectors, namespace=None):
    return await asyncio.to_thread(_pinecone_upsert_sync, vectors, namespace)

async def store_pdf_in_pinecone_cached(pdf_text: str) -> str:
    doc_hash = hashlib.md5(pdf_text.encode("utf-8")).hexdigest()
    namespace = f"doc-{doc_hash}"
    if doc_hash in DOC_CACHE:
        return namespace

    chunks = chunk_text(pdf_text)
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    embeddings = await asyncio.gather(*[get_gemini_embedding_async(c, sem) for c in chunks])

    vectors = [
        {"id": str(uuid.uuid4()), "values": emb, "metadata": {"text": chunk}}
        for chunk, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        await pinecone_upsert_async(batch, namespace=namespace)

    DOC_CACHE[doc_hash] = {"namespace": namespace, "chunks": len(chunks), "time": time.time()}
    save_cache()
    return namespace
