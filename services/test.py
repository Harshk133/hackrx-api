# import os
# import aiohttp
# from io import BytesIO
# from pypdf import PdfReader
# import google.generativeai as genai
# from fastapi import FastAPI, HTTPException, Depends, Header
# from pydantic import BaseModel
# from typing import List
# from pinecone import Pinecone, ServerlessSpec
# import uuid
# import uvicorn

# # ===== CONFIG =====
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
# INDEX_NAME = "doc-rag-index"
# API_AUTH_TOKEN = "YOUR_TEAM_TOKEN"  # Token for authentication

# print("[CONFIG] Setting up Gemini and Pinecone...")
# genai.configure(api_key=GEMINI_API_KEY)
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Create Pinecone index if not exists
# print("[PINECONE] Checking if index exists...")
# if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
#     print(f"[PINECONE] Index '{INDEX_NAME}' not found. Creating...")
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,  # Gemini embeddings dimension
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
# else:
#     print(f"[PINECONE] Index '{INDEX_NAME}' already exists.")

# index = pc.Index(INDEX_NAME)

# # ===== AUTH DEPENDENCY =====
# def verify_token(authorization: str = Header(...)):
#     """Verify Bearer token from request headers."""
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid Authorization header format.")
#     token = authorization.split(" ")[1]
#     if token != API_AUTH_TOKEN:
#         raise HTTPException(status_code=403, detail="Invalid or expired token.")
#     print("[AUTH] Authentication successful.")

# # ===== PDF Extraction =====
# async def extract_text_from_pdf_url(url: str) -> str:
#     print(f"[PDF] Downloading PDF from: {url}")
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             if response.status != 200:
#                 raise Exception(f"Failed to download PDF: HTTP {response.status}")
#             pdf_bytes = await response.read()
#     print("[PDF] Download complete. Extracting text...")

#     pdf_file = BytesIO(pdf_bytes)
#     reader = PdfReader(pdf_file)

#     text_parts = []
#     for i, page in enumerate(reader.pages, start=1):
#         text = page.extract_text()
#         if text and text.strip():
#             print(f"[PDF] Extracted text from page {i}.")
#             text_parts.append(text.strip())

#     print(f"[PDF] Total pages processed: {len(text_parts)}")
#     return "\n".join(text_parts)

# # ===== Text Splitter =====
# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
#     print("[CHUNK] Splitting text into chunks...")
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     print(f"[CHUNK] Total chunks created: {len(chunks)}")
#     return chunks

# # ===== Gemini Embedding =====
# def get_gemini_embedding(text: str):
#     print("[EMBED] Generating embedding with Gemini...")
#     model = "models/text-embedding-004"
#     result = genai.embed_content(model=model, content=text)
#     return result["embedding"]

# # ===== Store PDF in Pinecone =====
# def store_pdf_in_pinecone(pdf_text: str):
#     print("[STORE] Storing embeddings into Pinecone...")
#     chunks = chunk_text(pdf_text, chunk_size=1000, overlap=200)
#     vectors = []
#     for i, chunk in enumerate(chunks, start=1):
#         print(f"[STORE] Processing chunk {i}/{len(chunks)}...")
#         emb = get_gemini_embedding(chunk)
#         vectors.append({
#             "id": str(uuid.uuid4()),
#             "values": emb,
#             "metadata": {"text": chunk}
#         })

#     index.upsert(vectors=vectors)
#     print(f"[STORE] Uploaded {len(vectors)} vectors to Pinecone.")

# # ===== Retrieve Closest Clause from Pinecone =====
# def retrieve_clauses_from_pinecone(query: str, top_k: int = 3):
#     print(f"[RETRIEVE] Searching Pinecone for query: '{query}'")
#     query_emb = get_gemini_embedding(query)
#     results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
#     matches = [match["metadata"]["text"] for match in results["matches"]]
#     print(f"[RETRIEVE] Found {len(matches)} relevant matches.")
#     return matches

# # ===== Gemini Q&A =====
# def answer_question_from_clause(clauses: List[str], question: str) -> str:
#     print(f"[ANSWER] Generating answer for question: '{question}'")
#     context = "\n".join(clauses)
#     prompt = f"""
# You are an assistant that answers questions based ONLY on the provided clauses from a policy document.
# If the answer is not present in the clauses, reply exactly with: "Data is not present".

# Clauses:
# \"\"\"{context}\"\"\"

# Question: {question}
# Answer:
# """
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     print(f"[ANSWER] Response: {response.text.strip()}")
#     return response.text.strip()

# # ===== FastAPI Setup =====
# app = FastAPI()

# class RequestData(BaseModel):
#     documents: str
#     questions: List[str]

# class ResponseData(BaseModel):
#     answers: List[str]

# @app.post("/api/v1/hackrx/run", response_model=ResponseData, dependencies=[Depends(verify_token)])
# async def process_pdf_and_questions(data: RequestData):
#     try:
#         print("[API] Received request...")
#         pdf_text = await extract_text_from_pdf_url(data.documents)
#         store_pdf_in_pinecone(pdf_text)

#         answers = []
#         for question in data.questions:
#             clauses = retrieve_clauses_from_pinecone(question)
#             ans = answer_question_from_clause(clauses, question)
#             answers.append(ans)

#         print("[API] Request processing complete.")
#         return ResponseData(answers=answers)

#     except Exception as e:
#         print(f"[ERROR] {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ===== Run Server =====
# if __name__ == "__main__":
#     print("[SERVER] Starting FastAPI server on port 8000...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import os
# import aiohttp
# from io import BytesIO
# from pypdf import PdfReader
# import google.generativeai as genai
# from fastapi import FastAPI, HTTPException, Depends, Header
# from pydantic import BaseModel
# from typing import List
# from pinecone import Pinecone, ServerlessSpec
# import uuid
# import uvicorn
# from dotenv import load_dotenv

# # ===== Load Environment Variables =====
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = os.getenv("INDEX_NAME", "doc-rag-index")
# API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# if not all([GEMINI_API_KEY, PINECONE_API_KEY, API_AUTH_TOKEN]):
#     raise RuntimeError("Missing required environment variables. Check your .env file.")

# # ===== CONFIG =====
# print("[CONFIG] Setting up Gemini and Pinecone...")
# genai.configure(api_key=GEMINI_API_KEY)
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Create Pinecone index if not exists
# print("[PINECONE] Checking if index exists...")
# if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
#     print(f"[PINECONE] Index '{INDEX_NAME}' not found. Creating...")
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,  # Gemini embeddings dimension
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
# else:
#     print(f"[PINECONE] Index '{INDEX_NAME}' already exists.")

# index = pc.Index(INDEX_NAME)

# # ===== AUTH DEPENDENCY =====
# def verify_token(authorization: str = Header(...)):
#     """Verify Bearer token from request headers."""
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid Authorization header format.")
#     token = authorization.split(" ")[1]
#     if token != API_AUTH_TOKEN:
#         raise HTTPException(status_code=403, detail="Invalid or expired token.")
#     print("[AUTH] Authentication successful.")

# # ===== PDF Extraction =====
# async def extract_text_from_pdf_url(url: str) -> str:
#     print(f"[PDF] Downloading PDF from: {url}")
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             if response.status != 200:
#                 raise Exception(f"Failed to download PDF: HTTP {response.status}")
#             pdf_bytes = await response.read()
#     print("[PDF] Download complete. Extracting text...")

#     pdf_file = BytesIO(pdf_bytes)
#     reader = PdfReader(pdf_file)

#     text_parts = []
#     for i, page in enumerate(reader.pages, start=1):
#         text = page.extract_text()
#         if text and text.strip():
#             print(f"[PDF] Extracted text from page {i}.")
#             text_parts.append(text.strip())

#     print(f"[PDF] Total pages processed: {len(text_parts)}")
#     return "\n".join(text_parts)

# # ===== Text Splitter =====
# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
#     print("[CHUNK] Splitting text into chunks...")
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     print(f"[CHUNK] Total chunks created: {len(chunks)}")
#     return chunks

# # ===== Gemini Embedding =====
# def get_gemini_embedding(text: str):
#     print("[EMBED] Generating embedding with Gemini...")
#     model = "models/text-embedding-004"
#     result = genai.embed_content(model=model, content=text)
#     return result["embedding"]

# # ===== Store PDF in Pinecone =====
# def store_pdf_in_pinecone(pdf_text: str):
#     print("[STORE] Storing embeddings into Pinecone...")
#     chunks = chunk_text(pdf_text, chunk_size=1000, overlap=200)
#     vectors = []
#     for i, chunk in enumerate(chunks, start=1):
#         print(f"[STORE] Processing chunk {i}/{len(chunks)}...")
#         emb = get_gemini_embedding(chunk)
#         vectors.append({
#             "id": str(uuid.uuid4()),
#             "values": emb,
#             "metadata": {"text": chunk}
#         })

#     index.upsert(vectors=vectors)
#     print(f"[STORE] Uploaded {len(vectors)} vectors to Pinecone.")

# # ===== Retrieve Closest Clause from Pinecone =====
# def retrieve_clauses_from_pinecone(query: str, top_k: int = 3):
#     print(f"[RETRIEVE] Searching Pinecone for query: '{query}'")
#     query_emb = get_gemini_embedding(query)
#     results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
#     matches = [match["metadata"]["text"] for match in results["matches"]]
#     print(f"[RETRIEVE] Found {len(matches)} relevant matches.")
#     return matches

# # ===== Gemini Q&A =====
# def answer_question_from_clause(clauses: List[str], question: str) -> str:
#     print(f"[ANSWER] Generating answer for question: '{question}'")
#     context = "\n".join(clauses)
#     prompt = f"""
# You are an assistant that answers questions based ONLY on the provided clauses from a policy document.
# If the answer is not present in the clauses, reply exactly with: "Data is not present".

# Clauses:
# \"\"\"{context}\"\"\"

# Question: {question}
# Answer:
# """
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     print(f"[ANSWER] Response: {response.text.strip()}")
#     return response.text.strip()

# # ===== FastAPI Setup =====
# app = FastAPI()

# class RequestData(BaseModel):
#     documents: str
#     questions: List[str]

# class ResponseData(BaseModel):
#     answers: List[str]

# @app.post("/api/v1/hackrx/run", response_model=ResponseData, dependencies=[Depends(verify_token)])
# async def process_pdf_and_questions(data: RequestData):
#     try:
#         print("[API] Received request...")
#         pdf_text = await extract_text_from_pdf_url(data.documents)
#         store_pdf_in_pinecone(pdf_text)

#         answers = []
#         for question in data.questions:
#             clauses = retrieve_clauses_from_pinecone(question)
#             ans = answer_question_from_clause(clauses, question)
#             answers.append(ans)

#         print("[API] Request processing complete.")
#         return ResponseData(answers=answers)

#     except Exception as e:
#         print(f"[ERROR] {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ===== Run Server =====
# if __name__ == "__main__":
#     print("[SERVER] Starting FastAPI server on port 8000...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import aiohttp
import asyncio
from io import BytesIO
from pypdf import PdfReader
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
import uuid
import uvicorn
from dotenv import load_dotenv
import hashlib
import json
from pathlib import Path
import time

# ===== Load Environment Variables =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "doc-rag-index")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# Performance tuning (can adjust in .env)
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "12"))
UPsert_BATCH_SIZE = int(os.getenv("UPsert_BATCH_SIZE", "100"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.03"))

if not all([GEMINI_API_KEY, PINECONE_API_KEY, API_AUTH_TOKEN]):
    raise RuntimeError("Missing required environment variables. Check your .env file.")

# ===== CONFIG =====
print("[CONFIG] Setting up Gemini and Pinecone...")
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
print("[PINECONE] Checking if index exists...")
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    print(f"[PINECONE] Index '{INDEX_NAME}' not found. Creating...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Gemini text-embedding-004 dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"[PINECONE] Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)

# Local cache (doc_hash -> namespace)
CACHE_PATH = Path("doc_cache.json")
if CACHE_PATH.exists():
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        DOC_CACHE = json.load(f)
else:
    DOC_CACHE = {}

def save_cache():
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(DOC_CACHE, f)

# ===== AUTH DEPENDENCY =====
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format.")
    token = authorization.split(" ")[1]
    if token != API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token.")
    print("[AUTH] Authentication successful.")

# ===== PDF Extraction =====
async def extract_text_from_pdf_url(url: str) -> str:
    print(f"[PDF] Downloading PDF: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download PDF: HTTP {resp.status}")
            pdf_bytes = await resp.read()
    print("[PDF] Download done. Extracting text...")
    pdf_file = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
            print(f"[PDF] Page {i} extracted ({len(text)} chars).")
    print(f"[PDF] Extracted {len(pages)} pages.")
    return "\n".join(pages)

# ===== Text Splitter =====
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    print("[CHUNK] Splitting text into chunks...")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    print(f"[CHUNK] Created {len(chunks)} chunks.")
    return chunks

# ===== Embedding helpers =====
def _get_gemini_embedding_sync(text: str):
    model = "models/text-embedding-004"
    res = genai.embed_content(model=model, content=text)
    return res["embedding"]

async def get_gemini_embedding_async(text: str, sem: asyncio.Semaphore):
    async with sem:
        # offload sync call to thread to allow concurrency
        emb = await asyncio.to_thread(_get_gemini_embedding_sync, text)
        return emb

# ===== Pinecone upsert helper =====
def _pinecone_upsert_sync(vectors: List[Dict], namespace: str = None):
    # client expects list of dicts with id/values/metadata as used earlier
    if namespace:
        return index.upsert(vectors=vectors, namespace=namespace)
    return index.upsert(vectors=vectors)

async def pinecone_upsert_async(vectors: List[Dict], namespace: str = None):
    return await asyncio.to_thread(_pinecone_upsert_sync, vectors, namespace)

# ===== Store PDF in Pinecone (cached, parallel embeddings, batched upsert) =====
async def store_pdf_in_pinecone_cached(pdf_text: str) -> str:
    doc_hash = hashlib.md5(pdf_text.encode("utf-8")).hexdigest()
    namespace = f"doc-{doc_hash}"
    if doc_hash in DOC_CACHE:
        print(f"[STORE] Document cached. Namespace: {namespace}")
        return namespace

    print(f"[STORE] New document. Namespace: {namespace}")
    chunks = chunk_text(pdf_text)
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    # embed chunks in parallel
    embed_tasks = [get_gemini_embedding_async(c, sem) for c in chunks]
    embeddings = await asyncio.gather(*embed_tasks)

    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        vectors.append({"id": str(uuid.uuid4()), "values": emb, "metadata": {"text": chunk}})

    # Batch upsert to Pinecone
    for i in range(0, len(vectors), UPsert_BATCH_SIZE):
        batch = vectors[i:i + UPsert_BATCH_SIZE]
        print(f"[STORE] Upserting batch {i//UPsert_BATCH_SIZE + 1} size={len(batch)}")
        await pinecone_upsert_async(batch, namespace=namespace)

    DOC_CACHE[doc_hash] = {"namespace": namespace, "chunks": len(chunks), "time": time.time()}
    save_cache()
    print(f"[STORE] Stored doc in namespace {namespace} (chunks={len(chunks)})")
    return namespace

# ===== Retrieve clauses with similarity debugging and adaptive filtering =====
async def retrieve_clauses_from_pinecone_async(query: str, namespace: str, top_k: int = TOP_K):
    print(f"[RETRIEVE] Querying for: '{query}' (top_k={top_k}) in ns={namespace}")
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    q_emb = await get_gemini_embedding_async(query, sem)

    def _query_sync():
        return index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)

    results = await asyncio.to_thread(_query_sync)
    matches_raw = results.get("matches", [])
    print(f"[RETRIEVE] raw matches returned: {len(matches_raw)}")
    # Normalize matches to dicts with score/text
    matches = []
    for m in matches_raw:
        score = m.get("score", None)
        if score is None:
            # try alternative keys
            score = m.get("similarity") or m.get("value") or 0.0
        text = (m.get("metadata") or {}).get("text", "")
        vid = m.get("id") or m.get("vector_id") or m.get("identifier")
        matches.append({"id": vid, "score": score or 0.0, "text": text, "raw": m})

    # Print for debug
    for i, mm in enumerate(matches, 1):
        print(f"[RETRIEVE] Match{i}: id={mm['id']}, score={mm['score']:.4f}, text_len={len(mm['text'])}")

    if not matches:
        return []

    # adaptive: require max_score >= MIN_SCORE_THRESHOLD
    scores = [m["score"] for m in matches]
    max_score = max(scores)
    print(f"[RETRIEVE] max_score={max_score:.4f}, min_threshold={MIN_SCORE_THRESHOLD}")

    if max_score < MIN_SCORE_THRESHOLD:
        print("[RETRIEVE] max_score below threshold -> returning empty (Data is not present).")
        return []

    # keep matches that are >= max( MIN_SCORE_THRESHOLD, 0.55*max_score )
    cutoff = max(MIN_SCORE_THRESHOLD, 0.55 * max_score)
    kept = [m["text"] for m in matches if m["score"] >= cutoff]
    print(f"[RETRIEVE] kept {len(kept)} matches after cutoff={cutoff:.4f}")
    return kept

# ===== Gemini answer generator (sync wrapped) =====
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

async def answer_question_from_clause_async(clauses: List[str], question: str):
    if not clauses:
        return "Data is not present"
    # join top clauses (limit to TOP_K for context length)
    context = "\n".join(clauses[:TOP_K])
    # call sync generator in thread pool
    ans = await asyncio.to_thread(_generate_answer_sync, context, question)
    return ans

# ===== FastAPI app =====
app = FastAPI()

class RequestData(BaseModel):
    documents: str
    questions: List[str]

class ResponseData(BaseModel):
    answers: List[str]

# Diagnostics
@app.get("/debug/index_stats")
def debug_index_stats():
    stats = pc.describe_index_stats(index_name=INDEX_NAME)
    return stats

@app.get("/debug/cache")
def debug_cache():
    return {"cache": DOC_CACHE}

@app.post("/api/v1/hackrx/run", response_model=ResponseData, dependencies=[Depends(verify_token)])
async def process_pdf_and_questions(data: RequestData):
    start_total = time.perf_counter()
    try:
        print("[API] Request received.")
        t0 = time.perf_counter()
        pdf_text = await extract_text_from_pdf_url(data.documents)
        t1 = time.perf_counter()
        print(f"[TIMING] pdf_extract={(t1-t0):.2f}s")

        # store embeddings (cached) -> returns namespace
        t0 = time.perf_counter()
        namespace = await store_pdf_in_pinecone_cached(pdf_text)
        t1 = time.perf_counter()
        print(f"[TIMING] store_pdf={(t1-t0):.2f}s (may be ~0 if cached)")

        # process questions in parallel
        async def handle_question(q: str):
            q0 = time.perf_counter()
            clauses = await retrieve_clauses_from_pinecone_async(q, namespace=namespace, top_k=TOP_K)
            q1 = time.perf_counter()
            ans = await answer_question_from_clause_async(clauses, q)
            q2 = time.perf_counter()
            print(f"[TIMING] question='{q[:40]}...' retrieve={(q1-q0):.2f}s answer={(q2-q1):.2f}s total={(q2-q0):.2f}s")
            return ans

        answers = await asyncio.gather(*[handle_question(q) for q in data.questions])
        total = time.perf_counter() - start_total
        print(f"[API] Completed. total_time={total:.2f}s")
        return ResponseData(answers=answers)

    except Exception as e:
        print("[ERROR]", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    print("[SERVER] Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)











# import os
# import aiohttp
# import asyncio
# from io import BytesIO
# from pypdf import PdfReader
# import google.generativeai as genai
# from fastapi import FastAPI, HTTPException, Depends, Header
# from pydantic import BaseModel
# from typing import List
# from pinecone import Pinecone, ServerlessSpec
# import uuid
# import uvicorn
# from dotenv import load_dotenv
# import hashlib
# import json
# from pathlib import Path
# from functools import partial

# # ===== Load Environment Variables =====
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = os.getenv("INDEX_NAME", "doc-rag-index")
# API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# # Performance tuning
# EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))   # parallel embedding calls
# UPsert_BATCH_SIZE = int(os.getenv("UPsert_BATCH_SIZE", "50"))  # batch size for Pinecone upsert
# TOP_K = int(os.getenv("TOP_K", "3"))

# if not all([GEMINI_API_KEY, PINECONE_API_KEY, API_AUTH_TOKEN]):
#     raise RuntimeError("Missing required environment variables. Check your .env file.")

# # ===== CONFIG =====
# print("[CONFIG] Setting up Gemini and Pinecone...")
# genai.configure(api_key=GEMINI_API_KEY)
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Create Pinecone index if not exists
# print("[PINECONE] Checking if index exists...")
# if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
#     print(f"[PINECONE] Index '{INDEX_NAME}' not found. Creating...")
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,  # Gemini embeddings dimension
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
# else:
#     print(f"[PINECONE] Index '{INDEX_NAME}' already exists.")

# index = pc.Index(INDEX_NAME)

# # Local cache file to mark processed documents (maps doc_hash -> namespace)
# CACHE_PATH = Path("doc_cache.json")
# if CACHE_PATH.exists():
#     with CACHE_PATH.open("r", encoding="utf-8") as f:
#         DOC_CACHE = json.load(f)
# else:
#     DOC_CACHE = {}

# def save_cache():
#     with CACHE_PATH.open("w", encoding="utf-8") as f:
#         json.dump(DOC_CACHE, f)

# # ===== AUTH DEPENDENCY =====
# def verify_token(authorization: str = Header(...)):
#     """Verify Bearer token from request headers."""
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid Authorization header format.")
#     token = authorization.split(" ")[1]
#     if token != API_AUTH_TOKEN:
#         raise HTTPException(status_code=403, detail="Invalid or expired token.")
#     print("[AUTH] Authentication successful.")

# # ===== PDF Extraction =====
# async def extract_text_from_pdf_url(url: str) -> str:
#     print(f"[PDF] Downloading PDF from: {url}")
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             if response.status != 200:
#                 raise Exception(f"Failed to download PDF: HTTP {response.status}")
#             pdf_bytes = await response.read()
#     print("[PDF] Download complete. Extracting text...")

#     pdf_file = BytesIO(pdf_bytes)
#     reader = PdfReader(pdf_file)

#     text_parts = []
#     for i, page in enumerate(reader.pages, start=1):
#         text = page.extract_text()
#         if text and text.strip():
#             print(f"[PDF] Extracted text from page {i}.")
#             text_parts.append(text.strip())

#     print(f"[PDF] Total pages processed: {len(text_parts)}")
#     return "\n".join(text_parts)

# # ===== Text Splitter =====
# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
#     print("[CHUNK] Splitting text into chunks...")
#     chunks = []
#     start = 0
#     length = len(text)
#     while start < length:
#         end = min(start + chunk_size, length)
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     print(f"[CHUNK] Total chunks created: {len(chunks)}")
#     return chunks

# # ===== Gemini Embedding (sync) =====
# def _get_gemini_embedding_sync(text: str):
#     # synchronous wrapper to call Gemini embedding
#     model = "models/text-embedding-004"
#     res = genai.embed_content(model=model, content=text)
#     return res["embedding"]

# # async wrapper using to_thread
# async def get_gemini_embedding_async(text: str, semaphore: asyncio.Semaphore):
#     async with semaphore:
#         print("[EMBED] Generating embedding (async)...")
#         emb = await asyncio.to_thread(_get_gemini_embedding_sync, text)
#         return emb

# # ===== Batch upsert helper (sync) =====
# def _pinecone_upsert_sync(vectors, namespace: str = None):
#     # this is sync; call via to_thread
#     if namespace:
#         return index.upsert(vectors=vectors, namespace=namespace)
#     return index.upsert(vectors=vectors)

# async def pinecone_upsert_async(vectors, namespace: str = None):
#     print(f"[PINECONE] Upserting batch of {len(vectors)} vectors (ns={namespace})...")
#     return await asyncio.to_thread(_pinecone_upsert_sync, vectors, namespace)

# # ===== Store PDF in Pinecone (async, parallel embeddings + batched upsert) =====
# async def store_pdf_in_pinecone_cached(pdf_text: str) -> str:
#     """
#     Stores embeddings in Pinecone only if not already stored.
#     Uses doc hash as namespace and caches it locally.
#     Returns namespace used.
#     """
#     # compute doc hash
#     doc_hash = hashlib.md5(pdf_text.encode("utf-8")).hexdigest()
#     namespace = f"doc-{doc_hash}"

#     if doc_hash in DOC_CACHE:
#         print(f"[STORE] Document already processed (cache). Namespace: {namespace}")
#         return namespace

#     print(f"[STORE] New document. Will store under namespace: {namespace}")
#     chunks = chunk_text(pdf_text, chunk_size=1000, overlap=200)

#     sem = asyncio.Semaphore(EMBED_CONCURRENCY)
#     tasks = [get_gemini_embedding_async(chunk, sem) for chunk in chunks]
#     embeddings = await asyncio.gather(*tasks)  # list of vectors

#     # prepare vectors in batches
#     vectors = []
#     for chunk, emb in zip(chunks, embeddings):
#         vectors.append((str(uuid.uuid4()), emb, {"text": chunk}))

#     # upsert in batches
#     for i in range(0, len(vectors), UPsert_BATCH_SIZE):
#         batch = vectors[i:i + UPsert_BATCH_SIZE]
#         # convert to pinecone expected format for this client: list[ {id, values, metadata} ] or tuple depending on client
#         # Using dict format to be consistent:
#         batch_payload = [{"id": v[0], "values": v[1], "metadata": v[2]} for v in batch]
#         await pinecone_upsert_async(batch_payload, namespace=namespace)

#     # mark cached
#     DOC_CACHE[doc_hash] = {"namespace": namespace, "chunks": len(chunks)}
#     save_cache()
#     print(f"[STORE] Completed storing document. Namespace: {namespace}, chunks: {len(chunks)}")
#     return namespace

# # ===== Retrieve Closest Clause from Pinecone (async) =====
# # ===== Retrieve Closest Clause with Semantic Filtering =====
# SIMILARITY_THRESHOLD = 0.80  # adjust based on quality

# # async def retrieve_clauses_from_pinecone_async(query: str, namespace: str, top_k: int = TOP_K):
# #     print(f"[RETRIEVE] Searching Pinecone for query: '{query}' in namespace: {namespace}")
# #     # embed question
# #     sem = asyncio.Semaphore(EMBED_CONCURRENCY)
# #     q_emb = await get_gemini_embedding_async(query, sem)
# #     # index.query is sync; run in thread
# #     def _query_sync():
# #         # include_metadata True to get the chunk text back
# #         return index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
# #     results = await asyncio.to_thread(_query_sync)
# #     matches = []
# #     for m in results.get("matches", []):
# #         md = m.get("metadata", {})
# #         text = md.get("text", "")
# #         matches.append(text)
# #     print(f"[RETRIEVE] Found {len(matches)} matches for question.")
# #     return matches

# # # ===== Gemini Q&A (sync) =====
# # def _generate_answer_sync(context: str, question: str):
# #     prompt = f"""
# # You are an assistant that answers questions based ONLY on the provided clauses from a policy document.
# # If the answer is not present in the clauses, reply exactly with: "Data is not present".

# # Clauses:
# # \"\"\"{context}\"\"\"

# # Question: {question}
# # Answer:
# # """
# #     model = genai.GenerativeModel("gemini-2.0-flash")
# #     resp = model.generate_content(prompt)
# #     return resp.text.strip()

# # async def answer_question_from_clause_async(clauses: List[str], question: str):
# #     print(f"[ANSWER] Generating answer for question: '{question}'")
# #     context = "\n".join(clauses)
# #     ans = await asyncio.to_thread(_generate_answer_sync, context, question)
# #     print(f"[ANSWER] Answer generated (len={len(ans)}).")
# #     return ans
# async def retrieve_clauses_from_pinecone_async(query: str, namespace: str, top_k: int = TOP_K):
#     print(f"[RETRIEVE] Searching Pinecone for: '{query}' in namespace: {namespace}")
    
#     # Embed question
#     sem = asyncio.Semaphore(EMBED_CONCURRENCY)
#     q_emb = await get_gemini_embedding_async(query, sem)
    
#     # Run Pinecone query
#     def _query_sync():
#         return index.query(
#             vector=q_emb,
#             top_k=top_k,
#             include_metadata=True,
#             namespace=namespace
#         )
#     results = await asyncio.to_thread(_query_sync)
    
#     matches = []
#     for m in results.get("matches", []):
#         score = m.get("score", 0)
#         text = m.get("metadata", {}).get("text", "")
#         if score >= SIMILARITY_THRESHOLD:  # only keep strong matches
#             matches.append((score, text))
    
#     matches.sort(key=lambda x: x[0], reverse=True)  # sort by similarity
#     final_clauses = [m[1] for m in matches]
    
#     print(f"[RETRIEVE] Found {len(final_clauses)} clauses after filtering by score ≥ {SIMILARITY_THRESHOLD}.")
#     return final_clauses


# # ===== Gemini Q&A with Clause Validation =====
# async def answer_question_from_clause_async(clauses: List[str], question: str):
#     if not clauses:
#         print(f"[ANSWER] No relevant clauses found for: '{question}'")
#         return "Data is not present"
    
#     context = "\n".join(clauses[:TOP_K])  # keep top relevant clauses only
    
#     def _generate_answer_sync():
#         prompt = f"""
# You are an assistant that answers questions based ONLY on the provided document clauses.
# If the answer is not present in the clauses, reply exactly: "Data is not present".

# Clauses:
# \"\"\"{context}\"\"\"

# Question: {question}
# Answer:
# """
#         model = genai.GenerativeModel("gemini-2.0-flash")
#         resp = model.generate_content(prompt)
#         return resp.text.strip()
    
#     ans = await asyncio.to_thread(_generate_answer_sync)
#     return ans


# # ===== Updated API Flow =====

# # ===== FastAPI Setup =====
# app = FastAPI()

# class RequestData(BaseModel):
#     documents: str
#     questions: List[str]

# class ResponseData(BaseModel):
#     answers: List[str]

# # @app.post("/api/v1/hackrx/run", response_model=ResponseData, dependencies=[Depends(verify_token)])
# # async def process_pdf_and_questions(data: RequestData):
# #     try:
# #         print("[API] Received request...")
# #         # 1. Extract PDF text
# #         pdf_text = await extract_text_from_pdf_url(data.documents)

# #         # 2. Store in Pinecone (cached) -> returns namespace
# #         namespace = await store_pdf_in_pinecone_cached(pdf_text)

# #         # 3. Parallel Q&A
# #         async def handle_question(q):
# #             clauses = await retrieve_clauses_from_pinecone_async(q, namespace=namespace, top_k=TOP_K)
# #             ans = await answer_question_from_clause_async(clauses, q)
# #             return ans

# #         # run all questions in parallel
# #         answers = await asyncio.gather(*[handle_question(q) for q in data.questions])

# #         print("[API] Request processing complete.")
# #         return ResponseData(answers=answers)

# #     except Exception as e:
# #         print(f"[ERROR] {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))
# @app.post("/api/v1/hackrx/run", response_model=ResponseData, dependencies=[Depends(verify_token)])
# async def process_pdf_and_questions(data: RequestData):
#     try:
#         print("[API] Received request...")
        
#         # Step 1: Extract text
#         pdf_text = await extract_text_from_pdf_url(data.documents)
        
#         # Step 2: Store embeddings in Pinecone (cached)
#         namespace = await store_pdf_in_pinecone_cached(pdf_text)
        
#         # Step 3: Process questions in parallel
#         async def handle_question(q):
#             clauses = await retrieve_clauses_from_pinecone_async(q, namespace=namespace, top_k=TOP_K)
#             ans = await answer_question_from_clause_async(clauses, q)
#             return ans
        
#         answers = await asyncio.gather(*[handle_question(q) for q in data.questions])
        
#         print("[API] Completed request.")
#         return ResponseData(answers=answers)
    
#     except Exception as e:
#         print(f"[ERROR] {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ===== Run Server =====
# if __name__ == "__main__":
#     print("[SERVER] Starting FastAPI server on port 8000...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
