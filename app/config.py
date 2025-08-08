import os
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "doc-rag-index")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# Performance tuning
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "12"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.03"))

if not all([GEMINI_API_KEY, PINECONE_API_KEY, API_AUTH_TOKEN]):
    raise RuntimeError("Missing required environment variables.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Gemini text-embedding-004 dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Cache path
CACHE_PATH = Path("doc_cache.json")
