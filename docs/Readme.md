# PDF Question Answering API (Gemini + Pinecone + FastAPI)

This project is a **modular FastAPI application** that:
1. Downloads and extracts text from a PDF URL.
2. Splits the text into chunks.
3. Generates **embeddings** using Google's **Gemini API**.
4. Stores embeddings in **Pinecone** for semantic search.
5. Retrieves relevant document chunks for a given query.
6. Generates answers using **Gemini** based only on the retrieved text.
7. Caches processed documents locally to avoid redundant processing.

---

## **Features**
- 📄 PDF download and text extraction
- 🔍 Semantic search with **Pinecone**
- 🤖 Q&A with **Google Gemini**
- ⚡ Async processing for better performance
- 🗂 Modular, maintainable codebase
- 🔑 Token-based API authentication
- 🛠 Debug endpoints for index stats and cache

---

## **Folder Structure**
```plaintext
project_root/
│
├── app/
│   ├── __init__.py               # Marks package
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                  # Loads env vars, sets up Pinecone & Gemini
│   ├── auth.py                    # Bearer token authentication
│   ├── cache.py                   # Local document cache management
│   ├── pdf_utils.py               # Download and extract text from PDFs
│   ├── text_splitter.py           # Split large text into overlapping chunks
│   ├── embeddings.py              # Gemini embedding generation
│   ├── pinecone_utils.py          # Store and manage embeddings in Pinecone
│   ├── retrieval.py               # Retrieve relevant chunks from Pinecone
│   ├── answer_generator.py        # Generate final answers using Gemini
│   └── routes.py                  # API endpoints
│
├── doc_cache.json                 # Local document cache (auto-generated)
├── .env.template                  # Example environment variables
├── requirements.txt               # Python dependencies
├── run.py                         # Runs the FastAPI app with uvicorn
└── README.md                      # This documentation
