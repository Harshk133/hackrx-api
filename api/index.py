# from app.main import app  # Import FastAPI instance

import sys, os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum  # to make FastAPI work with Vercel

# Add root path to imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.main import app  # import your FastAPI app

# Wrap with Mangum for AWS Lambda style (needed for Vercel serverless)
handler = Mangum(app)

# Optional test endpoint (to confirm deployment)
@app.get("/ping")
def ping():
    return JSONResponse({"message": "pong"})

