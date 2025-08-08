from fastapi import Header, HTTPException
from .config import API_AUTH_TOKEN

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format.")
    token = authorization.split(" ")[1]
    if token != API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token.")
