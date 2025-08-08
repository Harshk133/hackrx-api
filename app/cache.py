import json
from .config import CACHE_PATH

if CACHE_PATH.exists():
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        DOC_CACHE = json.load(f)
else:
    DOC_CACHE = {}

def save_cache():
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(DOC_CACHE, f)
