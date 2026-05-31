import os
from functools import lru_cache
from pinecone import Pinecone
from backend.app.config import get_settings

@lru_cache
def get_pinecone_index():
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set.")
    
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_host:
        host = settings.pinecone_host.replace("https://", "").replace("http://", "").strip("/")
        return pc.Index(host=host)
    else:
        # Fallback to describing index if host not provided
        host = pc.describe_index(settings.pinecone_index_name).host
        return pc.Index(host=host)
