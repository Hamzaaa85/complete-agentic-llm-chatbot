import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings (NVIDIA API)
    llm_api_key: str = ""
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_chat_model: str = "openai/gpt-oss-120b"

    # NVIDIA Reranking settings
    nvidia_api_key: str = ""
    rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    rerank_top_k: int = 5
    rerank_relevance_threshold: float = -4.0  # As discussed, keep it low/un-strict

    # Grader/Self-Correction settings
    grader_chat_model: str = "openai/gpt-oss-20"  # Fast model for self-reflection
    max_search_attempts: int = 3

    # OpenAI Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 1024

    # Postgres Database
    database_url: str = ""
    db_pool_min: int = 2
    db_pool_max: int = 20

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "new-business-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_host: str | None = None
    pinecone_namespace: str | None = None

@lru_cache
def get_settings() -> Settings:
    return Settings()
