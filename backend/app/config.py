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
