from langchain_openai import OpenAIEmbeddings
from backend.app.config import get_settings

def get_embeddings_model() -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
        openai_api_key=settings.openai_api_key,
    )

def embed_query(text: str) -> list[float]:
    model = get_embeddings_model()
    return model.embed_query(text)
