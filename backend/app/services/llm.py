from functools import lru_cache
from langchain_openai import ChatOpenAI
from backend.app.config import get_settings

@lru_cache
def get_chat_model() -> ChatOpenAI:
    """Shared Async Chat Model configured for LangGraph."""
    settings = get_settings()
    
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set in .env")

    # We use ChatOpenAI from langchain-openai because it seamlessly supports
    # Async API calls, Tool Calling, and OpenAI-compatible endpoints like NVIDIA's v1 api.
    return ChatOpenAI(
        model=settings.llm_chat_model,
        temperature=0,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        streaming=True,
    )

@lru_cache
def get_grader_model() -> ChatOpenAI:
    """Shared Async Chat Model configured for Grader Node (Fast & Cheap)."""
    settings = get_settings()
    
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set in .env")

    return ChatOpenAI(
        model=settings.grader_chat_model,
        temperature=0,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        # We don't need streaming for structured output typically
        streaming=False, 
    )
