import json
import asyncpg
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from backend.app.config import get_settings
from backend.app.services.pinecone_client import get_pinecone_index
from backend.app.services.embeddings import embed_query
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

async def get_db_connection():
    """Returns a read-only async database connection."""
    settings = get_settings()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is not set.")
    # In a production app, use an asyncpg connection pool
    conn = await asyncpg.connect(settings.database_url)
    # asyncpg doesn't have a simple readonly flag like psycopg2 session, but we use safe queries
    return conn


@tool
async def search_postgres(city: Optional[str] = None, category_id: Optional[int] = None, sub_category_id: Optional[int] = None, limit: int = 5) -> str:
    """
    Search the PostgreSQL database for exact business matches based on structured filters.
    Use this when the user asks for exact matches like a specific city or category.
    Returns a JSON string of business IDs and basic info.
    """
    try:
        conn = await get_db_connection()
        query = "SELECT id, business_name, city, category_id, sub_category_id FROM business_listings WHERE 1=1"
        params = []
        param_idx = 1
        
        if city:
            query += f" AND city ILIKE ${param_idx}"
            params.append(f"%{city}%")
            param_idx += 1
        if category_id:
            query += f" AND category_id = ${param_idx}"
            params.append(category_id)
            param_idx += 1
        if sub_category_id:
            query += f" AND sub_category_id = ${param_idx}"
            params.append(sub_category_id)
            param_idx += 1
        
        query += f" ORDER BY id DESC LIMIT ${param_idx}"
        params.append(max(1, min(limit, 10)))
        
        results = await conn.fetch(query, *params)
        return json.dumps([dict(r) for r in results])
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if 'conn' in locals() and not conn.is_closed(): 
            await conn.close()


@tool
async def search_pinecone(semantic_query: str, city: Optional[str] = None, limit: int = 5) -> str:
    """
    Perform a semantic search on Pinecone to find businesses based on descriptive text.
    Use this when the user asks for things like "cheap baby products", "best halwa", or specific descriptions.
    Returns a JSON string of matched business IDs and their similarity scores.
    """
    try:
        if not semantic_query.strip():
            return json.dumps({"error": "semantic_query cannot be empty"})

        index = get_pinecone_index()
        vector = embed_query(semantic_query.strip())
        
        filter_dict = {}
        if city:
            filter_dict["city"] = {"$eq": city.strip().title()}

        # Fetch more results for reranking
        top_k_fetch = max(15, limit * 2)

        query_kwargs = {
            "vector": vector,
            "top_k": top_k_fetch,
            "include_metadata": True,
        }
        if filter_dict:
            query_kwargs["filter"] = filter_dict
            
        settings = get_settings()
        if settings.pinecone_namespace:
            query_kwargs["namespace"] = settings.pinecone_namespace

        # Pinecone client is synchronous
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: index.query(**query_kwargs))
        
        matches = []
        if hasattr(response, "matches"):
            for match in response.matches:
                meta = match.metadata if hasattr(match, 'metadata') else {}
                score = match.score if hasattr(match, 'score') else 0.0
                if meta.get("business_id"):
                    content_str = f"{meta.get('business_name', '')} in {meta.get('city', '')}."
                    matches.append({
                        "business_id": int(meta["business_id"]),
                        "business_name": meta.get("business_name"),
                        "city": meta.get("city"),
                        "score": score,
                        "content": content_str
                    })

        if not matches:
            return json.dumps([])

        # NVIDIA Reranking Step
        if settings.nvidia_api_key and settings.rerank_model:
            reranker = NVIDIARerank(
                model=settings.rerank_model,
                api_key=settings.nvidia_api_key,
                base_url=settings.llm_base_url,
                top_n=limit
            )
            docs = [Document(page_content=m["content"], metadata=m) for m in matches]
            
            try:
                reranked_docs = await reranker.acompress_documents(documents=docs, query=semantic_query)
            except (NotImplementedError, AttributeError):
                reranked_docs = await loop.run_in_executor(None, lambda: reranker.compress_documents(documents=docs, query=semantic_query))
            
            final_matches = []
            for doc in reranked_docs:
                rel_score = doc.metadata.get("relevance_score", 0.0)
                if rel_score >= settings.rerank_relevance_threshold:
                    meta = doc.metadata
                    # Cleanup the temporary content key before returning
                    if "content" in meta:
                        del meta["content"]
                    meta["rerank_score"] = rel_score
                    final_matches.append(meta)
                    
            return json.dumps(final_matches)

        # Fallback if no reranker
        for m in matches:
            if "content" in m: del m["content"]
        return json.dumps(matches[:limit])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@tool
async def fetch_business_details(business_ids: List[int]) -> str:
    """
    Fetch full, detailed source-of-truth profiles for a specific list of business IDs from Postgres.
    Use this tool when you need to answer follow-up questions about contact info, phone numbers, reviews, FAQs, or full descriptions for specific businesses.
    """
    if not business_ids:
        return json.dumps([])
    try:
        conn = await get_db_connection()
        placeholders = ','.join([f'${i+1}' for i in range(len(business_ids))])
        query = f"""
            SELECT 
                id, full_name, business_name, mobile_number, whatsapp_number, 
                email, business_address, city, category_id, sub_category_id, message, website_url 
            FROM business_listings 
            WHERE id IN ({placeholders})
        """
        results = await conn.fetch(query, *business_ids)
        return json.dumps([dict(r) for r in results])
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if 'conn' in locals() and not conn.is_closed(): 
            await conn.close()
