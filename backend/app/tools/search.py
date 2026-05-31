import json
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from backend.app.config import get_settings
from backend.app.services.pinecone_client import get_pinecone_index
from backend.app.services.embeddings import embed_query

def get_db_connection():
    """Returns a read-only database connection."""
    settings = get_settings()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is not set.")
    # In a production app, use a connection pool (e.g. psycopg2.pool)
    conn = psycopg2.connect(settings.database_url)
    conn.set_session(readonly=True, autocommit=True)
    return conn


@tool
def search_postgres(city: Optional[str] = None, category_id: Optional[int] = None, sub_category_id: Optional[int] = None, limit: int = 5) -> str:
    """
    Search the PostgreSQL database for exact business matches based on structured filters.
    Use this when the user asks for exact matches like a specific city or category.
    Returns a JSON string of business IDs and basic info.
    """
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT id, business_name, city, category_id, sub_category_id FROM business_listings WHERE 1=1"
            params = []
            if city:
                query += " AND city ILIKE %s"
                params.append(city)
            if category_id:
                query += " AND category_id = %s"
                params.append(category_id)
            if sub_category_id:
                query += " AND sub_category_id = %s"
                params.append(sub_category_id)
            
            query += " ORDER BY id DESC LIMIT %s"
            params.append(max(1, min(limit, 10)))
            
            cur.execute(query, tuple(params))
            results = cur.fetchall()
            return json.dumps([dict(r) for r in results])
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if 'conn' in locals(): conn.close()


@tool
def search_pinecone(semantic_query: str, city: Optional[str] = None, limit: int = 5) -> str:
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

        query_kwargs = {
            "vector": vector,
            "top_k": max(1, min(limit, 10)),
            "include_metadata": True,
        }
        if filter_dict:
            query_kwargs["filter"] = filter_dict
            
        settings = get_settings()
        if settings.pinecone_namespace:
            query_kwargs["namespace"] = settings.pinecone_namespace

        response = index.query(**query_kwargs)
        
        matches = []
        if hasattr(response, "matches"):
            for match in response.matches:
                meta = match.metadata if hasattr(match, 'metadata') else {}
                score = match.score if hasattr(match, 'score') else 0.0
                if meta.get("business_id"):
                    matches.append({
                        "business_id": int(meta["business_id"]),
                        "business_name": meta.get("business_name"),
                        "city": meta.get("city"),
                        "score": score
                    })
        return json.dumps(matches)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fetch_business_details(business_ids: List[int]) -> str:
    """
    Fetch full, detailed source-of-truth profiles for a specific list of business IDs from Postgres.
    Use this tool when you need to answer follow-up questions about contact info, phone numbers, reviews, FAQs, or full descriptions for specific businesses.
    """
    if not business_ids:
        return json.dumps([])
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # We fetch basic details here. In a true enterprise setup, we would join SEO, reviews, and FAQs as well.
            placeholders = ','.join(['%s'] * len(business_ids))
            query = f"""
                SELECT 
                    id, full_name, business_name, mobile_number, whatsapp_number, 
                    email, business_address, city, category_id, sub_category_id, message, website_url 
                FROM business_listings 
                WHERE id IN ({placeholders})
            """
            cur.execute(query, tuple(business_ids))
            results = cur.fetchall()
            return json.dumps([dict(r) for r in results])
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if 'conn' in locals(): conn.close()
