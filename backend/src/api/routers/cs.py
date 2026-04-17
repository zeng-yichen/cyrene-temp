"""CS Dashboard API — client health, metrics, analytics."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cs", tags=["cs"])


@router.get("/clients")
async def list_clients():
    """List clients with health summary."""
    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()
        result = (
            sb.table("users")
            .select("id, first_name, last_name, email, company_id, gets_content, customer_type, title")
            .eq("gets_content", True)
            .execute()
        )
        return {"clients": result.data or []}
    except Exception as e:
        logger.warning("Failed to fetch clients: %s", e)
        return {"clients": []}


@router.get("/clients/{client_id}")
async def get_client_detail(client_id: str):
    """Get detailed client info."""
    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()
        user_result = (
            sb.table("users")
            .select("*")
            .eq("id", client_id)
            .limit(1)
            .execute()
        )
        posts_result = (
            sb.table("posts")
            .select("id, hook, status, post_date, created_at")
            .eq("face_of_content_user_id", client_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return {
            "user": user_result.data[0] if user_result.data else None,
            "posts": posts_result.data or [],
        }
    except Exception as e:
        logger.warning("Failed to fetch client detail: %s", e)
        return {"user": None, "posts": []}


class QueryRequest(BaseModel):
    query: str


@router.post("/query/execute")
async def execute_query(req: QueryRequest):
    """Execute an analytics query (via LLM SQL generation)."""
    return {"result": [], "query": req.query, "note": "Query execution not yet implemented"}
