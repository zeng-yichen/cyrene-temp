"""Research API — Anaxa, Cerydra, Hysilens, Mydei."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/research", tags=["research"])


class WebQueryRequest(BaseModel):
    highlighted_text: str
    query: str


class DocumentQueryRequest(BaseModel):
    company: str
    question: str
    draft_text: str | None = None


class SourceRequest(BaseModel):
    snippet: str
    company: str
    model_choice: str = "Gemini"


class ABMRequest(BaseModel):
    company: str


@router.post("/web")
async def web_query(req: WebQueryRequest):
    """Anaxa — ask the web about highlighted text."""
    from backend.src.agents.anaxa import Anaxa
    anaxa = Anaxa()
    result = anaxa.query_with_search(req.highlighted_text, req.query)
    return {"result": result}


@router.post("/documents")
async def document_query(req: DocumentQueryRequest):
    """Cerydra — document-grounded Q&A."""
    from backend.src.agents.cerydra import Cerydra
    cerydra = Cerydra()
    result = cerydra.query_documents(req.company, req.question, req.draft_text)
    return {"result": result}


@router.post("/source")
async def source_identification(req: SourceRequest):
    """Hysilens — identify the source of a snippet."""
    from backend.src.agents.hysilens import Hysilens
    hysilens = Hysilens()
    result = hysilens.find_source(req.snippet, req.company, req.model_choice)
    return {"result": result}


@router.post("/abm")
async def abm_research(req: ABMRequest):
    """Mydei — ABM target research."""
    from backend.src.agents.mydei import Mydei
    mydei = Mydei()
    result = mydei.generate_abm_briefing(req.company)
    return {"result": result}
