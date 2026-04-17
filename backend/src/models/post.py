"""Post and draft models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Post(BaseModel):
    id: str | None = None
    face_of_content_user_id: str | None = None
    post_text: str = ""
    hook: str = ""
    status: str = "draft"
    post_date: str | None = None
    source_type: str | None = None
    created_at: str | None = None


class Draft(BaseModel):
    id: str | None = None
    company_id: str | None = None
    user_id: str | None = None
    content: str = ""
    title: str = ""
    status: Literal["draft", "review", "approved", "scheduled", "publishing", "published", "failed", "archived"] = "draft"
    scheduled_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DraftComment(BaseModel):
    id: str | None = None
    draft_id: str
    user_id: str | None = None
    message: str = ""
    highlighted_text: str | None = None
    paragraph_index: int | None = None
    resolved: bool = False
    parent_id: str | None = None
    created_at: datetime | None = None


class DraftSnapshot(BaseModel):
    id: str | None = None
    draft_id: str
    content: str = ""
    version: int = 0
    created_at: datetime | None = None


class PostParseResult(BaseModel):
    theme: str = ""
    content: str = ""


class FactCheckResult(BaseModel):
    post_index: int
    report: str = ""
    corrected_post: str | None = None
    has_corrections: bool = False
