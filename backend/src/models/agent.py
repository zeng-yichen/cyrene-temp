"""Agent execution models."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRun(BaseModel):
    id: str
    client_slug: str
    agent: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    prompt: str | None = None
    output: str | None = None
    error: str | None = None
    config_snapshot: dict[str, Any] | None = None
    started_at: float | None = None
    completed_at: float | None = None
    created_at: float = Field(default_factory=time.time)


class AgentTool(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    output: str | None = None
    error: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
