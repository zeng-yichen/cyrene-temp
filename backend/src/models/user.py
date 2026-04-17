"""User and organization models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class User(BaseModel):
    id: str
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    company_id: str | None = None
    role: str = ""
    gets_content: bool = False
    customer_type: str = ""
    linkedin_url: str | None = None
    title: str | None = None


class Company(BaseModel):
    id: str
    name: str = ""
    slug: str = ""


class UserAccess(BaseModel):
    accessor_id: str
    target_id: str
    access_type: str = "read"


class Role(BaseModel):
    id: str
    name: str = ""
    permissions: list[str] = []


class AuthUser(BaseModel):
    """Authenticated user extracted from JWT."""
    user_id: str
    email: str = ""
    role: Literal["internal", "external", "unauthorized"] = "external"
    company_id: str | None = None
