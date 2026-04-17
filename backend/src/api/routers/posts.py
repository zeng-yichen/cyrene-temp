"""Posts API — CRUD, rewrite, fact-check, push to Ordinal."""

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.src.core.config import get_settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/posts", tags=["posts"])


def _parse_publish_at_iso(raw: str | None) -> datetime | None:
    """Parse client ISO string (often ends with Z) to timezone-aware UTC."""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def _castorice_entry_from_row(row: dict) -> dict | None:
    """Build Hyacinthia castorice_result dict from a local_posts row, or None if empty."""
    citations: list[str] = []
    cc_raw = row.get("citation_comments")
    if cc_raw:
        try:
            parsed = json.loads(cc_raw)
            if isinstance(parsed, list):
                citations = [str(x) for x in parsed]
        except json.JSONDecodeError:
            logger.warning("[posts] Invalid citation_comments JSON for post %s", row.get("id"))
    why_post = (row.get("why_post") or "").strip()
    if not citations and not why_post:
        return None
    entry: dict = {"citation_comments": citations}
    if why_post:
        entry["why_post"] = why_post
    return entry


def _utc_publish_at_nine(d: datetime) -> datetime:
    """Match push_drafts cadence: calendar day at 09:00 naive UTC for Ordinal publishAt."""
    return datetime(d.year, d.month, d.day, 9, 0, 0)


def _linkedin_asset_ids_for_row(en, company_slug: str, row: dict | None) -> list[str]:
    """Upload approved draft image to Ordinal by public URL; returns asset UUIDs for linkedIn.assetIds."""
    if not row:
        return []
    stem = (row.get("linked_image_id") or "").strip()
    if not stem:
        return []
    base = (get_settings().public_base_url or "").strip().rstrip("/")
    if not base:
        logger.warning(
            "[posts] Draft has linked_image_id=%s but PUBLIC_BASE_URL is unset — "
            "Ordinal cannot fetch the image; attach skipped.",
            stem,
        )
        return []
    public_url = f"{base}/api/images/{company_slug}/{stem}"
    aid = en.upload_asset_from_public_url(company_slug, public_url)
    return [aid] if aid else []


class CreatePostRequest(BaseModel):
    company: str
    content: str
    title: str | None = None
    scheduled_at: str | None = None
    status: str = "draft"


class PatchPostRequest(BaseModel):
    """PATCH body — use model_dump(exclude_unset=True) so omitted fields are not cleared."""

    company: str
    content: str | None = None
    title: str | None = None
    status: str | None = None
    linked_image_id: str | None = None


class RewriteRequest(BaseModel):
    company: str
    post_text: str
    style_instruction: str = ""
    client_context: str = ""


class FactCheckRequest(BaseModel):
    company: str
    post_text: str


class PushApproval(BaseModel):
    """Ordinal POST /approvals item (camelCase keys for API)."""

    userId: str
    message: str | None = None
    dueDate: str | None = None
    isBlocking: bool = False


class PushRequest(BaseModel):
    company: str
    content: str = ""
    post_id: str | None = Field(
        default=None,
        description="When set, load content, citation_comments, and why_post from this local draft row.",
    )
    model_name: str = "stelle"
    posts_per_month: int = 12
    start_date: str | None = None
    citation_comments: list[str] = []
    publish_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC instant for publishAt on Ordinal (e.g. from Date.toISOString()).",
    )
    approvals: list[PushApproval] = []


class PushAllRequest(BaseModel):
    """Push every local draft for a company to Ordinal on cadence slots (UTC)."""

    company: str
    posts_per_month: int = Field(
        12,
        description="12 → Mon/Wed/Thu slots; 8 → Tue/Thu slots (same as Hyacinthia).",
    )
    approvals: list[PushApproval] = []

    @field_validator("posts_per_month")
    @classmethod
    def posts_per_month_must_be_8_or_12(cls, v: int) -> int:
        if v not in (8, 12):
            raise ValueError("posts_per_month must be 8 or 12")
        return v


@router.get("")
async def list_posts(company: str | None = None, limit: int = 50):
    """List local draft posts for a client."""
    from backend.src.db.local import list_local_posts
    posts = list_local_posts(company=company, limit=limit)
    return {"posts": posts}


@router.post("")
async def create_post(req: CreatePostRequest):
    """Create a new local draft post."""
    from backend.src.db.local import create_local_post
    post_id = str(uuid.uuid4())
    post = create_local_post(
        post_id=post_id,
        company=req.company,
        content=req.content,
        title=req.title,
        status=req.status,
    )
    return {"post": post}


@router.patch("/{post_id}")
async def update_post(post_id: str, req: PatchPostRequest):
    from backend.src.db.local import get_local_post, update_local_post_fields

    raw = req.model_dump(exclude_unset=True)
    raw.pop("company", None)
    if not raw:
        post = get_local_post(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return {"post": post}
    post = update_local_post_fields(post_id, raw)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"post": post}


@router.delete("/{post_id}")
async def delete_post(post_id: str):
    from backend.src.db.local import delete_local_post
    delete_local_post(post_id)
    return {"deleted": True}


@router.post("/{post_id}/rewrite")
async def rewrite_post(post_id: str, req: RewriteRequest):
    """Rewrite a post via Cyrene."""
    from backend.src.agents.demiurge import Cyrene
    cyrene = Cyrene()
    result = cyrene.rewrite_single_post(
        post_text=req.post_text,
        style_instruction=req.style_instruction,
        client_context=req.client_context,
    )
    return {"result": result}


@router.post("/{post_id}/fact-check")
async def fact_check_post(post_id: str, req: FactCheckRequest):
    """Fact-check and annotate a post via Castorice. Saves annotated version locally."""
    from backend.src.agents.castorice import Castorice
    from backend.src.db.vortex import castorice_annotated_path
    result = Castorice().fact_check_post(req.company, req.post_text)

    # Persist the annotated version for CE review
    annotated = result.get("annotated_post", "")
    if annotated:
        ann_path = castorice_annotated_path(req.company)
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(annotated, encoding="utf-8")
        logger.info("[posts] Annotated post saved to %s", ann_path)

    return {
        "report": result.get("report", ""),
        "corrected_post": result.get("corrected_post", ""),
        "annotated_post": annotated,
        "citation_comments": result.get("citation_comments", []),
    }


@router.post("/push")
async def push_to_ordinal(req: PushRequest):
    """Push posts to Ordinal via Hyacinthia. Citation comments and why-post blurb
    come from the local draft when post_id is set; otherwise citation_comments
    in the request body are used (legacy)."""
    from backend.src.agents.hyacinthia import Hyacinthia
    from backend.src.db.local import get_local_post, set_local_post_ordinal_post_id

    en = Hyacinthia()
    start = datetime.fromisoformat(req.start_date) if req.start_date else None
    sched = _parse_publish_at_iso(req.publish_at)
    approval_dicts = [a.model_dump(exclude_none=True) for a in req.approvals]

    push_content = req.content
    castorice_results = None
    inline_single = False
    single_title: str | None = None
    local_row: dict | None = None

    pid = (req.post_id or "").strip() if req.post_id else ""
    if pid:
        local_row = get_local_post(pid)
        if not local_row:
            raise HTTPException(status_code=404, detail="Post not found")
        row_company = (local_row.get("company") or "").strip().lower()
        if row_company != req.company.strip().lower():
            raise HTTPException(status_code=403, detail="Post does not belong to this company")
        push_content = local_row.get("content") or ""
        if not (local_row.get("why_post") or "").strip() and push_content.strip():
            try:
                from backend.src.agents.stelle import _compose_rationale
                why = _compose_rationale(push_content, "draft", req.company, req.company)
                if why:
                    local_row["why_post"] = why
                    from backend.src.db.local import get_connection
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE local_posts SET why_post = ? WHERE id = ?",
                            (why, pid),
                        )
                    logger.info("[posts] Lazy-generated why_post for %s", pid[:12])
            except Exception as e:
                logger.debug("[posts] Lazy why_post generation failed: %s", e)
        entry = _castorice_entry_from_row(local_row)
        if entry:
            castorice_results = [entry]
        inline_single = True
        single_title = local_row.get("title")
    elif req.citation_comments:
        castorice_results = [{"citation_comments": req.citation_comments}]

    if not (push_content or "").strip():
        raise HTTPException(
            status_code=400,
            detail="No post content to push (provide content or a valid post_id with saved body).",
        )

    company_key = req.company.strip()
    per_post_opt: list | None = None
    if inline_single and local_row:
        li_assets = _linkedin_asset_ids_for_row(en, company_key, local_row)
        if li_assets:
            per_post_opt = [{"linkedin_asset_ids": li_assets}]

    success, result, ordinal_ids = en.push_drafts(
        company_keyword=req.company,
        model_name=req.model_name,
        content=push_content,
        posts_per_month=req.posts_per_month,
        start_date=start,
        castorice_results=castorice_results,
        schedule_publish_at=sched,
        default_approvals=approval_dicts if approval_dicts else None,
        prefer_rewritten_file=not inline_single,
        inline_single_post=inline_single,
        single_post_title=single_title,
        per_post=per_post_opt,
    )
    if pid and success and ordinal_ids and local_row:
        new_oid = (ordinal_ids[0] or "").strip()
        if new_oid:
            old_oid = (local_row.get("ordinal_post_id") or "").strip()
            if old_oid and old_oid != new_oid:
                en.remove_draft_map_entry(company_key, old_oid)
            set_local_post_ordinal_post_id(pid, new_oid)
            try:
                from backend.src.agents.ruan_mei import RuanMei

                RuanMei(company_key).link_ordinal_post_id(pid, new_oid)
            except Exception:
                logger.debug("[posts] RuanMei link_ordinal_post_id skipped", exc_info=True)

    return {
        "success": success,
        "result": result,
        "ordinal_post_ids": ordinal_ids,
    }


@router.post("/push-all")
async def push_all_drafts_to_ordinal(req: PushAllRequest):
    """Push each local draft to Ordinal on the next available cadence days (UTC, 09:00)."""
    from backend.src.agents.hyacinthia import Hyacinthia
    from backend.src.agents.ruan_mei import RuanMei
    from backend.src.db.local import list_local_posts, set_local_post_ordinal_post_id

    en = Hyacinthia()
    rm_link = RuanMei(req.company.strip())
    approval_dicts = [a.model_dump(exclude_none=True) for a in req.approvals]

    rows = list_local_posts(company=req.company.strip(), limit=500)
    drafts = [r for r in rows if (r.get("status") or "draft") == "draft" and (r.get("content") or "").strip()]
    drafts.sort(key=lambda r: float(r.get("created_at") or 0))

    if not drafts:
        raise HTTPException(status_code=400, detail="No draft posts with content to push for this company.")

    start0 = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Use Temporal Orchestrator when available (data-driven day+hour),
    # fall back to fixed cadence + 09:00 UTC otherwise.
    _orchestrated = False
    publish_days = []
    co = req.company.strip()
    try:
        from backend.src.services.temporal_orchestrator import compute_publish_dates_optimized
        publish_days = compute_publish_dates_optimized(co, len(drafts), start0)
        if publish_days:
            _orchestrated = True
    except Exception:
        pass
    if not publish_days:
        publish_days = en._compute_publish_dates(start0, len(drafts), req.posts_per_month)

    cadence = "Temporal Orchestrator" if _orchestrated else ("Mon/Wed/Thu" if req.posts_per_month == 12 else "Tue/Thu")

    pushed = 0
    errors: list[str] = []
    first_url: str | None = None

    for i, row in enumerate(drafts):
        pub = publish_days[i] if _orchestrated else _utc_publish_at_nine(publish_days[i])
        if not (row.get("why_post") or "").strip() and (row.get("content") or "").strip():
            try:
                from backend.src.agents.stelle import _compose_rationale
                why = _compose_rationale(row["content"], "draft", co, co)
                if why:
                    row["why_post"] = why
                    from backend.src.db.local import get_connection
                    with get_connection() as conn:
                        conn.execute("UPDATE local_posts SET why_post = ? WHERE id = ?", (why, row["id"]))
            except Exception:
                pass
        cr = _castorice_entry_from_row(row)
        li_assets = _linkedin_asset_ids_for_row(en, co, row)
        # Parse generation metadata from local_post for draft_map persistence
        _gen_meta = None
        _gen_meta_raw = row.get("generation_metadata")
        if _gen_meta_raw:
            try:
                _gen_meta = json.loads(_gen_meta_raw) if isinstance(_gen_meta_raw, str) else _gen_meta_raw
            except Exception:
                pass
        res = en.push_single_post(
            company_keyword=co,
            content=(row.get("content") or "").strip(),
            publish_date=pub,
            title=(row.get("title") or None),
            approvals=approval_dicts if approval_dicts else None,
            castorice_result=cr,
            linkedin_asset_ids=li_assets if li_assets else None,
            generation_metadata=_gen_meta,
        )
        if res.get("success"):
            pushed += 1
            if not first_url:
                first_url = res.get("url")
            new_oid = (res.get("post_id") or "").strip()
            local_id = row.get("id")
            if new_oid and local_id:
                old_oid = (row.get("ordinal_post_id") or "").strip()
                if old_oid and old_oid != new_oid:
                    en.remove_draft_map_entry(req.company.strip(), old_oid)
                set_local_post_ordinal_post_id(local_id, new_oid)
                try:
                    rm_link.link_ordinal_post_id(local_id, new_oid)
                except Exception:
                    logger.debug("[posts] RuanMei link (push-all) skipped", exc_info=True)
        else:
            errors.append(f"{row.get('id', '?')}: {res.get('error', 'unknown error')}")

    return {
        "success": pushed > 0,
        "pushed": pushed,
        "total": len(drafts),
        "failed": len(drafts) - pushed,
        "cadence": cadence,
        "first_url": first_url,
        "errors": errors,
    }
