"""Transcripts router — add-only file management for ``memory/{slug}/transcripts/``.

Content engineers use this to drop raw context (interview transcripts, reference
docs, notes) into a client's knowledge directory. Everything Stelle reads is
seeded here; the add-only UX is intentional so we don't lose provenance.

ACL model:
- GET (list, download): enforced by the global ``require_client_from_path``
  dependency (admin ⇒ all, scoped user ⇒ their slug allowlist).
- POST (upload / paste): same ACL — any user with access to the client can add.
- DELETE: **admin-only**, checked explicitly in the handler. Scoped users cannot
  delete via the UI; they must ask an admin. This keeps an immutable audit trail
  for content engineers while giving admins an escape hatch.

Metadata: we maintain a hidden JSONL sidecar at ``transcripts/.uploads.jsonl``
that records ``{filename, source_label, uploaded_by, uploaded_at, size,
original_filename, content_type}`` on every upload. The list endpoint joins the
real directory with this sidecar so the UI can show a human-readable "source"
label instead of meaningless UUID filenames.

Why JSONL and not one meta file per transcript:
- Append-only = safe for concurrent writes (no lockfile needed).
- Deletes don't need to rewrite anything — the list endpoint just filters to
  files that still exist on disk.
- Easy to grep / audit by hand.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.src.auth.cf_access import AuthedUser
from backend.src.db import vortex

logger = logging.getLogger("amphoreus.transcripts")

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])

# --- Config ---------------------------------------------------------------

# 100 MB hard cap per upload. Generous enough for long PDF transcripts while
# still bounding what a scoped user can do if their credentials are stolen.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Extensions we'll accept. Everything else returns 400. Intentionally narrow:
# agents that read this dir glob by extension, and we don't want random
# binaries parked in a context-loaded directory.
ALLOWED_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".vtt",
        ".srt",
        ".pdf",
        ".docx",
        ".doc",
        ".rtf",
        ".json",
        ".csv",
        ".html",
    }
)

# Sidecar file living inside the transcripts dir. Dot-prefixed so it's hidden
# from the list endpoint (we filter dotfiles) and from ``ls`` on the volume.
UPLOADS_LOG_NAME = ".uploads.jsonl"


# --- Helpers --------------------------------------------------------------


def _sanitize_filename(filename: str) -> str:
    """Strip path separators and return the trailing component, preserving extension.

    This is the last line of defence against ``..`` / absolute paths in the
    ``filename`` path param on DELETE and download. We also call it on upload
    to sanitize ``UploadFile.filename`` before extracting the extension.
    """
    name = Path(filename).name  # strips any leading dirs
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    # Reject anything that tries to be a dotfile (would hide from listing) or
    # contains path separators after the .name strip (belt and braces).
    if name.startswith(".") or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def _extension_of(filename: str) -> str:
    """Return the lowercase extension (with leading dot), or ''."""
    return Path(filename).suffix.lower()


def _require_admin(request: Request) -> AuthedUser:
    """Explicit admin gate. Used on DELETE only."""
    user: AuthedUser | None = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="unauthenticated")
    if not getattr(request.state, "user_is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete transcripts.",
        )
    return user


def _current_user_email(request: Request) -> str:
    user: AuthedUser | None = getattr(request.state, "user", None)
    return user.email if user else ""


def _transcripts_dir(company: str) -> Path:
    """Return the transcripts dir for ``company``, creating it if needed.

    Some clients that haven't been touched by the sync loop yet may
    not have a directory at all — we create it lazily on the first upload
    so onboarding is seamless.
    """
    target = vortex.transcripts_dir(company)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _uploads_log_path(company: str) -> Path:
    return _transcripts_dir(company) / UPLOADS_LOG_NAME


def _append_upload_log(company: str, entry: dict[str, Any]) -> None:
    """Best-effort append to the JSONL sidecar. Non-fatal on failure."""
    try:
        with open(_uploads_log_path(company), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        logger.exception("[transcripts] failed to append upload log for %s", company)


def _load_upload_log(company: str) -> dict[str, dict[str, Any]]:
    """Return a ``{filename: latest_metadata_dict}`` map from the JSONL sidecar.

    Later lines overwrite earlier ones for the same filename, which gives us
    free support for future metadata edits (just append a new line).
    """
    log_path = _uploads_log_path(company)
    if not log_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fname = entry.get("filename")
                if isinstance(fname, str):
                    out[fname] = entry
    except Exception:
        logger.exception("[transcripts] failed to read upload log for %s", company)
    return out


# --- Response / request models -------------------------------------------


class TranscriptFile(BaseModel):
    filename: str
    size_bytes: int
    modified_at: float  # unix epoch seconds
    source_label: str | None = None
    uploaded_by: str | None = None
    uploaded_at: float | None = None
    original_filename: str | None = None
    content_type: str | None = None


class ListResponse(BaseModel):
    company: str
    files: list[TranscriptFile]


class PasteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_UPLOAD_BYTES)
    source_label: str = Field(..., min_length=1, max_length=200)


class UploadResponse(BaseModel):
    status: str
    filename: str
    size_bytes: int
    source_label: str


class DeleteResponse(BaseModel):
    status: str
    filename: str


# --- Routes ---------------------------------------------------------------


@router.get("/{company}", response_model=ListResponse)
async def list_transcripts(company: str) -> ListResponse:
    """List every file in ``memory/{company}/transcripts/``.

    Joins the filesystem with the upload log so the UI sees a ``source_label``
    where available. Files that pre-date the upload log (or were dropped on
    disk via SSH) show up with ``source_label=None`` — that's expected.
    """
    tdir = _transcripts_dir(company)
    log = _load_upload_log(company)
    files: list[TranscriptFile] = []
    for entry in sorted(tdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not entry.is_file():
            continue
        # Hide the log sidecar + any dotfile from the listing.
        if entry.name.startswith("."):
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        meta = log.get(entry.name, {})
        files.append(
            TranscriptFile(
                filename=entry.name,
                size_bytes=st.st_size,
                modified_at=st.st_mtime,
                source_label=meta.get("source_label"),
                uploaded_by=meta.get("uploaded_by"),
                uploaded_at=meta.get("uploaded_at"),
                original_filename=meta.get("original_filename"),
                content_type=meta.get("content_type"),
            )
        )
    return ListResponse(company=company, files=files)


@router.post("/{company}/upload", response_model=UploadResponse)
async def upload_transcript(
    company: str,
    request: Request,
    file: UploadFile = File(...),
    source_label: str = Form(...),
) -> UploadResponse:
    """Multipart file upload. Streams to disk under a random UUID filename."""
    if not source_label.strip():
        raise HTTPException(status_code=400, detail="source_label is required")
    original = _sanitize_filename(file.filename or "upload.bin")
    ext = _extension_of(original)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension '{ext}' not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tdir = _transcripts_dir(company)
    random_name = f"{uuid.uuid4().hex}{ext}"
    dest = tdir / random_name

    # Stream in 1 MB chunks so a 100 MB upload doesn't sit in RAM.
    total = 0
    CHUNK = 1024 * 1024
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.exception("[transcripts] upload failed for %s", company)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}") from e

    _append_upload_log(
        company,
        {
            "filename": random_name,
            "source_label": source_label.strip(),
            "uploaded_by": _current_user_email(request),
            "uploaded_at": time.time(),
            "size": total,
            "original_filename": original,
            "content_type": file.content_type or mimetypes.guess_type(original)[0] or "",
        },
    )
    logger.info(
        "[transcripts] upload %s/%s (%d bytes, label=%r) by %s",
        company,
        random_name,
        total,
        source_label.strip()[:60],
        _current_user_email(request),
    )
    return UploadResponse(
        status="uploaded",
        filename=random_name,
        size_bytes=total,
        source_label=source_label.strip(),
    )


@router.post("/{company}/paste", response_model=UploadResponse)
async def paste_transcript(
    company: str,
    req: PasteRequest,
    request: Request,
) -> UploadResponse:
    """Save a blob of pasted text as a new ``.txt`` transcript."""
    text_bytes = req.text.encode("utf-8")
    if len(text_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Pasted text exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap",
        )
    tdir = _transcripts_dir(company)
    random_name = f"{uuid.uuid4().hex}.txt"
    dest = tdir / random_name
    try:
        with open(dest, "wb") as out:
            out.write(text_bytes)
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.exception("[transcripts] paste failed for %s", company)
        raise HTTPException(status_code=500, detail=f"Paste failed: {e}") from e

    _append_upload_log(
        company,
        {
            "filename": random_name,
            "source_label": req.source_label.strip(),
            "uploaded_by": _current_user_email(request),
            "uploaded_at": time.time(),
            "size": len(text_bytes),
            "original_filename": None,
            "content_type": "text/plain",
        },
    )
    logger.info(
        "[transcripts] paste %s/%s (%d bytes, label=%r) by %s",
        company,
        random_name,
        len(text_bytes),
        req.source_label.strip()[:60],
        _current_user_email(request),
    )
    return UploadResponse(
        status="pasted",
        filename=random_name,
        size_bytes=len(text_bytes),
        source_label=req.source_label.strip(),
    )


@router.get("/{company}/{filename}")
async def download_transcript(company: str, filename: str) -> FileResponse:
    """Download / preview a single transcript. ACL'd by the global dep."""
    safe = _sanitize_filename(filename)
    path = _transcripts_dir(company) / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Transcript not found")
    return FileResponse(path, filename=safe)


@router.delete("/{company}/{filename}", response_model=DeleteResponse)
async def delete_transcript(
    company: str,
    filename: str,
    request: Request,
) -> DeleteResponse:
    """Admin-only: permanently delete a transcript.

    Scoped users get 403 here even if the ``{company}`` path param resolves to
    one of their allowed clients. We intentionally don't move to trash — the
    audit log captures who + when, and admins shouldn't be hitting this often.
    """
    _require_admin(request)
    safe = _sanitize_filename(filename)
    path = _transcripts_dir(company) / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Refusing to delete non-file")
    try:
        path.unlink()
    except Exception as e:
        logger.exception("[transcripts] delete failed for %s/%s", company, safe)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}") from e

    logger.info(
        "[transcripts] delete %s/%s by %s (admin)",
        company,
        safe,
        _current_user_email(request),
    )
    return DeleteResponse(status="deleted", filename=safe)


# Keep shutil imported even if unused right now — future "move to trash" code
# path will want it and linters will complain otherwise.
_ = shutil
