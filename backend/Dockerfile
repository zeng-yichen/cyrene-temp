#
# Amphoreus backend image.
#
# Build context: repo root. Build with:
#   docker build -f backend/Dockerfile -t amphoreus-backend .
# Or, on Fly:
#   fly deploy --config backend/fly.toml --dockerfile backend/Dockerfile --remote-only
#
# (The --dockerfile flag is required because fly.toml's [build] dockerfile
# field is resolved relative to the fly.toml directory, which would produce
# backend/backend/Dockerfile. The CLI flag lets us specify a repo-root-
# relative path while still using the repo root as the build context.)
#
# On Fly.io this image runs as root so docker-entrypoint.sh can symlink
# the mounted volume (/data) into the in-image paths that the application
# code expects (/app/memory, /app/backend/data, /app/backend/static/images,
# /app/products, /app/output). SQLite lives at /data/sqlite/amphoreus.db
# via the SQLITE_PATH env var.
#
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps:
#   curl, ca-certificates — general
#   libportaudio2         — required by sounddevice (Tribbie audio capture)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# uv: fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy backend source. Context = repo root, so this copies `backend/` into
# /app/backend, preserving the module layout for `uvicorn backend.src.main:app`.
# .dockerignore (at repo root) excludes __pycache__, .venv, memory/, data/, etc.
COPY backend /app/backend

# Install dependencies + the amphoreus package into the system site-packages.
# We intentionally do NOT install the [dev] extras in production.
RUN uv pip install --system --no-cache /app/backend

# Entrypoint wires the persistent volume into the in-image paths.
COPY backend/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Basic healthcheck — Fly also auto-restarts on failure via fly.toml.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Bind to :: (dual-stack v4+v6) so Fly's 6PN private network (IPv6-only)
# can reach the backend via amphoreus.internal:8000. Binding to 0.0.0.0
# only listens on IPv4 and makes .internal DNS unusable.
CMD ["uvicorn", "backend.src.main:app", "--host", "::", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
