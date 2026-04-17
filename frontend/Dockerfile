#
# Amphoreus frontend image.
#
# Build context: frontend/. Build with:
#   docker build -f frontend/Dockerfile -t amphoreus-frontend frontend/
#
# Runtime architecture:
#   - Browser hits this container over HTTPS (Cloudflare Access in front)
#   - Same-origin requests to /api/* are rewritten server-side by Next.js to
#     $BACKEND_URL (internal Fly 6PN address of amphoreus-backend)
#   - The backend is never publicly reachable
#
# Build-time env:
#   NEXT_PUBLIC_API_URL=""   → api.ts uses relative fetches (same origin)
#
# Runtime env (set via fly secrets / [env]):
#   BACKEND_URL=http://amphoreus-backend.internal:8000
#   NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY  (if used by client)
#

# ---------- builder ----------
FROM node:22-slim AS builder

WORKDIR /app

# Install deps with clean lockfile
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source
COPY . .

# next build captures environment into the routes manifest AT BUILD TIME —
# runtime env var changes will be ignored by rewrites(). So BACKEND_URL is
# a build ARG with a production-correct default. Override locally with:
#   docker build --build-arg BACKEND_URL=http://amph-be-test:8000 ...
ARG BACKEND_URL=http://amphoreus.internal:8000
ENV BACKEND_URL=${BACKEND_URL}
ENV NEXT_PUBLIC_API_URL=""
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# ---------- runner ----------
FROM node:22-slim AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Non-root user
RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs

# Next.js standalone output: a minimal self-contained server + node_modules
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# public/ is optional — include only if it exists. The `|| true` makes the
# COPY tolerant of a missing dir without failing the build.
# (Handled by .dockerignore: if public/ doesn't exist, nothing is copied.)

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:3000/').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["node", "server.js"]
