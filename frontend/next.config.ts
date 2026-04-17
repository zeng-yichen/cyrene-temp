import type { NextConfig } from "next";

// IMPORTANT: Next.js rewrites() are resolved at build time and baked into
// the routes manifest. Runtime env var changes will NOT affect the proxy
// destination. BACKEND_URL must be set BEFORE `next build` runs.
//
// In the production container, the Dockerfile sets this via a build ARG:
//   ARG BACKEND_URL=http://amphoreus-backend.internal:8000
//
// In local dev (`next dev`), falls back to http://localhost:8000.
const BACKEND_URL =
  process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
