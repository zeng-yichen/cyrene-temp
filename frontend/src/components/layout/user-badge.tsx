"use client";

/**
 * Fixed top-right badge that shows the logged-in user's email and a sign-out
 * link that hits Cloudflare Access's `/cdn-cgi/access/logout` endpoint. When
 * auth is disabled (local dev), shows a bright warning pill so nobody mistakes
 * the bypass for production behavior.
 */

import { useMe } from "@/lib/auth-context";

export function UserBadge() {
  const { me, loading, error } = useMe();

  // Loading and dev-bypass warnings live in the BOTTOM-LEFT corner with
  // `pointer-events-none` so they can never block clicks on app chrome.
  // The authenticated badge (with sign-out link) goes top-right because it
  // needs to be interactive and near the typical "account menu" area.

  if (loading) {
    return (
      <div className="pointer-events-none fixed bottom-3 left-3 z-50 rounded-full border border-stone-200 bg-white/70 px-2 py-0.5 text-[10px] text-stone-400 shadow-sm backdrop-blur">
        auth…
      </div>
    );
  }

  if (error) {
    return (
      <div className="pointer-events-none fixed bottom-3 left-3 z-50 rounded-full border border-red-200 bg-red-50 px-2 py-0.5 text-[10px] font-medium text-red-700 shadow-sm">
        auth error
      </div>
    );
  }

  if (!me) return null;

  // Local-dev bypass warning — tiny, bottom-left, click-through.
  if (!me.auth_enabled) {
    return (
      <div
        className="pointer-events-none fixed bottom-3 left-3 z-50 rounded-full border border-amber-300 bg-amber-100/90 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-900 shadow-sm"
        title="Local dev: CF Access JWT verification disabled. NEVER run this way in prod."
      >
        dev auth bypass
      </div>
    );
  }

  // Authenticated badge — interactive, top-right.
  const scopeLabel = me.is_admin
    ? "admin"
    : me.allowed_clients === "*"
      ? "all"
      : `${me.allowed_clients.length} clients`;

  return (
    <div className="fixed right-4 top-4 z-50 flex items-center gap-2 rounded-full border border-stone-200 bg-white/90 px-3 py-1 text-xs text-stone-700 shadow-sm backdrop-blur">
      <span className="font-medium">{me.email}</span>
      <span className="rounded-full bg-stone-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-stone-500">
        {scopeLabel}
      </span>
      <a
        href="/cdn-cgi/access/logout"
        className="text-stone-400 hover:text-stone-700"
        title="Sign out of Cloudflare Access"
      >
        sign out
      </a>
    </div>
  );
}
