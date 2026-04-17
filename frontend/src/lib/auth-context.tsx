"use client";

/**
 * Stage 2 auth context.
 *
 * Fetches `/api/me` on mount, caches the result in React state, and exposes
 * the caller's identity + ACL-scoped client list to the rest of the app.
 *
 * The backend enforces authz — this context is purely for UX (greeting the
 * user by email, hiding client tabs they don't have access to, showing a
 * banner when auth is in local-dev bypass mode, etc.).
 *
 * On the happy path the `/api/me` fetch succeeds silently. On failure (401
 * session expired, network error, backend down) we render the children
 * anyway with `me = null`; the individual API calls will re-throw and
 * apiFetch's 401 handler will trigger a full-page reload through CF Access.
 */

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { meApi, type MeResponse } from "./api";

type AuthContextValue = {
  me: MeResponse | null;
  loading: boolean;
  error: string | null;
  canAccess: (clientSlug: string) => boolean;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<MeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await meApi.get();
      setMe(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMe(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      me,
      loading,
      error,
      refresh,
      canAccess: (slug: string) => {
        if (!me) return false;
        if (me.is_admin) return true;
        if (me.allowed_clients === "*") return true;
        return me.allowed_clients.includes(slug);
      },
    }),
    [me, loading, error],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useMe(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useMe must be used inside <AuthProvider>");
  }
  return ctx;
}
