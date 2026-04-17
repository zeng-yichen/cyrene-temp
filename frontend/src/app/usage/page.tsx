"use client";

/**
 * /usage — admin-only spend dashboard.
 *
 * Reads /api/usage/summary + /api/usage/by-client. Backend enforces admin
 * via _require_admin(); this page also gates render on me.is_admin for
 * a fast-fail UX (scoped users see a "nothing to see here" message
 * instead of hitting a 403).
 *
 * The top-right date range defaults to last 30 days. Adjusting it refetches.
 * Clicking a user row drills into /usage/[email] — TODO in v2; for now the
 * summary view is enough to answer "who's burning budget".
 */

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import {
  usageApi,
  type UsageClientRow,
  type UsageSummary,
  type UsageUserRow,
} from "@/lib/api";
import { useMe } from "@/lib/auth-context";

const DAY = 24 * 60 * 60 * 1000;

function isoNow(): string {
  return new Date().toISOString();
}

function isoDaysAgo(days: number): string {
  return new Date(Date.now() - days * DAY).toISOString();
}

function formatUsd(n: number): string {
  if (!Number.isFinite(n)) return "$0.00";
  if (n >= 100) return `$${n.toFixed(0)}`;
  if (n >= 1) return `$${n.toFixed(2)}`;
  return `$${n.toFixed(4)}`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export default function UsagePage() {
  const { me, loading: meLoading } = useMe();
  const isAdmin = me?.is_admin ?? false;

  const [rangeDays, setRangeDays] = useState<number>(30);
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [byClient, setByClient] = useState<UsageClientRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { since, until } = useMemo(
    () => ({ since: isoDaysAgo(rangeDays), until: isoNow() }),
    [rangeDays],
  );

  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([
      usageApi.summary({ since, until }),
      usageApi.byClient({ since, until }),
    ])
      .then(([s, c]) => {
        if (cancelled) return;
        setSummary(s);
        setByClient(c.by_client);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [isAdmin, since, until]);

  if (meLoading) {
    return <div className="p-12 text-stone-400">Loading…</div>;
  }

  if (!isAdmin) {
    return (
      <div className="mx-auto max-w-xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Usage</h1>
        <p className="mt-3 text-sm text-stone-500">
          This page is admin-only. If you think you should see spend data, ask
          an admin to add you.
        </p>
        <Link
          href="/home"
          className="mt-6 inline-block text-sm text-stone-700 underline hover:text-stone-900"
        >
          ← Back to home
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Usage</h1>
          <p className="mt-1 text-sm text-stone-500">
            Per-user and per-client LLM spend across Cyrene. Anthropic calls
            only in v1 — streaming, embeddings, and Perplexity come next.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-stone-500">Range</label>
          <select
            value={rangeDays}
            onChange={(e) => setRangeDays(Number(e.target.value))}
            className="rounded-md border border-stone-300 bg-white px-2 py-1 text-sm"
          >
            <option value={1}>Last 24h</option>
            <option value={7}>Last 7d</option>
            <option value={30}>Last 30d</option>
            <option value={90}>Last 90d</option>
            <option value={365}>Last 365d</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          Failed to load usage: {error}
        </div>
      )}

      {loading && !summary && (
        <div className="mt-8 text-sm text-stone-400">Loading usage data…</div>
      )}

      {summary && (
        <>
          {/* Grand totals */}
          <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-4">
            <StatCard label="Total cost" value={formatUsd(summary.total.cost_usd)} />
            <StatCard label="Calls" value={summary.total.n_calls.toLocaleString()} />
            <StatCard
              label="Input tokens"
              value={formatTokens(summary.total.input_tokens)}
            />
            <StatCard
              label="Output tokens"
              value={formatTokens(summary.total.output_tokens)}
            />
          </div>

          {/* By user */}
          <section className="mt-10">
            <h2 className="text-lg font-medium text-stone-900">By user</h2>
            <div className="mt-3 overflow-hidden rounded-xl border border-stone-200 bg-white">
              <table className="w-full text-sm">
                <thead className="bg-stone-50 text-left text-xs uppercase tracking-wide text-stone-500">
                  <tr>
                    <th className="px-4 py-2 font-medium">User</th>
                    <th className="px-4 py-2 font-medium">Calls</th>
                    <th className="px-4 py-2 font-medium">Input</th>
                    <th className="px-4 py-2 font-medium">Output</th>
                    <th className="px-4 py-2 text-right font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {summary.by_user.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-4 py-6 text-center text-stone-400">
                        No usage recorded in this window.
                      </td>
                    </tr>
                  )}
                  {summary.by_user.map((row) => (
                    <UserRow key={row.user_email ?? "null"} row={row} />
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* By client */}
          <section className="mt-10">
            <h2 className="text-lg font-medium text-stone-900">By client</h2>
            <div className="mt-3 overflow-hidden rounded-xl border border-stone-200 bg-white">
              <table className="w-full text-sm">
                <thead className="bg-stone-50 text-left text-xs uppercase tracking-wide text-stone-500">
                  <tr>
                    <th className="px-4 py-2 font-medium">Client</th>
                    <th className="px-4 py-2 font-medium">Calls</th>
                    <th className="px-4 py-2 font-medium">Input</th>
                    <th className="px-4 py-2 font-medium">Output</th>
                    <th className="px-4 py-2 text-right font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {(byClient ?? []).length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-4 py-6 text-center text-stone-400">
                        No usage recorded in this window.
                      </td>
                    </tr>
                  )}
                  {(byClient ?? []).map((row) => (
                    <tr key={row.client_slug ?? "null"}>
                      <td className="px-4 py-2 font-mono text-xs text-stone-700">
                        {row.client_slug ?? <em className="text-stone-400">unattributed</em>}
                      </td>
                      <td className="px-4 py-2 text-stone-600">{row.n_calls.toLocaleString()}</td>
                      <td className="px-4 py-2 text-stone-600">{formatTokens(row.input_tokens)}</td>
                      <td className="px-4 py-2 text-stone-600">{formatTokens(row.output_tokens)}</td>
                      <td className="px-4 py-2 text-right font-medium text-stone-900">
                        {formatUsd(row.cost_usd)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      <p className="mt-10 text-xs text-stone-400">
        Cost numbers are computed at record-time from a static price table in{" "}
        <code className="rounded bg-stone-100 px-1">backend/src/usage/pricing.py</code>.
        Edit that file when provider pricing changes; historical rows are not
        backfilled.
      </p>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
      <div className="text-xs uppercase tracking-wide text-stone-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold tracking-tight text-stone-900">
        {value}
      </div>
    </div>
  );
}

function UserRow({ row }: { row: UsageUserRow }) {
  const label = row.user_email ?? "unattributed (system)";
  return (
    <tr>
      <td className="px-4 py-2 font-mono text-xs text-stone-700">
        {row.user_email ? (
          label
        ) : (
          <em className="text-stone-400">{label}</em>
        )}
      </td>
      <td className="px-4 py-2 text-stone-600">{row.n_calls.toLocaleString()}</td>
      <td className="px-4 py-2 text-stone-600">{formatTokens(row.input_tokens)}</td>
      <td className="px-4 py-2 text-stone-600">{formatTokens(row.output_tokens)}</td>
      <td className="px-4 py-2 text-right font-medium text-stone-900">
        {formatUsd(row.cost_usd)}
      </td>
    </tr>
  );
}
