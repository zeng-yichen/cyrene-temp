"use client";

/**
 * Cyrene home page — client picker + three workflow boxes.
 *
 * Layout:
 *   1. Ghostwriter (Stelle)       → /ghostwriter/{slug}
 *   2. Interview Prep (Aglaea+Tribbie) → /interview/{slug}
 *   3. Transcripts (file mgmt)    → /transcripts/{slug}
 *
 * The dropdown is populated from /api/clients, which is already ACL-scoped
 * by the backend — admins see everything, scoped users only see the clients
 * they've been granted access to.
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { clientsApi, deployApi } from "@/lib/api";
import { useMe } from "@/lib/auth-context";

export default function HomePage() {
  const { me } = useMe();
  const isAdmin = me?.is_admin ?? false;
  const isLocal = me ? !me.auth_enabled : false;
  const [company, setCompanyRaw] = useState("");
  const [clientList, setClientList] = useState<string[]>([]);
  const [loadingClients, setLoadingClients] = useState(true);

  // Deploy state
  const [deploying, setDeploying] = useState<string | null>(null); // "code" | "data" | null
  const [deployLog, setDeployLog] = useState("");

  // Ban state
  const [bannedUsers, setBannedUsers] = useState<string[]>([]);
  const [allUsers, setAllUsers] = useState<{ email: string; role: string }[]>([]);
  const [banInput, setBanInput] = useState("");

  useEffect(() => {
    if (isLocal) {
      deployApi.listBanned().then((r) => setBannedUsers(r.banned)).catch(() => {});
      deployApi.listUsers().then((r) => setAllUsers(r.users)).catch(() => {});
    }
  }, [isLocal]);

  async function refreshBanState() {
    const [b, u] = await Promise.all([deployApi.listBanned(), deployApi.listUsers()]);
    setBannedUsers(b.banned);
    setAllUsers(u.users);
  }

  async function handleBan() {
    const email = banInput.trim().toLowerCase();
    if (!email) return;
    try {
      await deployApi.ban(email);
      setBanInput("");
      await refreshBanState();
    } catch (e) {
      setDeployLog(`Ban error: ${e}`);
    }
  }

  async function handleUnban(email: string) {
    try {
      await deployApi.unban(email);
      await refreshBanState();
    } catch (e) {
      setDeployLog(`Unban error: ${e}`);
    }
  }

  const pollDeploy = useCallback(async (key: string, label: string) => {
    setDeploying(label);
    setDeployLog("Starting...");
    const poll = async () => {
      for (let i = 0; i < 120; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        try {
          const s = await deployApi.status(key);
          if (s.status === "completed") {
            setDeployLog(`✓ ${label} succeeded.`);
            setDeploying(null);
            return;
          }
          if (s.status === "failed") {
            setDeployLog(`✗ ${label} failed:\n${s.log?.slice(-500) || "unknown error"}`);
            setDeploying(null);
            return;
          }
          setDeployLog(`Running... (${(i + 1) * 3}s)`);
        } catch {
          // keep polling
        }
      }
      setDeployLog("Timed out waiting for deploy.");
      setDeploying(null);
    };
    poll();
  }, []);

  async function handleDeployCode() {
    if (deploying) return;
    try {
      const res = await deployApi.code("both");
      if (res.status === "already_running") {
        setDeployLog("A deploy is already running.");
        return;
      }
      pollDeploy(res.key, "Code deploy");
    } catch (e) {
      setDeployLog(`Error: ${e}`);
    }
  }

  async function handlePushData() {
    if (deploying) return;
    try {
      const res = await deployApi.data();
      if (res.status === "already_running") {
        setDeployLog("A data push is already running.");
        return;
      }
      pollDeploy(res.key, "Data push");
    } catch (e) {
      setDeployLog(`Error: ${e}`);
    }
  }

  // Persist selected client across navigation
  function setCompany(value: string) {
    setCompanyRaw(value);
    if (value) {
      localStorage.setItem("amphoreus_selected_client", value);
    }
  }

  useEffect(() => {
    // Restore last selected client
    const saved = localStorage.getItem("amphoreus_selected_client");
    if (saved) setCompanyRaw(saved);

    clientsApi
      .list()
      .then((data) => setClientList(data.clients.map((c) => c.slug)))
      .catch(() => {})
      .finally(() => setLoadingClients(false));
  }, []);

  const slug = company.trim().toLowerCase().replace(/\s+/g, "-");
  const hasCompany = slug.length > 0;

  const workflows = [
    {
      title: "Stelle",
      href: `/ghostwriter/${slug}`,
      description:
        "Generate a batch of LinkedIn posts. Stelle reads transcripts, mines them for angles, drafts against Irontomb, and iterates until each post clears the adversarial bar.",
    },
    {
      title: "Cyrene",
      href: `/interview/${slug}`,
      description:
        "Strategic review + interview prep. Cyrene studies engagement trajectories, ICP exposure trends, warm prospects, and produces a data-backed brief: what to ask in the next interview, who to DM, what Stelle should prioritize.",
    },
    {
      title: "Transcripts",
      href: `/transcripts/${slug}`,
      description:
        "Upload files or paste text into the client's context. Everything Stelle reads lives here.",
    },
  ];

  return (
    <div className="mx-auto max-w-5xl px-6 py-16">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight">Cyrene</h1>
          <p className="mt-2 text-sm italic text-stone-400">
            This will be a LinkedIn journey like none that has come before.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isLocal && (
            <>
              <button
                onClick={handleDeployCode}
                disabled={!!deploying}
                className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 shadow-sm transition-colors hover:bg-blue-100 disabled:opacity-50"
              >
                {deploying === "Code deploy" ? "Deploying..." : "Push Code → Fly"}
              </button>
              <button
                onClick={handlePushData}
                disabled={!!deploying}
                className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 shadow-sm transition-colors hover:bg-emerald-100 disabled:opacity-50"
              >
                {deploying === "Data push" ? "Pushing..." : "Push Data → Fly"}
              </button>
            </>
          )}
          {isAdmin && (
            <Link
              href="/usage"
              className="rounded-md border border-stone-200 bg-white px-3 py-1.5 text-xs font-medium text-stone-600 shadow-sm transition-colors hover:border-stone-300 hover:text-stone-900"
            >
              Usage →
            </Link>
          )}
        </div>
      </div>

      {deployLog && (
        <div className="mt-4 rounded-lg border border-stone-200 bg-stone-900 px-4 py-3 text-xs font-mono text-stone-300 whitespace-pre-wrap">
          {deployLog}
          {!deploying && (
            <button
              onClick={() => setDeployLog("")}
              className="ml-3 text-stone-500 hover:text-stone-300"
            >
              ✕
            </button>
          )}
        </div>
      )}

      <div className="mt-10 flex items-end gap-4">
        <div className="flex-1">
          <label htmlFor="company" className="block text-sm font-medium text-stone-700">
            Client
          </label>
          <select
            id="company"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            className="mt-1 w-full rounded-lg border border-stone-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-500"
          >
            <option value="">
              {loadingClients ? "Loading clients…" : "Select a client"}
            </option>
            {clientList.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <p className="pb-2 text-xs text-stone-400">
          Maps to <code className="rounded bg-stone-100 px-1">memory/{slug || "…"}/</code>
        </p>
      </div>

      <div className="mt-10 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {workflows.map((w) =>
          hasCompany ? (
            <Link
              key={w.title}
              href={w.href}
              className="group rounded-xl border border-stone-200 bg-white p-6 shadow-sm transition-all hover:border-stone-300 hover:shadow-md"
            >
              <h2 className="text-lg font-medium text-stone-900 group-hover:text-stone-700">
                {w.title}
              </h2>
              <p className="mt-1 text-sm text-stone-500">{w.description}</p>
            </Link>
          ) : (
            <div
              key={w.title}
              title="Select a client above first"
              className="cursor-not-allowed rounded-xl border border-stone-200 bg-stone-50 p-6 opacity-40"
            >
              <h2 className="text-lg font-medium text-stone-400">{w.title}</h2>
              <p className="mt-1 text-sm text-stone-400">{w.description}</p>
            </div>
          )
        )}
      </div>

      {isLocal && (
        <div className="mt-12 rounded-xl border border-stone-200 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold text-stone-700">User Management</h2>
          <div className="mt-3 flex items-center gap-2">
            <select
              value={banInput}
              onChange={(e) => setBanInput(e.target.value)}
              className="flex-1 rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-sm shadow-sm focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-500"
            >
              <option value="">Select a user…</option>
              {allUsers
                .filter((u) => !bannedUsers.includes(u.email))
                .map((u) => (
                  <option key={u.email} value={u.email}>
                    {u.email}{u.role === "admin" ? " (admin)" : ""}
                  </option>
                ))}
            </select>
            <button
              onClick={handleBan}
              disabled={!banInput}
              className="rounded-md border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-medium text-red-700 shadow-sm transition-colors hover:bg-red-100 disabled:opacity-40"
            >
              Ban
            </button>
          </div>
          {bannedUsers.length > 0 && (
            <ul className="mt-3 space-y-1">
              {bannedUsers.map((email) => (
                <li
                  key={email}
                  className="flex items-center justify-between rounded-lg bg-red-50 px-3 py-1.5 text-sm"
                >
                  <span className="text-red-800">{email}</span>
                  <button
                    onClick={() => handleUnban(email)}
                    className="text-xs font-medium text-red-600 hover:text-red-800"
                  >
                    Unban
                  </button>
                </li>
              ))}
            </ul>
          )}
          {bannedUsers.length === 0 && (
            <p className="mt-2 text-xs text-stone-400">No banned users.</p>
          )}
        </div>
      )}
    </div>
  );
}
