"use client";

/**
 * Transcripts page — add-only file management for `memory/{company}/transcripts/`.
 *
 * What it does:
 *   - Lists every file currently in the client's transcripts directory.
 *   - Lets any authorized user UPLOAD a file (drag/drop or picker) with a short
 *     "source" label so the list view isn't a wall of UUIDs.
 *   - Lets any authorized user PASTE text into a textarea and save it as a
 *     fresh .txt file.
 *   - Shows a DELETE button on each row, but only if the current user is an
 *     admin. Scoped users see the list but can't delete.
 *
 * Add-only for non-admins is deliberate: content engineers need to trust that
 * the context they feed the system doesn't silently disappear behind them.
 * Admins can still nuke junk via the UI if needed.
 */

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { transcriptsApi, type TranscriptFile } from "@/lib/api";
import { useMe } from "@/lib/auth-context";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

export default function TranscriptsPage() {
  const params = useParams();
  const company = params.company as string;
  const { me } = useMe();
  const isAdmin = me?.is_admin ?? false;

  const [files, setFiles] = useState<TranscriptFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [pasteLabel, setPasteLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [uploadLabel, setUploadLabel] = useState("");

  const reload = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await transcriptsApi.list(company);
      setFiles(res.files);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [company]);

  useEffect(() => {
    void reload();
  }, [reload]);

  function flashSuccess(msg: string) {
    setFlash(msg);
    setErrorMsg(null);
    setTimeout(() => setFlash(null), 4000);
  }

  function flashError(msg: string) {
    setErrorMsg(msg);
    setFlash(null);
    setTimeout(() => setErrorMsg(null), 6000);
  }

  async function handleFilePicked(file: File) {
    setPendingFile(file);
    setUploadLabel(file.name);
  }

  async function doUpload() {
    if (!pendingFile || !uploadLabel.trim()) return;
    setBusy(true);
    try {
      const res = await transcriptsApi.upload(company, pendingFile, uploadLabel.trim());
      flashSuccess(`Uploaded ${res.filename} (${formatSize(res.size_bytes)})`);
      setPendingFile(null);
      setUploadLabel("");
      if (fileInputRef.current) fileInputRef.current.value = "";
      await reload();
    } catch (e) {
      flashError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function doPaste() {
    if (!pasteText.trim() || !pasteLabel.trim()) return;
    setBusy(true);
    try {
      const res = await transcriptsApi.paste(company, pasteText, pasteLabel.trim());
      flashSuccess(`Saved ${res.filename} (${formatSize(res.size_bytes)})`);
      setPasteText("");
      setPasteLabel("");
      setShowPasteModal(false);
      await reload();
    } catch (e) {
      flashError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function doDelete(filename: string) {
    if (!isAdmin) return;
    if (!confirm(`Delete ${filename}? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await transcriptsApi.delete(company, filename);
      flashSuccess(`Deleted ${filename}`);
      await reload();
    } catch (e) {
      flashError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-6 py-10">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-stone-200 pb-4">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">
          &larr;
        </Link>
        <h1 className="text-2xl font-semibold tracking-tight">Transcripts</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        <div className="flex-1" />
        {!isAdmin && (
          <span className="text-xs text-stone-400">Add-only mode (scoped user)</span>
        )}
      </header>

      {/* Flash messages */}
      {flash && (
        <div className="mt-4 rounded border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-800">
          {flash}
        </div>
      )}
      {errorMsg && (
        <div className="mt-4 rounded border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          {errorMsg}
        </div>
      )}

      {/* Add controls */}
      <section className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Upload file */}
        <div className="rounded-xl border border-stone-200 bg-white p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500">
            Upload a file
          </h2>
          <p className="mt-1 text-xs text-stone-400">
            Transcript, PDF, Markdown, etc. Max 100 MB.
          </p>
          <input
            ref={fileInputRef}
            type="file"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFilePicked(f);
            }}
            className="mt-3 block w-full text-sm text-stone-600 file:mr-3 file:rounded-lg file:border-0 file:bg-stone-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-stone-700 hover:file:bg-stone-200"
          />
          {pendingFile && (
            <div className="mt-3 space-y-2">
              <label className="block text-xs font-medium text-stone-500">
                Source label <span className="text-stone-400">(required)</span>
              </label>
              <input
                type="text"
                value={uploadLabel}
                onChange={(e) => setUploadLabel(e.target.value)}
                placeholder="e.g. Call with Jordan 2026-04-08"
                className="w-full rounded-lg border border-stone-300 px-3 py-1.5 text-sm focus:border-stone-500 focus:outline-none"
              />
              <div className="flex items-center gap-2">
                <button
                  onClick={doUpload}
                  disabled={busy || !uploadLabel.trim()}
                  className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-40"
                >
                  {busy ? "Uploading…" : "Save upload"}
                </button>
                <button
                  onClick={() => {
                    setPendingFile(null);
                    setUploadLabel("");
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                  disabled={busy}
                  className="text-xs text-stone-400 hover:text-stone-600"
                >
                  Cancel
                </button>
                <span className="ml-auto text-xs text-stone-400">
                  {formatSize(pendingFile.size)}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Paste text */}
        <div className="rounded-xl border border-stone-200 bg-white p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500">
            Paste text
          </h2>
          <p className="mt-1 text-xs text-stone-400">
            Quickly drop notes, a raw transcript, or any plain text into this client's context.
          </p>
          <button
            onClick={() => setShowPasteModal(true)}
            className="mt-3 rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800"
          >
            Paste &amp; save
          </button>
        </div>
      </section>

      {/* File list */}
      <section className="mt-8 flex-1">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500">
          Files ({files.length})
        </h2>
        {loading ? (
          <p className="mt-4 text-sm text-stone-400">Loading…</p>
        ) : loadError ? (
          <p className="mt-4 text-sm text-red-600">Failed to load: {loadError}</p>
        ) : files.length === 0 ? (
          <p className="mt-4 rounded border border-dashed border-stone-300 bg-stone-50 p-6 text-center text-sm text-stone-400">
            No files yet. Upload one or paste text above to get started.
          </p>
        ) : (
          <div className="mt-3 overflow-hidden rounded-xl border border-stone-200 bg-white">
            <table className="w-full text-sm">
              <thead className="border-b border-stone-200 bg-stone-50 text-xs font-medium uppercase tracking-wide text-stone-500">
                <tr>
                  <th className="px-4 py-2 text-left">Source</th>
                  <th className="px-4 py-2 text-left">Filename</th>
                  <th className="px-4 py-2 text-right">Size</th>
                  <th className="px-4 py-2 text-left">Uploaded</th>
                  <th className="px-4 py-2 text-left">By</th>
                  <th className="px-4 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100">
                {files.map((f) => (
                  <tr key={f.filename} className="hover:bg-stone-50">
                    <td className="px-4 py-2 text-stone-800">
                      {f.source_label ?? (
                        <span className="italic text-stone-400">(no label)</span>
                      )}
                    </td>
                    <td className="px-4 py-2">
                      <a
                        href={transcriptsApi.downloadUrl(company, f.filename)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-mono text-xs text-indigo-600 hover:underline"
                      >
                        {f.filename}
                      </a>
                    </td>
                    <td className="px-4 py-2 text-right text-xs text-stone-500">
                      {formatSize(f.size_bytes)}
                    </td>
                    <td className="px-4 py-2 text-xs text-stone-500">
                      {formatTime(f.uploaded_at ?? f.modified_at)}
                    </td>
                    <td className="px-4 py-2 text-xs text-stone-500">
                      {f.uploaded_by ?? <span className="italic text-stone-400">—</span>}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {isAdmin && (
                        <button
                          onClick={() => doDelete(f.filename)}
                          disabled={busy}
                          className="rounded px-2 py-1 text-xs text-red-500 transition-colors hover:bg-red-50 hover:text-red-700 disabled:opacity-40"
                        >
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Paste modal */}
      {showPasteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-2xl rounded-xl bg-white p-6 shadow-xl">
            <h3 className="text-lg font-semibold">Paste &amp; save text</h3>
            <p className="mt-1 text-xs text-stone-500">
              Saved as a new <code className="rounded bg-stone-100 px-1">.txt</code> file in{" "}
              <code className="rounded bg-stone-100 px-1">memory/{company}/transcripts/</code>.
            </p>
            <label className="mt-4 block text-xs font-medium text-stone-600">
              Source label
            </label>
            <input
              type="text"
              value={pasteLabel}
              onChange={(e) => setPasteLabel(e.target.value)}
              placeholder="e.g. Slack thread from Jordan about pricing"
              className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-1.5 text-sm focus:border-stone-500 focus:outline-none"
            />
            <label className="mt-4 block text-xs font-medium text-stone-600">Text</label>
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              rows={12}
              placeholder="Paste transcript, notes, anything…"
              className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 font-mono text-xs focus:border-stone-500 focus:outline-none"
            />
            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                onClick={() => setShowPasteModal(false)}
                disabled={busy}
                className="rounded-lg px-4 py-1.5 text-sm text-stone-500 hover:text-stone-800"
              >
                Cancel
              </button>
              <button
                onClick={doPaste}
                disabled={busy || !pasteText.trim() || !pasteLabel.trim()}
                className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-40"
              >
                {busy ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
