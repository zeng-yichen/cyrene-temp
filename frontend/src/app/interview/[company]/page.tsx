"use client";

/**
 * Interview Prep — merged Interview Briefing + Interview Companion.
 *
 * Two phases on one page:
 *
 *   Phase A (Prep):
 *     - Click "Generate briefing" → runs Aglaea via /api/briefings/generate,
 *       streams the progress to a log panel, then populates the briefing panel
 *       with the resulting markdown.
 *     - Existing briefing (if any) is loaded on mount and shown immediately.
 *
 *   Phase B (Live):
 *     - Once a briefing exists and BlackHole is detected, "Start recording"
 *       kicks off Tribbie via /api/interview/start. Live transcript streams
 *       into the left panel, follow-up suggestions appear on the right.
 *
 * The old standalone /briefings/{company} route is gone; this page owns the
 * full interview-prep lifecycle.
 */

import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { briefingsApi, cyreneApi, ghostwriterApi, interviewApi } from "@/lib/api";

interface TranscriptSegment {
  text: string;
  timestamp: number;
}

interface Suggestion {
  text: string;
  timestamp: number;
}

interface BriefingLogLine {
  type: string;
  text: string;
  timestamp: number;
}

export default function InterviewPrepPage() {
  const params = useParams();
  const company = params.company as string;


  // --- Live interview (Tribbie) state ---
  const [isRecording, setIsRecording] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("Ready");
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [trashed, setTrashed] = useState(false);
  const [hasBlackhole, setHasBlackhole] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- Briefing (Aglaea) state ---
  const [hasBriefing, setHasBriefing] = useState<boolean | null>(null);
  const [briefingContent, setBriefingContent] = useState<string | null>(null);
  const [isGeneratingBrief, setIsGeneratingBrief] = useState(false);
  const [briefingLog, setBriefingLog] = useState<BriefingLogLine[]>([]);
  const [showBriefingLog, setShowBriefingLog] = useState(false);

  // --- Cyrene (Strategic Review) state ---
  const [cyreneBrief, setCyreneBrief] = useState<any>(null);
  const [isRunningCyrene, setIsRunningCyrene] = useState(false);
  const [cyreneLog, setCyreneLog] = useState<BriefingLogLine[]>([]);
  const [showCyreneLog, setShowCyreneLog] = useState(false);
  const [showCyreneBrief, setShowCyreneBrief] = useState(false);

  const transcriptRef = useRef<HTMLDivElement>(null);
  const briefingLogRef = useRef<HTMLDivElement>(null);

  // Detect BlackHole, briefing existence, and briefing content on mount
  useEffect(() => {
    interviewApi
      .listDevices()
      .then((res) => setHasBlackhole(res.has_blackhole))
      .catch(() => setHasBlackhole(false));

    briefingsApi
      .check(company)
      .then((res) => {
        setHasBriefing(res.exists);
        if (res.exists) {
          briefingsApi
            .get(company)
            .then((r) => setBriefingContent(r.content))
            .catch(() => setBriefingContent(null));
        }
      })
      .catch(() => setHasBriefing(false));
  }, [company]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  // Auto-scroll briefing log
  useEffect(() => {
    if (briefingLogRef.current) {
      briefingLogRef.current.scrollTop = briefingLogRef.current.scrollHeight;
    }
  }, [briefingLog]);

  // Load existing Cyrene brief on mount
  useEffect(() => {
    cyreneApi
      .getBrief(company)
      .then((brief) => {
        setCyreneBrief(brief);
        setShowCyreneBrief(true);
      })
      .catch(() => setCyreneBrief(null));
  }, [company]);

  // Persist the live Cyrene job per-company so navigating away or closing
  // the tab doesn't lose the stream — on remount we rejoin.
  const activeCyreneKey = `amphoreus_cyrene_job_${company}`;

  const consumeCyreneStream = useCallback(
    async (jobId: string, afterId = 0) => {
      try {
        for await (const data of cyreneApi.streamJob(jobId, afterId)) {
          const text =
            (data.data as { message?: string } | undefined)?.message ||
            (data.data as { output?: string } | undefined)?.output ||
            JSON.stringify(data.data).slice(0, 200);
          setCyreneLog((prev) => [
            ...prev,
            { type: data.type, text, timestamp: Date.now() },
          ]);

          if (data.type === "done") {
            localStorage.removeItem(activeCyreneKey);
            try {
              const output = (data.data as { output?: string } | undefined)?.output;
              if (output) {
                setCyreneBrief(JSON.parse(output));
              } else {
                const fresh = await cyreneApi.getBrief(company);
                setCyreneBrief(fresh);
              }
            } catch {
              try {
                const fresh = await cyreneApi.getBrief(company);
                setCyreneBrief(fresh);
              } catch { /* ignore */ }
            }
            setShowCyreneBrief(true);
            setShowCyreneLog(false);
            setIsRunningCyrene(false);
            return;
          }
          if (data.type === "error") {
            localStorage.removeItem(activeCyreneKey);
            setIsRunningCyrene(false);
            return;
          }
        }
        setIsRunningCyrene(false);
      } catch (e) {
        setCyreneLog((prev) => [
          ...prev,
          { type: "error", text: String(e), timestamp: Date.now() },
        ]);
        setIsRunningCyrene(false);
      }
    },
    [activeCyreneKey, company],
  );

  async function handleRunCyrene() {
    if (isRunningCyrene) return;
    setIsRunningCyrene(true);
    setCyreneLog([]);
    setShowCyreneLog(true);
    setShowCyreneBrief(false);

    try {
      const { job_id } = await cyreneApi.run(company);
      localStorage.setItem(activeCyreneKey, job_id);
      setCyreneLog((prev) => [
        ...prev,
        { type: "status", text: `Cyrene review started: ${job_id}`, timestamp: Date.now() },
      ]);
      await consumeCyreneStream(job_id, 0);
    } catch (e) {
      setCyreneLog((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsRunningCyrene(false);
    }
  }

  // On mount, if a Cyrene review is in-flight for this company, rejoin it.
  useEffect(() => {
    const savedJobId = localStorage.getItem(activeCyreneKey);
    if (!savedJobId) return;
    let cancelled = false;

    (async () => {
      try {
        // Events are stored in a shared run_events table, so reuse the
        // ghostwriter runs/events endpoint to hydrate history.
        const res = await ghostwriterApi.getRunEvents(savedJobId);
        if (cancelled) return;
        const status = res.run?.status;
        if (status === "completed" || status === "failed") {
          localStorage.removeItem(activeCyreneKey);
          return;
        }
        setIsRunningCyrene(true);
        setShowCyreneLog(true);
        setShowCyreneBrief(false);
        const hist: BriefingLogLine[] = res.events.map((ev: any) => {
          const parsed =
            typeof ev.data === "string"
              ? (() => {
                  try {
                    return JSON.parse(ev.data);
                  } catch {
                    return {};
                  }
                })()
              : ev.data || {};
          const text =
            parsed?.message ||
            parsed?.output ||
            JSON.stringify(parsed).slice(0, 200);
          return {
            type: ev.event_type,
            text,
            timestamp: ev.timestamp * 1000,
          };
        });
        setCyreneLog([
          { type: "status", text: `Rejoining Cyrene review ${savedJobId}…`, timestamp: Date.now() },
          ...hist,
        ]);
        const maxId = res.events.reduce(
          (m: number, e: any) => (e.id > m ? e.id : m),
          0,
        );
        await consumeCyreneStream(savedJobId, maxId);
      } catch {
        localStorage.removeItem(activeCyreneKey);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeCyreneKey, consumeCyreneStream]);

  async function handleGenerateBriefing() {
    if (isGeneratingBrief) return;
    setIsGeneratingBrief(true);
    setBriefingLog([]);
    setShowBriefingLog(true);

    try {
      const { job_id } = await briefingsApi.generate(company, company);
      setBriefingLog((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);

      for await (const data of briefingsApi.streamJob(job_id)) {
        const text = extractBriefingText(data);
        setBriefingLog((prev) => [
          ...prev,
          {
            type: data.type,
            text,
            timestamp: ((data.timestamp as number | undefined) ?? Date.now() / 1000) * 1000,
          },
        ]);

        if (data.type === "done") {
          const output = (data.data as { output?: string } | undefined)?.output || "";
          if (output) {
            setBriefingContent(output);
            setHasBriefing(true);
          } else {
            // Try to fetch the persisted briefing if the done event didn't include it.
            try {
              const res = await briefingsApi.get(company);
              setBriefingContent(res.content);
              setHasBriefing(true);
            } catch {
              /* ignore */
            }
          }
          setIsGeneratingBrief(false);
          setShowBriefingLog(false);
          return;
        }
        if (data.type === "error") {
          setIsGeneratingBrief(false);
          return;
        }
      }
      setIsGeneratingBrief(false);
    } catch (e) {
      setBriefingLog((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsGeneratingBrief(false);
    }
  }

  async function handleStart() {
    if (isRecording) return;
    setIsRecording(true);
    setTranscript([]);
    setSuggestions([]);
    setSavedPath(null);
    setTrashed(false);
    setError(null);
    setStatus("Starting...");

    try {
      const { job_id } = await interviewApi.start(company);
      setJobId(job_id);
      setStatus("Connecting...");

      for await (const data of interviewApi.streamJob(job_id)) {
        const ts = ((data.timestamp as number | undefined) ?? Date.now() / 1000) * 1000;

        if (data.type === "status") {
          setStatus((data.data as { message?: string } | undefined)?.message || "");
        } else if (data.type === "text_delta") {
          const text = (data.data as { text?: string } | undefined)?.text || "";
          if (text) setTranscript((prev) => [...prev, { text, timestamp: ts }]);
        } else if (data.type === "tool_result") {
          const suggestion = (data.data as { result?: string } | undefined)?.result || "";
          if (suggestion) {
            setSuggestions((prev) => [{ text: suggestion, timestamp: ts }, ...prev]);
          }
        } else if (data.type === "error") {
          setError((data.data as { message?: string } | undefined)?.message || "Unknown error");
          setIsRecording(false);
          setStatus("Error");
          return;
        } else if (data.type === "done") {
          const filePath = (data.data as { output?: string } | undefined)?.output || null;
          const msg = (data.data as { message?: string } | undefined)?.message || "Session complete";
          setSavedPath(filePath);
          setIsRecording(false);
          setStatus(msg);
          return;
        }
      }
      setIsRecording(false);
    } catch (e) {
      setError(String(e));
      setIsRecording(false);
      setStatus("Error");
    }
  }

  async function handleStop() {
    if (!isRecording || !jobId) return;
    setStatus("Stopping — saving transcript...");
    try {
      await interviewApi.stop(jobId, company);
    } catch {
      // Backend signals the session; done event will arrive via SSE
    }
  }

  const canStart = hasBlackhole !== false && hasBriefing === true && !isGeneratingBrief;

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-stone-200 bg-white px-6 py-3">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">
          &larr;
        </Link>
        <h1 className="text-lg font-semibold">Cyrene</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>

        {isRecording && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-red-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
            Recording
          </span>
        )}
        {isGeneratingBrief && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-indigo-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
            Generating briefing…
          </span>
        )}

        <div className="flex-1" />

        {/* Cyrene strategic review */}
        {!isRecording && (
          <>
            {cyreneBrief && !showCyreneBrief && !showCyreneLog && (
              <button
                onClick={() => setShowCyreneBrief(true)}
                className="rounded-lg border border-emerald-300 bg-white px-4 py-1.5 text-sm font-medium text-emerald-700 transition-colors hover:bg-emerald-50"
              >
                Show strategic brief
              </button>
            )}
            <button
              onClick={handleRunCyrene}
              disabled={isRunningCyrene}
              className="rounded-lg border border-emerald-300 bg-emerald-50 px-4 py-1.5 text-sm font-medium text-emerald-700 transition-colors hover:bg-emerald-100 disabled:cursor-wait disabled:opacity-50"
            >
              {isRunningCyrene
                ? "Running review…"
                : cyreneBrief
                ? "Re-run strategic review"
                : "Run strategic review"}
            </button>
          </>
        )}

        {/* Progress report — opens a page that streams generation progress
            and swaps in the rendered HTML on completion. */}
        {!isRecording && (
          <button
            onClick={() => window.open(`/report/${company}`, "_blank")}
            className="rounded-lg border border-stone-300 bg-white px-4 py-1.5 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50"
          >
            Progress report
          </button>
        )}

        {/* Record / stop */}
        {!isRecording ? (
          <button
            onClick={handleStart}
            disabled={!canStart}
            title={
              hasBriefing !== true
                ? "Generate a briefing first"
                : hasBlackhole === false
                ? "BlackHole audio device required"
                : "Start the live interview companion"
            }
            className="rounded-lg bg-red-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Start recording
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800"
          >
            Stop &amp; save
          </button>
        )}
      </header>

      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-stone-200 bg-stone-50 px-6 py-2 text-xs">
        <span className="font-medium text-stone-600">Status:</span>
        <span className="text-stone-500">{status}</span>
        {savedPath && !trashed && (
          <>
            <span className="ml-3 font-medium text-emerald-600">
              Saved: {savedPath.split("/").pop()}
            </span>
            <button
              onClick={async () => {
                try {
                  await interviewApi.trashTranscript(savedPath);
                  setTrashed(true);
                  setStatus("Transcript moved to Trash");
                } catch {
                  setStatus("Failed to trash transcript");
                }
              }}
              title="Move transcript to Trash"
              className="ml-2 rounded px-2 py-0.5 text-stone-400 transition-colors hover:bg-red-50 hover:text-red-500"
            >
              🗑 Trash
            </button>
          </>
        )}
        {trashed && (
          <span className="ml-3 text-stone-400 line-through">
            {savedPath?.split("/").pop()}
          </span>
        )}
      </div>

      {/* No strategic review warning */}
      {!cyreneBrief && !isRunningCyrene && (
        <div className="border-b border-emerald-200 bg-emerald-50 px-6 py-3 text-sm text-emerald-800">
          <span className="font-semibold text-emerald-900">No strategic review yet for {company}. </span>
          Click <strong>Run strategic review</strong> above. Cyrene will study engagement data, ICP
          trends, and warm prospects, then produce a data-backed brief with interview questions,
          DM targets, and content priorities.
        </div>
      )}

      {/* BlackHole setup guide */}
      {hasBlackhole === false && (
        <div className="border-b border-amber-200 bg-amber-50 px-6 py-3 text-sm">
          <p className="font-semibold text-amber-900">BlackHole audio device not detected</p>
          <ol className="mt-1.5 list-decimal space-y-0.5 pl-4 text-amber-800">
            <li>
              Run: <code className="rounded bg-amber-100 px-1 font-mono text-xs">brew install blackhole-2ch</code>
            </li>
            <li>Reboot your Mac</li>
            <li>
              Open <strong>Audio MIDI Setup</strong> → click <strong>+</strong> →{" "}
              <strong>Create Multi-Output Device</strong>
            </li>
            <li>Check both <strong>BlackHole 2ch</strong> and your speakers/headphones</li>
            <li>
              Set this Multi-Output Device as your system output in{" "}
              <strong>System Settings → Sound</strong>
            </li>
          </ol>
          <p className="mt-2 text-xs text-amber-700">
            Then refresh this page — the Start Recording button will become active.
          </p>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="border-b border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700">
          <span className="font-medium">Error: </span>
          {error}
        </div>
      )}

      {/* Briefing generation log (collapsible while streaming) */}
      {showBriefingLog && (
        <div className="flex max-h-64 flex-col border-b border-stone-200 bg-stone-950 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-stone-800 px-4 py-1.5 text-stone-400">
            <span>Aglaea progress</span>
            <button
              onClick={() => setShowBriefingLog(false)}
              className="text-stone-500 hover:text-stone-300"
            >
              ×
            </button>
          </div>
          <div ref={briefingLogRef} className="flex-1 space-y-0.5 overflow-auto p-3">
            {briefingLog.map((line, i) => (
              <div key={i} className={getBriefingLineColor(line.type)}>
                <span className="mr-2 text-stone-600">
                  {new Date(line.timestamp).toLocaleTimeString()}
                </span>
                <span className="mr-2 text-stone-500">[{line.type}]</span>
                <span className="whitespace-pre-wrap">{line.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cyrene strategic review log (collapsible while streaming) */}
      {showCyreneLog && (
        <div className="flex max-h-64 flex-col border-b border-stone-200 bg-stone-950 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-stone-800 px-4 py-1.5 text-stone-400">
            <span>Cyrene strategic review progress</span>
            <button
              onClick={() => setShowCyreneLog(false)}
              className="text-stone-500 hover:text-stone-300"
            >
              &times;
            </button>
          </div>
          <div className="flex-1 space-y-0.5 overflow-auto p-3">
            {cyreneLog.map((line, i) => (
              <div key={i} className={line.type === "error" ? "text-red-400" : "text-stone-300"}>
                <span className="mr-2 text-stone-600">
                  {new Date(line.timestamp).toLocaleTimeString()}
                </span>
                <span className="mr-2 text-stone-500">[{line.type}]</span>
                <span className="whitespace-pre-wrap">{line.text.slice(0, 300)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cyrene strategic brief display (when a brief exists) */}
      {showCyreneBrief && cyreneBrief && !showCyreneLog && (
        <div className="border-b border-emerald-200 bg-emerald-50/50 px-6 py-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-emerald-900">
              Cyrene Strategic Brief
              {cyreneBrief._computed_at && (
                <span className="ml-2 font-normal text-emerald-600">
                  ({new Date(cyreneBrief._computed_at).toLocaleDateString()})
                </span>
              )}
            </h3>
            <button
              onClick={() => setShowCyreneBrief(false)}
              className="text-xs text-emerald-500 hover:text-emerald-700"
            >
              Hide
            </button>
          </div>

          <div className="mt-3 grid grid-cols-1 gap-4 text-xs sm:grid-cols-2 lg:grid-cols-3">
            {/* Interview Questions */}
            {cyreneBrief.interview_questions?.length > 0 && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">Interview Questions</h4>
                <ol className="list-decimal space-y-1 pl-4 text-stone-700">
                  {cyreneBrief.interview_questions.map((q: string, i: number) => (
                    <li key={i}>{q}</li>
                  ))}
                </ol>
              </div>
            )}

            {/* Content Priorities */}
            {cyreneBrief.content_priorities?.length > 0 && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">Content Priorities</h4>
                <ul className="list-disc space-y-1 pl-4 text-stone-700">
                  {cyreneBrief.content_priorities.map((p: string, i: number) => (
                    <li key={i}>{p}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* DM Targets */}
            {cyreneBrief.dm_targets?.length > 0 && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">
                  DM Targets ({cyreneBrief.dm_targets.length})
                </h4>
                <div className="space-y-2">
                  {cyreneBrief.dm_targets.slice(0, 5).map((t: any, i: number) => (
                    <div key={i} className="text-stone-700">
                      <span className="font-medium">{t.name}</span>
                      {t.company && <span className="text-stone-500"> @ {t.company}</span>}
                      {t.suggested_angle && (
                        <p className="mt-0.5 text-stone-500 italic">{t.suggested_angle}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ICP Exposure */}
            {cyreneBrief.icp_exposure_assessment && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">ICP Exposure</h4>
                <p className="text-stone-700">{cyreneBrief.icp_exposure_assessment}</p>
              </div>
            )}

            {/* Stelle Timing */}
            {cyreneBrief.stelle_timing && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">Stelle Timing</h4>
                <p className="text-stone-700">{cyreneBrief.stelle_timing}</p>
              </div>
            )}

            {/* Content Avoid */}
            {cyreneBrief.content_avoid?.length > 0 && (
              <div className="rounded-lg border border-red-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-red-800">Avoid</h4>
                <ul className="list-disc space-y-1 pl-4 text-stone-700">
                  {cyreneBrief.content_avoid.map((a: string, i: number) => (
                    <li key={i}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Meta footer */}
          <div className="mt-3 flex gap-4 text-xs text-emerald-600">
            {cyreneBrief._turns_used && <span>{cyreneBrief._turns_used} turns</span>}
            {cyreneBrief._cost_usd && <span>${cyreneBrief._cost_usd}</span>}
            {cyreneBrief.next_run_trigger?.condition && (
              <span>Next: {cyreneBrief.next_run_trigger.condition}</span>
            )}
          </div>
        </div>
      )}

      {/* Two-panel layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel: Live Transcript */}
        <div className="flex flex-1 flex-col border-r border-stone-200">
          <div className="flex items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
              Live Transcript
            </span>
            <span className="text-xs text-stone-400">{transcript.length} segments</span>
          </div>
          <div ref={transcriptRef} className="flex-1 space-y-3 overflow-y-auto p-4">
            {transcript.length === 0 ? (
              <p className="text-sm text-stone-400">
                {isRecording
                  ? "Listening for speech..."
                  : "Transcript will appear here once recording starts."}
              </p>
            ) : (
              transcript.map((seg, i) => (
                <div key={i} className="flex gap-3">
                  <span className="mt-0.5 shrink-0 font-mono text-xs text-stone-400">
                    {new Date(seg.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                  </span>
                  <p className="text-sm leading-relaxed text-stone-800">{seg.text}</p>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right column: Briefing (top) + Suggestions (bottom) — always both visible */}
        <div className="flex w-96 flex-col xl:w-[420px]">
          {/* Interview Questions (from Cyrene brief) */}
          <div className="flex h-1/2 flex-col border-b border-stone-200">
            <div className="flex shrink-0 items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Interview Questions
              </span>
              {cyreneBrief?.interview_questions && (
                <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-600">
                  {cyreneBrief.interview_questions.length}
                </span>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {cyreneBrief?.interview_questions?.length > 0 ? (
                <ol className="list-decimal space-y-2 pl-4 text-sm text-stone-700">
                  {cyreneBrief.interview_questions.map((q: string, i: number) => (
                    <li key={i} className="leading-relaxed">{q}</li>
                  ))}
                </ol>
              ) : briefingContent ? (
                <div className="prose prose-sm prose-stone max-w-none text-xs [&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs [&_li]:my-0.5 [&_p]:my-1">
                  <ReactMarkdown>{briefingContent}</ReactMarkdown>
                </div>
              ) : (
                <p className="text-sm text-stone-400">
                  Run a strategic review to generate data-backed interview questions.
                </p>
              )}
            </div>
          </div>

          {/* Suggestions panel */}
          <div className="flex flex-1 flex-col overflow-hidden">
            <div className="flex shrink-0 items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Suggestions
              </span>
              {suggestions.length > 0 && (
                <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-semibold text-indigo-600">
                  {suggestions.length}
                </span>
              )}
            </div>
            <div className="flex-1 space-y-2 overflow-y-auto p-3">
              {suggestions.length === 0 ? (
                <p className="p-1 text-sm text-stone-400">
                  {isRecording
                    ? "Suggestions appear after substantial speech is detected."
                    : "Suggestions will appear here during recording."}
                </p>
              ) : (
                suggestions.map((s, i) => (
                  <div
                    key={i}
                    className={`rounded-lg border p-3 text-sm leading-relaxed ${
                      i === 0
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 shadow-sm"
                        : "border-stone-200 bg-white text-stone-700"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span className="mt-0.5 shrink-0 font-bold text-stone-400">
                        {i === 0 ? "›" : "·"}
                      </span>
                      <span className="whitespace-pre-line">{s.text}</span>
                    </div>
                    <p className="mt-1.5 text-xs text-stone-400">
                      {new Date(s.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function extractBriefingText(data: Record<string, unknown>): string {
  const d = data.data as Record<string, unknown> | undefined;
  if (!d) return JSON.stringify(data);
  if (data.type === "done") return ((d.output as string) || "").slice(0, 200) || "Briefing complete.";
  if (data.type === "error") return (d.message as string) || "Unknown error";
  if (data.type === "tool_call") {
    const args = d.arguments as Record<string, unknown> | undefined;
    return `${d.name}(${(args?.summary as string) || JSON.stringify(d.arguments) || ""})`;
  }
  if (data.type === "tool_result") return `${d.name} -> ${((d.result as string) || "").slice(0, 200)}`;
  return (d.text as string) || (d.message as string) || (d.name as string) || JSON.stringify(d);
}

function getBriefingLineColor(type: string): string {
  switch (type) {
    case "thinking":
      return "text-blue-400";
    case "tool_call":
      return "text-amber-400";
    case "tool_result":
      return "text-emerald-400";
    case "text_delta":
      return "text-stone-200";
    case "compaction":
      return "text-purple-400";
    case "error":
      return "text-red-400";
    case "done":
      return "text-green-400";
    case "status":
      return "text-cyan-400";
    default:
      return "text-stone-400";
  }
}
