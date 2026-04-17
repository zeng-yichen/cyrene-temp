"use client";

import { useParams } from "next/navigation";
import { useState, useRef, useEffect, useCallback } from "react";
import * as Popover from "@radix-ui/react-popover";
import {
  addMonths,
  eachDayOfInterval,
  endOfMonth,
  endOfWeek,
  format,
  isSameDay,
  isSameMonth,
  startOfMonth,
  startOfWeek,
} from "date-fns";
import { Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import { ghostwriterApi, postsApi, imagesApi } from "@/lib/api";
import Link from "next/link";

interface TerminalLine {
  type: string;
  text: string;
  timestamp: number;
}

interface RunEntry {
  id: string;
  agent: string;
  status: string;
  prompt: string | null;
  output: string | null;
  error: string | null;
  created_at: number;
  completed_at: number | null;
}

interface RunEvent {
  id: number;
  run_id: string;
  event_type: string;
  data: string | null;
  timestamp: number;
}

export default function GhostwriterIDE() {
  const params = useParams();
  const company = params.company as string;
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [activeTab, setActiveTab] = useState<"terminal" | "history" | "posts">("terminal");
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState<RunEntry | null>(null);
  const [runEvents, setRunEvents] = useState<RunEvent[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [posts, setPosts] = useState<any[]>([]);
  const [loadingPosts, setLoadingPosts] = useState(false);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [linkedinUsername, setLinkedinUsername] = useState<string | null | undefined>(undefined);
  const [linkedinInput, setLinkedinInput] = useState("");
  const [savingUsername, setSavingUsername] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  // Story mode: show Amphoreus story lines instead of raw tool calls
  const isLocalhost = typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
  const [storyMode, setStoryMode] = useState(!isLocalhost);
  const [storyLines, setStoryLines] = useState<string[]>([]);

  useEffect(() => {
    fetch("/amphoreus_story.json")
      .then((r) => r.json())
      .then((raw: string[]) => {
        // Pre-process: strip decorative separators, drop blanks, then
        // split paragraphs into individual sentences so each SSE event
        // reveals one sentence at a time.
        const sentences: string[] = [];
        for (const line of raw) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (/^[═─]{3,}$/.test(trimmed)) continue; // decorative separator
          // Book titles / chapter headers (all caps, no sentence-ending
          // punctuation) stay as one unit.
          const isHeader = /^(BOOK |EPILOGUE|THE AMPHOREUS|As remembered|End of)/i.test(trimmed)
            || trimmed === trimmed.toUpperCase();
          if (isHeader) {
            sentences.push(trimmed);
            continue;
          }
          // Split on sentence-ending punctuation followed by a space.
          // Keep the punctuation with the preceding sentence.
          const parts = trimmed.match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g);
          if (parts) {
            for (const p of parts) {
              const s = p.trim();
              if (s) sentences.push(s);
            }
          } else {
            sentences.push(trimmed);
          }
        }
        setStoryLines(sentences);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines, runEvents]);

  useEffect(() => {
    ghostwriterApi
      .getLinkedInUsername(company)
      .then((res) => setLinkedinUsername(res.username))
      .catch(() => setLinkedinUsername(null));
  }, [company]);

  async function handleSaveUsername() {
    if (!linkedinInput.trim()) return;
    setSavingUsername(true);
    try {
      const res = await ghostwriterApi.saveLinkedInUsername(company, linkedinInput);
      setLinkedinUsername(res.username);
      setLinkedinInput("");
    } finally {
      setSavingUsername(false);
    }
  }

  const loadRuns = useCallback(async () => {
    setLoadingRuns(true);
    try {
      const res = await ghostwriterApi.getRuns(company);
      setRuns(res.runs);
    } catch {
      setRuns([]);
    } finally {
      setLoadingRuns(false);
    }
  }, [company]);

  const loadPosts = useCallback(async () => {
    setLoadingPosts(true);
    try {
      const res = await postsApi.list(company, 200);
      setPosts(res.posts);
    } catch {
      setPosts([]);
    } finally {
      setLoadingPosts(false);
    }
  }, [company]);

  const loadRunDetail = useCallback(async (run: RunEntry) => {
    setSelectedRun(run);
    setLoadingEvents(true);
    try {
      const res = await ghostwriterApi.getRunEvents(run.id);
      setRunEvents(res.events);
    } catch {
      setRunEvents([]);
    } finally {
      setLoadingEvents(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "history") { loadRuns(); setSelectedRun(null); }
    if (activeTab === "posts") loadPosts();
  }, [activeTab, loadRuns, loadPosts]);

  // localStorage key for tracking a live Stelle run per client — lets the
  // user navigate away (or close + reopen) and rejoin the live terminal.
  const activeJobKey = `amphoreus_stelle_job_${company}`;

  const consumeStream = useCallback(
    async (jobId: string, afterId = 0) => {
      try {
        for await (const data of ghostwriterApi.streamJob(jobId, afterId)) {
          // Always store the raw event. Story-mode vs debug-mode
          // presentation is decided at render time so toggling the
          // button instantly swaps all lines (past and future).
          const text = extractText(data);
          setLines((prev) => [
            ...prev,
            {
              type: data.type === "done" ? "done" : data.type === "error" ? "error" : (data.type as string),
              text,
              timestamp: (data.timestamp || Date.now() / 1000) * 1000,
            },
          ]);
          if (data.type === "done" || data.type === "error") {
            localStorage.removeItem(activeJobKey);
            setIsGenerating(false);
          }
        }
        setIsGenerating(false);
      } catch (e) {
        setLines((prev) => [
          ...prev,
          { type: "error", text: String(e), timestamp: Date.now() },
        ]);
        setIsGenerating(false);
      }
    },
    [activeJobKey],
  );

  async function handleGenerate() {
    if (isGenerating) return;
    setIsGenerating(true);
    setLines([]);
    setActiveTab("terminal");

    try {
      const { job_id } = await ghostwriterApi.generate(
        company,
        prompt || undefined,
      );
      localStorage.setItem(activeJobKey, job_id);
      setLines((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);
      await consumeStream(job_id, 0);
    } catch (e) {
      setLines((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsGenerating(false);
    }
  }

  // On mount: if a job was in-flight when the user navigated away or
  // closed the tab, rehydrate the terminal from run_events and resume
  // the live stream from the last event id.
  useEffect(() => {
    const savedJobId = localStorage.getItem(activeJobKey);
    if (!savedJobId) return;
    let cancelled = false;

    (async () => {
      try {
        const res = await ghostwriterApi.getRunEvents(savedJobId);
        if (cancelled) return;
        const status = res.run?.status;
        if (status === "completed" || status === "failed") {
          localStorage.removeItem(activeJobKey);
          return;
        }
        setIsGenerating(true);
        setActiveTab("terminal");
        const hist: TerminalLine[] = res.events.map((ev) => {
          const parsed =
            typeof ev.data === "string"
              ? (() => {
                  try {
                    return JSON.parse(ev.data as string);
                  } catch {
                    return {};
                  }
                })()
              : ev.data || {};
          return {
            type: ev.event_type,
            text: extractText({ type: ev.event_type, data: parsed }),
            timestamp: ev.timestamp * 1000,
          };
        });
        setLines([
          { type: "status", text: `Rejoining job ${savedJobId}…`, timestamp: Date.now() },
          ...hist,
        ]);
        const maxId = res.events.reduce((m, e) => (e.id > m ? e.id : m), 0);
        await consumeStream(savedJobId, maxId);
      } catch {
        localStorage.removeItem(activeJobKey);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeJobKey, consumeStream]);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-4 border-b border-stone-200 bg-white px-6 py-3">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">&larr;</Link>
        <h1 className="text-lg font-semibold">Stelle</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        <Link
          href={`/ghostwriter/${company}/calendar`}
          className="rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 transition-colors hover:border-stone-300 hover:text-stone-900"
        >
          Calendar &rarr;
        </Link>
        <div className="flex-1" />
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Optional prompt..."
          className="w-96 rounded-lg border border-stone-200 px-3 py-1.5 text-sm focus:border-stone-400 focus:outline-none"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <button
          onClick={handleGenerate}
          disabled={isGenerating || !linkedinUsername}
          title={!linkedinUsername ? "Set the LinkedIn username below before generating" : undefined}
          className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-50"
        >
          {isGenerating ? "Generating..." : "Generate"}
        </button>
        {isLocalhost && (
          <button
            onClick={() => setStoryMode((v) => !v)}
            className="rounded-lg border border-stone-300 px-3 py-1.5 text-xs text-stone-500 transition-colors hover:bg-stone-100"
            title={storyMode ? "Switch to debug view (tool calls)" : "Switch to story mode"}
          >
            {storyMode ? "Debug" : "Story"}
          </button>
        )}
      </header>

      {/* LinkedIn username prompt */}
      {linkedinUsername === null && (
        <div className="flex items-center gap-3 border-b border-amber-200 bg-amber-50 px-6 py-2.5">
          <span className="shrink-0 text-sm text-amber-800">
            <span className="font-semibold">LinkedIn username required</span> — needed to look up past posts and push to Ordinal.
          </span>
          <input
            value={linkedinInput}
            onChange={(e) => setLinkedinInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSaveUsername()}
            placeholder="e.g. andrew-hume (after linkedin.com/in/)"
            className="flex-1 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-sm focus:border-amber-500 focus:outline-none"
          />
          <button
            onClick={handleSaveUsername}
            disabled={savingUsername || !linkedinInput.trim()}
            className="shrink-0 rounded-lg bg-amber-500 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-amber-600 disabled:opacity-50"
          >
            {savingUsername ? "Saving..." : "Save"}
          </button>
        </div>
      )}
      {linkedinUsername && (
        <div className="flex items-center gap-2 border-b border-stone-100 bg-stone-50 px-6 py-1.5 text-xs text-stone-400">
          <span>LinkedIn:</span>
          <span className="font-medium text-stone-600">{linkedinUsername}</span>
          <button
            onClick={() => setLinkedinUsername(null)}
            className="ml-1 text-stone-300 hover:text-stone-500"
            title="Change username"
          >
            ✎
          </button>
        </div>
      )}

      <div className="flex border-b border-stone-200 bg-stone-50 px-6">
        {(["terminal", "posts", "history"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-700"
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto bg-stone-950 p-4 font-mono text-sm" ref={terminalRef}>
        {activeTab === "terminal" && (
          <div className="space-y-0.5">
            {lines.length === 0 && (
              <p className="text-stone-500">Ready. Click Generate to start.</p>
            )}
            {(() => {
              // In story mode, map non-terminal events (anything
              // that isn't done/error) to story lines by position.
              // done/error always render as themselves so the user
              // sees completion regardless of mode.
              let storyIdx = 0;
              return lines.map((line, i) => {
                const isTerminal = line.type === "done" || line.type === "error";
                if (storyMode && storyLines.length > 0 && !isTerminal) {
                  if (storyIdx >= storyLines.length) return null;
                  const storyText = storyLines[storyIdx];
                  storyIdx += 1;
                  return (
                    <div key={i} className={`${getLineColor("story")} font-mono text-sm leading-loose py-1`}>
                      {storyText}
                    </div>
                  );
                }
                return (
                  <div key={i} className={getLineColor(line.type)}>
                    <span className="mr-2 text-stone-600">
                      {new Date(line.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="mr-2 text-stone-500">[{line.type}]</span>
                    <span className="whitespace-pre-wrap">{line.text}</span>
                  </div>
                );
              });
            })()}
          </div>
        )}

        {activeTab === "history" && !selectedRun && (
          <div className="space-y-2">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs text-stone-400">Run history for {company}</span>
              <button
                onClick={loadRuns}
                className="rounded px-2 py-0.5 text-xs text-stone-500 hover:bg-stone-800 hover:text-stone-200"
              >
                Refresh
              </button>
            </div>
            {loadingRuns ? (
              <p className="text-stone-500">Loading...</p>
            ) : runs.length === 0 ? (
              <p className="text-stone-500">No runs yet.</p>
            ) : (
              runs.map((r) => (
                <button
                  key={r.id}
                  onClick={() => loadRunDetail(r)}
                  className="w-full rounded border border-stone-800 bg-stone-900 px-3 py-2 text-left text-sm transition-colors hover:border-stone-600 hover:bg-stone-800"
                >
                  <div className="flex items-center gap-3">
                    <span className={`text-xs font-medium ${statusColor(r.status)}`}>
                      {r.status}
                    </span>
                    <span className="text-xs text-stone-500">{r.agent}</span>
                    <span className="ml-auto text-xs text-stone-600">
                      {new Date(r.created_at * 1000).toLocaleString()}
                    </span>
                    {r.completed_at && (
                      <span className="text-xs text-stone-600">
                        ({Math.round(r.completed_at - r.created_at)}s)
                      </span>
                    )}
                  </div>
                  {r.prompt && (
                    <p className="mt-1 truncate text-xs text-stone-400">{r.prompt}</p>
                  )}
                  {r.error && (
                    <p className="mt-1 truncate text-xs text-red-400">{r.error}</p>
                  )}
                </button>
              ))
            )}
          </div>
        )}

        {activeTab === "history" && selectedRun && (
          <RunDetailView
            run={selectedRun}
            events={runEvents}
            loading={loadingEvents}
            onBack={() => setSelectedRun(null)}
          />
        )}

        {activeTab === "posts" && (
          <div>
            <PostsManager
              company={company}
              posts={posts}
              loading={loadingPosts}
              actionInProgress={actionInProgress}
              onAction={setActionInProgress}
              onRefresh={loadPosts}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function RunDetailView({
  run,
  events,
  loading,
  onBack,
}: {
  run: RunEntry;
  events: RunEvent[];
  loading: boolean;
  onBack: () => void;
}) {
  const toolCalls = events.filter((e) => e.event_type === "tool_call");
  const errors = events.filter((e) => e.event_type === "error");
  const duration = run.completed_at ? Math.round(run.completed_at - run.created_at) : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="rounded px-2 py-1 text-xs text-stone-400 hover:bg-stone-800 hover:text-stone-200"
        >
          &larr; Back
        </button>
        <span className={`text-xs font-medium ${statusColor(run.status)}`}>{run.status}</span>
        <span className="text-xs text-stone-500">{run.agent}</span>
        <span className="text-xs text-stone-600">{new Date(run.created_at * 1000).toLocaleString()}</span>
        {duration !== null && <span className="text-xs text-stone-600">({duration}s)</span>}
      </div>

      <div className="flex gap-4 text-xs text-stone-500">
        <span>{events.length} events</span>
        <span>{toolCalls.length} tool calls</span>
        {errors.length > 0 && <span className="text-red-400">{errors.length} errors</span>}
      </div>

      {run.prompt && (
        <div className="rounded border border-stone-800 bg-stone-900 p-3">
          <span className="text-xs text-stone-500">Prompt: </span>
          <span className="text-xs text-stone-300">{run.prompt}</span>
        </div>
      )}

      {loading ? (
        <p className="text-stone-500">Loading events...</p>
      ) : events.length === 0 ? (
        <p className="text-stone-500">No events recorded for this run.</p>
      ) : (
        <div className="space-y-0.5">
          {events.map((evt) => {
            const parsed = parseEventData(evt);
            return (
              <div key={evt.id} className={`flex gap-2 ${getLineColor(evt.event_type)}`}>
                <span className="shrink-0 text-stone-600">
                  {new Date(evt.timestamp * 1000).toLocaleTimeString()}
                </span>
                <span className="shrink-0 text-stone-500">[{evt.event_type}]</span>
                <span className="whitespace-pre-wrap break-all">{parsed}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function parseEventData(evt: RunEvent): string {
  if (!evt.data) return "";
  try {
    const d = JSON.parse(evt.data);
    switch (evt.event_type) {
      case "tool_call":
        return `${d.name || ""}(${d.arguments?.summary || d.arguments || ""})`;
      case "tool_result":
        return `${d.name || ""} -> ${(d.result || "").slice(0, 300)}`;
      case "thinking":
        return d.text || "";
      case "text_delta":
        return d.text || "";
      case "error":
        return d.message || JSON.stringify(d);
      case "done":
        return (d.output || "").slice(0, 500) || "Generation complete.";
      case "status":
        return d.message || "";
      case "compaction":
        return d.message || "Context compaction";
      default:
        return JSON.stringify(d).slice(0, 300);
    }
  } catch {
    return evt.data.slice(0, 300);
  }
}

function extractText(data: Record<string, unknown>): string {
  const d = data.data as Record<string, unknown> | undefined;
  if (!d) return JSON.stringify(data);

  if (data.type === "done") return (d.output as string) || "Generation complete.";
  if (data.type === "error") return (d.message as string) || "Unknown error";
  if (data.type === "tool_call") return `${d.name}(${(d.arguments as Record<string, unknown>)?.summary || d.arguments || ""})`;
  if (data.type === "tool_result") return `${d.name} -> ${((d.result as string) || "").slice(0, 200)}`;

  return (d.text as string) || (d.message as string) || (d.name as string) || JSON.stringify(d);
}

function statusColor(status: string): string {
  if (status === "completed") return "text-green-400";
  if (status === "failed") return "text-red-400";
  return "text-yellow-400";
}

function getLineColor(type: string): string {
  switch (type) {
    case "thinking": return "text-blue-400";
    case "tool_call": return "text-amber-400";
    case "tool_result": return "text-emerald-400";
    case "text_delta": return "text-stone-200";
    case "compaction": return "text-purple-400";
    case "error": return "text-red-400";
    case "done": return "text-green-400";
    case "status": return "text-cyan-400";
    case "story": return "text-amber-200/90";
    default: return "text-stone-400";
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/** Default picker: 7 days ahead at 09:00 local (datetime-local string). */
function defaultPublishDatetimeLocal(): string {
  const d = new Date();
  d.setDate(d.getDate() + 7);
  d.setHours(9, 0, 0, 0);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

function splitDatetimeLocal(s: string): { date: string; time: string } {
  if (s && s.length >= 16) {
    return { date: s.slice(0, 10), time: s.slice(11, 16) };
  }
  const d = defaultPublishDatetimeLocal();
  return { date: d.slice(0, 10), time: d.slice(11, 16) };
}

function mergeDatetimeLocal(dateYmd: string, timeHm: string): string {
  return `${dateYmd}T${timeHm}`;
}

function parseLocalYmd(ymd: string): Date {
  const [y, m, d] = ymd.split("-").map(Number);
  return new Date(y, (m || 1) - 1, d || 1);
}

/** Calendar popover + time input; keeps value as `YYYY-MM-DDTHH:mm` (local). */
function PublishSchedulePicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const { date: dateYmd, time: timeHm } = splitDatetimeLocal(value);
  const selectedDate = parseLocalYmd(dateYmd);
  const [open, setOpen] = useState(false);
  const [viewMonth, setViewMonth] = useState(() => startOfMonth(selectedDate));

  useEffect(() => {
    if (open) {
      setViewMonth(startOfMonth(parseLocalYmd(splitDatetimeLocal(value).date)));
    }
  }, [open, value]);

  const monthStart = startOfMonth(viewMonth);
  const monthEnd = endOfMonth(viewMonth);
  const calStart = startOfWeek(monthStart, { weekStartsOn: 0 });
  const calEnd = endOfWeek(monthEnd, { weekStartsOn: 0 });
  const days = eachDayOfInterval({ start: calStart, end: calEnd });
  const weekDays = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-end gap-2">
        <Popover.Root open={open} onOpenChange={setOpen}>
          <Popover.Trigger asChild>
            <button
              type="button"
              className="mt-1 flex min-h-[2.25rem] min-w-[10.5rem] items-center justify-between gap-2 rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-left text-sm text-stone-200 hover:border-stone-500"
            >
              <span className="flex items-center gap-2">
                <Calendar className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                {format(selectedDate, "MMM d, yyyy")}
              </span>
            </button>
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content
              className="z-[60] w-[min(100vw-2rem,18rem)] rounded-lg border border-stone-600 bg-stone-900 p-3 shadow-xl"
              sideOffset={4}
              align="start"
            >
              <div className="mb-2 flex items-center justify-between gap-1">
                <button
                  type="button"
                  aria-label="Previous month"
                  className="rounded p-1 text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                  onClick={() => setViewMonth((m) => addMonths(m, -1))}
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="text-xs font-medium text-stone-200">
                  {format(viewMonth, "MMMM yyyy")}
                </span>
                <button
                  type="button"
                  aria-label="Next month"
                  className="rounded p-1 text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                  onClick={() => setViewMonth((m) => addMonths(m, 1))}
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
              <div className="mb-1 grid grid-cols-7 gap-0.5 text-center text-[10px] font-medium uppercase tracking-wide text-stone-500">
                {weekDays.map((d) => (
                  <div key={d}>{d}</div>
                ))}
              </div>
              <div className="grid grid-cols-7 gap-0.5">
                {days.map((day) => {
                  const inMonth = isSameMonth(day, viewMonth);
                  const isSelected = isSameDay(day, selectedDate);
                  const ymd = format(day, "yyyy-MM-dd");
                  return (
                    <button
                      key={ymd}
                      type="button"
                      onClick={() => {
                        onChange(mergeDatetimeLocal(ymd, timeHm));
                        setViewMonth(startOfMonth(day));
                        setOpen(false);
                      }}
                      className={
                        "flex h-8 items-center justify-center rounded text-xs tabular-nums " +
                        (!inMonth
                          ? "text-stone-600 hover:bg-stone-800/80 hover:text-stone-400"
                          : isSelected
                            ? "bg-cyan-800 font-medium text-white"
                            : "text-stone-300 hover:bg-stone-800")
                      }
                    >
                      {format(day, "d")}
                    </button>
                  );
                })}
              </div>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
        <label className="block text-xs font-medium text-stone-400">
          Time (local)
          <input
            type="time"
            value={timeHm}
            onChange={(e) => onChange(mergeDatetimeLocal(dateYmd, e.target.value))}
            className="mt-1 block w-full min-w-[7rem] rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-sm text-stone-200"
          />
        </label>
      </div>
      <p className="text-[11px] text-stone-500">
        Selected:{" "}
        <time dateTime={value || mergeDatetimeLocal(dateYmd, timeHm)}>
          {format(parseLocalYmd(dateYmd), "EEEE, MMMM d, yyyy")} at {timeHm}
        </time>
      </p>
    </div>
  );
}

function PostsManager({
  company,
  posts,
  loading,
  actionInProgress,
  onAction,
  onRefresh,
}: {
  company: string;
  posts: any[];
  loading: boolean;
  actionInProgress: string | null;
  onAction: (id: string | null) => void;
  onRefresh: () => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [factReport, setFactReport] = useState<{ id: string; report: string } | null>(null);
  const [citationData, setCitationData] = useState<Record<string, string[]>>({});
  const [imageJobId, setImageJobId] = useState<string | null>(null);
  const [imageLines, setImageLines] = useState<{ type: string; text: string }[]>([]);
  const [generatingImageFor, setGeneratingImageFor] = useState<string | null>(null);
  const [pendingImageByPost, setPendingImageByPost] = useState<Record<string, string>>({});
  const [imageFeedbackByPost, setImageFeedbackByPost] = useState<Record<string, string>>({});
  const [pushModalPost, setPushModalPost] = useState<any | null>(null);
  const [pushAllOpen, setPushAllOpen] = useState(false);
  const [pushAllPostsPerMonth, setPushAllPostsPerMonth] = useState<8 | 12>(12);
  const [pushPublishLocal, setPushPublishLocal] = useState("");
  const [pushApproverIds, setPushApproverIds] = useState<Set<string>>(new Set());
  const [pushApprovalsBlocking, setPushApprovalsBlocking] = useState(true);
  const [ordinalUsers, setOrdinalUsers] = useState<any[]>([]);
  const [loadingOrdinalUsers, setLoadingOrdinalUsers] = useState(false);
  const [ordinalUsersError, setOrdinalUsersError] = useState<string | null>(null);

  useEffect(() => {
    if (!pushModalPost && !pushAllOpen) return;
    setOrdinalUsersError(null);
    setLoadingOrdinalUsers(true);
    ghostwriterApi
      .getOrdinalUsers(company)
      .then((res) => setOrdinalUsers(res.users || []))
      .catch(() => {
        setOrdinalUsers([]);
        setOrdinalUsersError("Could not load Ordinal workspace users (check API key in ordinal_auth_rows).");
      })
      .finally(() => setLoadingOrdinalUsers(false));
  }, [pushModalPost, pushAllOpen, company]);

  function openPushModal(post: any) {
    setPushPublishLocal(defaultPublishDatetimeLocal());
    setPushApproverIds(new Set());
    setPushApprovalsBlocking(true);
    setPushModalPost(post);
  }

  function openPushAllModal() {
    setPushApproverIds(new Set());
    setPushApprovalsBlocking(true);
    setPushAllPostsPerMonth(12);
    setPushAllOpen(true);
  }

  async function handleConfirmPushAll() {
    onAction("__push_all__");
    try {
      const approvals = Array.from(pushApproverIds).map((userId) => ({
        userId,
        isBlocking: pushApprovalsBlocking,
      }));
      const res = await postsApi.pushAll(company, pushAllPostsPerMonth, { approvals });
      const errLines = res.errors?.length ? `\n\nErrors:\n${res.errors.slice(0, 8).join("\n")}` : "";
      if (res.success) {
        window.alert(
          `Pushed ${res.pushed} of ${res.total} draft(s) on ${res.cadence} cadence (UTC 09:00 per slot).` +
            (res.first_url ? `\n\nOpen: ${res.first_url}` : "") +
            errLines
        );
        setPushAllOpen(false);
        onRefresh();
      } else {
        window.alert(`Push-all failed (${res.pushed}/${res.total}).${errLines}`);
      }
    } catch (e) {
      window.alert(`Push-all failed: ${e}`);
    } finally {
      onAction(null);
    }
  }

  function toggleApprover(id: string) {
    setPushApproverIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleGenerateImage(
    post: any,
    options?: { feedback?: string; referenceImageId?: string }
  ) {
    const pid = post.id;
    setGeneratingImageFor(pid);
    setImageLines([]);
    setImageJobId(null);
    try {
      const { job_id } = await imagesApi.generate(company, post.content, undefined, {
        feedback: options?.feedback,
        referenceImageId: options?.referenceImageId,
        localPostId: pid,
      });
      setImageJobId(job_id);
      for await (const data of imagesApi.streamJob(job_id)) {
        const d = (data.data || {}) as { image_id?: string; text?: string; message?: string; name?: string; output?: string };
        const text =
          d.text || d.message || d.name || d.output?.slice?.(0, 200) || JSON.stringify(d);
        setImageLines((prev) => [...prev, { type: data.type, text }]);
        if (data.type === "done" && d.image_id) {
          setPendingImageByPost((prev) => ({ ...prev, [pid]: d.image_id as string }));
        }
        if (data.type === "done" || data.type === "error") {
          setGeneratingImageFor(null);
        }
      }
      setGeneratingImageFor(null);
    } catch (e) {
      setImageLines((prev) => [...prev, { type: "error", text: String(e) }]);
      setGeneratingImageFor(null);
    }
  }

  async function handleApproveLinkedImage(post: any) {
    const img = pendingImageByPost[post.id] || post.linked_image_id;
    if (!img) {
      window.alert("Generate an image first, then approve it for Ordinal push.");
      return;
    }
    onAction(post.id);
    try {
      await postsApi.update(post.id, company, { linked_image_id: img });
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  async function handleRegenerateWithFeedback(post: any) {
    const ref = pendingImageByPost[post.id] || post.linked_image_id;
    const fb = (imageFeedbackByPost[post.id] || "").trim();
    if (!fb && !ref) {
      window.alert("Generate an image first, or add revision notes (optionally with an existing image as reference).");
      return;
    }
    await handleGenerateImage(post, {
      feedback: fb || "Improve the composite image to better match the post; apply concrete visual changes.",
      referenceImageId: ref,
    });
  }

  async function handleDelete(postId: string) {
    if (!confirm("Delete this post?")) return;
    onAction(postId);
    try {
      await postsApi.delete(postId);
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  async function handleRewrite(post: any) {
    onAction(post.id);
    try {
      const res = await postsApi.rewrite(post.id, company, post.content);
      onRefresh();
      if (res.result) {
        setEditingId(post.id);
        setEditText(typeof res.result === "string" ? res.result : res.result.content || "");
      }
    } finally {
      onAction(null);
    }
  }

  async function handleFactCheck(post: any) {
    onAction(post.id);
    try {
      const res = await postsApi.factCheck(post.id, company, post.content);
      setFactReport({ id: post.id, report: res.report });
      if (res.citation_comments?.length) {
        setCitationData((prev) => ({ ...prev, [post.id]: res.citation_comments }));
      }
    } finally {
      onAction(null);
    }
  }

  async function handleSaveEdit(postId: string) {
    onAction(postId);
    try {
      await postsApi.update(postId, company, { content: editText });
      setEditingId(null);
      setEditText("");
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  async function handleConfirmPush() {
    if (!pushModalPost) return;
    const post = pushModalPost;
    const t = new Date(pushPublishLocal);
    if (Number.isNaN(t.getTime())) {
      window.alert("Invalid date/time.");
      return;
    }
    onAction(post.id);
    try {
      const approvals = Array.from(pushApproverIds).map((userId) => ({
        userId,
        isBlocking: pushApprovalsBlocking,
      }));
      const res = await postsApi.push(company, post.content, [], {
        postId: post.id,
        publishAt: t.toISOString(),
        approvals,
      });
      if (res.success) {
        const oid = res.ordinal_post_ids?.[0];
        window.alert(
          (res.result ? `Pushed to Ordinal.\n\nOpen: ${res.result}` : "Pushed to Ordinal.") +
            (oid ? `\n\nOrdinal post id (saved on this draft): ${oid}` : "") +
            (post.linked_image_id || pendingImageByPost[post.id]
              ? "\n\nApproved image is attached when PUBLIC_BASE_URL on the API points to a URL Ordinal can reach (see server .env)."
              : "")
        );
        setPushModalPost(null);
        onRefresh();
      } else {
        window.alert(`Push failed:\n${String(res.result || "Unknown error")}`);
      }
    } catch (e) {
      window.alert(`Push failed: ${e}`);
    } finally {
      onAction(null);
    }
  }

  if (loading) return <p className="text-stone-500">Loading posts...</p>;

  return (
    <div className="space-y-3">
      {pushModalPost && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div
            className="max-h-[min(90vh,36rem)] w-full max-w-lg overflow-y-auto rounded-xl border border-stone-700 bg-stone-900 p-5 shadow-xl"
            role="dialog"
            aria-labelledby="push-ordinal-title"
          >
            <h4 id="push-ordinal-title" className="mb-3 text-sm font-semibold text-stone-100">
              Push to Ordinal
            </h4>
            {(pushModalPost.linked_image_id || pendingImageByPost[pushModalPost.id]) && (
              <p className="mb-3 rounded border border-amber-900/40 bg-amber-950/20 px-2 py-1.5 text-xs text-amber-200/90">
                Image: Ordinal will attach the approved or preview PNG when{" "}
                <code className="text-amber-100/80">PUBLIC_BASE_URL</code> on the API is a URL Ordinal can fetch.
                {!pushModalPost.linked_image_id && pendingImageByPost[pushModalPost.id] && (
                  <span className="block pt-1 text-amber-300/80">
                    Save with &quot;Use for Ordinal&quot; first if you want this push to use the new preview.
                  </span>
                )}
              </p>
            )}
            <p className="mb-3 text-xs text-stone-500">
              Only <span className="text-stone-300">this draft</span> is pushed (the saved post body from the list below),
              with its stored citations and why-post as thread comments—not your Cyrene markdown file.
            </p>
            <p className="mb-3 text-xs text-stone-500">
              Schedule time is sent to Ordinal as <code className="text-stone-400">publishAt</code> (UTC).
              Optional approvers receive requests per{" "}
              <a
                className="text-cyan-500 hover:underline"
                href="https://docs.tryordinal.com/api-reference/approvals/create-approval-requests"
                target="_blank"
                rel="noreferrer"
              >
                Ordinal approvals API
              </a>
              .
            </p>
            <div className="mb-3">
              <div className="mb-1 text-xs font-medium text-stone-400">Publish date &amp; time (local)</div>
              <PublishSchedulePicker value={pushPublishLocal} onChange={setPushPublishLocal} />
            </div>
            <div className="mb-3">
              <div className="mb-1 text-xs font-medium text-stone-400">Approvers (optional)</div>
              {loadingOrdinalUsers && (
                <p className="text-xs text-stone-500">Loading workspace users…</p>
              )}
              {ordinalUsersError && (
                <p className="text-xs text-amber-400">{ordinalUsersError}</p>
              )}
              {!loadingOrdinalUsers && !ordinalUsersError && ordinalUsers.length === 0 && (
                <p className="text-xs text-stone-500">No users returned.</p>
              )}
              <ul className="max-h-40 space-y-1 overflow-y-auto rounded border border-stone-800 p-2">
                {ordinalUsers.map((u: any) => {
                  const uid = u.id || u.userId;
                  if (!uid) return null;
                  const name = [u.firstName, u.lastName].filter(Boolean).join(" ") || "User";
                  return (
                    <li key={uid}>
                      <label className="flex cursor-pointer items-center gap-2 text-xs text-stone-300">
                        <input
                          type="checkbox"
                          checked={pushApproverIds.has(uid)}
                          onChange={() => toggleApprover(uid)}
                        />
                        <span>
                          {name} {u.email ? <span className="text-stone-500">({u.email})</span> : null}
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              {pushApproverIds.size > 0 && (
                <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-stone-400">
                  <input
                    type="checkbox"
                    checked={pushApprovalsBlocking}
                    onChange={(e) => setPushApprovalsBlocking(e.target.checked)}
                  />
                  Blocking approvals
                </label>
              )}
            </div>
            <div className="flex justify-end gap-2 border-t border-stone-800 pt-3">
              <button
                type="button"
                onClick={() => setPushModalPost(null)}
                className="rounded bg-stone-800 px-3 py-1.5 text-xs text-stone-300 hover:bg-stone-700"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleConfirmPush()}
                disabled={actionInProgress === pushModalPost.id}
                className="rounded bg-cyan-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-cyan-700 disabled:opacity-50"
              >
                {actionInProgress === pushModalPost.id ? "Pushing…" : "Push"}
              </button>
            </div>
          </div>
        </div>
      )}

      {pushAllOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div
            className="max-h-[min(90vh,36rem)] w-full max-w-lg overflow-y-auto rounded-xl border border-stone-700 bg-stone-900 p-5 shadow-xl"
            role="dialog"
            aria-labelledby="push-all-ordinal-title"
          >
            <h4 id="push-all-ordinal-title" className="mb-3 text-sm font-semibold text-stone-100">
              Push all drafts to Ordinal
            </h4>
            <p className="mb-3 text-xs text-stone-500">
              Each listed draft becomes its own Ordinal post. Publish dates are the next{" "}
              <strong className="font-medium text-stone-300">Mon / Wed / Thu</strong> (12/mo) or{" "}
              <strong className="font-medium text-stone-300">Tue / Thu</strong> (8/mo) slots starting from today
              (UTC calendar day), at <code className="text-stone-400">09:00 UTC</code> per slot. Oldest drafts get
              the earliest slots.
            </p>
            <label className="mb-3 block text-xs font-medium text-stone-400">
              Posting cadence
              <select
                value={pushAllPostsPerMonth}
                onChange={(e) => setPushAllPostsPerMonth(Number(e.target.value) as 8 | 12)}
                className="mt-1 w-full rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-sm text-stone-200"
              >
                <option value={12}>12 posts / month — Mon, Wed, Thu</option>
                <option value={8}>8 posts / month — Tue, Thu</option>
              </select>
            </label>
            <div className="mb-3">
              <div className="mb-1 text-xs font-medium text-stone-400">Approvers (optional)</div>
              {loadingOrdinalUsers && (
                <p className="text-xs text-stone-500">Loading workspace users…</p>
              )}
              {ordinalUsersError && (
                <p className="text-xs text-amber-400">{ordinalUsersError}</p>
              )}
              {!loadingOrdinalUsers && !ordinalUsersError && ordinalUsers.length === 0 && (
                <p className="text-xs text-stone-500">No users returned.</p>
              )}
              <ul className="max-h-40 space-y-1 overflow-y-auto rounded border border-stone-800 p-2">
                {ordinalUsers.map((u: any) => {
                  const uid = u.id || u.userId;
                  if (!uid) return null;
                  const name = [u.firstName, u.lastName].filter(Boolean).join(" ") || "User";
                  return (
                    <li key={uid}>
                      <label className="flex cursor-pointer items-center gap-2 text-xs text-stone-300">
                        <input
                          type="checkbox"
                          checked={pushApproverIds.has(uid)}
                          onChange={() => toggleApprover(uid)}
                        />
                        <span>
                          {name} {u.email ? <span className="text-stone-500">({u.email})</span> : null}
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              {pushApproverIds.size > 0 && (
                <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-stone-400">
                  <input
                    type="checkbox"
                    checked={pushApprovalsBlocking}
                    onChange={(e) => setPushApprovalsBlocking(e.target.checked)}
                  />
                  Blocking approvals
                </label>
              )}
            </div>
            <div className="flex justify-end gap-2 border-t border-stone-800 pt-3">
              <button
                type="button"
                onClick={() => setPushAllOpen(false)}
                className="rounded bg-stone-800 px-3 py-1.5 text-xs text-stone-300 hover:bg-stone-700"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleConfirmPushAll()}
                disabled={actionInProgress === "__push_all__"}
                className="rounded bg-cyan-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-cyan-700 disabled:opacity-50"
              >
                {actionInProgress === "__push_all__" ? "Pushing…" : `Push all (${posts.length})`}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-stone-300">
          {posts.length} post{posts.length !== 1 ? "s" : ""}
        </h3>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => openPushAllModal()}
            disabled={posts.length === 0 || actionInProgress !== null}
            className="rounded bg-cyan-950 px-2.5 py-1 text-xs font-medium text-cyan-300 ring-1 ring-cyan-800 hover:bg-cyan-900 disabled:opacity-40"
          >
            Push all to Ordinal
          </button>
          <button onClick={onRefresh} className="text-xs text-stone-500 hover:text-stone-300">
            Refresh
          </button>
        </div>
      </div>

      {posts.length === 0 && (
        <p className="text-sm text-stone-500">
          No posts yet. Generate some with the Ghostwriter.
        </p>
      )}

      {factReport && (
        <div className="rounded-lg border border-cyan-800 bg-cyan-950/30 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-sm font-medium text-cyan-300">Fact-Check Report</h4>
            <button onClick={() => setFactReport(null)} className="text-xs text-stone-500 hover:text-stone-300">
              Dismiss
            </button>
          </div>
          <pre className="whitespace-pre-wrap text-xs text-stone-300">{factReport.report}</pre>
        </div>
      )}

      {imageLines.length > 0 && (
        <div className="rounded-lg border border-amber-800 bg-amber-950/20 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-sm font-medium text-amber-300">
              Image Assembly {generatingImageFor ? "(running...)" : "(complete)"}
            </h4>
            <button onClick={() => setImageLines([])} className="text-xs text-stone-500 hover:text-stone-300">
              Dismiss
            </button>
          </div>
          <div className="max-h-48 space-y-0.5 overflow-y-auto font-mono text-xs">
            {imageLines.map((line, i) => (
              <div key={i} className={
                line.type === "error" ? "text-red-400" :
                line.type === "tool_call" ? "text-amber-400" :
                line.type === "tool_result" ? "text-emerald-400" :
                line.type === "done" ? "text-green-400" :
                line.type === "status" ? "text-cyan-400" :
                "text-stone-400"
              }>
                <span className="mr-1 text-stone-600">[{line.type}]</span>
                {line.text}
              </div>
            ))}
          </div>
        </div>
      )}

      {posts.map((post) => (
        <div
          key={post.id}
          className="rounded-lg border border-stone-800 bg-stone-900 p-4"
        >
          {editingId === post.id ? (
            <div className="space-y-2">
              <textarea
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                rows={8}
                className="w-full rounded border border-stone-700 bg-stone-950 p-3 text-sm text-stone-200 focus:border-stone-500 focus:outline-none"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => handleSaveEdit(post.id)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-700 px-3 py-1 text-xs text-white hover:bg-stone-600 disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  onClick={() => { setEditingId(null); setEditText(""); }}
                  className="rounded bg-stone-800 px-3 py-1 text-xs text-stone-400 hover:bg-stone-700"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Permansor score badge */}
              {post.permansor_score != null && (
                <div className="mb-2 flex items-center gap-2">
                  <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${
                    post.permansor_score >= 4 ? "bg-emerald-950 text-emerald-400" :
                    post.permansor_score >= 3.5 ? "bg-amber-950 text-amber-400" :
                    "bg-red-950 text-red-400"
                  }`}>
                    Permansor {post.permansor_score.toFixed(1)}/5
                  </span>
                  {post.pre_revision_content && (
                    <span className="rounded bg-stone-800 px-2 py-0.5 text-[10px] text-stone-400">
                      Revised by SELF-REFINE
                    </span>
                  )}
                </div>
              )}

              {/* Pre-revision toggle */}
              {post.pre_revision_content && (
                <details className="mb-3">
                  <summary className="cursor-pointer text-xs text-stone-500 hover:text-stone-300">
                    Show original draft (before Permansor revision)
                  </summary>
                  <div className="mt-2 max-h-[min(50vh,20rem)] overflow-y-auto rounded border border-stone-700/50 bg-stone-950/30 p-3">
                    <pre className="whitespace-pre-wrap break-words text-sm text-stone-400">
                      {post.pre_revision_content}
                    </pre>
                  </div>
                </details>
              )}

              {/* Current (revised) post content */}
              <div className="mb-3 max-h-[min(70vh,32rem)] overflow-y-auto rounded border border-stone-800/80 bg-stone-950/50 p-3">
                <pre className="whitespace-pre-wrap break-words text-sm text-stone-200">
                  {post.content || ""}
                </pre>
              </div>
              {(pendingImageByPost[post.id] || post.linked_image_id) && (
                <div className="mb-3 rounded-lg border border-amber-900/50 bg-stone-950/50 p-3">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <p className="text-xs font-medium text-amber-200/90">LinkedIn / Ordinal image</p>
                    {post.linked_image_id && (
                      <span
                        className="rounded bg-amber-950/80 px-2 py-0.5 text-[10px] font-medium text-amber-400"
                        title="This draft’s PNG will be uploaded to Ordinal on push when PUBLIC_BASE_URL is set."
                      >
                        Approved for push
                      </span>
                    )}
                    {pendingImageByPost[post.id] &&
                      pendingImageByPost[post.id] !== post.linked_image_id && (
                        <span className="rounded bg-stone-800 px-2 py-0.5 text-[10px] text-stone-400">
                          New preview — click &quot;Use for Ordinal&quot; to save
                        </span>
                      )}
                  </div>
                  <img
                    src={imagesApi.getUrl(
                      company,
                      pendingImageByPost[post.id] || post.linked_image_id
                    )}
                    alt=""
                    className="mb-3 max-h-64 max-w-full rounded border border-stone-700"
                  />
                  <label className="block text-xs text-stone-400">
                    Revision notes for Phainon (concrete visual changes work best)
                    <textarea
                      value={imageFeedbackByPost[post.id] ?? ""}
                      onChange={(e) =>
                        setImageFeedbackByPost((p) => ({ ...p, [post.id]: e.target.value }))
                      }
                      rows={3}
                      className="mt-1 w-full rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-sm text-stone-200"
                      placeholder="e.g. warmer palette, less text on the image, tighter crop…"
                    />
                  </label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => void handleRegenerateWithFeedback(post)}
                      disabled={generatingImageFor === post.id || actionInProgress === post.id}
                      className="rounded bg-stone-800 px-2.5 py-1 text-xs text-amber-300 hover:bg-stone-700 disabled:opacity-50"
                    >
                      {generatingImageFor === post.id ? "Regenerating…" : "Regenerate with notes"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleApproveLinkedImage(post)}
                      disabled={actionInProgress === post.id}
                      className="rounded bg-amber-900/60 px-2.5 py-1 text-xs font-medium text-amber-100 hover:bg-amber-900 disabled:opacity-50"
                    >
                      Use for Ordinal
                    </button>
                  </div>
                  <p className="mt-2 text-[11px] text-stone-500">
                    Ordinal fetches the image from a public URL — set{" "}
                    <code className="text-stone-400">PUBLIC_BASE_URL</code> on the API host (tunnel or prod) so
                    uploads succeed.
                  </p>
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2 border-t border-stone-800 pt-3">
                {post.ordinal_post_id && (
                  <span
                    className="max-w-full truncate rounded bg-stone-800/80 px-2 py-0.5 font-mono text-[10px] text-stone-400"
                    title={`Latest Ordinal post id (updates if you re-push): ${post.ordinal_post_id}`}
                  >
                    Ordinal: {post.ordinal_post_id}
                  </span>
                )}
                {citationData[post.id] && (
                  <span
                    title={`${citationData[post.id].length} source annotation${citationData[post.id].length !== 1 ? "s" : ""} ready — will be posted as Ordinal comments on push`}
                    className="rounded bg-emerald-900/40 px-2 py-0.5 text-xs font-medium text-emerald-400"
                  >
                    Annotated
                  </span>
                )}
                <button
                  onClick={() => { setEditingId(post.id); setEditText(post.content || ""); }}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleRewrite(post)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700 disabled:opacity-50"
                >
                  {actionInProgress === post.id ? "..." : "Rewrite"}
                </button>
                <button
                  onClick={() => handleFactCheck(post)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700 disabled:opacity-50"
                >
                  {actionInProgress === post.id ? "..." : "Fact-check"}
                </button>
                <button
                  onClick={() => handleGenerateImage(post)}
                  disabled={generatingImageFor === post.id || actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-amber-400 hover:bg-stone-700 disabled:opacity-50"
                >
                  {generatingImageFor === post.id ? "Generating..." : "Generate Image"}
                </button>
                <button
                  onClick={() => openPushModal(post)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-cyan-400 hover:bg-stone-700 disabled:opacity-50"
                >
                  Push to Ordinal
                </button>
                {post.pre_revision_content && (
                  <button
                    onClick={() => openPushModal({ ...post, content: post.pre_revision_content, title: `${post.title || "Draft"} (original)` })}
                    disabled={actionInProgress === post.id}
                    className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-400 hover:bg-stone-700 disabled:opacity-50"
                  >
                    Push Original
                  </button>
                )}
                <div className="flex-1" />
                <span className="text-xs text-stone-600">
                  {post.status || "draft"}
                  {post.created_at ? ` \u00b7 ${new Date(post.created_at * 1000).toLocaleDateString()}` : ""}
                </span>
                <button
                  onClick={() => handleDelete(post.id)}
                  disabled={actionInProgress === post.id}
                  className="rounded px-2 py-1 text-xs text-red-500 hover:bg-red-950 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}



