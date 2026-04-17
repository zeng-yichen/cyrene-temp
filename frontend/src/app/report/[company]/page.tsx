"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { reportApi } from "@/lib/api";

interface ProgressLine {
  type: string;
  text: string;
  timestamp: number;
}

export default function ProgressReportPage() {
  const params = useParams();
  const company = params.company as string;

  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [lines, setLines] = useState<ProgressLine[]>([]);
  const [renderedUrl, setRenderedUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);

  const kickoff = useCallback(async () => {
    setStatus("running");
    setLines([]);
    setError(null);
    setRenderedUrl(null);
    try {
      const { job_id } = await reportApi.generate(company);
      setJobId(job_id);
      setLines([{ type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() }]);
      for await (const data of reportApi.streamJob(job_id)) {
        const text =
          (data.data as { message?: string } | undefined)?.message ||
          (data.data as { output?: string } | undefined)?.output ||
          "";
        setLines((prev) => [
          ...prev,
          { type: data.type, text, timestamp: Date.now() },
        ]);
        if (data.type === "done") {
          const out = (data.data as { output?: string } | undefined)?.output || "";
          setRenderedUrl(out || `/api/report/${company}/rendered`);
          setStatus("done");
          return;
        }
        if (data.type === "error") {
          setError(
            (data.data as { message?: string } | undefined)?.message || "Unknown error",
          );
          setStatus("error");
          return;
        }
      }
      setStatus("error");
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }, [company]);

  // Auto-start on mount (opened in a new tab → should just run)
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    kickoff();
  }, [kickoff]);

  // When the HTML is ready, swap the page to full-bleed iframe with a
  // floating download button in the top-right corner.
  if (status === "done" && renderedUrl) {
    const today = new Date().toISOString().slice(0, 10);
    const filename = `progress-report-${company}-${today}.html`;
    return (
      <>
        <iframe
          src={renderedUrl}
          title={`${company} progress report`}
          className="fixed inset-0 h-screen w-screen border-0"
        />
        <a
          href={renderedUrl}
          download={filename}
          className="fixed right-4 top-4 z-10 rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-sm font-medium text-stone-700 shadow-md transition-colors hover:border-stone-400 hover:bg-stone-50"
        >
          Download HTML
        </a>
      </>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-center gap-3">
        <Link href={`/interview/${company}`} className="text-sm text-stone-400 hover:text-stone-600">
          &larr;
        </Link>
        <h1 className="text-2xl font-semibold">Progress report</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        {status === "running" && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-indigo-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
            Generating…
          </span>
        )}
      </div>

      <p className="mt-4 text-sm text-stone-500">
        Aggregates {company}&apos;s posting data over the last 2 weeks, then asks Opus
        to render a presentation-ready HTML report. Takes ~3-4 minutes end-to-end.
      </p>

      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <p className="font-semibold">Report generation failed</p>
          <p className="mt-1 whitespace-pre-wrap">{error}</p>
          <button
            onClick={kickoff}
            className="mt-3 rounded-md border border-red-300 bg-white px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100"
          >
            Retry
          </button>
        </div>
      )}

      <div className="mt-6 rounded-lg border border-stone-200 bg-stone-950 font-mono text-xs">
        <div className="flex items-center justify-between border-b border-stone-800 px-4 py-2 text-stone-400">
          <span>Progress</span>
          {jobId && <span className="text-stone-600">job {jobId.slice(0, 8)}</span>}
        </div>
        <div className="max-h-[60vh] space-y-0.5 overflow-auto p-3">
          {lines.length === 0 && (
            <p className="text-stone-500">Connecting…</p>
          )}
          {lines.map((line, i) => (
            <div key={i} className={lineColor(line.type)}>
              <span className="mr-2 text-stone-600">
                {new Date(line.timestamp).toLocaleTimeString()}
              </span>
              <span className="mr-2 text-stone-500">[{line.type}]</span>
              <span className="whitespace-pre-wrap">{line.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function lineColor(type: string): string {
  switch (type) {
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
