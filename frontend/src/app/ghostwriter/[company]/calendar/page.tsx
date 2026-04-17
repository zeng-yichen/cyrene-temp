"use client";

/**
 * Post Calendar — visual scheduling interface for LinkedIn post batches.
 *
 * Month grid (Mon-Fri), cadence toggle (12/mo = Mon/Tue/Thu, 8/mo = Tue/Thu),
 * drag-and-drop between cells, auto-assign, push-all-to-Ordinal.
 */

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ghostwriterApi } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CalendarPost {
  id: string;
  hook: string;
  content: string;
  content_preview: string;
  status: string;
  scheduled_date: string | null;
  publication_order: number | null;
  ordinal_post_id: string | null;
  why_post: string | null;
}

type Cadence = "3pw" | "2pw";

// Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
const CADENCE_DAYS: Record<Cadence, Set<number>> = {
  "3pw": new Set([0, 1, 3]), // Mon, Tue, Thu
  "2pw": new Set([1, 3]),    // Tue, Thu
};

const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getMonthDays(year: number, month: number): Date[] {
  const days: Date[] = [];
  const first = new Date(year, month, 1);
  const last = new Date(year, month + 1, 0);

  // Pad start to Monday
  const startPad = (first.getDay() + 6) % 7; // 0=Mon
  for (let i = startPad - 1; i >= 0; i--) {
    const d = new Date(year, month, 1 - i - 1 + startPad - i);
    // Actually, let's just compute correctly
  }

  // Simpler: iterate from the Monday of the week containing the 1st
  const startDay = new Date(first);
  startDay.setDate(startDay.getDate() - ((first.getDay() + 6) % 7));

  for (let i = 0; i < 35; i++) {
    const d = new Date(startDay);
    d.setDate(d.getDate() + i);
    // Only weekdays (Mon-Fri)
    const dow = d.getDay();
    if (dow >= 1 && dow <= 5) {
      days.push(d);
    }
  }
  return days;
}

function formatDate(d: Date): string {
  return d.toISOString().split("T")[0]; // YYYY-MM-DD
}

function statusColor(status: string): string {
  switch (status) {
    case "draft":
      return "border-amber-300 bg-amber-50";
    case "pushed":
    case "scheduled":
      return "border-blue-300 bg-blue-50";
    case "published":
    case "posted":
      return "border-emerald-300 bg-emerald-50";
    default:
      return "border-stone-200 bg-white";
  }
}

function statusBadge(status: string): string {
  switch (status) {
    case "draft":
      return "bg-amber-100 text-amber-700";
    case "pushed":
    case "scheduled":
      return "bg-blue-100 text-blue-700";
    case "published":
    case "posted":
      return "bg-emerald-100 text-emerald-700";
    default:
      return "bg-stone-100 text-stone-600";
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CalendarPage() {
  const params = useParams();
  const company = params.company as string;

  const [posts, setPosts] = useState<CalendarPost[]>([]);
  const [cadence, setCadence] = useState<Cadence>("3pw");
  const [currentMonth, setCurrentMonth] = useState(() => {
    const now = new Date();
    return { year: now.getFullYear(), month: now.getMonth() };
  });
  const [dragPostId, setDragPostId] = useState<string | null>(null);
  const [expandedPost, setExpandedPost] = useState<CalendarPost | null>(null);
  const [loading, setLoading] = useState(true);
  const [pushing, setPushing] = useState(false);
  const [assigning, setAssigning] = useState(false);

  const monthStr = `${currentMonth.year}-${String(currentMonth.month + 1).padStart(2, "0")}`;
  const monthLabel = new Date(currentMonth.year, currentMonth.month).toLocaleDateString("en-US", {
    month: "long",
    year: "numeric",
  });

  // Fetch posts
  const fetchPosts = useCallback(async () => {
    try {
      const data = await ghostwriterApi.getCalendar(company);
      setPosts(data.posts);
    } catch {
      setPosts([]);
    } finally {
      setLoading(false);
    }
  }, [company]);

  useEffect(() => {
    fetchPosts();
  }, [fetchPosts]);

  // Build month grid
  const days = getMonthDays(currentMonth.year, currentMonth.month);

  // Map posts to dates
  const postsByDate: Record<string, CalendarPost[]> = {};
  const unscheduled: CalendarPost[] = [];
  for (const p of posts) {
    if (p.scheduled_date) {
      const key = p.scheduled_date;
      if (!postsByDate[key]) postsByDate[key] = [];
      postsByDate[key].push(p);
    } else if (!p.ordinal_post_id) {
      // Only unpushed posts without a date go in the unscheduled sidebar.
      // Pushed posts stay where they are (on the calendar if dated, or
      // hidden if they were pushed without a date via an older flow).
      unscheduled.push(p);
    }
  }

  // Cadence slots for this month
  const cadenceDays = CADENCE_DAYS[cadence];

  // Navigate months
  function prevMonth() {
    setCurrentMonth((prev) => {
      const m = prev.month - 1;
      return m < 0 ? { year: prev.year - 1, month: 11 } : { year: prev.year, month: m };
    });
  }
  function nextMonth() {
    setCurrentMonth((prev) => {
      const m = prev.month + 1;
      return m > 11 ? { year: prev.year + 1, month: 0 } : { year: prev.year, month: m };
    });
  }

  // Drag handlers
  function onDragStart(postId: string) {
    setDragPostId(postId);
  }

  async function onDropOnDate(dateStr: string) {
    if (!dragPostId) return;
    setDragPostId(null);
    try {
      await ghostwriterApi.schedulePost(company, dragPostId, dateStr);
      await fetchPosts();
    } catch (e) {
      console.error("Failed to schedule post:", e);
    }
  }

  async function onDropOnUnscheduled() {
    if (!dragPostId) return;
    setDragPostId(null);
    try {
      await ghostwriterApi.schedulePost(company, dragPostId, null);
      await fetchPosts();
    } catch (e) {
      console.error("Failed to unschedule post:", e);
    }
  }

  // Auto-assign
  async function handleAutoAssign() {
    setAssigning(true);
    try {
      await ghostwriterApi.autoAssign(company, cadence);
      await fetchPosts();
    } catch (e) {
      console.error("Auto-assign failed:", e);
    } finally {
      setAssigning(false);
    }
  }

  // Push all
  async function handlePushAll() {
    setPushing(true);
    try {
      const result = await ghostwriterApi.pushAll(company);
      alert(`Pushed ${result.pushed} post(s) to Ordinal.`);
      await fetchPosts();
    } catch (e) {
      console.error("Push failed:", e);
    } finally {
      setPushing(false);
    }
  }

  const nDrafts = posts.filter((p) => p.status === "draft").length;
  const nScheduled = posts.filter((p) => p.scheduled_date && p.status === "draft").length;
  const nPushed = posts.filter((p) => p.ordinal_post_id).length;

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center text-stone-400">
        Loading calendar...
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-stone-200 bg-white px-6 py-3">
        <Link
          href={`/ghostwriter/${company}`}
          className="text-sm text-stone-400 hover:text-stone-600"
        >
          &larr; Stelle
        </Link>
        <h1 className="text-lg font-semibold">Calendar</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">
          {company}
        </span>

        <div className="flex-1" />

        {/* Stats */}
        <div className="flex gap-3 text-xs text-stone-500">
          <span>{nDrafts} draft{nDrafts !== 1 ? "s" : ""}</span>
          <span>{nScheduled} scheduled</span>
          <span>{nPushed} pushed</span>
        </div>

        {/* Cadence toggle */}
        <div className="flex rounded-lg border border-stone-200 text-xs">
          <button
            onClick={() => setCadence("3pw")}
            className={`px-3 py-1.5 ${
              cadence === "3pw"
                ? "bg-stone-900 text-white"
                : "bg-white text-stone-600 hover:bg-stone-50"
            } rounded-l-lg`}
          >
            12/mo
          </button>
          <button
            onClick={() => setCadence("2pw")}
            className={`px-3 py-1.5 ${
              cadence === "2pw"
                ? "bg-stone-900 text-white"
                : "bg-white text-stone-600 hover:bg-stone-50"
            } rounded-r-lg`}
          >
            8/mo
          </button>
        </div>

        {/* Auto-assign */}
        {unscheduled.length > 0 && (
          <button
            onClick={handleAutoAssign}
            disabled={assigning}
            className="rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-1.5 text-sm font-medium text-indigo-700 transition-colors hover:bg-indigo-100 disabled:opacity-50"
          >
            {assigning ? "Assigning..." : `Auto-assign ${unscheduled.length} posts`}
          </button>
        )}

        {/* Push all */}
        {nScheduled > 0 && (
          <button
            onClick={handlePushAll}
            disabled={pushing}
            className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-emerald-700 disabled:opacity-50"
          >
            {pushing ? "Pushing..." : `Push ${nScheduled} to Ordinal`}
          </button>
        )}
      </header>

      {/* Month navigation */}
      <div className="flex items-center justify-center gap-4 border-b border-stone-100 bg-stone-50 py-2">
        <button onClick={prevMonth} className="text-stone-400 hover:text-stone-700">
          &larr;
        </button>
        <span className="text-sm font-medium text-stone-700">{monthLabel}</span>
        <button onClick={nextMonth} className="text-stone-400 hover:text-stone-700">
          &rarr;
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Calendar grid */}
        <div className="flex-1 overflow-auto p-4">
          {/* Day headers */}
          <div className="grid grid-cols-5 gap-1 text-center text-xs font-semibold text-stone-500">
            {DAY_LABELS.map((d) => (
              <div key={d} className="py-1">
                {d}
              </div>
            ))}
          </div>

          {/* Day cells */}
          <div className="grid grid-cols-5 gap-1">
            {days.map((day) => {
              const dateStr = formatDate(day);
              const isCurrentMonth = day.getMonth() === currentMonth.month;
              const dow = (day.getDay() + 6) % 7; // 0=Mon
              const isCadenceDay = cadenceDays.has(dow);
              const dayPosts = postsByDate[dateStr] || [];
              const isToday = dateStr === formatDate(new Date());

              return (
                <div
                  key={dateStr}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => onDropOnDate(dateStr)}
                  className={`min-h-[100px] rounded-lg border p-1.5 transition-colors ${
                    !isCurrentMonth
                      ? "border-stone-100 bg-stone-50/50 opacity-40"
                      : isCadenceDay
                      ? "border-indigo-200 bg-indigo-50/30"
                      : "border-stone-100 bg-white"
                  } ${isToday ? "ring-2 ring-indigo-400" : ""} ${
                    dragPostId ? "hover:bg-indigo-100/50" : ""
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xs font-medium ${
                        isToday ? "text-indigo-600" : "text-stone-400"
                      }`}
                    >
                      {day.getDate()}
                    </span>
                    {isCadenceDay && isCurrentMonth && (
                      <span className="h-1.5 w-1.5 rounded-full bg-indigo-300" />
                    )}
                  </div>

                  {/* Post cards in this cell */}
                  <div className="mt-1 space-y-1">
                    {dayPosts.map((post) => {
                      const isPushed = !!post.ordinal_post_id;
                      return (
                      <div
                        key={post.id}
                        draggable={!isPushed}
                        onDragStart={isPushed ? undefined : () => onDragStart(post.id)}
                        onClick={() => setExpandedPost(post)}
                        className={`rounded border p-1.5 text-xs leading-tight shadow-sm ${
                          isPushed
                            ? "cursor-pointer opacity-75"
                            : "cursor-grab transition-shadow hover:shadow-md active:cursor-grabbing"
                        } ${statusColor(post.status)}`}
                        title="Click to view full post"
                      >
                        <div className="flex items-start gap-1">
                          <span
                            className={`mt-0.5 shrink-0 rounded px-1 py-0.5 text-[10px] font-semibold ${statusBadge(
                              post.status
                            )}`}
                          >
                            {post.status === "draft"
                              ? "D"
                              : post.status === "pushed" || post.status === "scheduled"
                              ? "S"
                              : "P"}
                          </span>
                          <span className="line-clamp-2 text-stone-700">
                            {post.hook || post.content_preview?.slice(0, 60)}
                          </span>
                        </div>
                      </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Unscheduled sidebar */}
        <div
          className="w-72 shrink-0 overflow-auto border-l border-stone-200 bg-stone-50 p-3"
          onDragOver={(e) => e.preventDefault()}
          onDrop={() => onDropOnUnscheduled()}
        >
          <h3 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
            Unscheduled ({unscheduled.length})
          </h3>
          <p className="mt-1 text-[10px] text-stone-400">
            Drag posts onto calendar days to schedule. Or use Auto-assign.
          </p>

          <div className="mt-3 space-y-2">
            {unscheduled
              .sort((a, b) => (a.publication_order ?? 99) - (b.publication_order ?? 99))
              .map((post) => (
                <div
                  key={post.id}
                  draggable
                  onDragStart={() => onDragStart(post.id)}
                  onClick={() => setExpandedPost(post)}
                  className={`cursor-grab rounded-lg border p-2.5 text-xs shadow-sm transition-shadow hover:shadow-md active:cursor-grabbing ${statusColor(
                    post.status
                  )}`}
                  title="Click to view full post"
                >
                  <div className="flex items-start gap-1.5">
                    {post.publication_order != null && (
                      <span className="shrink-0 rounded bg-stone-200 px-1.5 py-0.5 text-[10px] font-bold text-stone-600">
                        #{post.publication_order}
                      </span>
                    )}
                    <div>
                      <p className="font-medium leading-tight text-stone-800">
                        {post.hook || post.content_preview?.slice(0, 80)}
                      </p>
                      {post.why_post && (
                        <p className="mt-1 text-[10px] italic text-stone-400 line-clamp-2">
                          {post.why_post}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}

            {unscheduled.length === 0 && (
              <p className="py-6 text-center text-xs text-stone-400">
                All posts are scheduled.
                <br />
                Drag posts here to unschedule.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Post detail modal */}
      {expandedPost && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={() => setExpandedPost(null)}
        >
          <div
            className="max-h-[80vh] w-full max-w-2xl overflow-auto rounded-xl border border-stone-200 bg-white p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between">
              <div>
                <span
                  className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${statusBadge(
                    expandedPost.status
                  )}`}
                >
                  {expandedPost.status}
                </span>
                {expandedPost.scheduled_date && (
                  <span className="ml-2 text-xs text-stone-500">
                    Scheduled: {expandedPost.scheduled_date}
                  </span>
                )}
                {expandedPost.ordinal_post_id && (
                  <span className="ml-2 text-xs text-blue-500">
                    Ordinal: {expandedPost.ordinal_post_id.slice(0, 8)}...
                  </span>
                )}
              </div>
              <button
                onClick={() => setExpandedPost(null)}
                className="text-stone-400 hover:text-stone-700"
              >
                &times;
              </button>
            </div>

            {expandedPost.why_post && (
              <p className="mb-3 rounded-lg bg-stone-50 p-3 text-xs italic text-stone-500">
                {expandedPost.why_post}
              </p>
            )}

            <div className="whitespace-pre-wrap text-sm leading-relaxed text-stone-800">
              {expandedPost.content}
            </div>

            {/* Push individual post */}
            {expandedPost.status === "draft" && !expandedPost.ordinal_post_id && (
              <div className="mt-4 flex items-center gap-3 border-t border-stone-100 pt-4">
                {!expandedPost.scheduled_date && (
                  <span className="text-xs text-amber-600">
                    No scheduled date — assign one on the calendar first, or push without a date.
                  </span>
                )}
                <div className="flex-1" />
                <button
                  onClick={async () => {
                    try {
                      const data = await ghostwriterApi.pushSingle(company, expandedPost.id);
                      if (data.status === "pushed") {
                        alert("Pushed to Ordinal!");
                      } else {
                        alert(`Status: ${data.status}`);
                      }
                      setExpandedPost(null);
                      await fetchPosts();
                    } catch (e) {
                      alert(`Push failed: ${e}`);
                    }
                  }}
                  className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-emerald-700"
                >
                  Push to Ordinal{expandedPost.scheduled_date ? ` (${expandedPost.scheduled_date})` : ""}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
