/**
 * API client for the Amphoreus backend.
 */

// API base URL:
//   - In production (built with NEXT_PUBLIC_API_URL=""), this is an empty
//     string, so fetch("/api/...") goes same-origin → Next.js rewrites proxy
//     to the internal backend (see next.config.ts).
//   - In local dev (var unset), falls back to http://localhost:8000.
// We use ?? instead of || so an explicit empty string is preserved.
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function getAuthHeaders(): Promise<Record<string, string>> {
  // TODO: Get Supabase session token
  const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    // Critical for Stage 2: forward CF_Authorization cookie so backend can
    // read the Cloudflare Access JWT. Without this, same-origin rewrites skip
    // the cookie and backend 401s every request.
    credentials: "include",
    headers: { ...headers, ...options.headers },
  });
  if (res.status === 401) {
    // Cloudflare Access session expired. Force a full reload — CF will
    // intercept the navigation and re-issue a login PIN challenge.
    if (typeof window !== "undefined") {
      window.location.href = "/";
    }
    throw new Error("Session expired — redirecting to login");
  }
  if (res.status === 403) {
    let detail = "Forbidden";
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`Forbidden: ${detail}`);
  }
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }
  return res.json();
}

// --- Shared SSE stream helper ---

async function* streamSSE(url: string, initialAfterId = 0) {
  const headers = await getAuthHeaders();

  // Track the last event ID we saw so we can resume from it on reconnect.
  // The backend's run_events table has auto-incrementing IDs; after a
  // reconnect we only need events newer than what we already yielded.
  // initialAfterId > 0 is how a fresh page mount picks up mid-run events
  // it's already rendered (from the events REST endpoint).
  let lastEventId = initialAfterId;
  let reconnects = 0;
  const MAX_RECONNECTS = 30; // ~5 min of retry at 10s intervals
  let sawTerminal = false;

  while (reconnects <= MAX_RECONNECTS && !sawTerminal) {
    try {
      const connectUrl =
        lastEventId > 0
          ? `${API_BASE}${url}?after_id=${lastEventId}`
          : `${API_BASE}${url}`;
      // credentials:"include" forwards the CF_Authorization cookie on
      // Fly + Cloudflare Access so the reconnect doesn't 401 silently.
      const res = await fetch(connectUrl, { headers, credentials: "include" });
      if (!res.ok || !res.body) {
        // If the job is already done, the endpoint might 404 — stop.
        if (res.status === 404) return;
        reconnects++;
        await new Promise((r) => setTimeout(r, 10_000));
        continue;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          // Skip SSE comments (keepalives start with ":")
          if (part.startsWith(":")) continue;
          const line = part.replace(/^data: /, "").trim();
          if (!line) continue;
          try {
            const parsed = JSON.parse(line);
            // Track event ID if present for resume
            if (parsed._event_id && parsed._event_id > lastEventId) {
              lastEventId = parsed._event_id;
            }
            yield parsed;
            // Terminal events end the stream
            if (parsed.type === "done" || parsed.type === "error") {
              sawTerminal = true;
            }
          } catch {
            /* skip malformed */
          }
        }
      }

      // Stream ended (reader returned done). If we already got a terminal
      // event, we're finished. Otherwise the connection dropped mid-stream
      // — reconnect after a brief pause.
      if (sawTerminal) return;
      reconnects++;
      await new Promise((r) => setTimeout(r, 3_000));
    } catch {
      // Network error — retry
      reconnects++;
      await new Promise((r) => setTimeout(r, 10_000));
    }
  }
}

// --- Ghostwriter ---

export const ghostwriterApi = {
  generate: (
    company: string,
    prompt?: string,
    model?: string,
  ) =>
    apiFetch<{ job_id: string; status: string }>("/api/ghostwriter/generate", {
      method: "POST",
      body: JSON.stringify({ company, prompt, model }),
    }),

  streamJob: (jobId: string, afterId = 0) =>
    streamSSE(`/api/ghostwriter/stream/${jobId}`, afterId),

  getJob: (jobId: string) =>
    apiFetch<{ job_id: string; status: string; output?: string; error?: string }>(
      `/api/ghostwriter/jobs/${jobId}`
    ),

  getRuns: (company: string, limit = 20) =>
    apiFetch<{ runs: any[] }>(`/api/ghostwriter/${company}/runs?limit=${limit}`),

  getRunEvents: (runId: string) =>
    apiFetch<{ run: any; events: any[] }>(`/api/ghostwriter/runs/${runId}/events`),

  rollback: (company: string, runId: string) =>
    apiFetch(`/api/ghostwriter/${company}/rollback/${runId}`, { method: "POST" }),

  getFiles: (company: string, path = "") =>
    apiFetch<{ files: any[] }>(`/api/ghostwriter/sandbox/${company}/files?path=${path}`),

  inlineEdit: (company: string, postText: string, instruction: string) =>
    apiFetch<{ result: string }>("/api/ghostwriter/inline-edit", {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, instruction }),
    }),

  getLinkedInUsername: (company: string) =>
    apiFetch<{ username: string | null }>(`/api/ghostwriter/${company}/linkedin-username`),

  saveLinkedInUsername: (company: string, username: string) =>
    apiFetch<{ status: string; username: string }>(`/api/ghostwriter/${company}/linkedin-username`, {
      method: "POST",
      body: JSON.stringify({ username }),
    }),

  getOrdinalUsers: (company: string) =>
    apiFetch<{ users: any[] }>(`/api/ghostwriter/${company}/ordinal-users`),

  // Calendar
  getCalendar: (company: string, month?: string) =>
    apiFetch<{ company: string; month: string | null; posts: any[] }>(
      `/api/ghostwriter/${company}/calendar${month ? `?month=${month}` : ""}`
    ),

  schedulePost: (company: string, postId: string, scheduledDate: string | null) =>
    apiFetch<any>(`/api/ghostwriter/${company}/posts/${postId}/schedule`, {
      method: "PATCH",
      body: JSON.stringify({ scheduled_date: scheduledDate }),
    }),

  autoAssign: (company: string, cadence: string, startDate?: string) =>
    apiFetch<{ assigned: number; posts: any[] }>(
      `/api/ghostwriter/${company}/calendar/auto-assign`,
      {
        method: "POST",
        body: JSON.stringify({ cadence, start_date: startDate }),
      }
    ),

  pushAll: (company: string) =>
    apiFetch<{ pushed: number; results: any[] }>(
      `/api/ghostwriter/${company}/calendar/push-all`,
      { method: "POST" }
    ),

  pushSingle: (company: string, postId: string) =>
    apiFetch<{ id: string; status: string; ordinal_post_id?: string }>(
      `/api/ghostwriter/${company}/calendar/push-single`,
      {
        method: "POST",
        body: JSON.stringify({ post_id: postId }),
      }
    ),
};

// --- Briefings ---

export const briefingsApi = {
  generate: (clientName: string, company: string) =>
    apiFetch<{ job_id: string }>("/api/briefings/generate", {
      method: "POST",
      body: JSON.stringify({ client_name: clientName, company }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/briefings/stream/${jobId}`),

  check: (company: string) =>
    apiFetch<{ exists: boolean }>(`/api/briefings/check/${company}`),

  get: (company: string) =>
    apiFetch<{ content: string }>(`/api/briefings/content/${company}`),
};

// --- Cyrene (Strategic Growth Agent) ---

export const cyreneApi = {
  run: (company: string) =>
    apiFetch<{ job_id: string }>(`/api/strategy/cyrene/${company}`, {
      method: "POST",
    }),

  streamJob: (jobId: string, afterId = 0) =>
    streamSSE(`/api/strategy/stream/${jobId}`, afterId),

  getBrief: (company: string) =>
    apiFetch<any>(`/api/strategy/cyrene/${company}/brief`),
};

// --- Progress Report ---

export const reportApi = {
  getData: (company: string, weeks = 2) =>
    apiFetch<any>(`/api/report/${company}?weeks=${weeks}`),

  getHtml: (company: string, weeks = 2) =>
    apiFetch<{ html: string }>(`/api/report/${company}/html?weeks=${weeks}`),

  generate: (company: string, weeks = 2) =>
    apiFetch<{ job_id: string; status: string }>(
      `/api/report/${company}/generate?weeks=${weeks}`,
      { method: "POST" },
    ),

  streamJob: (jobId: string, afterId = 0) =>
    streamSSE(`/api/report/stream/${jobId}`, afterId),

  renderedUrl: (company: string) => `${API_BASE}/api/report/${company}/rendered`,
};

// --- Posts ---

export const postsApi = {
  list: (company?: string, limit = 50) =>
    apiFetch<{ posts: any[] }>(`/api/posts?${company ? `company=${company}&` : ""}limit=${limit}`),

  create: (company: string, content: string, title?: string) =>
    apiFetch("/api/posts", {
      method: "POST",
      body: JSON.stringify({ company, content, title }),
    }),

  update: (
    postId: string,
    company: string,
    fields: {
      content?: string;
      status?: string;
      title?: string;
      linked_image_id?: string | null;
    }
  ) =>
    apiFetch(`/api/posts/${postId}`, {
      method: "PATCH",
      body: JSON.stringify({ company, ...fields }),
    }),

  delete: (postId: string) =>
    apiFetch(`/api/posts/${postId}`, { method: "DELETE" }),

  rewrite: (postId: string, company: string, postText: string, styleInstruction?: string) =>
    apiFetch<{ result: any }>(`/api/posts/${postId}/rewrite`, {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, style_instruction: styleInstruction }),
    }),

  factCheck: (postId: string, company: string, postText: string) =>
    apiFetch<{
      report: string;
      corrected_post?: string;
      annotated_post?: string;
      citation_comments?: string[];
    }>(`/api/posts/${postId}/fact-check`, {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText }),
    }),

  push: (
    company: string,
    content: string,
    citationComments: string[] = [],
    options?: {
      postId?: string;
      publishAt?: string;
      approvals?: { userId: string; message?: string; dueDate?: string; isBlocking?: boolean }[];
    }
  ) => {
    const base = {
      company,
      ...(options?.publishAt ? { publish_at: options.publishAt } : {}),
      approvals: options?.approvals ?? [],
    };
    const body =
      options?.postId != null && options.postId !== ""
        ? { ...base, post_id: options.postId, content: "" }
        : { ...base, content, citation_comments: citationComments };
    return apiFetch<{ success: boolean; result: string; ordinal_post_ids?: string[] }>(
      "/api/posts/push",
      {
        method: "POST",
        body: JSON.stringify(body),
      }
    );
  },

  pushAll: (
    company: string,
    postsPerMonth: 8 | 12,
    options?: {
      approvals?: { userId: string; message?: string; dueDate?: string; isBlocking?: boolean }[];
    }
  ) =>
    apiFetch<{
      success: boolean;
      pushed: number;
      total: number;
      failed: number;
      cadence: string;
      first_url: string | null;
      errors: string[];
    }>("/api/posts/push-all", {
      method: "POST",
      body: JSON.stringify({
        company,
        posts_per_month: postsPerMonth,
        approvals: options?.approvals ?? [],
      }),
    }),
};

// --- Images ---

export const imagesApi = {
  generate: (
    company: string,
    postText: string,
    model?: string,
    options?: {
      feedback?: string;
      referenceImageId?: string;
      localPostId?: string;
    }
  ) =>
    apiFetch<{ job_id: string; status: string }>("/api/images/generate", {
      method: "POST",
      body: JSON.stringify({
        company,
        post_text: postText,
        model: model ?? "claude-opus-4-6",
        feedback: options?.feedback ?? "",
        reference_image_id: options?.referenceImageId ?? "",
        local_post_id: options?.localPostId ?? "",
      }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/images/stream/${jobId}`),

  list: (company: string, limit = 50) =>
    apiFetch<{ images: any[] }>(`/api/images/${company}?limit=${limit}`),

  getUrl: (company: string, imageId: string) =>
    `${API_BASE}/api/images/${company}/${imageId}`,
};

// --- Research ---

export const researchApi = {
  web: (highlightedText: string, query: string) =>
    apiFetch<{ result: string }>("/api/research/web", {
      method: "POST",
      body: JSON.stringify({ highlighted_text: highlightedText, query }),
    }),

  documents: (company: string, question: string, draftText?: string) =>
    apiFetch<{ result: string }>("/api/research/documents", {
      method: "POST",
      body: JSON.stringify({ company, question, draft_text: draftText }),
    }),

  source: (snippet: string, company: string) =>
    apiFetch<{ result: string }>("/api/research/source", {
      method: "POST",
      body: JSON.stringify({ snippet, company }),
    }),

  abm: (company: string) =>
    apiFetch<{ result: string }>("/api/research/abm", {
      method: "POST",
      body: JSON.stringify({ company }),
    }),
};

// --- Clients ---

export const clientsApi = {
  list: () => apiFetch<{ clients: { slug: string }[] }>("/api/clients"),
};

// --- Me (Stage 2 auth) ---

export type MeResponse = {
  email: string;
  is_admin: boolean;
  allowed_clients: "*" | string[];
  auth_enabled: boolean;
};

export const meApi = {
  get: () => apiFetch<MeResponse>("/api/me"),
};

// --- Auth ---

export const authApi = {
  getPermissions: () => apiFetch<{ user_id: string; role: string }>("/api/auth/permissions"),
  getProfile: () => apiFetch<{ profile: any }>("/api/auth/profile"),
};

// --- CS Dashboard ---

export const csApi = {
  listClients: () => apiFetch<{ clients: any[] }>("/api/cs/clients"),
  getClient: (clientId: string) => apiFetch<{ user: any; posts: any[] }>(`/api/cs/clients/${clientId}`),
};

// --- Transcripts (add-only file management) ---

export type TranscriptFile = {
  filename: string;
  size_bytes: number;
  modified_at: number;
  source_label: string | null;
  uploaded_by: string | null;
  uploaded_at: number | null;
  original_filename: string | null;
  content_type: string | null;
};

export const transcriptsApi = {
  list: (company: string) =>
    apiFetch<{ company: string; files: TranscriptFile[] }>(`/api/transcripts/${company}`),

  // Multipart upload — uses FormData directly, not apiFetch, because apiFetch
  // sets Content-Type: application/json which would break the boundary header.
  upload: async (company: string, file: File, sourceLabel: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("source_label", sourceLabel);
    const res = await fetch(`${API_BASE}/api/transcripts/${company}/upload`, {
      method: "POST",
      credentials: "include",
      body: form,
    });
    if (res.status === 401) {
      if (typeof window !== "undefined") window.location.href = "/";
      throw new Error("Session expired");
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Upload failed (${res.status}): ${detail}`);
    }
    return res.json() as Promise<{
      status: string;
      filename: string;
      size_bytes: number;
      source_label: string;
    }>;
  },

  paste: (company: string, text: string, sourceLabel: string) =>
    apiFetch<{
      status: string;
      filename: string;
      size_bytes: number;
      source_label: string;
    }>(`/api/transcripts/${company}/paste`, {
      method: "POST",
      body: JSON.stringify({ text, source_label: sourceLabel }),
    }),

  delete: (company: string, filename: string) =>
    apiFetch<{ status: string; filename: string }>(
      `/api/transcripts/${company}/${encodeURIComponent(filename)}`,
      { method: "DELETE" }
    ),

  downloadUrl: (company: string, filename: string) =>
    `${API_BASE}/api/transcripts/${company}/${encodeURIComponent(filename)}`,
};

// --- Usage / Spend (admin-only) ---

export type UsageUserRow = {
  user_email: string | null;
  n_calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cost_usd: number;
};

export type UsageClientRow = {
  client_slug: string | null;
  n_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

export type UsageModelRow = {
  model: string;
  provider: string;
  n_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

export type UsageSummary = {
  since: number;
  until: number;
  total: {
    n_calls: number;
    input_tokens: number;
    output_tokens: number;
    cost_usd: number;
  };
  by_user: UsageUserRow[];
};

function usageQueryString(opts: { since?: string; until?: string }): string {
  const params = new URLSearchParams();
  if (opts.since) params.set("since", opts.since);
  if (opts.until) params.set("until", opts.until);
  const s = params.toString();
  return s ? `?${s}` : "";
}

export const usageApi = {
  summary: (opts: { since?: string; until?: string } = {}) =>
    apiFetch<UsageSummary>(`/api/usage/summary${usageQueryString(opts)}`),

  byClient: (opts: { since?: string; until?: string } = {}) =>
    apiFetch<{ since: number; until: number; by_client: UsageClientRow[] }>(
      `/api/usage/by-client${usageQueryString(opts)}`
    ),

  byUser: (email: string, opts: { since?: string; until?: string } = {}) =>
    apiFetch<{
      user_email: string | null;
      since: number;
      until: number;
      by_model: UsageModelRow[];
    }>(`/api/usage/by-user/${encodeURIComponent(email)}${usageQueryString(opts)}`),
};

// --- Deploy (localhost only) ---

export const deployApi = {
  code: (target: string = "both") =>
    apiFetch<{ status: string; target: string; key: string }>("/api/deploy/code", {
      method: "POST",
      body: JSON.stringify({ target }),
    }),

  data: (client?: string) =>
    apiFetch<{ status: string; client: string | null; key: string }>("/api/deploy/data", {
      method: "POST",
      body: JSON.stringify({ client: client ?? null }),
    }),

  status: (key: string) =>
    apiFetch<{ key: string; status: string; log: string; returncode?: number }>(
      `/api/deploy/status/${encodeURIComponent(key)}`
    ),

  ban: (email: string) =>
    apiFetch<{ status: string; email: string }>("/api/deploy/ban", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),

  unban: (email: string) =>
    apiFetch<{ status: string; email: string }>("/api/deploy/unban", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),

  listBanned: () =>
    apiFetch<{ banned: string[] }>("/api/deploy/banned"),

  listUsers: () =>
    apiFetch<{ users: { email: string; role: string }[] }>("/api/deploy/users"),
};

// --- Interview Companion (Tribbie) ---

export const interviewApi = {
  listDevices: () =>
    apiFetch<{ devices: any[]; has_blackhole: boolean; error?: string }>("/api/interview/devices"),

  start: (company: string, clientName?: string) =>
    apiFetch<{ job_id: string; status: string }>("/api/interview/start", {
      method: "POST",
      body: JSON.stringify({ company, client_name: clientName }),
    }),

  stop: (jobId: string, company: string) =>
    apiFetch<{ status: string; job_id: string }>("/api/interview/stop", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId, company }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/interview/stream/${jobId}`),

  trashTranscript: (path: string) =>
    apiFetch<{ status: string; destination: string }>("/api/interview/trash-transcript", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
};
