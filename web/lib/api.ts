const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export interface Citation {
  chunk_id: string;
  doc_id: string;
  source_filename: string;
  heading_path: string;
  cited_text: string;
}

export interface QueryMeta {
  session_id: string | null;
  rewritten_query: string | null;
  retrieved_count: number;
}

export interface QueryDone {
  text: string;
  citations: Citation[];
  model: string;
  stop_reason: string;
}

export type Pipeline = "improved" | "naive";

export interface StreamQueryBody {
  query: string;
  session_id?: string;
  pipeline?: Pipeline;
}

export interface StreamQueryCallbacks {
  onMeta?: (meta: QueryMeta) => void;
  onToken?: (text: string) => void;
  onDone?: (done: QueryDone) => void;
  onError?: (error: string) => void;
}

/** Parses the api's raw SSE proxy of rag-core's stream: "meta" once, "token"
 * zero-or-more times, exactly one "done" (see services/api/src/routes/query.ts
 * and services/rag-core/app/main.py's internal_query_stream). Uses fetch +
 * a manual reader rather than EventSource, since EventSource can't send a
 * POST body. */
export async function streamQuery(body: StreamQueryBody, callbacks: StreamQueryCallbacks, signal?: AbortSignal) {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    callbacks.onError?.(`request failed (${res.status})${detail ? `: ${detail}` : ""}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      const eventLine = block.split("\n").find((l) => l.startsWith("event: "));
      const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
      if (!eventLine || !dataLine) continue;

      const event = eventLine.slice("event: ".length);
      const data = JSON.parse(dataLine.slice("data: ".length));

      if (event === "meta") callbacks.onMeta?.(data as QueryMeta);
      else if (event === "token") callbacks.onToken?.((data as { text: string }).text);
      else if (event === "done") callbacks.onDone?.(data as QueryDone);
      // Emitted only if the upstream connection breaks after streaming has
      // already started (see services/api/src/routes/query.ts) - there's no
      // "done" event coming after this, so surface it the same way as a
      // request-level failure rather than leaving the caller waiting forever.
      else if (event === "error") callbacks.onError?.((data as { error: string }).error);
    }
  }
}

export interface IngestResult {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  failure_reason: string | null;
  skipped_duplicate: boolean;
}

export interface IngestResponse {
  root: string;
  processed: number;
  ready: number;
  skipped_duplicate: number;
  failed: number;
  unprocessable: number;
  results: IngestResult[];
}

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.error ?? `upload failed (${res.status})`);
  }
  return res.json();
}

export interface SourceRecord {
  doc_id: string;
  filename: string;
  mime_type: string;
  status: string;
  page_count: number | null;
  version: string | null;
  supersedes: string | null;
  chunk_count: number;
  ingested_at: string;
}

export async function getSource(docId: string): Promise<SourceRecord> {
  const res = await fetch(`${API_URL}/sources/${encodeURIComponent(docId)}`);
  if (!res.ok) throw new Error(`source lookup failed (${res.status})`);
  return res.json();
}
