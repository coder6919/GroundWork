import { config } from "./config.js";
import { logger } from "./logger.js";

export class RagCoreError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "RagCoreError";
  }
}

interface CallOptions {
  auth?: boolean;
  method?: string;
  body?: RequestInit["body"];
  headers?: Record<string, string>;
  /** Default 5s - generous enough for health/metadata calls, too short for
   * generation or ingestion, which override it explicitly. */
  timeoutMs?: number;
}

/** Performs the HTTP call and returns the raw Response - the body is never
 * consumed here so callers can either `.json()` it or pipe `.body` straight
 * through (Stage 9's SSE proxy needs the latter). */
async function rawCall(path: string, opts: CallOptions = {}): Promise<Response> {
  const url = `${config.RAG_CORE_URL}${path}`;
  const headers: Record<string, string> = { ...opts.headers };

  if (opts.auth) {
    if (!config.RAG_CORE_SHARED_SECRET) {
      throw new RagCoreError("RAG_CORE_SHARED_SECRET not configured", 503);
    }
    headers["X-Internal-Secret"] = config.RAG_CORE_SHARED_SECRET;
  }

  const started = Date.now();
  let res: Response;
  try {
    res = await fetch(url, {
      method: opts.method ?? "GET",
      headers,
      body: opts.body,
      signal: AbortSignal.timeout(opts.timeoutMs ?? 5000),
    });
  } catch (err) {
    logger.error({ event: "rag_core_call", path, error: String(err) }, "rag-core unreachable");
    throw new RagCoreError(`rag-core unreachable: ${String(err)}`);
  }

  logger.info({ event: "rag_core_call", path, status: res.status, ms: Date.now() - started });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new RagCoreError(`rag-core ${path} responded ${res.status}${detail ? `: ${detail}` : ""}`, res.status);
  }
  return res;
}

async function call<T>(path: string, opts: CallOptions = {}): Promise<T> {
  const res = await rawCall(path, opts);
  return (await res.json()) as T;
}

export interface RagCoreHealth {
  status: string;
  service?: string;
  version?: string;
}

export interface IngestResultOut {
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
  results: IngestResultOut[];
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

export interface QueryRequestBody {
  query: string;
  top_k?: number;
  source_filename?: string;
  doc_version?: string;
  ingested_after?: string;
  ingested_before?: string;
  include_superseded?: boolean;
  session_id?: string;
  pipeline?: "improved" | "naive";
}

export interface UploadedFile {
  buffer: Buffer;
  originalname: string;
  mimetype: string;
}

export const ragCore = {
  health: () => call<RagCoreHealth>("/health"),
  ready: () => call<Record<string, unknown>>("/ready"),
  /** Authenticated: exercises the shared-secret trust boundary end to end. */
  ping: () => call<Record<string, unknown>>("/internal/ping", { auth: true }),

  /** File bytes only, never a server-side path - the public /ingest route
   * must never let a caller point ingestion at an arbitrary rag-core path
   * (see DIRECTION.md's hosted-demo rule). Generous timeout: embedding a
   * real document can take a while. */
  ingestUpload: (file: UploadedFile) => {
    const form = new FormData();
    form.append("file", new Blob([file.buffer], { type: file.mimetype }), file.originalname);
    return call<IngestResponse>("/internal/ingest/upload", {
      auth: true,
      method: "POST",
      body: form,
      timeoutMs: 120_000,
    });
  },

  getSource: (id: string) => call<SourceRecord>(`/internal/sources/${encodeURIComponent(id)}`, { auth: true }),

  /** Returns the raw Response so the route handler can pipe the SSE body
   * straight through to the client - never buffered or parsed here.
   * Generous timeout: generation can legitimately take tens of seconds. */
  queryStream: (body: QueryRequestBody): Promise<Response> =>
    rawCall("/internal/query/stream", {
      auth: true,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      timeoutMs: 60_000,
    }),
};
