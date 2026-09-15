import { z } from "zod";

/**
 * Environment-driven configuration. Keys are read from the environment only and
 * never hardcoded. All fields have safe defaults so the service boots in Stage 0
 * with just RAG_CORE_SHARED_SECRET set.
 */
const schema = z.object({
  ENV: z.string().default("development"),
  LOG_LEVEL: z.string().default("info"),
  API_PORT: z.coerce.number().int().positive().default(8080),
  RAG_CORE_URL: z.string().url().default("http://rag-core:8000"),
  RAG_CORE_SHARED_SECRET: z.string().default(""),
  INGEST_ENABLED: z
    .string()
    .default("true")
    .transform((v) => v === "true" || v === "1"),
  INGEST_API_KEY: z.string().default(""),
  // --- Stage 9: rate limiting + upload limits ---
  RATE_LIMIT_WINDOW_MS: z.coerce.number().int().positive().default(60_000),
  RATE_LIMIT_MAX_QUERY: z.coerce.number().int().positive().default(20), // per window, per IP
  RATE_LIMIT_MAX_INGEST: z.coerce.number().int().positive().default(5), // per window, per IP
  MAX_UPLOAD_BYTES: z.coerce.number().int().positive().default(20 * 1024 * 1024), // 20MB
  // --- Stage 10: the web frontend is a separate origin (Next.js dev server /
  // Vercel), so the browser needs an explicit CORS allow rather than same-
  // origin defaults. A single origin, not a wildcard - this api also accepts
  // an X-Ingest-Key header, which must never be readable cross-origin from an
  // unapproved site.
  WEB_ORIGIN: z.string().default("http://localhost:3000"),
});

const parsed = schema.safeParse(process.env);
if (!parsed.success) {
  // eslint-disable-next-line no-console
  console.error("Invalid environment configuration:", parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = parsed.data;

/** Non-fatal configuration warnings surfaced at startup. */
export function configWarnings(): string[] {
  const w: string[] = [];
  if (!config.RAG_CORE_SHARED_SECRET) {
    w.push("RAG_CORE_SHARED_SECRET is empty - authenticated calls to rag-core will fail closed.");
  }
  if (config.ENV !== "development" && config.INGEST_ENABLED && !config.INGEST_API_KEY) {
    w.push(
      "INGEST_ENABLED=true without INGEST_API_KEY outside development - a public ingestion endpoint would be unprotected.",
    );
  }
  return w;
}
