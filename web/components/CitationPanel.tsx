"use client";

import { useState } from "react";

import type { Citation, SourceRecord } from "@/lib/api";
import { getSource } from "@/lib/api";

import styles from "./CitationPanel.module.css";

export function CitationPanel({ citations }: { citations: Citation[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [sources, setSources] = useState<Record<string, SourceRecord | "loading" | "error">>({});

  if (citations.length === 0) return null;

  async function toggle(citation: Citation, key: string) {
    if (expanded === key) {
      setExpanded(null);
      return;
    }
    setExpanded(key);
    if (!sources[citation.doc_id]) {
      setSources((prev) => ({ ...prev, [citation.doc_id]: "loading" }));
      try {
        const record = await getSource(citation.doc_id);
        setSources((prev) => ({ ...prev, [citation.doc_id]: record }));
      } catch {
        setSources((prev) => ({ ...prev, [citation.doc_id]: "error" }));
      }
    }
  }

  return (
    <div className={styles.panel}>
      <p className={styles.label}>Sources</p>
      <ul className={styles.list}>
        {citations.map((c, i) => {
          const key = `${c.chunk_id}-${i}`;
          const isOpen = expanded === key;
          const source = sources[c.doc_id];
          return (
            <li key={key} className={styles.item}>
              <button type="button" className={styles.trigger} onClick={() => toggle(c, key)} aria-expanded={isOpen}>
                <span className={styles.filename}>{c.source_filename}</span>
                {c.heading_path && <span className={styles.heading}>{c.heading_path}</span>}
              </button>
              <blockquote className={styles.quote}>&ldquo;{c.cited_text}&rdquo;</blockquote>
              {isOpen && (
                <div className={styles.detail}>
                  {source === "loading" && <span className={styles.muted}>Loading source…</span>}
                  {source === "error" && <span className={styles.muted}>Couldn&rsquo;t load source metadata.</span>}
                  {source && source !== "loading" && source !== "error" && (
                    <dl className={styles.meta}>
                      <div>
                        <dt>Status</dt>
                        <dd>{source.status}</dd>
                      </div>
                      <div>
                        <dt>Chunks</dt>
                        <dd>{source.chunk_count}</dd>
                      </div>
                      {source.version && (
                        <div>
                          <dt>Version</dt>
                          <dd>{source.version}</dd>
                        </div>
                      )}
                      <div>
                        <dt>Ingested</dt>
                        <dd>{new Date(source.ingested_at).toLocaleString()}</dd>
                      </div>
                    </dl>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
