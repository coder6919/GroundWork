"use client";

import { useRef, useState } from "react";

import type { Citation, Pipeline } from "@/lib/api";
import { streamQuery } from "@/lib/api";

import { CitationPanel } from "./CitationPanel";
import { ConfidenceIndicator } from "./ConfidenceIndicator";
import styles from "./ChatPanel.module.css";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  stopReason?: string;
  rewrittenQuery?: string | null;
  streaming?: boolean;
}

const SAMPLE_QUESTIONS = [
  "What is the standard shipping cost to the US?",
  "How many days per week can employees work remotely?",
  "How long does it take to get a laptop as a new hire?",
];

export function ChatPanel({ pipeline }: { pipeline: Pipeline }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const sessionId = useRef(crypto.randomUUID());

  async function send(question: string) {
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setInput("");

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", text: trimmed };
    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = { id: assistantId, role: "assistant", text: "", streaming: true };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    await streamQuery(
      { query: trimmed, session_id: sessionId.current, pipeline },
      {
        onToken: (text) => {
          setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, text: m.text + text } : m)));
        },
        onDone: (done) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    text: done.text,
                    citations: done.citations,
                    stopReason: done.stop_reason,
                    streaming: false,
                  }
                : m,
            ),
          );
        },
        onError: (error) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, text: `Error: ${error}`, streaming: false } : m)),
          );
        },
      },
    ).finally(() => setBusy(false));
  }

  return (
    <div className={styles.panel}>
      {messages.length === 0 && (
        <div className={styles.empty}>
          <p className={styles.emptyLede}>Ask a question about the sample corpus, or try one:</p>
          <div className={styles.samples}>
            {SAMPLE_QUESTIONS.map((q) => (
              <button key={q} type="button" className={styles.sample} onClick={() => send(q)}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className={styles.log}>
        {messages.map((m) => (
          <div key={m.id} className={`${styles.message} ${m.role === "user" ? styles.user : styles.assistant}`}>
            {m.role === "assistant" && <span className={styles.roleLabel}>Groundwork</span>}
            <p className={styles.text}>
              {m.text}
              {m.streaming && <span className={styles.caret} aria-hidden="true" />}
            </p>
            {!m.streaming && m.stopReason && (
              <ConfidenceIndicator stopReason={m.stopReason} citationCount={m.citations?.length ?? 0} />
            )}
            {!m.streaming && m.citations && <CitationPanel citations={m.citations} />}
          </div>
        ))}
      </div>

      <form
        className={styles.form}
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          className={styles.input}
          placeholder="Ask about the sample corpus…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          disabled={busy}
          rows={1}
        />
        <button type="submit" className="btn btn--primary" disabled={busy || !input.trim()}>
          {busy ? "Asking…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
