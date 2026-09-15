"use client";

import { useState } from "react";

import type { Citation } from "@/lib/api";
import { streamQuery } from "@/lib/api";

import styles from "./LiveDemoCard.module.css";

const DEFAULT_QUESTION = "How many days per week can employees work remotely?";

type Status = "idle" | "streaming" | "done" | "error";

export function LiveDemoCard() {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [stopReason, setStopReason] = useState<string | null>(null);

  async function run() {
    if (status === "streaming") return;
    setStatus("streaming");
    setAnswer("");
    setCitations([]);
    setStopReason(null);

    await streamQuery(
      { query: question, pipeline: "improved" },
      {
        onToken: (text) => setAnswer((prev) => prev + text),
        onDone: (done) => {
          setAnswer(done.text);
          setCitations(done.citations);
          setStopReason(done.stop_reason);
          setStatus("done");
        },
        onError: () => setStatus("error"),
      },
    );
  }

  return (
    <div className={styles.card}>
      <div className={styles.bar}>
        <span className={styles.method}>POST</span>
        <span className={styles.path}>/query</span>
        <span className={`${styles.chip} ${styles[status]}`}>
          {status === "idle" && "ready"}
          {status === "streaming" && "streaming…"}
          {status === "done" && "200 OK"}
          {status === "error" && "error"}
        </span>
      </div>

      <div className={styles.body}>
        <label className={styles.promptRow}>
          <span className={styles.promptMark}>&gt;</span>
          <input
            className={styles.promptInput}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={status === "streaming"}
            aria-label="Question to ask the live demo"
          />
        </label>

        {(answer || status === "streaming") && (
          <p className={styles.answer}>
            {answer}
            {status === "streaming" && <span className={styles.caret} aria-hidden="true" />}
          </p>
        )}

        {status === "done" && (
          <p className={styles.meta}>
            {stopReason === "end_turn" ? "grounded" : stopReason} · {citations.length} citation
            {citations.length === 1 ? "" : "s"}
          </p>
        )}
      </div>

      <button type="button" className={styles.run} onClick={run} disabled={status === "streaming"}>
        {status === "streaming" ? "Running…" : "Run against the live corpus"}
      </button>
    </div>
  );
}
