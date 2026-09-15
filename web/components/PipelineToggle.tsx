"use client";

import type { Pipeline } from "@/lib/api";

import styles from "./PipelineToggle.module.css";

interface PipelineToggleProps {
  value: Pipeline;
  onChange: (value: Pipeline) => void;
}

export function PipelineToggle({ value, onChange }: PipelineToggleProps) {
  return (
    <div className={styles.wrap} role="radiogroup" aria-label="Retrieval pipeline">
      <button
        type="button"
        role="radio"
        aria-checked={value === "improved"}
        className={`${styles.option} ${value === "improved" ? styles.active : ""}`}
        onClick={() => onChange("improved")}
      >
        Improved
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={value === "naive"}
        className={`${styles.option} ${value === "naive" ? styles.active : ""}`}
        onClick={() => onChange("naive")}
      >
        Naive
      </button>
      <p className={styles.hint}>
        {value === "improved"
          ? "Hybrid retrieval + rerank, cited answers, refusal below the confidence floor."
          : "Fixed-length chunking, flat context, no citations, no refusal gating — the baseline this project improves on."}
      </p>
    </div>
  );
}
