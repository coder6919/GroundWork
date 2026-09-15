import styles from "./ConfidenceIndicator.module.css";

interface ConfidenceIndicatorProps {
  stopReason: string;
  citationCount: number;
}

const COPY: Record<string, { label: string; tone: "grounded" | "warning" | "info" | "neutral" }> = {
  end_turn: { label: "Grounded", tone: "grounded" },
  low_confidence: { label: "Low confidence — declined to answer", tone: "warning" },
  no_context: { label: "No matching context — declined to answer", tone: "warning" },
  clarification_needed: { label: "Needs clarification", tone: "info" },
  naive_pipeline: { label: "No confidence gating (naive baseline)", tone: "neutral" },
  generation_error: { label: "Generation failed — try again", tone: "warning" },
};

export function ConfidenceIndicator({ stopReason, citationCount }: ConfidenceIndicatorProps) {
  const copy = COPY[stopReason] ?? { label: stopReason, tone: "neutral" as const };
  return (
    <div className={`${styles.badge} ${styles[copy.tone]}`}>
      <span className={styles.dot} aria-hidden="true" />
      <span>{copy.label}</span>
      {copy.tone === "grounded" && citationCount > 0 && (
        <span className={styles.count}>
          · {citationCount} citation{citationCount === 1 ? "" : "s"}
        </span>
      )}
    </div>
  );
}
