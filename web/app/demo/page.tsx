"use client";

import { useState } from "react";

import { ChatPanel } from "@/components/ChatPanel";
import { PipelineToggle } from "@/components/PipelineToggle";
import { UploadDropzone } from "@/components/UploadDropzone";
import type { Pipeline } from "@/lib/api";

import styles from "./page.module.css";

export default function DemoPage() {
  const [pipeline, setPipeline] = useState<Pipeline>("improved");

  return (
    <main>
      <div className={`container ${styles.header}`}>
        <p className="mono-label">Live demo</p>
        <h1 className={styles.title}>Ask the sample corpus</h1>
        <p className={styles.lede}>
          Answers come from the ingested documents only — grounded, cited, and gated by confidence. Switch to the
          naive pipeline to see what this project improves on.
        </p>
      </div>

      <div className={`container ${styles.section}`} id="pipeline">
        <PipelineToggle value={pipeline} onChange={setPipeline} />
      </div>

      <div className={`container ${styles.section}`} id="upload">
        <h2 className={styles.sectionTitle}>Add a document</h2>
        <p className={styles.sectionHint}>Uploads join the demo corpus for this session — .txt, .md, or .pdf.</p>
        <UploadDropzone />
      </div>

      <div className={`container ${styles.section}`}>
        <h2 className={styles.sectionTitle}>Ask</h2>
        <ChatPanel key={pipeline} pipeline={pipeline} />
      </div>
    </main>
  );
}
