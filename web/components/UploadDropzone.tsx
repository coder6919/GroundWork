"use client";

import { useCallback, useRef, useState } from "react";

import { uploadDocument } from "@/lib/api";

import styles from "./UploadDropzone.module.css";

type FileState = {
  name: string;
  status: "uploading" | "ready" | "error";
  chunkCount?: number;
  error?: string;
};

const ACCEPTED = ".txt,.md,.markdown,.pdf";

export function UploadDropzone({ onIngested }: { onIngested?: (filename: string) => void }) {
  const [dragOver, setDragOver] = useState(false);
  const [files, setFiles] = useState<FileState[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0) return;
      const file = fileList[0];
      setFiles((prev) => [{ name: file.name, status: "uploading" }, ...prev]);
      try {
        const result = await uploadDocument(file);
        const first = result.results[0];
        setFiles((prev) =>
          prev.map((f) =>
            f.name === file.name && f.status === "uploading"
              ? {
                  name: file.name,
                  status: first?.status === "failed" || first?.status === "unprocessable" ? "error" : "ready",
                  chunkCount: first?.chunk_count,
                  error: first?.failure_reason ?? undefined,
                }
              : f,
          ),
        );
        if (first && first.status !== "failed" && first.status !== "unprocessable") {
          onIngested?.(file.name);
        }
      } catch (err) {
        setFiles((prev) =>
          prev.map((f) =>
            f.name === file.name && f.status === "uploading"
              ? { name: file.name, status: "error", error: err instanceof Error ? err.message : "upload failed" }
              : f,
          ),
        );
      }
    },
    [onIngested],
  );

  return (
    <div>
      <div
        className={`${styles.zone} ${dragOver ? styles.dragOver : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload a document"
      >
        <p className={styles.title}>Drop a file, or click to choose</p>
        <p className={styles.hint}>.txt · .md · .pdf</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className={styles.input}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <ul className={styles.list}>
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`} className={styles.row}>
              <span className={styles.filename}>{f.name}</span>
              {f.status === "uploading" && <span className={styles.statusLoading}>Ingesting…</span>}
              {f.status === "ready" && (
                <span className={styles.statusReady}>
                  Ready · {f.chunkCount} chunk{f.chunkCount === 1 ? "" : "s"}
                </span>
              )}
              {f.status === "error" && <span className={styles.statusError}>{f.error ?? "Failed"}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
