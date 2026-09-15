"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import styles from "./CommandPalette.module.css";

interface Command {
  id: string;
  label: string;
  hint: string;
  href: string;
}

const COMMANDS: Command[] = [
  { id: "home", label: "Home", hint: "Product overview", href: "/" },
  { id: "how", label: "How it works", hint: "Ingestion → retrieval → grounded answer", href: "/#how-it-works" },
  { id: "demo", label: "Try the demo", hint: "Ask a question, see citations", href: "/demo" },
  { id: "upload", label: "Upload a document", hint: "Add a file to the demo corpus", href: "/demo#upload" },
  { id: "toggle", label: "Naive vs. improved", hint: "Compare pipelines side by side", href: "/demo#pipeline" },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Reset the active row whenever the query changes - adjusted during render
  // (not in an Effect) per https://react.dev/learn/you-might-not-need-an-effect
  // "Adjusting some state when a prop changes".
  const [prevQuery, setPrevQuery] = useState(query);
  if (query !== prevQuery) {
    setPrevQuery(query);
    setActiveIndex(0);
  }

  const results = COMMANDS.filter((c) => c.label.toLowerCase().includes(query.toLowerCase()));

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    setActiveIndex(0);
  }, []);

  const select = useCallback(
    (cmd: Command) => {
      close();
      router.push(cmd.href);
    },
    [close, router],
  );

  useEffect(() => {
    function onKeydown(e: KeyboardEvent) {
      const isCmdK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      if (isCmdK) {
        e.preventDefault();
        setOpen((prev) => !prev);
        return;
      }
      if (!open) return;
      if (e.key === "Escape") {
        close();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter" && results[activeIndex]) {
        e.preventDefault();
        select(results[activeIndex]);
      }
    }
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [open, results, activeIndex, close, select]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  return (
    <>
      <button
        type="button"
        className={styles.pill}
        onClick={() => setOpen(true)}
        aria-label="Search (Cmd K)"
      >
        <span className={styles.pillIcon} aria-hidden="true" />
        <span className={styles.pillText}>Jump to…</span>
        <span className={styles.pillKbd}>
          <kbd>⌘</kbd>
          <kbd>K</kbd>
        </span>
      </button>

      <div className={`${styles.overlay} ${open ? styles.overlayOpen : ""}`} aria-hidden={!open}>
        <div className={styles.backdrop} onClick={close} />
        <div className={styles.panel} role="dialog" aria-modal="true" aria-label="Command palette">
          <div className={styles.field}>
            <span className={styles.fieldIcon} aria-hidden="true" />
            <input
              ref={inputRef}
              className={styles.input}
              placeholder="Jump to a page or section…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search commands"
            />
            <kbd className={styles.escHint}>esc</kbd>
          </div>
          <div className={styles.results} role="listbox">
            {results.length === 0 && <p className={styles.empty}>No matches.</p>}
            {results.map((cmd, i) => (
              <button
                key={cmd.id}
                type="button"
                role="option"
                aria-selected={i === activeIndex}
                className={`${styles.item} ${i === activeIndex ? styles.itemActive : ""}`}
                onMouseEnter={() => setActiveIndex(i)}
                onClick={() => select(cmd)}
              >
                <span className={styles.itemLabel}>{cmd.label}</span>
                <span className={styles.itemHint}>{cmd.hint}</span>
              </button>
            ))}
          </div>
          <div className={styles.footer}>
            <span>
              <kbd>↑</kbd>
              <kbd>↓</kbd> navigate
            </span>
            <span>
              <kbd>↵</kbd> open
            </span>
            <span>
              <kbd>esc</kbd> close
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
