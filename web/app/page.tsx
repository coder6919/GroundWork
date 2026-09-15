import Link from "next/link";

import { LiveDemoCard } from "@/components/LiveDemoCard";
import { Reveal } from "@/components/Reveal";

import styles from "./page.module.css";

const STEPS = [
  {
    n: "01",
    title: "Ingest",
    body: "Markdown, plain text, and PDF (with table extraction and OCR fallback) are chunked structure-aware — headings and tables stay intact, never cut mid-block.",
  },
  {
    n: "02",
    title: "Retrieve",
    body: "Hybrid dense + sparse search (Voyage embeddings + BM25, fused with RRF) surfaces a candidate pool, then Cohere reranks it down to the top matches.",
  },
  {
    n: "03",
    title: "Generate",
    body: "Claude answers from the retrieved context only, with native API citations tracing every claim back to its source chunk — no hand-rolled citation parsing.",
  },
  {
    n: "04",
    title: "Gate",
    body: "Below a calibrated confidence floor, the model call is skipped entirely and the answer refuses outright — cost saved, no guess offered.",
  },
];

export default function Home() {
  return (
    <main>
      <section className={`container ${styles.hero}`}>
        <div className={styles.heroText}>
          <h1 className={styles.h1}>Answers your documents can prove</h1>
          <p className={styles.lede}>
            Groundwork retrieves from your knowledge base, cites every claim back to its source, and refuses to
            guess when it isn&rsquo;t confident. Run the live demo below against a real sample corpus — no sign-up.
          </p>
          <div className={styles.heroActions}>
            <Link href="/demo" className="btn btn--primary">
              Try the full demo
            </Link>
            <a href="#how-it-works" className="btn btn--ghost">
              How it works
            </a>
          </div>
        </div>
        <div className={styles.heroDemo}>
          <LiveDemoCard />
        </div>
      </section>

      <section id="how-it-works" className={`container ${styles.steps}`}>
        <h2 className={styles.sectionTitle}>Retrieve-then-generate, made honest</h2>
        <div className={styles.stepsGrid}>
          {STEPS.map((step) => (
            <Reveal key={step.n} className={styles.step}>
              <span className={styles.stepN}>{step.n}</span>
              <h3 className={styles.stepTitle}>{step.title}</h3>
              <p className={styles.stepBody}>{step.body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <section className={`container ${styles.features}`}>
        <div className={styles.featuresGrid}>
          <Reveal className={styles.featureWide}>
            <p className="mono-label">Citations</p>
            <h3 className={styles.featureTitle}>Every claim traces to a chunk</h3>
            <p className={styles.featureBody}>
              Citations use the Messages API&rsquo;s native citations feature — each retrieved chunk is its own
              document block, and Claude&rsquo;s response citations map straight back to the source file, heading,
              and exact cited text. Click through to see the original document&rsquo;s metadata.
            </p>
          </Reveal>
          <Reveal className={styles.featureNarrow}>
            <p className="mono-label">Refusal</p>
            <h3 className={styles.featureTitle}>Declines below the floor</h3>
            <p className={styles.featureBody}>
              A calibrated rerank-score floor gates every answer. Below it, the model call never fires — the
              refusal is free, not a disclaimer bolted onto a guess.
            </p>
          </Reveal>
          <Reveal className={styles.featureNarrow}>
            <p className="mono-label">Hybrid retrieval</p>
            <h3 className={styles.featureTitle}>Dense + sparse, reranked</h3>
            <p className={styles.featureBody}>
              RRF-fused vector and BM25 search, cut to the top matches by Cohere rerank — not vector search alone.
            </p>
          </Reveal>
          <Reveal className={styles.featureWide}>
            <p className="mono-label">Naive vs. improved</p>
            <h3 className={styles.featureTitle}>The contrast is live, not asserted</h3>
            <p className={styles.featureBody}>
              A second pipeline runs the same question through fixed-length chunking, flat context, and no
              citations or refusal gating — the baseline this project improves on. Toggle it in the demo and see
              the difference yourself, on the same question.
            </p>
          </Reveal>
        </div>
      </section>

      <section className={styles.band}>
        <div className={`container ${styles.bandInner}`}>
          <p className={styles.bandLabel}>How it&rsquo;s built</p>
          <p className={styles.bandArch}>web (Next.js) → api (Node/Express) → rag-core (Python/FastAPI) → Qdrant</p>
          <p className={styles.bandBody}>
            Generation on Claude Sonnet 5, embeddings on Voyage, rerank on Cohere. The Python core and the public
            Node layer are deliberately separate services — retrieval, reranking, and generation never leave
            rag-core, and it&rsquo;s never reachable from the public internet.
          </p>
        </div>
      </section>

      <section className={`container ${styles.cta}`}>
        <h2 className={styles.ctaTitle}>See it answer a real question</h2>
        <Link href="/demo" className="btn btn--primary">
          Try the demo
        </Link>
      </section>
    </main>
  );
}
