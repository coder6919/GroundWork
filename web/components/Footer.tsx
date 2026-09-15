import styles from "./Footer.module.css";

export function Footer() {
  return (
    <footer className={styles.foot}>
      <div className={`container ${styles.inner}`}>
        <p>© 2026 Groundwork · Apache-2.0 · a RAG knowledge agent, built end to end as a portfolio project</p>
      </div>
    </footer>
  );
}
