import Link from "next/link";

import { CommandPalette } from "./CommandPalette";
import styles from "./Nav.module.css";

export function Nav() {
  return (
    <header className={styles.nav}>
      <div className={`container ${styles.inner}`}>
        <Link href="/" className={styles.brand}>
          Groundwork
        </Link>
        <div className={styles.right}>
          <CommandPalette />
          <Link href="/demo" className="btn btn--primary">
            Try the demo
          </Link>
        </div>
      </div>
    </header>
  );
}
