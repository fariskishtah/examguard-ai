"""ExamGuard AI entry point: `python -m examguard` or `examguard`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from examguard.config import default_data_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="examguard", description="ExamGuard AI desktop application")
    ap.add_argument("--home", type=Path, help="runtime data directory (default: per-user app-support folder)")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--demo", type=Path, metavar="VIDEO", help=argparse.SUPPRESS)
    ap.add_argument("--verify-persistence", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--selftest", type=Path, metavar="VIDEO", help="process VIDEO through the real GUI once, print JSON, exit")
    a = ap.parse_args(argv)
    if a.version:
        print("ExamGuard AI 0.1.0")
        return 0
    home = a.home or default_data_dir()
    (home / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(home / "logs" / "examguard.log"), logging.StreamHandler()])
    if a.demo:
        from examguard.ui.selftest import run_demo
        return run_demo(home, a.demo)
    if a.verify_persistence:
        from examguard.ui.selftest import run_verify_persistence
        return run_verify_persistence(home)
    if a.selftest:
        from examguard.ui.selftest import run_selftest
        return run_selftest(home, a.selftest)
    from examguard.ui.app import ExamGuardApp
    app = ExamGuardApp(home)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
