"""Run all inversion example scripts sequentially.

This script finds all files named `inversion_*.py` in the same
directory and executes them one by one using the current Python
interpreter. It logs the outcome of each run and can optionally stop
on the first error.

Usage:
    python run_all_inversions.py         # run all, continue on errors
    python run_all_inversions.py --stop  # stop on first non-zero exit
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("run_all_inversions")


def find_inversion_scripts(dir_path: Path):
    return sorted([p for p in dir_path.glob("inversion_*.py") if p.name != Path(__file__).name])


def run_script(path: Path, python_exe: str) -> int:
    logger.info("Running %s", path.name)
    proc = subprocess.run([python_exe, str(path)])
    logger.info("%s finished (exit=%s)", path.name, proc.returncode)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all inversion example scripts sequentially.")
    parser.add_argument("--stop", "-s", action="store_true", help="Stop on first non-zero exit code")
    parser.add_argument("--dir", "-d", default=Path(__file__).resolve().parent, type=Path, help="Directory containing inversion scripts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    python_exe = sys.executable
    scripts = find_inversion_scripts(args.dir)

    if not scripts:
        logger.warning("No inversion_*.py scripts found in %s", args.dir)
        return 0

    failures = []
    for script in scripts:
        rc = run_script(script, python_exe)
        if rc != 0:
            failures.append((script.name, rc))
            if args.stop:
                logger.error("Stopping on first failure: %s (exit=%s)", script.name, rc)
                break

    if failures:
        logger.error("Completed with %d failure(s): %s", len(failures), failures)
        return 1

    logger.info("All inversion scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
