"""Regenerate the sample runs in examples/ (offline: fixture data + mock agents).

    python scripts/make_examples.py

Produces one single-date rebalance and one short backtest, writes them to
examples/ under stable names, and rewrites absolute local paths to repo-relative
ones so the committed files don't leak anything machine-specific.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
AS_OF = "2024-09-30"
START, END = "2024-01-31", "2024-06-30"


def _run(args: list[str], runs_dir: str) -> None:
    env = {**os.environ, "STRICT_MODEL": "0", "RUNS_DIR": runs_dir}
    subprocess.run([sys.executable, "-m", *args], env=env, check=True, cwd=ROOT)


def _relativize(obj, tmp: str):
    """Replace absolute paths: run logs -> basename, repo files -> repo-relative."""
    if isinstance(obj, dict):
        return {k: _relativize(v, tmp) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize(v, tmp) for v in obj]
    if isinstance(obj, str):
        if obj.startswith(tmp):
            return os.path.basename(obj)
        if obj.startswith(ROOT + os.sep):
            return os.path.relpath(obj, ROOT)
    return obj


def _write(src: str, dest_name: str, tmp: str) -> None:
    with open(src) as fh:
        data = json.load(fh)
    dest = os.path.join(EXAMPLES, dest_name)
    with open(dest, "w") as fh:
        json.dump(_relativize(data, tmp), fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(dest, ROOT)} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main() -> None:
    os.makedirs(EXAMPLES, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)
        _run(["agentic_bl.run_single", "--portfolio-id", "example", "--as-of", AS_OF], tmp)
        (single,) = glob.glob(os.path.join(tmp, f"example_*{AS_OF}_*.json"))
        _write(single, f"example_rebalance_{AS_OF}.json", tmp)

        bt_dir = os.path.join(tmp, "bt")
        _run(["agentic_bl.backtest", "--start", START, "--end", END, "--portfolio-id", "example"], bt_dir)
        (bt,) = glob.glob(os.path.join(bt_dir, "backtest_*.json"))
        _write(bt, f"backtest_example_{START}_{END}.json", tmp)


if __name__ == "__main__":
    main()
