from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.research_state import write_json


SAFE_PREFIXES = (
    "src/",
    "scripts/",
    "configs/",
    "run_auto_world_model_loop.py",
    "run_stage13_",
    "outputs/reports/",
    "README_RESULTS.md",
    "research_state.json",
)


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def safe_changed_files() -> list[str]:
    files = []
    for line in git_status().splitlines():
        path = line[3:].strip()
        if path.startswith(SAFE_PREFIXES):
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a safe git snapshot for auto-loop reports/code.")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--message", default="Add auto orchestrator loop")
    args = parser.parse_args()
    files = safe_changed_files()
    payload = {"safe_files": files, "committed": False, "commit_hash": None}
    if args.commit and files:
        subprocess.run(["git", "add", *files], check=True)
        subprocess.run(["git", "commit", "-m", args.message], check=True)
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, stdout=subprocess.PIPE, check=True)
        payload["committed"] = True
        payload["commit_hash"] = result.stdout.strip()
    write_json("outputs/reports/auto_git_snapshot.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
