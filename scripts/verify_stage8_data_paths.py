#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_root(name: str, root: str | None, patterns: list[str]) -> dict:
    if not root:
        return {"dataset": name, "path_provided": False, "exists": False, "matched_files": 0, "status": "not_provided"}
    path = Path(root)
    matches = []
    if path.exists():
        for pattern in patterns:
            matches.extend(path.rglob(pattern))
    return {
        "dataset": name,
        "path_provided": True,
        "path": str(path),
        "exists": path.exists(),
        "matched_files": len(matches),
        "status": "ok" if path.exists() and matches else "no_matching_files",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdd-root")
    parser.add_argument("--opentraj-root")
    parser.add_argument("--trajnet-root", default="data/stage5b_raw/trajnetplusplusdataset")
    parser.add_argument("--eth-ucy-root", default="data/stage5b_raw/trajnetplusplusdataset")
    parser.add_argument("--aerialmpt-root", default="data/aerialmpt/extracted")
    args = parser.parse_args()
    records = [
        inspect_root("sdd", args.sdd_root, ["annotations.txt", "*.csv"]),
        inspect_root("opentraj", args.opentraj_root, ["*.csv", "*.txt"]),
        inspect_root("trajnet_full", args.trajnet_root, ["*.ndjson", "*.txt"]),
        inspect_root("eth_ucy_full", args.eth_ucy_root, ["*.ndjson", "*.txt"]),
        inspect_root("aerialmpt_long", args.aerialmpt_root, ["*.csv"]),
    ]
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8_path_verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

