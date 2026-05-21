#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_path(path: str | None, kind: str) -> dict:
    if not path:
        return {"kind": kind, "path": None, "exists": False, "status": "not_provided"}
    root = Path(path)
    files = []
    if root.exists():
        if kind == "sdd":
            files = list(root.rglob("annotations.txt"))
        elif kind == "opentraj":
            files = list(root.rglob("*.csv")) + list(root.rglob("*.txt"))
        else:
            files = list(root.rglob("*"))
    return {"kind": kind, "path": str(root), "exists": root.exists(), "candidate_files": len(files), "status": "ok" if root.exists() and files else "missing_or_empty"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdd-root")
    parser.add_argument("--opentraj-root")
    parser.add_argument("--trajnet-root", default="data/stage5b_world_state/trajnet")
    parser.add_argument("--eth-ucy-root", default="data/stage5b_world_state/eth_ucy")
    parser.add_argument("--aerialmpt-root")
    args = parser.parse_args()
    rows = [
        inspect_path(args.sdd_root, "sdd"),
        inspect_path(args.opentraj_root, "opentraj"),
        inspect_path(args.trajnet_root, "trajnet_full"),
        inspect_path(args.eth_ucy_root, "eth_ucy_full"),
        inspect_path(args.aerialmpt_root, "aerialmpt_long"),
    ]
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8p5_path_verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
