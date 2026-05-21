from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_path(path: str | None, markers: list[str]) -> dict:
    if not path:
        return {"provided": False, "exists": False, "marker_count": 0, "status": "not_provided"}
    root = Path(path).expanduser()
    marker_count = 0
    if root.exists():
        for marker in markers:
            marker_count += len(list(root.rglob(marker)))
    return {
        "provided": True,
        "path": str(root),
        "exists": root.exists(),
        "marker_count": marker_count,
        "status": "verified" if root.exists() and marker_count > 0 else ("exists_but_unrecognized" if root.exists() else "missing"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local Stage 10 pedestrian/drone dataset paths.")
    parser.add_argument("--sdd-root")
    parser.add_argument("--opentraj-root")
    parser.add_argument("--trajnet-root")
    parser.add_argument("--eth-ucy-root")
    parser.add_argument("--aerialmpt-root")
    args = parser.parse_args()
    rows = {
        "sdd": inspect_path(args.sdd_root, ["annotations.txt"]),
        "opentraj": inspect_path(args.opentraj_root, ["*.csv", "*.txt"]),
        "trajnet_full": inspect_path(args.trajnet_root, ["*.ndjson", "*.txt"]),
        "eth_ucy_full": inspect_path(args.eth_ucy_root, ["*.txt", "*.csv"]),
        "aerialmpt_long": inspect_path(args.aerialmpt_root, ["*_gts.txt", "image_list.txt"]),
    }
    out = Path("outputs/reports/stage10_path_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
