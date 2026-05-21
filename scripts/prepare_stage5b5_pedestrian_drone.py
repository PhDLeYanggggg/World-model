from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajnet-full-root", default="data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original")
    parser.add_argument("--sdd-root", default="data/stage5b_raw/sdd")
    parser.add_argument("--aerialmpt-root", default="data/aerialmpt/extracted")
    args = parser.parse_args()
    records = {
        "trajnet_full": probe(args.trajnet_full_root, ["*.txt"]),
        "eth_ucy_full": probe(Path(args.trajnet_full_root) / "biwi", ["*.txt"]),
        "sdd": probe(args.sdd_root, ["*.txt", "*.csv"]),
        "aerialmpt_long": probe(args.aerialmpt_root, ["*.csv", "*.txt", "*.json"]),
    }
    out = Path("outputs/reports/stage5b5_prepare_records.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    return 0


def probe(root, patterns):
    path = Path(root)
    files = []
    if path.exists():
        for pattern in patterns:
            files += [str(p) for p in path.rglob(pattern)]
    return {"path": str(path), "exists": path.exists(), "file_count": len(files), "sample_files": files[:8]}


if __name__ == "__main__":
    raise SystemExit(main())
