from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajnet-root", default="data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/stanford")
    parser.add_argument("--eth-ucy-root", default="data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/biwi")
    parser.add_argument("--tgsim-i90", default="https://data.transportation.gov/resource/9uas-hf8b.csv?$limit=50000")
    args = parser.parse_args()
    records = {
        "trajnet": {"path": args.trajnet_root, "exists": Path(args.trajnet_root).exists()},
        "eth_ucy": {"path": args.eth_ucy_root, "exists": Path(args.eth_ucy_root).exists()},
        "tgsim_i90": {"path": args.tgsim_i90, "exists": True, "kind": "public_url"},
    }
    out = Path("outputs/reports/stage5b_prepare_records.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    return 0 if all(row.get("exists", False) for row in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
