from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import urlretrieve


DATASETS = {
    "trajnet": {
        "kind": "git",
        "url": "https://github.com/vita-epfl/trajnetplusplusdataset.git",
        "target": "data/stage5b_raw/trajnetplusplusdataset",
        "notes": "Public GitHub repository with TrajNet++ original trajectory subsets.",
    },
    "eth_ucy": {
        "kind": "derived_from_trajnet",
        "target": "data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/biwi",
        "notes": "Stage 5B uses the BIWI/ETH-style subset bundled in the TrajNet++ repository when present.",
    },
    "tgsim_other": {
        "kind": "url",
        "url": "https://data.transportation.gov/resource/9uas-hf8b.csv?$limit=50000",
        "target": "data/stage5b_raw/tgsim_i90_sample.csv",
        "notes": "Public Socrata CSV sample used as an additional TGSIM corridor benchmark.",
    },
    "sdd": {
        "kind": "gated_placeholder",
        "target": "data/stage5b_raw/sdd",
        "notes": "Stanford Drone Dataset requires license-aware manual preparation; not downloaded by default.",
    },
    "opendd": {
        "kind": "gated_placeholder",
        "target": "data/stage5b_raw/opendd",
        "notes": "OpenDD access and license must be verified by the user before local conversion.",
    },
    "ngsim": {
        "kind": "manual_placeholder",
        "target": "data/stage5b_raw/ngsim",
        "notes": "NGSIM source files must be supplied by the user or an official allowed portal.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", help="Dataset key or all.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without downloading.")
    parser.add_argument("--execute-download", action="store_true", help="Actually download public allowed data.")
    parser.add_argument("--max-gb", type=float, default=5.0)
    args = parser.parse_args()

    selected = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failures = []
    records = []
    for name in selected:
        info = DATASETS.get(name)
        if not info:
            failures.append({"dataset": name, "reason": "unknown_dataset"})
            continue
        target = Path(info["target"])
        record = {"dataset": name, "target": str(target), "kind": info["kind"], "executed": False, "status": "planned", "notes": info["notes"]}
        should_execute = args.execute_download and not args.dry_run
        if not should_execute:
            records.append(record)
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if info["kind"] == "git":
                if target.exists():
                    record["status"] = "already_exists"
                else:
                    subprocess.run(["git", "clone", "--depth", "1", info["url"], str(target)], check=True)
                    record["status"] = "downloaded"
                record["executed"] = True
            elif info["kind"] == "url":
                urlretrieve(info["url"], target)
                record["status"] = "downloaded"
                record["executed"] = True
            elif info["kind"] == "derived_from_trajnet":
                record["status"] = "available" if target.exists() else "needs_trajnet_download"
            else:
                record["status"] = "placeholder_requires_user_action"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            failures.append({"dataset": name, "reason": str(exc)})
        records.append(record)

    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage5b_download_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Stage 5B Download Failures", ""]
    if failures:
        lines += ["| dataset | reason |", "| --- | --- |"]
        lines += [f"| {row['dataset']} | {row['reason']} |" for row in failures]
    else:
        lines.append("No download failures in the requested operation. Registry-only or gated placeholders remain placeholders.")
    (out / "stage5b_download_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": records, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
