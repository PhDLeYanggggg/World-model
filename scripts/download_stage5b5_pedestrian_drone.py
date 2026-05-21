from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SOURCES = {
    "trajnet_full": {
        "kind": "git",
        "url": "https://github.com/vita-epfl/trajnetplusplusdataset.git",
        "target": "data/stage5b_raw/trajnetplusplusdataset",
        "license_note": "Public TrajNet++ repository; original bundled datasets may have separate licenses.",
    },
    "eth_ucy_full": {
        "kind": "bundled_or_manual",
        "target": "data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original",
        "license_note": "Uses ETH/UCY-style bundled original files when present; full official source should be prepared manually if needed.",
    },
    "sdd": {
        "kind": "license_placeholder",
        "target": "data/stage5b_raw/sdd",
        "license_note": "Stanford Drone Dataset is commonly distributed for non-commercial research; user must review and accept official terms before download.",
    },
    "opentraj": {
        "kind": "manual_placeholder",
        "target": "data/stage5b_raw/opentraj",
        "license_note": "OpenTraj aggregates datasets with mixed licenses; prepare only sources whose terms are accepted.",
    },
    "aerialmpt_long": {
        "kind": "local_probe",
        "target": "data/aerialmpt/extracted",
        "license_note": "Uses local AerialMPT if longer sequences exist; current bauma3 limitation remains separate.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-download", action="store_true")
    parser.add_argument("--max-gb", type=float, default=10.0)
    args = parser.parse_args()
    selected = list(SOURCES) if args.dataset == "all" else [args.dataset]
    records = []
    failures = []
    for name in selected:
        spec = SOURCES.get(name)
        if not spec:
            failures.append({"dataset": name, "reason": "unknown_dataset"})
            continue
        target = Path(spec["target"])
        record = {"dataset": name, **spec, "executed": False, "status": "planned"}
        if not args.execute_download or args.dry_run:
            record["exists"] = target.exists()
            records.append(record)
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if spec["kind"] == "git":
                if target.exists():
                    record["status"] = "already_exists"
                else:
                    subprocess.run(["git", "clone", "--depth", "1", spec["url"], str(target)], check=True)
                    record["status"] = "downloaded"
                record["executed"] = True
            else:
                record["status"] = "placeholder_requires_user_action"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            failures.append({"dataset": name, "reason": str(exc)})
        records.append(record)
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage5b5_download_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Stage 5B.5 Download Failures", ""]
    if failures:
        lines += ["| dataset | reason |", "| --- | --- |"]
        lines += [f"| {row['dataset']} | {row['reason']} |" for row in failures]
    else:
        lines.append("No hard failures for the requested operation. License-gated/manual sources remain placeholders until the user prepares them.")
    (out / "stage5b5_download_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": records, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
