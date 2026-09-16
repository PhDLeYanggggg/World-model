"""Write a fresh recording-identity audit without touching data or checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_recording_lineage import CACHE_DIR, SPLITS, audit_caches, markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/publication_readiness_2026_09")
    parser.add_argument("--cache-layout", choices=("stage43", "stage35"), default="stage43")
    args = parser.parse_args()
    if args.cache_layout == "stage43":
        paths = {s: ROOT / CACHE_DIR / f"stage43_full_waypoint_supervision_{s}.npz" for s in SPLITS}
        stem = "recording_lineage_audit"
    else:
        paths = {s: ROOT / "data/stage35_selective_transfer" / f"expanded_external_{s}.npz" for s in SPLITS}
        stem = "stage35_recording_lineage_audit"
    payload = audit_caches(paths, ROOT)
    payload["cache_layout"] = args.cache_layout
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / f"{stem}.md").write_text(markdown_report(payload))
    print(json.dumps({k: payload[k] for k in [
        "recording_boundary_pass", "legacy_teacher_boundary_pass",
        "training_allowed_with_unchanged_legacy_caches", "new_evaluation_rows_from_legacy_teacher_train",
    ]}, indent=2))


if __name__ == "__main__":
    main()
