from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_unification.build_stage5_episodes import build_stage5_real_dataset
from src.evaluation.data_quality_audit import audit_world_state_table, write_data_quality_report
from src.evaluation.leakage_audit import leakage_audit_dataset, write_leakage_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 5 world-state episodes.")
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--tgsim-data", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = {}
    audits = []
    leakage = []
    if args.datasets in {"all_available", "tgsim"} and args.tgsim_data:
        built = build_stage5_real_dataset("tgsim", args.tgsim_data, quick=args.quick)
        results["tgsim"] = built
        import pandas as pd

        table = pd.read_csv("data/stage5_world_state/tgsim/world_state.csv")
        audits.append(audit_world_state_table("tgsim", table))
        leakage.append(leakage_audit_dataset("tgsim", built["metadata"], official_velocity_source="causal_fd"))
    else:
        results["note"] = "No real data path provided; registry-only dry run. Pass --tgsim-data <url-or-csv> to build TGSIM episodes."
    Path("outputs/reports/stage5_data").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/report_stage5_data_lake.md").write_text("# Stage 5 Data Lake\n\n```json\n" + json.dumps(results, indent=2) + "\n```\n", encoding="utf-8")
    if audits:
        write_data_quality_report(audits)
    if leakage:
        write_leakage_report(leakage)
    print(json.dumps(results, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
