from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.evaluation.hardbench_builder import build_hardbench, markdown_table, write_outputs as write_hardbench_outputs


REPORT_DIR = Path("outputs/reports")
OUT_DIR = Path("data/baseline_failure_bench")


def threshold(dataset: str, horizon: int, unit: str = "metric") -> float:
    if dataset in {"trajnet", "eth_ucy"}:
        if horizon <= 10:
            return 1.0
        if horizon <= 50:
            return 2.5
        return 5.0
    if horizon >= 100:
        return 5.0
    if horizon >= 50:
        return 2.5
    return 1.0


def classify_failure(record: Dict) -> str:
    events = set(record.get("events", []))
    if "turning" in events or "route_change" in events:
        return "turn_failure"
    if "stop_go" in events:
        return "stop_go_failure"
    if "close_interaction" in events or "near_collision" in events or "high_density" in events:
        return "interaction_failure"
    if "crossing_paths" in events:
        return "crossing_failure"
    if "acceleration_change" in events or "deceleration_change" in events:
        return "acceleration_failure"
    if record.get("future_horizon", 0) >= 50:
        return "long_horizon_drift"
    return "unknown_failure"


def build_failure_bench() -> Dict:
    hardbench_path = Path("data/hardbench_v1/hardbench_v1_records.json")
    if hardbench_path.exists():
        records = json.loads(hardbench_path.read_text(encoding="utf-8"))
    else:
        payload = write_hardbench_outputs(build_hardbench())
        records = payload["records"]
    out_records = []
    for record in records:
        h = int(record.get("future_horizon", 0))
        th = threshold(record["dataset"], h)
        fde = float(record.get("baseline_FDE", 0.0))
        if fde > th:
            status = "baseline_failure"
        elif fde > 0.7 * th:
            status = "baseline_near_failure"
        else:
            status = "baseline_success"
        out = dict(record)
        out.update(
            {
                "failure_threshold": th,
                "baseline_status": status,
                "baseline_success": status == "baseline_success",
                "baseline_failure": status == "baseline_failure",
                "baseline_near_failure": status == "baseline_near_failure",
                "failure_type": classify_failure(record) if status != "baseline_success" else "none",
                "failure_severity": round(fde / max(th, 1e-6), 6),
            }
        )
        out_records.append(out)
    return summarize(out_records)


def summarize(records: List[Dict]) -> Dict:
    failures = [r for r in records if r["baseline_failure"]]
    by_dataset = defaultdict(list)
    by_horizon = Counter()
    by_event = Counter()
    for r in records:
        by_dataset[r["dataset"]].append(r)
        if r["baseline_failure"]:
            if r["future_horizon"] >= 100:
                by_horizon["t100"] += 1
            elif r["future_horizon"] >= 50:
                by_horizon["t50"] += 1
            elif r["future_horizon"] >= 25:
                by_horizon["t25"] += 1
            else:
                by_horizon["t10"] += 1
            for event in r.get("events", []):
                by_event[event] += 1
    dataset_rows = {}
    for dataset, rows in by_dataset.items():
        dataset_rows[dataset] = {
            "samples": len(rows),
            "failure_samples": sum(r["baseline_failure"] for r in rows),
            "near_failure_samples": sum(r["baseline_near_failure"] for r in rows),
            "failure_rate": round(sum(r["baseline_failure"] for r in rows) / max(len(rows), 1), 6),
            "enough_for_training": sum(r["baseline_failure"] for r in rows) >= 10,
            "enough_for_evaluation": sum(r["baseline_failure"] for r in rows) >= 5,
        }
    return {
        "total_samples": len(records),
        "failure_samples": len(failures),
        "near_failure_samples": sum(r["baseline_near_failure"] for r in records),
        "baseline_failure_rate_by_dataset": dataset_rows,
        "baseline_failure_rate_by_event_type": dict(by_event),
        "baseline_failure_rate_by_horizon": dict(by_horizon),
        "enough_samples_for_training": len(failures) >= 30,
        "enough_samples_for_evaluation": len(failures) >= 15,
        "records": records,
    }


def write_outputs(payload: Dict) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "baseline_failure_bench_records.json").write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "records"}
    (OUT_DIR / "baseline_failure_bench_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "baseline_failure_bench_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [
        {
            "dataset": dataset,
            **row,
        }
        for dataset, row in summary["baseline_failure_rate_by_dataset"].items()
    ]
    text = "\n".join(
        [
            "# BaselineFailureBench Summary",
            "",
            f"total_samples: `{summary['total_samples']}`",
            f"failure_samples: `{summary['failure_samples']}`",
            f"near_failure_samples: `{summary['near_failure_samples']}`",
            f"enough_samples_for_training: `{summary['enough_samples_for_training']}`",
            f"enough_samples_for_evaluation: `{summary['enough_samples_for_evaluation']}`",
            "",
            markdown_table(rows),
        ]
    )
    (REPORT_DIR / "baseline_failure_bench_summary.md").write_text(text, encoding="utf-8")
    return payload

