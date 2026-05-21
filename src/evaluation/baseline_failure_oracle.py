from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def threshold_for(dataset: str, horizon: int, subset: str = "all") -> float:
    if dataset in {"trajnet", "eth_ucy"}:
        if horizon <= 10:
            return 1.0 if subset != "hard" else 1.5
        return 2.5
    if horizon >= 100:
        return 5.0 if subset != "hard" else 8.0
    if horizon >= 50:
        return 2.5
    return 1.0


def hard_lookup() -> Dict[tuple, Dict]:
    rows = load_json(REPORT_DIR / "stage5b5_hard_subset_summary.json", [])
    out = {}
    for dataset in rows:
        for ep in dataset.get("episodes", []):
            out[(dataset["dataset_name"], int(ep["episode_id"]))] = ep
    return out


def baseline_failure_rows() -> List[Dict]:
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    hard = hard_lookup()
    rows = []
    for dataset, brow in baselines.get("datasets", {}).items():
        baseline = brow["strongest_causal_baseline"]
        for split in ["train", "val", "test"]:
            for ep in load_dataset_episodes(dataset, split=split):
                states = ep["states"]
                meta = ep["meta"]
                episode_id = int(meta.get("episode_id", -1))
                past = int(meta.get("past_horizon", 10))
                future_len = states.shape[0] - past
                if future_len <= 0:
                    continue
                horizon = 100 if future_len >= 100 else future_len
                dt = float(meta.get("dt_s", 1.0))
                true = states[past : past + horizon]
                pred = rollout(states[:past], horizon, dt, baseline)[1:]
                fde = float(np.linalg.norm(pred[-1, :, 0:2] - true[-1, :, 0:2], axis=1).mean())
                hrow = hard.get((dataset, episode_id), {})
                subset = hrow.get("hardness", "easy")
                threshold = threshold_for(dataset, horizon, subset=subset)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "episode_id": episode_id,
                        "horizon": horizon,
                        "baseline": baseline,
                        "baseline_FDE": round(fde, 6),
                        "threshold": threshold,
                        "baseline_failure": bool(fde > threshold),
                        "hardness": subset,
                        "events": hrow.get("events", []),
                    }
                )
    return rows


def summarize_failures(rows: List[Dict], alpha_rows: List[Dict] | None = None) -> Dict:
    by_dataset = defaultdict(list)
    by_event = Counter()
    for row in rows:
        by_dataset[row["dataset"]].append(row)
        if row["baseline_failure"]:
            for event in row.get("events", []) or ["unknown"]:
                by_event[event] += 1
    dataset_summary = {}
    for dataset, ds_rows in by_dataset.items():
        dataset_summary[dataset] = {
            "episodes": len(ds_rows),
            "baseline_failure_rate": round(sum(r["baseline_failure"] for r in ds_rows) / max(len(ds_rows), 1), 6),
            "mean_baseline_FDE": round(float(np.mean([r["baseline_FDE"] for r in ds_rows])), 6) if ds_rows else 0.0,
        }
    alpha_corr = None
    easy_alpha = None
    hard_alpha = None
    if alpha_rows:
        failures = []
        alphas = []
        easy = []
        hard = []
        failure_lookup = {(r["dataset"], r["episode_id"], r["horizon"]): r["baseline_failure"] for r in rows}
        for ar in alpha_rows:
            key = (ar.get("dataset"), ar.get("episode_id"), ar.get("horizon"))
            if key not in failure_lookup:
                continue
            failures.append(float(failure_lookup[key]))
            alphas.append(float(ar.get("alpha", 0.0)))
            if ar.get("hardness") == "hard":
                hard.append(float(ar.get("alpha", 0.0)))
            elif ar.get("hardness") == "easy":
                easy.append(float(ar.get("alpha", 0.0)))
        if len(set(failures)) > 1 and len(alphas) > 1:
            alpha_corr = round(float(np.corrcoef(failures, alphas)[0, 1]), 6)
        easy_alpha = round(float(np.mean(easy)), 6) if easy else None
        hard_alpha = round(float(np.mean(hard)), 6) if hard else None
    return {
        "baseline_failure_rate": round(sum(r["baseline_failure"] for r in rows) / max(len(rows), 1), 6),
        "baseline_failure_by_dataset": dataset_summary,
        "baseline_failure_by_event_type": dict(by_event),
        "alpha_vs_baseline_failure_correlation": alpha_corr,
        "easy_alpha_mean": easy_alpha,
        "hard_alpha_mean": hard_alpha,
    }


def write_oracle_report(rows: List[Dict], alpha_rows: List[Dict] | None = None) -> Dict:
    summary = summarize_failures(rows, alpha_rows=alpha_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows[:2000]}
    (REPORT_DIR / "stage5b6_baseline_failure_oracle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Stage 5B.6 Baseline Failure Oracle",
        "",
        "The oracle labels are generated from ground truth for training diagnostics and reporting only. They are not inference inputs.",
        "",
        f"overall_baseline_failure_rate: `{summary['baseline_failure_rate']}`",
        f"alpha_vs_baseline_failure_correlation: `{summary['alpha_vs_baseline_failure_correlation']}`",
        f"easy_alpha_mean: `{summary['easy_alpha_mean']}`",
        f"hard_alpha_mean: `{summary['hard_alpha_mean']}`",
        "",
        "| dataset | episodes | failure_rate | mean_baseline_FDE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset, row in summary["baseline_failure_by_dataset"].items():
        lines.append(f"| {dataset} | {row['episodes']} | {row['baseline_failure_rate']} | {row['mean_baseline_FDE']} |")
    (REPORT_DIR / "stage5b6_baseline_failure_oracle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload

