from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.evaluation.stage5b5_benchmark import load_model as load_stage5b5_model
from src.evaluation.stage5b5_benchmark import predict_residual as predict_stage5b5_residual
from src.training.train_stage5b5_temporal_numpy import feature_vector


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hard_lookup() -> Dict[tuple, Dict]:
    rows = load_json(REPORT_DIR / "stage5b5_hard_subset_summary.json", [])
    out = {}
    for dataset in rows:
        for ep in dataset.get("episodes", []):
            out[(dataset["dataset_name"], int(ep["episode_id"]))] = ep
    return out


def reliability_label(hard_count: int) -> str:
    if hard_count >= 50:
        return "official_hard_gate"
    if hard_count >= 30:
        return "weak_gate_only"
    return "diagnostic_only"


def bootstrap_ci(values: List[float], seed: int = 17, samples: int = 1000) -> Dict:
    if not values:
        return {"mean": 0.0, "ci_low": None, "ci_high": None, "n": 0}
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return {"mean": round(float(vals.mean()), 6), "ci_low": None, "ci_high": None, "n": int(len(vals))}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        idx = rng.integers(0, len(vals), size=len(vals))
        means.append(float(vals[idx].mean()))
    return {
        "mean": round(float(vals.mean()), 6),
        "ci_low": round(float(np.quantile(means, 0.025)), 6),
        "ci_high": round(float(np.quantile(means, 0.975)), 6),
        "n": int(len(vals)),
    }


def per_episode_stage5b5_improvements(dataset: str, subset: str = "hard") -> List[float]:
    baselines = load_json(REPORT_DIR / "stage5b_baseline_metrics.json", {"datasets": {}})
    if dataset not in baselines.get("datasets", {}):
        return []
    baseline_name = baselines["datasets"][dataset]["strongest_causal_baseline"]
    model = load_stage5b5_model()
    hard = hard_lookup()
    values = []
    for ep in load_dataset_episodes(dataset, split="test"):
        states = ep["states"]
        meta = ep["meta"]
        episode_id = int(meta.get("episode_id", -1))
        hard_row = hard.get((dataset, episode_id), {})
        hardness = hard_row.get("hardness", "easy")
        if subset != "all" and hardness != subset:
            continue
        past = int(meta.get("past_horizon", 10))
        future_len = states.shape[0] - past
        if future_len <= 0:
            continue
        horizon = 100 if future_len >= 100 else future_len
        dt = float(meta.get("dt_s", 1.0))
        base = rollout(states[:past], horizon, dt, baseline_name)[1:]
        true = states[past : past + horizon]
        x = feature_vector(dataset, states[:past], hard_row, horizon, future_len)
        residual, _ = predict_stage5b5_residual(model, dataset, horizon, x)
        pred = base.copy()
        for step in range(horizon):
            pred[step, :, 0:2] += residual * ((step + 1) / horizon)
        b_fde = float(np.linalg.norm(base[-1, :, 0:2] - true[-1, :, 0:2], axis=1).mean())
        l_fde = float(np.linalg.norm(pred[-1, :, 0:2] - true[-1, :, 0:2], axis=1).mean())
        values.append((b_fde - l_fde) / max(abs(b_fde), 0.1))
    return values


def audit_dataset(row: Dict) -> Dict:
    episodes = row.get("episodes", [])
    counts = Counter(ep.get("hardness", "easy") for ep in episodes)
    scores = np.asarray([float(ep.get("hard_score", 0.0)) for ep in episodes], dtype=float)
    extreme_cut = float(np.quantile(scores, 0.9)) if len(scores) else 1.0
    extreme = [ep for ep in episodes if float(ep.get("hard_score", 0.0)) >= extreme_cut and episodes]
    hard_eps = [ep for ep in episodes if ep.get("hardness") == "hard"]
    event_counts = Counter(event for ep in hard_eps for event in ep.get("events", []))
    horizon_counts = {
        "t10": sum(1 for ep in hard_eps if int(ep.get("future_horizon", 0)) >= 10),
        "t25": sum(1 for ep in hard_eps if int(ep.get("future_horizon", 0)) >= 25),
        "t50": sum(1 for ep in hard_eps if int(ep.get("future_horizon", 0)) >= 50),
        "t100": sum(1 for ep in hard_eps if int(ep.get("future_horizon", 0)) >= 100),
    }
    hard_improvements = per_episode_stage5b5_improvements(row["dataset_name"], subset="hard")
    ci = bootstrap_ci(hard_improvements)
    min_detectable = None
    if len(hard_improvements) >= 2:
        min_detectable = round(float(1.96 * np.std(hard_improvements, ddof=1) / np.sqrt(len(hard_improvements))), 6)
    reliability = reliability_label(len(hard_eps))
    meaningful = reliability != "diagnostic_only" and ci["ci_low"] is not None and ci["ci_low"] > 0.0
    return {
        "dataset_name": row["dataset_name"],
        "all_episode_count": int(row.get("total_episodes", len(episodes))),
        "easy_episode_count": int(counts["easy"]),
        "medium_episode_count": int(counts["medium"]),
        "hard_episode_count": int(counts["hard"]),
        "extreme_episode_count": len(extreme),
        "hard_count_by_event_type": dict(event_counts),
        "hard_count_by_target_horizon": horizon_counts,
        "hard_count_by_actual_verified_t100": sum(1 for ep in hard_eps if ep.get("can_evaluate_t100")),
        "hard_reliability_label": reliability,
        "hard_subset_is_gate_eligible": reliability == "official_hard_gate",
        "bootstrap_improvement_ci": ci,
        "minimum_detectable_improvement": min_detectable,
        "previous_stage5b5_hard_win_statistically_meaningful": bool(meaningful),
        "note": "hard subset <30 episodes is diagnostic only" if len(hard_eps) < 30 else "usable for weak/official hard gate depending on count",
    }


def run_audit() -> List[Dict]:
    hard_rows = load_json(REPORT_DIR / "stage5b5_hard_subset_summary.json", [])
    return [audit_dataset(row) for row in hard_rows]


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = [
        "dataset_name",
        "all_episode_count",
        "hard_episode_count",
        "extreme_episode_count",
        "hard_count_by_actual_verified_t100",
        "hard_reliability_label",
        "hard_subset_is_gate_eligible",
        "bootstrap_mean",
        "bootstrap_ci",
        "minimum_detectable_improvement",
        "previous_stage5b5_hard_win_statistically_meaningful",
    ]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        ci = row["bootstrap_improvement_ci"]
        flat = dict(row)
        flat["bootstrap_mean"] = ci["mean"]
        flat["bootstrap_ci"] = f"[{ci['ci_low']}, {ci['ci_high']}]"
        lines.append("| " + " | ".join(str(flat.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage5b6_hard_reliability_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    text = "\n".join(
        [
            "# Stage 5B.6 Hard Benchmark Reliability Audit",
            "",
            "Reliability rule: `<30` hard episodes is diagnostic only, `30-49` is weak gate only, and `>=50` is official hard gate eligible.",
            "",
            markdown_table(rows),
            "Conclusion: Stage 5B.5 hard wins with one or a few episodes are not statistically reliable gate wins.",
        ]
    )
    (REPORT_DIR / "stage5b6_hard_reliability_audit.md").write_text(text, encoding="utf-8")
    return rows
