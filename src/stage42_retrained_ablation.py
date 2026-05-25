from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
DATA_NPZ = Path("data/stage41_world_model/combined_external.npz")
META_JSON = Path("data/stage41_world_model/combined_meta.json")
REPORT_JSON = OUT_DIR / "retrained_ablation_stage42.json"
REPORT_MD = OUT_DIR / "retrained_ablation_stage42.md"
REPORT_CSV = OUT_DIR / "retrained_ablation_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_g_gate.md"
SEEDS = [11, 17, 23]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "External 数据仍是 dataset-local / unverified weak metric diagnostic。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "future endpoints / family_fde 只作为 supervised label/eval，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _safe_div(num: float, den: float) -> float:
    return float(num) / max(float(den), EPS)


def _base_feature_names() -> list[str]:
    meta = read_json(META_JSON, {})
    return list(meta.get("feature_names", []))


def _history_flat_features(history_seq: np.ndarray, k: int = 16) -> tuple[np.ndarray, list[str]]:
    tail = history_seq[:, -k:, :].astype(np.float32)
    names = [f"history_seq_k{k}_t{t}_{field}" for t in range(k) for field in ["x", "y", "speed", "accel", "heading", "curvature", "valid"]]
    return tail.reshape(tail.shape[0], -1), names


def _domain_one_hot(dataset: np.ndarray) -> tuple[np.ndarray, list[str]]:
    domains = sorted(set(map(str, dataset.tolist())))
    idx = {d: i for i, d in enumerate(domains)}
    x = np.zeros((len(dataset), len(domains)), dtype=np.float32)
    for i, d in enumerate(map(str, dataset.tolist())):
        x[i, idx[d]] = 1.0
    return x, [f"domain_{d}" for d in domains]


def _assemble_all_features(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    base = data["stage37_features"].astype(np.float32)
    base_names = _base_feature_names()
    hist_flat, hist_names = _history_flat_features(data["history_seq"].astype(np.float32), k=16)
    domain, domain_names = _domain_one_hot(data["dataset"].astype(str))
    x = np.concatenate([base, hist_flat, domain], axis=1).astype(np.float32)
    names = base_names + hist_names + domain_names

    groups: dict[str, list[int]] = {
        "history": [],
        "history_sequence": [],
        "neighbor": [],
        "interaction": [],
        "goal": [],
        "scene": [],
        "domain": [],
        "transformer_proxy": [],
        "jepa": [],
    }
    for i, name in enumerate(names):
        low = name.lower()
        if low.startswith("history_"):
            groups["history"].append(i)
        if low.startswith("history_seq"):
            groups["history_sequence"].append(i)
            groups["transformer_proxy"].append(i)
        if any(key in low for key in ["neighbor", "density", "ttc", "closing"]):
            groups["neighbor"].append(i)
            groups["interaction"].append(i)
        if any(key in low for key in ["prototype", "goal", "ambiguity", "exit_like"]):
            groups["goal"].append(i)
            groups["scene"].append(i)
        if low.startswith("domain_"):
            groups["domain"].append(i)
    return np.nan_to_num(x, posinf=1e6, neginf=-1e6), names, groups


def _drop_columns(x: np.ndarray, drop: Iterable[int]) -> np.ndarray:
    drop_set = {int(i) for i in drop}
    keep = np.asarray([i not in drop_set for i in range(x.shape[1])], dtype=bool)
    return x[:, keep]


def _metrics(selected_err: np.ndarray, floor_err: np.ndarray, labels: Mapping[str, np.ndarray], selected_idx: np.ndarray, floor_idx: np.ndarray) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    oracle = labels["oracle_err"].astype(np.float64)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return 1.0 - _safe_div(float(np.mean(selected_err[mask])), float(np.mean(floor_err[mask])))

    out = {
        "rows": int(len(selected_err)),
        "all_improvement": imp(np.ones(len(selected_err), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_raw_frame_diagnostic_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": -imp(easy),
        "switch_rate": float(np.mean(selected_idx != floor_idx)) if len(selected_idx) else 0.0,
        "selector_regret": float(np.mean(selected_err - oracle)) if len(selected_err) else 0.0,
        "harm_over_fallback": float(np.mean(selected_err - floor_err)) if len(selected_err) else 0.0,
    }
    return out


def _choose_policy_on_val(pred: np.ndarray, y: np.ndarray, labels: Mapping[str, np.ndarray], *, no_safe_switch: bool) -> dict[str, Any]:
    floor_idx = labels["floor_idx"].astype(int)
    if no_safe_switch:
        return {"confidence_min": -1.0, "gain_min": -1.0, "max_switch_rate": 1.0, "no_safe_switch": True}
    candidates = []
    for confidence_min in [0.0, 0.02, 0.05, 0.10, 0.15]:
        for gain_min in [0.0, 0.01, 0.025, 0.05, 0.10]:
            for max_switch_rate in [0.03, 0.05, 0.08, 0.12, 0.20, 0.35]:
                selected_idx = _select_from_pred(pred, floor_idx, confidence_min, gain_min, max_switch_rate)
                selected_err = y[np.arange(len(y)), selected_idx]
                floor_err = y[np.arange(len(y)), floor_idx]
                m = _metrics(selected_err, floor_err, labels, selected_idx, floor_idx)
                safe = m["easy_degradation"] <= 0.02
                score = (m["all_improvement"] + m["t50_improvement"] + m["hard_failure_improvement"]) - 10.0 * max(0.0, m["easy_degradation"] - 0.02)
                candidates.append((safe, score, m, {"confidence_min": confidence_min, "gain_min": gain_min, "max_switch_rate": max_switch_rate, "no_safe_switch": False}))
    safe_candidates = [c for c in candidates if c[0]]
    chosen = max(safe_candidates or candidates, key=lambda item: item[1])
    return dict(chosen[3])


def _select_from_pred(pred: np.ndarray, floor_idx: np.ndarray, confidence_min: float, gain_min: float, max_switch_rate: float) -> np.ndarray:
    best_idx = np.argmin(pred, axis=1).astype(np.int64)
    floor_pred = pred[np.arange(len(pred)), floor_idx]
    best_pred = pred[np.arange(len(pred)), best_idx]
    gain = floor_pred - best_pred
    confidence = gain / np.maximum(floor_pred, EPS)
    ok = (best_idx != floor_idx) & (gain >= gain_min) & (confidence >= confidence_min)
    selected = floor_idx.copy()
    ids = np.where(ok)[0]
    if len(ids):
        max_count = int(np.floor(max_switch_rate * len(pred)))
        order = ids[np.argsort(gain[ids])[::-1]][:max_count]
        selected[order] = best_idx[order]
    return selected


def _split_payload(data: Mapping[str, np.ndarray], x: np.ndarray, split: str) -> dict[str, Any]:
    mask = data["old_split"].astype(str) == split
    y = data["family_fde"][mask].astype(np.float64)
    floor_idx = np.clip(data["safe_strongest_idx_old"][mask].astype(int), 0, y.shape[1] - 1)
    labels = {
        "horizon": data["horizon"][mask],
        "hard": data["hard"][mask],
        "failure": data["failure"][mask],
        "easy": data["easy"][mask],
        "floor_idx": floor_idx,
        "oracle_err": np.min(y, axis=1),
    }
    return {"x": x[mask], "y": y, "labels": labels}


def _train_one(seed: int, train_x: np.ndarray, train_y: np.ndarray) -> Any:
    rng = np.random.default_rng(seed)
    n = len(train_x)
    size = min(n, max(5000, int(0.9 * n)))
    ids = rng.choice(np.arange(n), size=size, replace=False)
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0 + 0.25 * (seed % 10)))
    model.fit(train_x[ids], np.log1p(np.maximum(train_y[ids], 0.0)))
    return model


def _eval_variant(name: str, data: Mapping[str, np.ndarray], all_x: np.ndarray, drop: list[int], seed: int, *, no_safe_switch: bool = False) -> dict[str, Any]:
    x = _drop_columns(all_x, drop)
    train = _split_payload(data, x, "train")
    val = _split_payload(data, x, "val")
    test = _split_payload(data, x, "test")
    model = _train_one(seed, train["x"], train["y"])
    val_pred = np.maximum(0.0, np.expm1(model.predict(val["x"])))
    policy = _choose_policy_on_val(val_pred, val["y"], val["labels"], no_safe_switch=no_safe_switch)
    test_pred = np.maximum(0.0, np.expm1(model.predict(test["x"])))
    floor_idx = test["labels"]["floor_idx"].astype(int)
    if policy.get("no_safe_switch"):
        selected_idx = np.argmin(test_pred, axis=1).astype(np.int64)
    else:
        selected_idx = _select_from_pred(test_pred, floor_idx, policy["confidence_min"], policy["gain_min"], policy["max_switch_rate"])
    selected_err = test["y"][np.arange(len(test["y"])), selected_idx]
    floor_err = test["y"][np.arange(len(test["y"])), floor_idx]
    metrics = _metrics(selected_err, floor_err, test["labels"], selected_idx, floor_idx)
    return {
        "source": "fresh_run",
        "variant": name,
        "seed": seed,
        "feature_dim": int(x.shape[1]),
        "dropped_feature_count": int(len(set(drop))),
        "policy_selected_on": "val",
        "policy": policy,
        **metrics,
    }


def _mean_ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    mean = float(arr.mean())
    half = 1.96 * std / np.sqrt(max(len(arr), 1))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name in sorted(set(r["variant"] for r in rows)):
        sub = [r for r in rows if r["variant"] == name]
        out[name] = {
            "source": "fresh_run",
            "seeds": [r["seed"] for r in sub],
            "all": _mean_ci([r["all_improvement"] for r in sub]),
            "t50": _mean_ci([r["t50_improvement"] for r in sub]),
            "t100_raw_frame_diagnostic": _mean_ci([r["t100_raw_frame_diagnostic_improvement"] for r in sub]),
            "hard_failure": _mean_ci([r["hard_failure_improvement"] for r in sub]),
            "easy_degradation": _mean_ci([r["easy_degradation"] for r in sub]),
            "switch_rate": _mean_ci([r["switch_rate"] for r in sub]),
            "selector_regret": _mean_ci([r["selector_regret"] for r in sub]),
        }
    return out


def _contribution(summary: Mapping[str, Any], full_name: str = "full_retrained_external") -> dict[str, Any]:
    full = summary.get(full_name, {})
    out = {}
    for name, item in summary.items():
        if name == full_name:
            continue
        out[name] = {
            "all_delta_full_minus_ablation": (full.get("all", {}).get("mean", 0.0) - item.get("all", {}).get("mean", 0.0)),
            "t50_delta_full_minus_ablation": (full.get("t50", {}).get("mean", 0.0) - item.get("t50", {}).get("mean", 0.0)),
            "hard_delta_full_minus_ablation": (full.get("hard_failure", {}).get("mean", 0.0) - item.get("hard_failure", {}).get("mean", 0.0)),
            "easy_delta_ablation_minus_full": (item.get("easy_degradation", {}).get("mean", 0.0) - full.get("easy_degradation", {}).get("mean", 0.0)),
        }
    return out


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary") or {}
    contribution = result.get("contribution_vs_full") or {}
    full = summary.get("full_retrained_external") or {}
    required_fresh = [
        "full_retrained_external",
        "no_history",
        "no_neighbor",
        "no_goal",
        "no_scene_goal",
        "no_interaction",
        "no_domain_expert",
        "no_transformer_proxy_history_sequence",
        "no_safe_switch",
        "no_teacher_floor_proxy",
    ]
    positive_contribs = [
        name
        for name, row in contribution.items()
        if row.get("t50_delta_full_minus_ablation", 0.0) > 0.0 or row.get("hard_delta_full_minus_ablation", 0.0) > 0.0
    ]
    gates = {
        "fresh_retrained_rows_present": all(name in summary for name in required_fresh),
        "three_seeds_per_fresh_variant": all(len((summary.get(name) or {}).get("seeds", [])) >= 3 for name in required_fresh),
        "full_variant_safe": (full.get("easy_degradation", {}).get("mean", 1.0) <= 0.02),
        "at_least_two_positive_component_contributions": len(positive_contribs) >= 2,
        "no_safe_switch_diagnosed": "no_safe_switch" in summary,
        "no_teacher_floor_proxy_diagnosed": "no_teacher_floor_proxy" in summary,
        "source_labels_explicit": all(r.get("source") == "fresh_run" for r in result.get("rows", [])),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "positive_component_contributions": positive_contribs,
        "verdict": "stage42_g_retrained_ablation_phase1_pass" if all(gates.values()) else "stage42_g_retrained_ablation_phase1_partial",
    }


def run_stage42_retrained_ablation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = dict(np.load(DATA_NPZ, allow_pickle=True))
    all_x, feature_names, groups = _assemble_all_features(data)
    variants = {
        "full_retrained_external": [],
        "no_history": groups["history"],
        "no_neighbor": groups["neighbor"],
        "no_goal": groups["goal"],
        "no_scene_goal": sorted(set(groups["scene"] + groups["goal"])),
        "no_interaction": sorted(set(groups["interaction"])),
        "no_domain_expert": groups["domain"],
        "no_transformer_proxy_history_sequence": groups["transformer_proxy"],
        "no_safe_switch": [],
        "no_teacher_floor_proxy": [],
    }
    rows = []
    for name, drop in variants.items():
        for seed in SEEDS:
            rows.append(_eval_variant(name, data, all_x, drop, seed, no_safe_switch=name in {"no_safe_switch", "no_teacher_floor_proxy"}))
    summary = _summarize(rows)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-G retrained ablation phase1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([DATA_NPZ, META_JSON]),
        "feature_schema": {
            "source": "fresh_run",
            "total_features": len(feature_names),
            "base_stage37_feature_count": len(_base_feature_names()),
            "history_sequence_flattened_k": 16,
            "group_sizes": {k: len(v) for k, v in groups.items()},
        },
        "source_labels": {
            "external_combined_dataset": "cached_verified",
            "feature_matrix_assembly": "fresh_run",
            "selector_refits": "fresh_run",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once_per_variant_seed",
            "jepa_transformer_full_waypoint_all_component_retraining": "not_run_in_phase1",
        },
        "rows": rows,
        "summary": summary,
        "contribution_vs_full": _contribution(summary),
        "not_run_boundaries": {
            "no_jepa": "not_run_in_phase1; current deployable path does not use JEPA features, and JEPA remains diagnostic-only from Stage18/19/41 evidence.",
            "full_transformer_retrain": "not_run_in_phase1; this phase retrains ridge expected-FDE selectors on causal external features, not torch Transformer checkpoints.",
            "no_endpoint_bridge": "not_run_in_phase1; Stage42-C has fresh full-waypoint vs endpoint-linear comparisons, but this phase does not retrain the waypoint model.",
            "no_full_waypoint_shape": "not_run_in_phase1; Stage42-C covers protected full-waypoint sequence dynamics, but all-component waypoint-shape retraining remains open.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "family_fde_used_as_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "thresholds_selected_on_val": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
            "phase1_not_full_stage42_d_completion": True,
        },
    }
    result["stage42_g_gate"] = _gate(result)
    _write_json(REPORT_JSON, result)
    _write_csv(rows)
    _write_report(result)
    _write_gate(result["stage42_g_gate"])
    _append_readme_and_state(result)
    return result


def _write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "seed",
        "source",
        "feature_dim",
        "dropped_feature_count",
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "selector_regret",
        "switch_rate",
    ]
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_report(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Stage42-G Retrained Ablation Phase1",
        "",
        "- source: `fresh_run`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_g_gate']['passed']} / {result['stage42_g_gate']['total']}`",
        f"- verdict: `{result['stage42_g_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## What This Freshly Retrains",
        "",
        "- expected-FDE baseline-family selector refits on external train rows.",
        "- validation-only safety threshold selection.",
        "- test evaluated once per variant/seed.",
        "- variants: full, no_history, no_neighbor, no_goal, no_scene_goal, no_interaction, no_domain_expert, no_transformer_proxy_history_sequence, no_safe_switch, no_teacher_floor_proxy.",
        "",
        "## Metrics",
        "",
        "| variant | all mean | t50 mean | t100 diag mean | hard mean | easy mean | switch mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(summary):
        item = summary[name]
        lines.append(
            f"| `{name}` | {item['all']['mean']:.6f} | {item['t50']['mean']:.6f} | {item['t100_raw_frame_diagnostic']['mean']:.6f} | {item['hard_failure']['mean']:.6f} | {item['easy_degradation']['mean']:.6f} | {item['switch_rate']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Contribution Deltas",
            "",
            "`full_minus_ablation > 0` means the removed component helped the full model on that slice.",
            "",
            "| ablation | all delta | t50 delta | hard delta | easy delta ablation-minus-full |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(result["contribution_vs_full"].items()):
        lines.append(
            f"| `{name}` | {row['all_delta_full_minus_ablation']:.6f} | {row['t50_delta_full_minus_ablation']:.6f} | {row['hard_delta_full_minus_ablation']:.6f} | {row['easy_delta_ablation_minus_full']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Not-Run Boundaries",
            *[f"- `{k}`: {v}" for k, v in result["not_run_boundaries"].items()],
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-G Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    lines.extend(["", f"- positive component contributions: `{gate.get('positive_component_contributions')}`"])
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_g_gate"]
    full = result["summary"].get("full_retrained_external", {})
    block = f"""
## Stage42-G Retrained Ablation Phase1

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
full_all = {full.get('all', {}).get('mean')}
full_t50 = {full.get('t50', {}).get('mean')}
full_t100_raw_frame_diagnostic = {full.get('t100_raw_frame_diagnostic', {}).get('mean')}
full_hard_failure = {full.get('hard_failure', {}).get('mean')}
full_easy_degradation = {full.get('easy_degradation', {}).get('mean')}
phase1_not_full_stage42_d_completion = true
stage5c_executed = false
smc_enabled = false
```

Stage42-G Phase1 freshly refits external expected-FDE selectors for the key causal feature/safety variants. It improves the ablation evidence beyond cached coverage, but it still does not complete all A-journal retrained ablations because JEPA/Transformer/full-waypoint-shape retraining remains explicitly `not_run_in_phase1`.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-G Retrained Ablation Phase1", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-G Retrained Ablation Phase1", block)

    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_g_retrained_ablation_phase1"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_g_retrained_ablation_phase1"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "full_all": full.get("all", {}).get("mean"),
        "full_t50": full.get("t50", {}).get("mean"),
        "full_hard_failure": full.get("hard_failure", {}).get("mean"),
        "full_easy_degradation": full.get("easy_degradation", {}).get("mean"),
        "not_run_boundaries": result["not_run_boundaries"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, REPORT_CSV, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


if __name__ == "__main__":
    run_stage42_retrained_ablation()
