from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_full_trajectory_world_state as ft
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_composite_tail_evidence.json"
REPORT_MD = OUT_DIR / "stage41_composite_tail_evidence.md"
BOOTSTRAP_N = 2000
SEED = 414103


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _ds(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    labels = data["labels"]
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _slice_mask(ds: Mapping[str, np.ndarray], slice_name: str) -> np.ndarray:
    horizon = ds["horizon"].astype(int)
    hard_failure = ds["hard"].astype(bool) | ds["failure"].astype(bool)
    if slice_name == "all":
        return np.ones(len(horizon), dtype=bool)
    if slice_name == "t50":
        return horizon == 50
    if slice_name == "t100":
        return horizon == 100
    if slice_name == "hard_failure":
        return hard_failure
    if slice_name.startswith("domain_t50:"):
        name = slice_name.split(":", 1)[1]
        return (ds["domain"].astype(str) == name) & (horizon == 50)
    if slice_name.startswith("domain:"):
        name = slice_name.split(":", 1)[1]
        return ds["domain"].astype(str) == name
    raise ValueError(f"unknown slice: {slice_name}")


def _bootstrap_improvement(
    selected: np.ndarray,
    floor: np.ndarray,
    mask: np.ndarray,
    *,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        boot = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[boot].mean()) / max(float(floor[boot].mean()), s41.EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _bootstrap_delta(
    selected: np.ndarray,
    reference: np.ndarray,
    floor: np.ndarray,
    mask: np.ndarray,
    *,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        boot = rng.choice(ids, size=len(ids), replace=True)
        base = max(float(floor[boot].mean()), s41.EPS)
        vals.append((1.0 - float(selected[boot].mean()) / base) - (1.0 - float(reference[boot].mean()) / base))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _selected_ade(data: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = blend._alpha_vector(data, policy)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    blended_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    selected_ade, _selected_fde = ft._trajectory_errors(blended_xy, data["labels"])
    return selected_ade.astype(np.float64), data["floor_ade"].astype(np.float64), alpha


def _teacher_repair_ade(data: Mapping[str, Any]) -> np.ndarray:
    selected = data["floor_ade"].astype(np.float64).copy()
    switch = data["teacher_repaired_switch"].astype(bool)
    selected[switch] = data["neural_ade"].astype(np.float64)[switch]
    return selected


def _bootstrap_report(selected: np.ndarray, floor: np.ndarray, ds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    slices = ["all", "t50", "t100", "hard_failure"]
    out = {
        name: _bootstrap_improvement(selected, floor, _slice_mask(ds, name), seed=SEED + i)
        for i, name in enumerate(slices)
    }
    domains = sorted(set(ds["domain"].astype(str).tolist()))
    out["by_domain"] = {
        name: _bootstrap_improvement(selected, floor, _slice_mask(ds, f"domain:{name}"), seed=SEED + 10 + i)
        for i, name in enumerate(domains)
    }
    out["by_domain_t50"] = {
        name: _bootstrap_improvement(selected, floor, _slice_mask(ds, f"domain_t50:{name}"), seed=SEED + 20 + i)
        for i, name in enumerate(domains)
    }
    return out


def _delta_report(selected: np.ndarray, reference: np.ndarray, floor: np.ndarray, ds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    slices = ["all", "t50", "t100", "hard_failure"]
    out = {
        name: _bootstrap_delta(selected, reference, floor, _slice_mask(ds, name), seed=SEED + 100 + i)
        for i, name in enumerate(slices)
    }
    domains = sorted(set(ds["domain"].astype(str).tolist()))
    out["by_domain"] = {
        name: _bootstrap_delta(selected, reference, floor, _slice_mask(ds, f"domain:{name}"), seed=SEED + 110 + i)
        for i, name in enumerate(domains)
    }
    out["by_domain_t50"] = {
        name: _bootstrap_delta(selected, reference, floor, _slice_mask(ds, f"domain_t50:{name}"), seed=SEED + 120 + i)
        for i, name in enumerate(domains)
    }
    return out


def _evidence_passes(metrics: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> bool:
    domains = metrics.get("by_domain") or {}
    positive_domains = sum(1 for row in domains.values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0)
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("t100_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("collision_delta_vs_floor_005", 1.0) <= blend.TEST_COLLISION_CEILING
        and positive_domains >= 2
        and (bootstrap.get("all") or {}).get("low", 0.0) > 0
        and (bootstrap.get("t50") or {}).get("low", 0.0) > 0
        and (bootstrap.get("t100") or {}).get("low", 0.0) > 0
        and (bootstrap.get("hard_failure") or {}).get("low", 0.0) > 0
    )


def _strict_teacher_delta_pass(delta: Mapping[str, Any]) -> bool:
    return bool(
        (delta.get("all") or {}).get("low", 0.0) > 0
        and (delta.get("t50") or {}).get("low", 0.0) > 0
        and (delta.get("hard_failure") or {}).get("low", 0.0) > 0
    )


def run_composite_tail_evidence() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    report = read_json(blend.REPORT_JSON, {})
    policy = report.get("safe_switch_test_policy") or {}
    if not policy:
        raise FileNotFoundError("Run bounded neural blend dynamics before composite-tail evidence.")
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    test = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    selected, floor, alpha = _selected_ade(test, policy)
    ds = _ds(test)
    metrics = blend._evaluate_blend(test, policy)["metrics"]
    teacher_selected = _teacher_repair_ade(test)
    teacher_metrics = s41._metrics(teacher_selected, floor, ds, test["teacher_repaired_switch"].astype(bool))
    bootstrap = _bootstrap_report(selected, floor, ds)
    delta_vs_teacher = _delta_report(selected, teacher_selected, floor, ds)
    no_tail_policy = {"type": "global", "alpha": policy.get("switch_alpha", 1.0), "gate": "teacher_repaired_switch"}
    no_tail = blend._evaluate_blend(test, no_tail_policy)["metrics"]
    result = {
        "source": "fresh_run",
        "protocol": "composite_tail_bounded_neural_blend_bootstrap_evidence",
        "policy": policy,
        "checkpoint": checkpoint,
        "test_metrics": metrics,
        "teacher_repair_metrics_recomputed": teacher_metrics,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap": bootstrap,
        "delta_vs_teacher_repair_bootstrap": delta_vs_teacher,
        "strict_delta_vs_teacher_repair_pass": _strict_teacher_delta_pass(delta_vs_teacher),
        "evidence_pass": _evidence_passes(metrics, bootstrap),
        "alpha_stats": {
            "mean": float(np.mean(alpha)),
            "positive_rate": float(np.mean(alpha > blend.EPS)),
            "full_switch_alpha_rate": float(np.mean(alpha >= float(policy.get("switch_alpha", 1.0)) - blend.EPS)),
            "tail_alpha_rate": float(np.mean((alpha > blend.EPS) & (alpha < float(policy.get("switch_alpha", 1.0)) - blend.EPS))),
        },
        "ablation_no_tail_metrics": no_tail,
        "ablation_no_tail_delta": {
            "all_delta": float(metrics.get("all_improvement", 0.0) - no_tail.get("all_improvement", 0.0)),
            "t50_delta": float(metrics.get("t50_improvement", 0.0) - no_tail.get("t50_improvement", 0.0)),
            "t100_delta": float(metrics.get("t100_improvement", 0.0) - no_tail.get("t100_improvement", 0.0)),
            "hard_delta": float(metrics.get("hard_failure_improvement", 0.0) - no_tail.get("hard_failure_improvement", 0.0)),
            "easy_delta": float(metrics.get("easy_degradation", 0.0) - no_tail.get("easy_degradation", 0.0)),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "bootstrap_after_freeze": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "Bootstrap evidence is for the frozen composite-tail policy after validation selection. It remains dataset-local raw-frame 2.5D, not metric, true 3D, foundation, Stage5C, or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Composite-Tail Bounded Neural Evidence",
        "",
        "- source: `fresh_run`",
        f"- evidence pass: `{result['evidence_pass']}`",
        f"- strict delta vs teacher repair pass: `{result['strict_delta_vs_teacher_repair_pass']}`",
        f"- policy: `{policy}`",
        f"- test metrics: `{metrics}`",
        f"- bootstrap n: `{BOOTSTRAP_N}`",
        f"- bootstrap: `{bootstrap}`",
        f"- delta vs teacher repair bootstrap: `{delta_vs_teacher}`",
        f"- alpha stats: `{result['alpha_stats']}`",
        f"- ablation no-tail metrics: `{no_tail}`",
        f"- ablation no-tail delta: `{result['ablation_no_tail_delta']}`",
        f"- no leakage: `{result['no_leakage']}`",
        "",
        "This does not reselect thresholds on test. It bootstraps the frozen composite-tail bounded neural dynamics policy and compares it with the frozen teacher-guided repair.",
    ]
    write_md(REPORT_MD, lines)
    return result


def main_composite_tail_evidence() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_composite_tail_evidence()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_composite_tail_evidence",
            status,
            started,
            [blend.REPORT_JSON, OUT_DIR / "stage41_teacher_guided_proposal_repair.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_composite_tail_evidence()
