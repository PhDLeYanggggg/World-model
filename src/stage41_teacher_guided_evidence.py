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
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_teacher_guided_proposal as tgp
from src import stage41_teacher_guided_proposal_repair as repair


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_teacher_guided_evidence.json"
REPORT_MD = OUT_DIR / "stage41_teacher_guided_evidence.md"
BOOTSTRAP_N = 2000
SEED = 4197


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


def _feature_slices(dim: int) -> dict[str, tuple[int, int]]:
    static_dim = int(ft._fresh_ds("test")["static"].shape[1])
    pos = static_dim
    slices = {"history_static": (0, static_dim)}
    for name, width in [
        ("prediction_signals", 5),
        ("current_group", 2),
        ("floor_group", 2),
        ("neural_group", 2),
        ("neighbor_count", 1),
        ("domain", 3),
        ("horizon", 5),
        ("group_consistency", 10),
        ("residual_signals", 8),
    ]:
        slices[name] = (pos, min(dim, pos + width))
        pos += width
    if pos != dim:
        slices["unknown_tail"] = (min(pos, dim), dim)
    return slices


def _mask_data(data: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    out = dict(data)
    x = np.asarray(data["x_teacher"]).copy()
    slices = _feature_slices(x.shape[1])
    for name in names:
        start, end = slices.get(name, (0, 0))
        if end > start:
            x[:, start:end] = 0.0
    out["x_teacher"] = x.astype(np.float32)
    return out


def _selected_checkpoint_policy_guard() -> tuple[str, dict[str, Any], float]:
    proposal = read_json(OUT_DIR / "stage41_teacher_guided_proposal.json", {})
    repair_report = read_json(OUT_DIR / "stage41_teacher_guided_proposal_repair.json", {})
    selected = str(proposal.get("selected_trial"))
    trial = ((proposal.get("trial_reports") or {}).get(selected) or {})
    ckpt = ((trial.get("train") or {}).get("checkpoint"))
    policy = proposal.get("selected_policy") or {}
    guard = (((repair_report.get("validation_guard") or {}).get("selected") or {}).get("min_sep"))
    if not ckpt or not policy or guard is None:
        raise FileNotFoundError("Run teacher-guided proposal and repair before evidence.")
    return str(ckpt), dict(policy), float(guard)


def _eval_switch(data: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    return tgp._evaluate_switch(data, switch.astype(bool), name)


def _guarded_switch(data: Mapping[str, Any], switch: np.ndarray, min_sep: float) -> tuple[np.ndarray, int]:
    return jrc._apply_proximity_guard(
        data["floor_xy"],
        data["neural_xy"],
        data["labels"],
        data["keys"],
        switch.astype(bool),
        min_sep,
    )


def _metrics_from_switch(data: Mapping[str, Any], switch: np.ndarray) -> dict[str, Any]:
    selected = data["floor_ade"].astype(np.float64).copy()
    selected[switch.astype(bool)] = data["neural_ade"].astype(np.float64)[switch.astype(bool)]
    ds = {
        "horizon": data["horizon"],
        "hard": data["hard"],
        "failure": data["failure"],
        "easy": data["easy"],
        "domain": data["domain"],
        "candidate_fde": data["candidate_fde"],
    }
    return s41._metrics(selected, data["floor_ade"].astype(np.float64), ds, switch.astype(bool))


def _bootstrap_ci(data: Mapping[str, Any], switch: np.ndarray, slice_name: str, n: int = BOOTSTRAP_N) -> dict[str, float]:
    selected = data["floor_ade"].astype(np.float64).copy()
    selected[switch.astype(bool)] = data["neural_ade"].astype(np.float64)[switch.astype(bool)]
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    if slice_name == "t50":
        ids = np.where(horizon == 50)[0]
    elif slice_name == "t100":
        ids = np.where(horizon == 100)[0]
    elif slice_name == "hard_failure":
        ids = np.where(hard_failure)[0]
    elif slice_name.startswith("domain:"):
        domain = slice_name.split(":", 1)[1]
        ids = np.where(data["domain"].astype(str) == domain)[0]
    else:
        ids = np.arange(len(horizon))
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids))}
    rng = np.random.default_rng(SEED + abs(hash(slice_name)) % 10000)
    vals: list[float] = []
    floor = data["floor_ade"].astype(np.float64)
    for _ in range(n):
        boot = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[boot].mean()) / max(float(floor[boot].mean()), s41.EPS))
    return {"low": float(np.percentile(vals, 2.5)), "mid": float(np.percentile(vals, 50)), "high": float(np.percentile(vals, 97.5)), "n": int(len(ids))}


def _run_variant(name: str, checkpoint: str, policy: Mapping[str, Any], min_sep: float, data: Mapping[str, Any]) -> dict[str, Any]:
    pred = tgp._predict(checkpoint, data)
    raw_switch = tgp._policy_switch(pred, policy)
    guarded, guarded_off = _guarded_switch(data, raw_switch, min_sep)
    ev = _eval_switch(data, guarded, name)
    return {
        "name": name,
        "metrics": ev["selected_metrics"],
        "multi_agent_metrics": ev["multi_agent_metrics"],
        "collision_delta_vs_floor_005": ev["collision_delta_005"],
        "switch_rate": float(np.mean(guarded)),
        "guarded_off": int(guarded_off),
    }


def _evidence_passes(metrics: Mapping[str, Any], ci: Mapping[str, Any], collision_delta: float) -> bool:
    by_domain = metrics.get("by_domain") or {}
    positive_domains = sum(1 for row in by_domain.values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0)
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("t100_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and collision_delta <= repair.TEST_COLLISION_CEILING
        and positive_domains >= 2
        and (ci.get("all") or {}).get("low", 0.0) > 0
        and (ci.get("t50") or {}).get("low", 0.0) > 0
        and (ci.get("hard_failure") or {}).get("low", 0.0) > 0
    )


def run_teacher_guided_evidence() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, policy, min_sep = _selected_checkpoint_policy_guard()
    test = tgp._bundle("test")
    pred = tgp._predict(checkpoint, test)
    raw_switch = tgp._policy_switch(pred, policy)
    guarded_switch, guarded_off = _guarded_switch(test, raw_switch, min_sep)
    full_eval = _eval_switch(test, guarded_switch, "teacher_guided_repair_evidence_full")
    full_metrics = full_eval["selected_metrics"]
    bootstrap = {
        "all": _bootstrap_ci(test, guarded_switch, "all"),
        "t50": _bootstrap_ci(test, guarded_switch, "t50"),
        "t100_raw_frame_diagnostic": _bootstrap_ci(test, guarded_switch, "t100"),
        "hard_failure": _bootstrap_ci(test, guarded_switch, "hard_failure"),
    }
    for domain in sorted(set(test["domain"].astype(str).tolist())):
        bootstrap[f"domain:{domain}"] = _bootstrap_ci(test, guarded_switch, f"domain:{domain}")

    neural_all_switch = np.ones(len(test["floor_ade"]), dtype=bool)
    neural_without_fallback = _eval_switch(test, neural_all_switch, "neural_without_fallback_all_rows")
    raw_policy_eval = _eval_switch(test, raw_switch, "teacher_guided_raw_without_proximity_repair")
    teacher_reference_switch = test["teacher_prob"] >= 0.5
    teacher_reference_guarded, teacher_reference_guarded_off = _guarded_switch(test, teacher_reference_switch, min_sep)
    teacher_reference_eval = _eval_switch(test, teacher_reference_guarded, "teacher_reference_guarded")

    ablation_specs = {
        "no_history_static": ["history_static"],
        "no_prediction_signals": ["prediction_signals"],
        "no_neighbor_interaction": ["current_group", "floor_group", "neural_group", "neighbor_count", "group_consistency"],
        "no_scene_goal_proxy": ["domain", "horizon"],
        "no_group_consistency": ["group_consistency"],
        "no_horizon": ["horizon"],
        "no_domain": ["domain"],
    }
    ablations: dict[str, Any] = {}
    for name, masked in ablation_specs.items():
        ablations[name] = _run_variant(name, checkpoint, policy, min_sep, _mask_data(test, masked))
        ablations[name]["masked_feature_groups"] = masked
        ablations[name]["delta_vs_full"] = {
            "all_delta": float(ablations[name]["metrics"].get("all_improvement", 0.0) - full_metrics.get("all_improvement", 0.0)),
            "t50_delta": float(ablations[name]["metrics"].get("t50_improvement", 0.0) - full_metrics.get("t50_improvement", 0.0)),
            "t100_delta": float(ablations[name]["metrics"].get("t100_improvement", 0.0) - full_metrics.get("t100_improvement", 0.0)),
            "hard_delta": float(ablations[name]["metrics"].get("hard_failure_improvement", 0.0) - full_metrics.get("hard_failure_improvement", 0.0)),
        }

    basis = read_json(OUT_DIR / "stage41_group_consistency_multiseed_repair.json", {})
    group_summary = basis.get("metric_summary") or {}
    current_basis = {
        "all_mean": (group_summary.get("all_improvement") or {}).get("mean"),
        "t50_mean": (group_summary.get("t50_improvement") or {}).get("mean"),
        "t100_mean": (group_summary.get("t100_improvement") or {}).get("mean"),
        "hard_mean": (group_summary.get("hard_failure_improvement") or {}).get("mean"),
        "easy_max": (group_summary.get("easy_degradation") or {}).get("max"),
        "collision_delta_max": (group_summary.get("collision_delta_vs_floor_005") or {}).get("max"),
    }
    lift_over_current = {
        "all_delta": float(full_metrics.get("all_improvement", 0.0) - float(current_basis.get("all_mean") or 0.0)),
        "t50_delta": float(full_metrics.get("t50_improvement", 0.0) - float(current_basis.get("t50_mean") or 0.0)),
        "t100_delta": float(full_metrics.get("t100_improvement", 0.0) - float(current_basis.get("t100_mean") or 0.0)),
        "hard_delta": float(full_metrics.get("hard_failure_improvement", 0.0) - float(current_basis.get("hard_mean") or 0.0)),
    }
    result = {
        "source": "fresh_run",
        "protocol_status": "teacher_guided_proposal_bootstrap_and_ablation_evidence",
        "checkpoint": checkpoint,
        "policy": policy,
        "proximity_guard_min_sep": min_sep,
        "test_guarded_off": int(guarded_off),
        "test_metrics": full_metrics,
        "multi_agent_metrics": full_eval["multi_agent_metrics"],
        "collision_delta_vs_floor_005": full_eval["collision_delta_005"],
        "switch_rate": float(np.mean(guarded_switch)),
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap": bootstrap,
        "current_group_consistency_basis": current_basis,
        "lift_over_current_group_consistency_basis": lift_over_current,
        "neural_without_fallback_metrics": neural_without_fallback["selected_metrics"],
        "neural_without_fallback_collision_delta_vs_floor_005": neural_without_fallback["collision_delta_005"],
        "raw_policy_without_proximity_repair_metrics": raw_policy_eval["selected_metrics"],
        "raw_policy_without_proximity_repair_collision_delta_vs_floor_005": raw_policy_eval["collision_delta_005"],
        "teacher_reference_guarded_metrics": teacher_reference_eval["selected_metrics"],
        "teacher_reference_guarded_collision_delta_vs_floor_005": teacher_reference_eval["collision_delta_005"],
        "teacher_reference_guarded_off": int(teacher_reference_guarded_off),
        "feature_slices": _feature_slices(test["x_teacher"].shape[1]),
        "ablations": ablations,
        "evidence_pass": _evidence_passes(full_metrics, bootstrap, full_eval["collision_delta_005"]),
        "no_leakage": {
            "teacher_switch_inference_input": False,
            "teacher_switch_train_label_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "proximity_guard_selected_on_val": True,
            "bootstrap_after_freeze": True,
            "ablation_after_freeze": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "Evidence covers frozen teacher-guided repair with bootstrap and feature masking. It remains dataset-local raw-frame 2.5D and does not execute Stage5C/SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Teacher-Guided Repair Evidence",
        "",
        "- source: `fresh_run`",
        f"- evidence pass: `{result['evidence_pass']}`",
        f"- test metrics: `{full_metrics}`",
        f"- collision delta @0.05: `{full_eval['collision_delta_005']}`",
        f"- switch rate: `{float(np.mean(guarded_switch))}`",
        f"- bootstrap n: `{BOOTSTRAP_N}`",
        f"- bootstrap: `{bootstrap}`",
        f"- lift over current group-consistency basis: `{lift_over_current}`",
        f"- neural without fallback metrics: `{neural_without_fallback['selected_metrics']}`",
        f"- raw policy without proximity repair: `{raw_policy_eval['selected_metrics']}`",
        f"- raw policy collision delta @0.05: `{raw_policy_eval['collision_delta_005']}`",
        "",
        "## Ablations",
        "",
        "| ablation | all | t50 | t100 raw-frame | hard/failure | easy | collision_delta_005 | delta_vs_full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in ablations.items():
        m = row["metrics"]
        lines.append(
            f"| `{name}` | {m.get('all_improvement')} | {m.get('t50_improvement')} | {m.get('t100_improvement')} | {m.get('hard_failure_improvement')} | {m.get('easy_degradation')} | {row.get('collision_delta_vs_floor_005')} | `{row.get('delta_vs_full')}` |"
        )
    lines.extend(
        [
            "",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This evidence does not re-select thresholds on test. Bootstrap and ablations are computed after the validation-selected policy and proximity guard are frozen.",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_teacher_guided_evidence() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_teacher_guided_evidence()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_teacher_guided_evidence",
            status,
            started,
            [OUT_DIR / "stage41_teacher_guided_proposal_repair.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_teacher_guided_evidence()
