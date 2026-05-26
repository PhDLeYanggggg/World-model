from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_full_trajectory_world_state as ft
from src import stage42_full_waypoint_dynamics as cm


OUT_DIR = Path("outputs/stage42_long_research")
FULL_TRAJ_JSON = Path("outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json")
CN_JSON = OUT_DIR / "bridge_shape_composer_stage42.json"

REPORT_JSON = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
REPORT_MD = OUT_DIR / "common_validation_bridge_shape_composer_stage42.md"
REPORT_CSV = OUT_DIR / "common_validation_bridge_shape_composer_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_co_gate.md"

PAPER_FILES = [
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CO 使用 common validation-aligned rows 选择 composer policy；test 只评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _safe_improvement(selected: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(ref[mask])), EPS)


def _metric(selected: np.ndarray, ref: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    domain = labels["domain"].astype(str)
    out = {
        "rows": int(len(selected)),
        "all_improvement": _safe_improvement(selected, ref, np.ones(len(selected), dtype=bool)),
        "t10_improvement": _safe_improvement(selected, ref, horizon == 10),
        "t25_improvement": _safe_improvement(selected, ref, horizon == 25),
        "t50_improvement": _safe_improvement(selected, ref, horizon == 50),
        "t100_raw_frame_diagnostic_improvement": _safe_improvement(selected, ref, horizon == 100),
        "hard_failure_improvement": _safe_improvement(selected, ref, hard_failure),
        "easy_degradation": -_safe_improvement(selected, ref, easy),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "harm_over_ref": float(np.mean(selected - ref)) if len(selected) else 0.0,
    }
    out["by_domain"] = {}
    for name in sorted(set(domain.tolist())):
        mask = domain == name
        out["by_domain"][name] = {
            "rows": int(np.sum(mask)),
            "all_improvement": _safe_improvement(selected, ref, mask),
            "t50_improvement": _safe_improvement(selected, ref, mask & (horizon == 50)),
            "t100_raw_frame_diagnostic_improvement": _safe_improvement(selected, ref, mask & (horizon == 100)),
            "hard_failure_improvement": _safe_improvement(selected, ref, mask & hard_failure),
            "easy_degradation": -_safe_improvement(selected, ref, mask & easy),
        }
    return out


def _score(metric: Mapping[str, Any]) -> float:
    return (
        float(metric.get("all_improvement", 0.0))
        + 1.4 * float(metric.get("t50_improvement", 0.0))
        + 0.8 * float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
        + 1.2 * float(metric.get("hard_failure_improvement", 0.0))
        - 40.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
    )


def _endpoint_bundle(split: str) -> dict[str, Any]:
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    data = blend._bundle(split, checkpoint, teacher_policy, min_sep)
    policy = cm._load_composite_policy()
    alpha = blend._alpha_vector(data, policy).astype(np.float64)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    selected_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, data["labels"])
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, data["labels"])
    return {
        "source": "fresh_recomputed_endpoint_linear_bridge_from_cached_verified_checkpoint",
        "labels": data["labels"],
        "floor_xy": floor_xy,
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": alpha > EPS,
    }


def _full_bundle(split: str) -> dict[str, Any]:
    result = read_json(FULL_TRAJ_JSON, {})
    paths = cm._full_trajectory_best_paths(result)
    if not paths:
        raise FileNotFoundError("Missing Stage41 full-trajectory checkpoints.")
    pred, labels = ft._predict_ensemble(paths, split)
    selected_ade, selected_fde, switch, floor_ade = ft._apply_policy(pred, labels, result["best_policy"])
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    selected_xy = floor_xy.copy()
    selected_xy[switch.astype(bool)] = neural_xy[switch.astype(bool)]
    floor_fde = ft._trajectory_errors(floor_xy, labels)[1]
    return {
        "source": "fresh_recomputed_full_waypoint_sequence_from_cached_verified_checkpoints",
        "labels": labels,
        "floor_xy": floor_xy,
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": switch.astype(bool),
        "checkpoint_paths": paths,
    }


def _alignment_report(endpoint: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    e = endpoint["labels"]
    f = full["labels"]
    checks = {
        "rows_match": len(e["horizon"]) == len(f["horizon"]),
        "horizon_match": np.array_equal(e["horizon"].astype(int), f["horizon"].astype(int)),
        "domain_match": np.array_equal(e["domain"].astype(str), f["domain"].astype(str)),
        "scene_match": np.array_equal(e["scene_id"].astype(str), f["scene_id"].astype(str)),
        "source_file_match": np.array_equal(e["source_file"].astype(str), f["source_file"].astype(str)),
        "current_xy_max_abs_diff": float(np.max(np.abs(e["current_xy"] - f["current_xy"]))) if len(e["horizon"]) else 0.0,
    }
    checks["aligned"] = bool(
        checks["rows_match"]
        and checks["horizon_match"]
        and checks["domain_match"]
        and checks["scene_match"]
        and checks["source_file_match"]
        and checks["current_xy_max_abs_diff"] <= 1e-8
    )
    return checks


def _slice_masks(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    out = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if np.any(mask):
                out[f"{d}|{h}"] = mask
    return out


def _compose(endpoint: Mapping[str, Any], full: Mapping[str, Any], choices: Mapping[str, bool]) -> dict[str, Any]:
    labels = endpoint["labels"]
    selected_xy = endpoint["selected_xy"].copy()
    use_full = np.zeros(len(labels["horizon"]), dtype=bool)
    for key, mask in _slice_masks(labels).items():
        if choices.get(key, False):
            selected_xy[mask] = full["selected_xy"][mask]
            use_full[mask] = True
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "use_full": use_full,
        "metric_vs_endpoint_ade": _metric(selected_ade, endpoint["selected_ade"], labels, use_full),
        "metric_vs_floor_ade": _metric(selected_ade, endpoint["floor_ade"], labels, use_full),
        "metric_vs_endpoint_fde": _metric(selected_fde, endpoint["selected_fde"], labels, use_full),
        "metric_vs_floor_fde": _metric(selected_fde, endpoint["floor_fde"], labels, use_full),
    }


def _fit_policy(endpoint_val: Mapping[str, Any], full_val: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = endpoint_val["labels"]
    candidates: list[dict[str, Any]] = []
    for margin in [0.0, 0.0025, 0.005, 0.01, 0.02]:
        for easy_max in [0.0, 0.01, 0.02]:
            choices: dict[str, bool] = {}
            slice_rows: dict[str, Any] = {}
            for key, mask in _slice_masks(labels).items():
                if int(np.sum(mask)) < 80:
                    choices[key] = False
                    slice_rows[key] = {"rows": int(np.sum(mask)), "use_full": False, "reason": "too_few_val_rows"}
                    continue
                full_gain = _safe_improvement(full_val["selected_ade"], endpoint_val["selected_ade"], mask)
                easy_mask = mask & labels["easy"].astype(bool)
                easy_harm = -_safe_improvement(full_val["selected_ade"], endpoint_val["selected_ade"], easy_mask) if np.any(easy_mask) else 0.0
                use_full = bool(full_gain > margin and easy_harm <= easy_max)
                choices[key] = use_full
                slice_rows[key] = {
                    "rows": int(np.sum(mask)),
                    "use_full": use_full,
                    "full_gain_vs_endpoint": float(full_gain),
                    "easy_harm_vs_endpoint": float(easy_harm),
                    "margin": margin,
                    "easy_max": easy_max,
                }
            ev = _compose(endpoint_val, full_val, choices)
            metric = ev["metric_vs_endpoint_ade"]
            eligible = bool(
                metric["all_improvement"] > 0.0
                and metric["easy_degradation"] <= 0.02
                and np.any(ev["use_full"])
            )
            candidates.append(
                {
                    "policy": {"type": "domain_horizon_full_waypoint_composer", "margin": margin, "easy_max": easy_max, "choices": choices},
                    "val_metric_vs_endpoint_ade": metric,
                    "val_metric_vs_floor_ade": ev["metric_vs_floor_ade"],
                    "val_switch_rate": float(np.mean(ev["use_full"])),
                    "slice_rows": slice_rows,
                    "eligible": eligible,
                    "score": _score(metric),
                }
            )
    fallback = {
        "policy": {"type": "keep_endpoint_linear_bridge_floor", "margin": None, "easy_max": None, "choices": {}},
        "val_metric_vs_endpoint_ade": _metric(endpoint_val["selected_ade"], endpoint_val["selected_ade"], labels, np.zeros(len(labels["horizon"]), dtype=bool)),
        "val_metric_vs_floor_ade": _metric(endpoint_val["selected_ade"], endpoint_val["floor_ade"], labels, endpoint_val["switch"]),
        "val_switch_rate": 0.0,
        "slice_rows": {},
        "eligible": True,
        "score": 0.0,
    }
    candidates.append(fallback)
    eligible_pool = [row for row in candidates if row["eligible"]]
    best = max(eligible_pool, key=lambda row: row["score"])
    return best["policy"], candidates


def _evaluate_policy(policy: Mapping[str, Any], endpoint: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    ev = _compose(endpoint, full, policy.get("choices", {}))
    return {
        "policy": policy,
        "metric_vs_endpoint_ade": ev["metric_vs_endpoint_ade"],
        "metric_vs_floor_ade": ev["metric_vs_floor_ade"],
        "metric_vs_endpoint_fde": ev["metric_vs_endpoint_fde"],
        "metric_vs_floor_fde": ev["metric_vs_floor_fde"],
        "use_full_rate": float(np.mean(ev["use_full"])),
        "use_full_rows": int(np.sum(ev["use_full"])),
    }


def _replace_section(path: Path, marker: str, lines: list[str]) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    block = "\n".join([start, *lines, end])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = prefix + "\n\n" + block + ("\n\n" + suffix if suffix else "\n")
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(new_text, encoding="utf-8")


def _paper_lines(test_eval: Mapping[str, Any], alignment: Mapping[str, Any]) -> list[str]:
    m = test_eval["metric_vs_endpoint_ade"]
    mf = test_eval["metric_vs_floor_ade"]
    return [
        "## Stage42-CO Common Validation Bridge / Shape Composer",
        "",
        "- source: `fresh_common_validation_eval_from_cached_verified_checkpoints`",
        "- common validation/test row alignment is verified for endpoint-linear bridge and full-waypoint sequence.",
        "- policy is selected on validation rows only; test is evaluated once.",
        f"- composer vs endpoint-linear bridge ADE: all `{_pct(m['all_improvement'])}`, t50 `{_pct(m['t50_improvement'])}`, t100 raw diagnostic `{_pct(m['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(m['hard_failure_improvement'])}`, easy `{_pct(m['easy_degradation'])}`.",
        f"- composer vs strongest causal floor ADE: all `{_pct(mf['all_improvement'])}`, t50 `{_pct(mf['t50_improvement'])}`, t100 raw diagnostic `{_pct(mf['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(mf['hard_failure_improvement'])}`.",
        f"- use_full_rate: `{_pct(test_eval['use_full_rate'])}`.",
        f"- alignment: `{alignment['aligned']}`.",
        "- claim boundary: no true 3D, no foundation, no metric/seconds-level, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(test_eval: Mapping[str, Any], alignment: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_lines(test_eval, alignment)
    out = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_CO_COMMON_VALIDATION_BRIDGE_SHAPE_COMPOSER", lines)
        text = path.read_text(encoding="utf-8")
        out.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_co": "Stage42-CO Common Validation Bridge / Shape Composer" in text,
                "contains_claim_boundary": "no metric/seconds-level" in text,
            }
        )
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    val = payload["alignment"]["val"]
    test = payload["alignment"]["test"]
    test_eval = payload["test_eval"]
    m = test_eval["metric_vs_endpoint_ade"]
    mf = test_eval["metric_vs_floor_ade"]
    gates = {
        "common_validation_rows_aligned": bool(val["aligned"]),
        "common_test_rows_aligned": bool(test["aligned"]),
        "policy_selected_on_validation": payload["policy_selection"]["selected_on"] == "validation_only",
        "test_evaluated_once": payload["policy_selection"]["test_evaluated_once"] is True,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "composer_safe_vs_endpoint": m["easy_degradation"] <= 0.02,
        "composer_nonnegative_vs_endpoint_all": m["all_improvement"] >= 0.0,
        "composer_keeps_or_improves_t50_vs_endpoint": m["t50_improvement"] >= 0.0,
        "composer_positive_vs_floor_all": mf["all_improvement"] > 0.0,
        "composer_positive_vs_floor_t50": mf["t50_improvement"] > 0.0,
        "paper_files_refreshed": all(row["contains_stage42_co"] for row in payload["paper_file_status"]),
        "metric_seconds_overclaim_blocked": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_co_common_validation_bridge_shape_composer_pass"
        if passed == total
        else "stage42_co_common_validation_bridge_shape_composer_partial_or_blocked"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(candidates: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy_type",
                "margin",
                "easy_max",
                "eligible",
                "score",
                "val_all_vs_endpoint",
                "val_t50_vs_endpoint",
                "val_t100_vs_endpoint",
                "val_hard_vs_endpoint",
                "val_easy_vs_endpoint",
                "val_switch_rate",
            ],
        )
        writer.writeheader()
        for row in candidates:
            policy = row["policy"]
            metric = row["val_metric_vs_endpoint_ade"]
            writer.writerow(
                {
                    "policy_type": policy["type"],
                    "margin": policy.get("margin"),
                    "easy_max": policy.get("easy_max"),
                    "eligible": row["eligible"],
                    "score": row["score"],
                    "val_all_vs_endpoint": metric["all_improvement"],
                    "val_t50_vs_endpoint": metric["t50_improvement"],
                    "val_t100_vs_endpoint": metric["t100_raw_frame_diagnostic_improvement"],
                    "val_hard_vs_endpoint": metric["hard_failure_improvement"],
                    "val_easy_vs_endpoint": metric["easy_degradation"],
                    "val_switch_rate": row["val_switch_rate"],
                }
            )


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_co_gate"]
    val = payload["val_eval"]["metric_vs_endpoint_ade"]
    test = payload["test_eval"]["metric_vs_endpoint_ade"]
    test_floor = payload["test_eval"]["metric_vs_floor_ade"]
    lines = [
        "# Stage42-CO Common Validation Bridge / Shape Composer",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Row Alignment",
        "",
        f"- validation aligned: `{payload['alignment']['val']['aligned']}`",
        f"- test aligned: `{payload['alignment']['test']['aligned']}`",
        f"- validation rows: `{payload['alignment']['val']['rows_match']}`",
        f"- test current_xy max diff: `{payload['alignment']['test']['current_xy_max_abs_diff']}`",
        "",
        "## Validation-Selected Policy",
        "",
        f"- policy type: `{payload['selected_policy']['type']}`",
        f"- margin: `{payload['selected_policy'].get('margin')}`",
        f"- easy_max: `{payload['selected_policy'].get('easy_max')}`",
        f"- candidate count: `{payload['policy_selection']['candidate_count']}`",
        f"- val use_full_rate: `{_pct(payload['val_eval']['use_full_rate'])}`",
        f"- val vs endpoint all/t50/t100/hard/easy: `{_pct(val['all_improvement'])}` / `{_pct(val['t50_improvement'])}` / `{_pct(val['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(val['hard_failure_improvement'])}` / `{_pct(val['easy_degradation'])}`",
        "",
        "## Test Once",
        "",
        f"- test use_full_rate: `{_pct(payload['test_eval']['use_full_rate'])}`",
        f"- test vs endpoint all/t50/t100/hard/easy: `{_pct(test['all_improvement'])}` / `{_pct(test['t50_improvement'])}` / `{_pct(test['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(test['hard_failure_improvement'])}` / `{_pct(test['easy_degradation'])}`",
        f"- test vs strongest floor all/t50/t100/hard/easy: `{_pct(test_floor['all_improvement'])}` / `{_pct(test_floor['t50_improvement'])}` / `{_pct(test_floor['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(test_floor['hard_failure_improvement'])}` / `{_pct(test_floor['easy_degradation'])}`",
        "",
        "## Interpretation",
        "",
        "- This audit resolves the Stage42-CN row-alignment blocker with fresh common validation/test evidence.",
        "- A switch is selected only on validation rows. If validation does not support full-waypoint replacement, the selected policy remains endpoint-linear bridge.",
        "- The result is still dataset-local/raw-frame 2.5D; no metric/seconds, Stage5C, or SMC claim is made.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CO Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{name}` | `{ok}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    endpoint_val = _endpoint_bundle("val")
    endpoint_test = _endpoint_bundle("test")
    full_val = _full_bundle("val")
    full_test = _full_bundle("test")
    alignment = {
        "val": _alignment_report(endpoint_val, full_val),
        "test": _alignment_report(endpoint_test, full_test),
    }
    if not alignment["val"]["aligned"] or not alignment["test"]["aligned"]:
        raise ValueError(f"Endpoint/full-waypoint rows are not aligned: {alignment}")
    policy, candidates = _fit_policy(endpoint_val, full_val)
    val_eval = _evaluate_policy(policy, endpoint_val, full_val)
    test_eval = _evaluate_policy(policy, endpoint_test, full_test)
    paper_status = _refresh_paper_files(test_eval, alignment["test"])
    payload: dict[str, Any] = {
        "source": "fresh_common_validation_eval_from_cached_verified_checkpoints",
        "stage": "Stage42-CO common validation bridge / shape composer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([FULL_TRAJ_JSON, CN_JSON]),
        "alignment": alignment,
        "selected_policy": policy,
        "policy_selection": {
            "selected_on": "validation_only",
            "test_evaluated_once": True,
            "candidate_count": len(candidates),
        },
        "val_eval": val_eval,
        "test_eval": test_eval,
        "candidate_summary": candidates,
        "paper_file_status": paper_status,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_co_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_csv(candidates)
    _write_md(payload)
    return payload


if __name__ == "__main__":
    result = run()
    gate = result["stage42_co_gate"]
    print(f"Stage42-CO common validation composer: {gate['verdict']} ({gate['passed']}/{gate['total']})")
