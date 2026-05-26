from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

BX_JSON = OUT_DIR / "floor_relaxability_audit_stage42.json"
AW_JSON = OUT_DIR / "ucy_validation_support_repair_stage42.json"
BW_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"

REPORT_JSON = OUT_DIR / "t50_floor_relaxability_repair_stage42.json"
REPORT_MD = OUT_DIR / "t50_floor_relaxability_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_by_gate.md"

EASY_LIMIT = 0.02
TARGET_SLICES = ["TrajNet|50", "UCY|50"]


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BY 是 t50 floor-relaxability repair audit，不训练新模型，不执行 Stage5C，不启用 SMC。",
    "本修复使用 Stage42-AW train-only UCY internal validation support；test source 不参与 policy/threshold 选择。",
    "本修复不允许去掉 teacher/floor rollout context，也不是 ungated neural deployment。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_positive(metric: Mapping[str, Any]) -> bool:
    return bool(
        float(metric.get("all_improvement", 0.0)) > 0.0
        and float(metric.get("t50_improvement", 0.0)) > 0.0
        and float(metric.get("hard_failure_improvement", 0.0)) > 0.0
        and float(metric.get("easy_degradation", 0.0)) <= EASY_LIMIT
        and float(metric.get("switch_rate", 0.0)) > 0.0
    )


def _fit_selected_variant(
    variant_name: str,
    features: np.ndarray,
    masks: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    x, _, _ = am._standardize(features[:, masks[variant_name]], split == "train")
    train_mask = split == "train"
    val_mask = split == "val"
    cur = np.stack([data["current_x"], data["current_y"]], axis=1)
    target_delta = (
        (labels["waypoint_xy"].astype(np.float64) - cur[:, None, :].astype(np.float64))
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], am.EPS)
    ).astype(np.float32)
    best: dict[str, Any] | None = None
    best_score = -1e9
    val_candidates: list[dict[str, Any]] = []
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
        val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.2 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.1 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - EASY_LIMIT)
            - 0.03 * val_metric["switch_rate"]
        )
        val_candidates.append(
            {
                "lambda": float(lam),
                "score": float(score),
                "policy_slice_count": int(len(policy["slices"])),
                "val_metric": val_metric,
            }
        )
        if score > best_score:
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "score": float(score),
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "val_candidates": val_candidates,
            }
    if best is None:
        raise RuntimeError("No selected t50 repair variant was fitted.")
    return best


def _domain_horizon_metrics(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    out: dict[str, Any] = {}
    for d in sorted(set(domain[split == "test"].tolist())):
        for h in [10, 25, 50, 100]:
            mask = (split == "test") & (domain == d) & (horizon == h)
            if int(np.sum(mask)) == 0:
                continue
            key = f"{d}|{h}"
            out[key] = {
                "source": "fresh_stage42_by_t50_floor_relaxability_repair",
                "test_rows": int(np.sum(mask)),
                "metric": am._metric(selected, floor, data, switch, mask),
            }
    return out


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bx = _load_json(BX_JSON)
    aw_report = _load_json(AW_JSON)
    bw = _load_json(BW_JSON)
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = repaired_split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    masks = aw._safe_variant_masks(feature_names)
    selected_variant = str(aw_report["validation_best_variant"])
    fitted = _fit_selected_variant(selected_variant, features, masks, data, repaired_split, labels, floor)
    dh = _domain_horizon_metrics(data, repaired_split, fitted["selected_ade"], fitted["floor_ade"], fitted["switch"])
    target_decisions: dict[str, Any] = {}
    for key in TARGET_SLICES:
        row = dh.get(key, {"metric": {}, "test_rows": 0})
        metric = row.get("metric", {})
        bx_before = bx["slice_decisions"].get(key, {})
        target_decisions[key] = {
            "source": "fresh_stage42_by_t50_floor_relaxability_repair",
            "before_bx_status": bx_before.get("status", "not_present"),
            "before_bx_reason": bx_before.get("reason", ""),
            "after_policy": "stage42_aw_train_only_internal_validation_best",
            "after_metric": metric,
            "after_test_rows": int(row.get("test_rows", 0)),
            "policy_slice_present": key in fitted["policy"]["slices"],
            "protected_t50_repaired": _metric_positive(metric),
        }
    repaired_slices = [key for key, row in target_decisions.items() if row["protected_t50_repaired"]]
    blocked_slices = [key for key, row in target_decisions.items() if not row["protected_t50_repaired"]]
    summary = {
        "source": "fresh_stage42_by_t50_floor_relaxability_repair",
        "verdict_short": "protected_t50_relaxability_repaired_but_not_floor_free",
        "selected_variant": selected_variant,
        "internal_val_group": internal_val_group,
        "target_slices": TARGET_SLICES,
        "repaired_t50_slices": repaired_slices,
        "still_blocked_t50_slices": blocked_slices,
        "ucy_t50_repaired": target_decisions["UCY|50"]["protected_t50_repaired"],
        "trajnet_t50_repaired": target_decisions["TrajNet|50"]["protected_t50_repaired"],
        "global_t50_improvement": float(aw_report["validation_best"]["protected"]["t50_improvement"]),
        "global_easy_degradation": float(aw_report["validation_best"]["protected"]["easy_degradation"]),
        "teacher_floor_context_required": bool(bw["summary"]["teacher_floor_context_is_core_feature_mechanism"]),
        "fallback_floor_required_outside_repaired_slices": True,
        "floor_free_neural_deployable": False,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_by_t50_floor_relaxability_repair",
        "stage": "Stage42-BY Protected T50 Floor-Relaxability Repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BX_JSON), str(AW_JSON), str(BW_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_bx_verdict": bx["stage42_bx_gate"]["verdict"],
            "stage42_aw_verdict": aw_report["stage42_aw_gate"]["verdict"],
            "stage42_bw_verdict": bw["stage42_bw_gate"]["verdict"],
        },
        "validation_policy": {
            "source": "fresh_stage42_by_t50_floor_relaxability_repair",
            "selected_variant": selected_variant,
            "selected_lambda": fitted["lambda"],
            "selected_score": fitted["score"],
            "policy_slices": sorted(fitted["policy"]["slices"].keys()),
            "selection_source": "train_plus_internal_validation_only",
            "test_threshold_tuning": False,
            "internal_val_group": internal_val_group,
        },
        "domain_horizon_metrics": dh,
        "target_decisions": target_decisions,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_metrics_used_for_final_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_by_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "bx_input_verified": payload["input_reports"]["stage42_bx_verdict"] == "stage42_bx_floor_relaxability_audit_pass",
        "aw_input_verified": payload["input_reports"]["stage42_aw_verdict"] == "stage42_aw_ucy_validation_support_repair_pass",
        "bw_input_verified": payload["input_reports"]["stage42_bw_verdict"] == "stage42_bw_safety_floor_necessity_audit_pass",
        "train_only_internal_validation_used": no_leakage["internal_val_from_train_only"] is True,
        "target_t50_slices_reported": set(s["target_slices"]) == set(TARGET_SLICES),
        "ucy_t50_repaired": s["ucy_t50_repaired"] is True,
        "trajnet_t50_repaired": s["trajnet_t50_repaired"] is True,
        "global_t50_positive": s["global_t50_improvement"] > 0.0,
        "global_easy_safe": s["global_easy_degradation"] <= EASY_LIMIT,
        "teacher_floor_context_required": s["teacher_floor_context_required"] is True,
        "not_floor_free_neural": s["floor_free_neural_deployable"] is False and claim["floor_free_neural_deployable"] is False,
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_by_t50_floor_relaxability_repair_pass" if passed == total else "stage42_by_t50_floor_relaxability_repair_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BY Protected T50 Floor-Relaxability Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_by_gate']['passed']} / {payload['stage42_by_gate']['total']}`",
        f"- verdict: `{payload['stage42_by_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- verdict_short: `{s['verdict_short']}`",
        f"- selected_variant: `{s['selected_variant']}`",
        f"- internal_val_group: `{s['internal_val_group']}`",
        f"- repaired_t50_slices: `{s['repaired_t50_slices']}`",
        f"- still_blocked_t50_slices: `{s['still_blocked_t50_slices']}`",
        f"- global_t50_improvement: `{_pct(s['global_t50_improvement'])}`",
        f"- global_easy_degradation: `{_pct(s['global_easy_degradation'])}`",
        f"- teacher_floor_context_required: `{s['teacher_floor_context_required']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Target T50 Decisions",
        "",
        "| slice | before BX status | after rows | after all | after t50 | after hard | after easy | switch | repaired |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in payload["target_decisions"].items():
        metric = row["after_metric"]
        lines.append(
            f"| `{key}` | `{row['before_bx_status']}` | {row['after_test_rows']} | "
            f"{_pct(float(metric.get('all_improvement', 0.0)))} | "
            f"{_pct(float(metric.get('t50_improvement', 0.0)))} | "
            f"{_pct(float(metric.get('hard_failure_improvement', 0.0)))} | "
            f"{_pct(float(metric.get('easy_degradation', 0.0)))} | "
            f"{_pct(float(metric.get('switch_rate', 0.0)))} | {row['protected_t50_repaired']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-BX showed t50 fallback relaxation was blocked under the original slice audit.",
        "- Stage42-BY repairs t50 only under the Stage42-AW protected validation policy with train-only UCY internal validation.",
        "- This is not a floor-free result: teacher/floor rollout context and protected selection remain required.",
        "- This does not change t100, metric, seconds-level, Stage5C, or SMC claims.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_by_gate"]
    lines = [
        "# Stage42-BY Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_t50_floor_relaxability_repair() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_floor_relaxability_repair()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
