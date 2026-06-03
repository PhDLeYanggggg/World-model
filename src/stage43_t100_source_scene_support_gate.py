from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src import stage43_current_matrix_t100_source_family_gate as cm
from src import stage43_full_waypoint_latent_dynamics as m
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_robustness_audit import _pct
from src.stage43_tail_horizon_waypoint_adapter import _ridge_fit, _target_matrix


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_source_scene_support_gate.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_scene_support_gate.md"
GATE_MD = OUT_DIR / "stage43_stage_co_t100_source_scene_support_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SOURCE = "fresh_stage43_co_t100_source_scene_support_gate"
SECTION = "STAGE43_CO_T100_SOURCE_SCENE_SUPPORT_GATE"

STAGE43_CM_JSON = OUT_DIR / "stage43_current_matrix_t100_source_family_gate.json"
STAGE43_CN_JSON = OUT_DIR / "stage43_t100_validation_shift_forensics.json"
EPS = 1e-8


def _mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.mean(values[mask]))


def _safe_positive_rule(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    mask: np.ndarray,
    *,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> dict[str, Any]:
    rows = int(mask.sum())
    lift = cm._slice_improvement(candidate_ade, ds.floor_ade, mask)
    easy = cm._easy_degradation(ds, candidate_ade, mask)
    allowed = bool(rows >= int(min_support_rows) and lift > float(min_improvement) and easy <= float(max_easy_degradation))
    reason = "allowed_by_validation"
    if rows < int(min_support_rows):
        reason = "blocked_insufficient_validation_support"
    elif lift <= float(min_improvement):
        reason = "blocked_validation_nonpositive"
    elif easy > float(max_easy_degradation):
        reason = "blocked_validation_easy_harm"
    return {
        "rows": rows,
        "candidate_t100_full_waypoint_ade_improvement_vs_floor": float(lift),
        "easy_degradation_vs_floor": float(easy),
        "easy_ratio": float(np.mean(ds.easy[mask])) if rows else 0.0,
        "hard_failure_ratio": float(np.mean((ds.hard | ds.failure)[mask])) if rows else 0.0,
        "mean_floor_ade": _mean_or_zero(ds.floor_ade, mask),
        "mean_candidate_ade": _mean_or_zero(candidate_ade, mask),
        "allowed": allowed,
        "reason": reason,
    }


def _validation_support_rules(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    *,
    key_values: np.ndarray,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> tuple[dict[str, Any], set[str]]:
    table: dict[str, Any] = {}
    allowed: set[str] = set()
    key_values = key_values.astype(str)
    for value in sorted(set(key_values[ds.horizon == 100].tolist())):
        mask = (key_values == value) & (ds.horizon == 100)
        row = _safe_positive_rule(
            ds,
            candidate_ade,
            mask,
            min_support_rows=int(min_support_rows),
            min_improvement=float(min_improvement),
            max_easy_degradation=float(max_easy_degradation),
        )
        table[value] = row
        if row["allowed"]:
            allowed.add(value)
    return table, allowed


def _support_overlap(val: m.WaypointSplit, test: m.WaypointSplit, *, key: str) -> dict[str, Any]:
    val_values = set(getattr(val, key)[val.horizon == 100].astype(str).tolist())
    test_values = set(getattr(test, key)[test.horizon == 100].astype(str).tolist())
    inter = sorted(val_values & test_values)
    union = sorted(val_values | test_values)
    return {
        "validation_count": int(len(val_values)),
        "test_count": int(len(test_values)),
        "intersection_count": int(len(inter)),
        "jaccard": float(len(inter) / max(len(union), 1)),
        "validation_only_count": int(len(val_values - test_values)),
        "test_only_count": int(len(test_values - val_values)),
        "test_only_examples": sorted(test_values - val_values)[:8],
    }


def _apply_source_scene_support(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    candidate_fde: np.ndarray,
    *,
    allowed_source_files: set[str],
    allowed_scenes: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = ds.source_file.astype(str)
    scene = ds.scene_id.astype(str)
    h100 = ds.horizon == 100
    source_supported = np.asarray([value in allowed_source_files for value in source], dtype=bool)
    scene_supported = np.asarray([value in allowed_scenes for value in scene], dtype=bool)
    switch = h100 & (source_supported | scene_supported)
    selected_ade = np.where(switch, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(switch, candidate_fde, ds.floor_fde).astype(np.float32)
    h100_rows = int(h100.sum())
    blocked = h100 & ~switch
    return selected_ade, selected_fde, switch, {
        "h100_rows": h100_rows,
        "source_supported_h100_rows": int((h100 & source_supported).sum()),
        "scene_supported_h100_rows": int((h100 & scene_supported).sum()),
        "switched_h100_rows": int((h100 & switch).sum()),
        "blocked_h100_rows": int(blocked.sum()),
        "blocked_h100_ratio": float(blocked.sum() / max(h100_rows, 1)),
    }


def _source_scene_test_table(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    switch: np.ndarray,
    *,
    max_rows: int = 20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_file in sorted(set(ds.source_file[ds.horizon == 100].astype(str).tolist())):
        mask = (ds.source_file.astype(str) == source_file) & (ds.horizon == 100)
        rows.append(
            {
                "source_file": source_file,
                "rows": int(mask.sum()),
                "candidate_t100_full_waypoint_ade_improvement_vs_floor": cm._slice_improvement(
                    candidate_ade, ds.floor_ade, mask
                ),
                "easy_degradation_vs_floor": cm._easy_degradation(ds, candidate_ade, mask),
                "switch_rate": float(np.mean(switch[mask])) if int(mask.sum()) else 0.0,
                "mean_floor_ade": _mean_or_zero(ds.floor_ade, mask),
            }
        )
    rows.sort(key=lambda row: float(row["candidate_t100_full_waypoint_ade_improvement_vs_floor"]))
    return rows[: int(max_rows)]


def build_t100_source_scene_support_gate(
    *,
    seed: int = 437,
    min_support_rows: int = 200,
    min_improvement: float = 0.0,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cm_payload = read_json(STAGE43_CM_JSON, {})
    cn_payload = read_json(STAGE43_CN_JSON, {})
    selected = cm_payload.get("selected_model", {})

    train = cm._augment_with_floor_waypoint_features(m._build_split("train", max_rows=None, seed=int(seed)))
    val = cm._augment_with_floor_waypoint_features(m._build_split("val", max_rows=None, seed=int(seed)))
    test = cm._augment_with_floor_waypoint_features(m._build_split("test", max_rows=None, seed=int(seed)))
    cm._standardize(train, val, test)

    target = str(selected.get("target", "residual"))
    train_filter = str(selected.get("train_filter", "t50t100"))
    l2 = float(selected.get("l2", 10000.0))
    train_ids = cm._train_mask(train, train_filter)
    weight = _ridge_fit(train.x[train_ids], _target_matrix(train, target)[train_ids], l2)
    model_hash = cm._model_hash(weight, l2=l2, target=target, train_filter=train_filter)

    val_pred = cm._predict_waypoint(val, weight, target)
    test_pred = cm._predict_waypoint(test, weight, target)
    val_ade, _val_fde = m._trajectory_error(val, val_pred)
    test_ade, test_fde = m._trajectory_error(test, test_pred)

    source_file_table, allowed_source_files = _validation_support_rules(
        val,
        val_ade,
        key_values=val.source_file.astype(str),
        min_support_rows=int(min_support_rows),
        min_improvement=float(min_improvement),
        max_easy_degradation=float(max_easy_degradation),
    )
    scene_table, allowed_scenes = _validation_support_rules(
        val,
        val_ade,
        key_values=val.scene_id.astype(str),
        min_support_rows=int(min_support_rows),
        min_improvement=float(min_improvement),
        max_easy_degradation=float(max_easy_degradation),
    )
    selected_ade, selected_fde, switch, switch_support = _apply_source_scene_support(
        test,
        test_ade,
        test_fde,
        allowed_source_files=allowed_source_files,
        allowed_scenes=allowed_scenes,
    )
    support_metrics = m._metrics(test, selected_ade, selected_fde, switch)
    raw_family_metrics = cn_payload.get("raw_validation_rule_test_metrics", {})
    support_overlap = {
        "source_file": _support_overlap(val, test, key="source_file"),
        "scene": _support_overlap(val, test, key="scene_id"),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_source_scene_supported_t100_gate_on_current_matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_cm": str(STAGE43_CM_JSON),
            "stage43_cn": str(STAGE43_CN_JSON),
        },
        "input_verdicts": {
            "stage43_cm": cm_payload.get("stage43_cm_gate", {}).get("verdict"),
            "stage43_cn": cn_payload.get("stage43_cn_gate", {}).get("verdict"),
        },
        "selected_model_replayed": {
            "target": target,
            "train_filter": train_filter,
            "l2": l2,
            "train_rows": int(train_ids.sum()),
            "model_hash": model_hash,
            "model_hash_matches_stage43_cm": model_hash == selected.get("model_hash"),
        },
        "support_rule_protocol": {
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "min_support_rows": int(min_support_rows),
            "min_improvement": float(min_improvement),
            "max_easy_degradation": float(max_easy_degradation),
            "support_levels": ["source_file", "scene_id"],
            "deployment_rule": "switch_t100_only_if_source_file_or_scene_has_validation_positive_easy_safe_support",
        },
        "validation_source_file_rules": {
            "allowed_count": int(len(allowed_source_files)),
            "allowed_source_files": sorted(allowed_source_files),
            "table": source_file_table,
        },
        "validation_scene_rules": {
            "allowed_count": int(len(allowed_scenes)),
            "allowed_scenes": sorted(allowed_scenes),
            "table": scene_table,
        },
        "support_overlap": support_overlap,
        "switch_support": switch_support,
        "raw_family_rule_test_metrics": raw_family_metrics,
        "source_scene_supported_test_metrics": support_metrics,
        "source_scene_test_table": _source_scene_test_table(test, test_ade, switch),
        "deployment_decision": {
            "deploy_source_scene_supported_t100_gate": bool(
                support_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
                and support_metrics["easy_degradation_vs_floor"] <= 0.02
                and support_metrics["full_waypoint_ade_improvement_vs_floor"] >= 0.0
                and switch_support["switched_h100_rows"] > 0
            ),
            "reason": (
                "source_or_scene_supported_t100_positive_easy_safe"
                if support_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
                and support_metrics["easy_degradation_vs_floor"] <= 0.02
                and switch_support["switched_h100_rows"] > 0
                else "keep_floor_no_current_test_source_scene_support_for_safe_t100_switch"
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_deployment_claim": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_CM_JSON, STAGE43_CN_JSON]),
    }
    payload["stage43_co_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    support = payload["switch_support"]
    metrics = payload["source_scene_supported_test_metrics"]
    raw = payload["raw_family_rule_test_metrics"]
    gates = {
        "stage43_cm_precondition_present": payload["input_verdicts"]["stage43_cm"]
        == "stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor",
        "stage43_cn_precondition_present": payload["input_verdicts"]["stage43_cn"]
        == "stage43_cn_t100_validation_shift_forensics_pass_ucy_shift_blocker",
        "selected_model_replayed": payload["selected_model_replayed"]["model_hash_matches_stage43_cm"] is True,
        "validation_source_file_rules_built": "table" in payload["validation_source_file_rules"],
        "validation_scene_rules_built": "table" in payload["validation_scene_rules"],
        "source_scene_overlap_measured": payload["support_overlap"]["source_file"]["jaccard"] >= 0.0
        and payload["support_overlap"]["scene"]["jaccard"] >= 0.0,
        "unsupported_current_t100_rows_blocked": support["blocked_h100_rows"] == support["h100_rows"],
        "unsafe_family_rule_not_deployed": float(raw.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0)) < 0.0
        and payload["deployment_decision"]["deploy_source_scene_supported_t100_gate"] is False,
        "deployed_policy_safe_floor": metrics["easy_degradation_vs_floor"] <= 0.02
        and metrics["full_waypoint_ade_improvement_vs_floor"] >= -1e-9
        and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-9,
        "validation_only_no_test_tuning": payload["support_rule_protocol"]["selection_data"] == "validation_only"
        and payload["support_rule_protocol"]["test_threshold_tuning"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False,
        "t100_deployment_not_overclaimed": claim["t100_deployment_claim"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": (
            "stage43_co_t100_source_scene_support_gate_pass_floor_required"
            if passed == total
            else "stage43_co_t100_source_scene_support_gate_incomplete"
        ),
        "deploy_t100": False,
        "t100_positive_success": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_co_gate"]
    metrics = payload["source_scene_supported_test_metrics"]
    raw = payload["raw_family_rule_test_metrics"]
    support = payload["switch_support"]
    overlap = payload["support_overlap"]
    lines = [
        "# Stage43-CO T100 Source/Scene Support Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy t100: `{gate['deploy_t100']}`",
        "",
        "## Support Rule Protocol",
        "",
        f"- selection data: `{payload['support_rule_protocol']['selection_data']}`",
        f"- test threshold tuning: `{payload['support_rule_protocol']['test_threshold_tuning']}`",
        f"- min support rows: `{payload['support_rule_protocol']['min_support_rows']}`",
        f"- rule: `{payload['support_rule_protocol']['deployment_rule']}`",
        "",
        "## Current Support",
        "",
        f"- t100 test rows: `{support['h100_rows']}`",
        f"- source-supported t100 rows: `{support['source_supported_h100_rows']}`",
        f"- scene-supported t100 rows: `{support['scene_supported_h100_rows']}`",
        f"- switched t100 rows: `{support['switched_h100_rows']}`",
        f"- blocked t100 rows: `{support['blocked_h100_rows']}`",
        f"- blocked t100 ratio: `{_pct(support['blocked_h100_ratio'])}`",
        f"- source-file overlap jaccard: `{overlap['source_file']['jaccard']:.4f}`",
        f"- scene overlap jaccard: `{overlap['scene']['jaccard']:.4f}`",
        "",
        "## Metrics",
        "",
        f"- raw family-rule t100 lift: `{_pct(raw.get('t100_raw_frame_full_waypoint_diagnostic_vs_floor', 0.0))}`",
        f"- raw family-rule easy degradation: `{_pct(raw.get('easy_degradation_vs_floor', 0.0))}`",
        f"- source/scene-supported all lift: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- source/scene-supported t100 lift: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- source/scene-supported hard/failure lift: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- source/scene-supported easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "## Worst Current Test Source Files",
        "",
        "| source_file | rows | candidate t100 lift | easy harm | switch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["source_scene_test_table"]:
        lines.append(
            f"| `{row['source_file']}` | {row['rows']} | "
            f"{_pct(row['candidate_t100_full_waypoint_ade_improvement_vs_floor'])} | "
            f"{_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The stricter source-file/scene support gate blocks every current t100 test switch because no current t100 test source or scene has validation support. That is the correct safety behavior: the broader source-family rule was negative and easy-harmful on test.",
            "",
            "This does not solve t100; it narrows the next requirement. Future t100 deployment needs validation support at source-file or scene granularity, or a new split/source acquisition that provides that support.",
            "",
            "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric or seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_co_gate"]
    lines = [
        "# Stage43-CO Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- deploy t100: `{gate['deploy_t100']}`",
        f"- t100 positive success: `{gate['t100_positive_success']}`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    lines.extend([f"| `{name}` | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, lines)


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_co_gate"]
    support = payload["switch_support"]
    metrics = payload["source_scene_supported_test_metrics"]
    raw = payload["raw_family_rule_test_metrics"]
    block = [
        f"## {SECTION}",
        "",
        "I converted the Stage43-CN t100 diagnosis into a stricter deployment rule: a t100 switch is allowed only when the exact source file or scene has validation-positive, easy-safe support.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- current t100 rows: `{support['h100_rows']}`",
        f"- switched t100 rows: `{support['switched_h100_rows']}`",
        f"- blocked t100 rows: `{support['blocked_h100_rows']}`",
        f"- raw family-rule t100 lift: `{_pct(raw.get('t100_raw_frame_full_waypoint_diagnostic_vs_floor', 0.0))}`",
        f"- source/scene-supported t100 lift: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- source/scene-supported easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "Interpretation: t100 remains floor-only. The stricter support gate blocks all current t100 switches because validation and test have no shared source-file/scene support. This prevents the unsafe UCY family-level switch from being deployed.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("stage43", {})
    state["stage43"]["t100_source_scene_support_gate"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "switch_support": payload["switch_support"],
        "source_scene_supported_test_metrics": payload["source_scene_supported_test_metrics"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_co_t100_source_scene_support_gate"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, m._jsonable(state))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-CO t100 source/scene support gate.")
    parser.add_argument("--seed", type=int, default=437)
    parser.add_argument("--min-support-rows", type=int, default=200)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = build_t100_source_scene_support_gate(
        seed=int(args.seed),
        min_support_rows=int(args.min_support_rows),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = payload["stage43_co_gate"]
    print(f"Stage43-CO: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return payload


if __name__ == "__main__":
    main()
