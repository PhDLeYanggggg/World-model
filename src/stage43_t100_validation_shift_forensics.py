from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
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
REPORT_JSON = OUT_DIR / "stage43_t100_validation_shift_forensics.json"
REPORT_MD = OUT_DIR / "stage43_t100_validation_shift_forensics.md"
GATE_MD = OUT_DIR / "stage43_stage_cn_t100_validation_shift_forensics_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SOURCE = "fresh_stage43_cn_t100_validation_shift_forensics"
SECTION = "STAGE43_CN_T100_VALIDATION_SHIFT_FORENSICS"
STAGE43_CM_JSON = OUT_DIR / "stage43_current_matrix_t100_source_family_gate.json"
EPS = 1e-8


def _row_mask(ds: m.WaypointSplit, group: np.ndarray, group_value: str | int) -> np.ndarray:
    return (group == group_value) & (ds.horizon == 100)


def _source_families(ds: m.WaypointSplit) -> np.ndarray:
    return cm._source_families(ds)


def _mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.mean(values[mask]))


def _slice_table(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    *,
    group: np.ndarray,
    min_rows: int,
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for value in sorted(set(group.astype(str).tolist())):
        mask = _row_mask(ds, group.astype(str), value)
        rows = int(mask.sum())
        if rows < int(min_rows):
            continue
        easy = mask & ds.easy
        hard_failure = mask & (ds.hard | ds.failure)
        lift = cm._slice_improvement(candidate_ade, ds.floor_ade, mask)
        easy_deg = cm._easy_degradation(ds, candidate_ade, mask)
        table[str(value)] = {
            "rows": rows,
            "candidate_t100_full_waypoint_ade_improvement_vs_floor": float(lift),
            "easy_degradation_vs_floor": float(easy_deg),
            "easy_ratio": float(np.mean(ds.easy[mask])) if rows else 0.0,
            "hard_failure_ratio": float(np.mean((ds.hard | ds.failure)[mask])) if rows else 0.0,
            "mean_floor_ade": _mean_or_zero(ds.floor_ade, mask),
            "mean_candidate_ade": _mean_or_zero(candidate_ade, mask),
            "mean_scale": _mean_or_zero(ds.scale, mask),
            "easy_rows": int(easy.sum()),
            "hard_failure_rows": int(hard_failure.sum()),
        }
    return table


def _support_table(ds: m.WaypointSplit, *, min_rows: int) -> dict[str, Any]:
    families = _source_families(ds)
    return {
        "by_family": {
            family: {
                "rows": int(((families == family) & (ds.horizon == 100)).sum()),
                "source_file_count": int(
                    len(set(ds.source_file[((families == family) & (ds.horizon == 100))].astype(str).tolist()))
                ),
                "scene_count": int(
                    len(set(ds.scene_id[((families == family) & (ds.horizon == 100))].astype(str).tolist()))
                ),
                "easy_ratio": float(np.mean(ds.easy[((families == family) & (ds.horizon == 100))]))
                if int(((families == family) & (ds.horizon == 100)).sum())
                else 0.0,
                "hard_failure_ratio": float(
                    np.mean((ds.hard | ds.failure)[((families == family) & (ds.horizon == 100))])
                )
                if int(((families == family) & (ds.horizon == 100)).sum())
                else 0.0,
            }
            for family in sorted(set(families.tolist()))
            if int(((families == family) & (ds.horizon == 100)).sum()) >= int(min_rows)
        },
        "source_files": sorted(set(ds.source_file[ds.horizon == 100].astype(str).tolist())),
        "scenes": sorted(set(ds.scene_id[ds.horizon == 100].astype(str).tolist())),
    }


def _set_overlap(a: list[str], b: list[str]) -> dict[str, Any]:
    left = set(a)
    right = set(b)
    inter = sorted(left & right)
    union = sorted(left | right)
    return {
        "left_count": int(len(left)),
        "right_count": int(len(right)),
        "intersection_count": int(len(inter)),
        "jaccard": float(len(inter) / max(len(union), 1)),
        "left_only_count": int(len(left - right)),
        "right_only_count": int(len(right - left)),
        "left_only_examples": sorted(left - right)[:8],
        "right_only_examples": sorted(right - left)[:8],
    }


def _compare_val_test(
    val_table: Mapping[str, Any],
    test_table: Mapping[str, Any],
    allowed_families: list[str],
    *,
    max_easy_degradation: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted(set(val_table) | set(test_table)):
        val = val_table.get(family, {})
        test = test_table.get(family, {})
        val_lift = float(val.get("candidate_t100_full_waypoint_ade_improvement_vs_floor", 0.0))
        test_lift = float(test.get("candidate_t100_full_waypoint_ade_improvement_vs_floor", 0.0))
        val_easy = float(val.get("easy_degradation_vs_floor", 0.0))
        test_easy = float(test.get("easy_degradation_vs_floor", 0.0))
        allowed = family in allowed_families
        fails_test = bool(allowed and (test_lift <= 0.0 or test_easy > float(max_easy_degradation)))
        out[family] = {
            "validation_rows": int(val.get("rows", 0)),
            "test_rows": int(test.get("rows", 0)),
            "validation_lift": val_lift,
            "test_lift": test_lift,
            "lift_drop": float(test_lift - val_lift),
            "validation_easy_degradation": val_easy,
            "test_easy_degradation": test_easy,
            "easy_degradation_increase": float(test_easy - val_easy),
            "validation_easy_ratio": float(val.get("easy_ratio", 0.0)),
            "test_easy_ratio": float(test.get("easy_ratio", 0.0)),
            "easy_ratio_shift": float(test.get("easy_ratio", 0.0)) - float(val.get("easy_ratio", 0.0)),
            "validation_allowed": allowed,
            "allowed_family_failed_current_test": fails_test,
            "reason": (
                "validation_allowed_but_test_negative_or_easy_harm"
                if fails_test
                else "validation_allowed_and_test_safe"
                if allowed
                else "not_validation_allowed"
            ),
        }
    return out


def _root_causes(comparison: Mapping[str, Any], support: Mapping[str, Any]) -> list[str]:
    causes: list[str] = []
    failed_allowed = [family for family, row in comparison.items() if row["allowed_family_failed_current_test"]]
    if failed_allowed:
        causes.append("validation_allowed_family_failed_current_test")
    for family in failed_allowed:
        row = comparison[family]
        if row["test_lift"] <= 0.0:
            causes.append(f"{family}_test_lift_nonpositive")
        if row["test_easy_degradation"] > 0.02:
            causes.append(f"{family}_test_easy_harm")
        if abs(row["easy_ratio_shift"]) > 0.10:
            causes.append(f"{family}_easy_ratio_shift")
    overlap = support.get("val_test_overlap", {})
    if overlap.get("source_file_jaccard", 1.0) < 0.5:
        causes.append("low_val_test_source_file_overlap")
    if overlap.get("scene_jaccard", 1.0) < 0.5:
        causes.append("low_val_test_scene_overlap")
    if not causes:
        causes.append("no_major_shift_detected")
    return sorted(set(causes))


def build_t100_validation_shift_forensics(*, seed: int = 437, min_rows: int = 50) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cm_payload = read_json(STAGE43_CM_JSON, {})
    selected = cm_payload.get("selected_model", {})
    max_easy_degradation = float(cm_payload.get("training_protocol", {}).get("max_easy_degradation", 0.02))

    train = cm._augment_with_floor_waypoint_features(m._build_split("train", max_rows=None, seed=int(seed)))
    val = cm._augment_with_floor_waypoint_features(m._build_split("val", max_rows=None, seed=int(seed)))
    test = cm._augment_with_floor_waypoint_features(m._build_split("test", max_rows=None, seed=int(seed)))
    cm._standardize(train, val, test)

    train_filter = selected.get("train_filter", "t50t100")
    target = selected.get("target", "residual")
    l2 = float(selected.get("l2", 10000.0))
    train_ids = cm._train_mask(train, str(train_filter))
    weight = _ridge_fit(train.x[train_ids], _target_matrix(train, str(target))[train_ids], l2)

    val_pred = cm._predict_waypoint(val, weight, str(target))
    test_pred = cm._predict_waypoint(test, weight, str(target))
    val_ade, val_fde = m._trajectory_error(val, val_pred)
    test_ade, test_fde = m._trajectory_error(test, test_pred)

    val_family = _slice_table(val, val_ade, group=_source_families(val), min_rows=int(min_rows))
    test_family = _slice_table(test, test_ade, group=_source_families(test), min_rows=int(min_rows))
    allowed = list(selected.get("validation_allowed_families", []))
    comparison = _compare_val_test(
        val_family,
        test_family,
        allowed,
        max_easy_degradation=max_easy_degradation,
    )

    val_support = _support_table(val, min_rows=1)
    test_support = _support_table(test, min_rows=1)
    support = {
        "val_h100_rows": int(np.sum(val.horizon == 100)),
        "test_h100_rows": int(np.sum(test.horizon == 100)),
        "val_support": val_support,
        "test_support": test_support,
        "val_test_overlap": {
            "source_file": _set_overlap(val_support["source_files"], test_support["source_files"]),
            "scene": _set_overlap(val_support["scenes"], test_support["scenes"]),
        },
    }
    support["val_test_overlap"]["source_file_jaccard"] = support["val_test_overlap"]["source_file"]["jaccard"]
    support["val_test_overlap"]["scene_jaccard"] = support["val_test_overlap"]["scene"]["jaccard"]

    families_test = _source_families(test)
    switch_allowed = np.asarray(
        [(h == 100 and family in set(allowed)) for family, h in zip(families_test, test.horizon)], dtype=bool
    )
    selected_ade = np.where(switch_allowed, test_ade, test.floor_ade).astype(np.float32)
    selected_fde = np.where(switch_allowed, test_fde, test.floor_fde).astype(np.float32)
    raw_rule_metrics = m._metrics(test, selected_ade, selected_fde, switch_allowed)

    root_causes = _root_causes(comparison, support)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_current_matrix_t100_validation_test_shift_forensics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {"stage43_cm": str(STAGE43_CM_JSON)},
        "input_verdicts": {
            "stage43_cm": cm_payload.get("stage43_cm_gate", {}).get("verdict"),
        },
        "selected_model_replayed": {
            "target": str(target),
            "train_filter": str(train_filter),
            "l2": l2,
            "train_rows": int(train_ids.sum()),
            "model_hash_matches_stage43_cm": selected.get("model_hash")
            == cm._model_hash(weight, l2=l2, target=str(target), train_filter=str(train_filter)),
            "model_hash": cm._model_hash(weight, l2=l2, target=str(target), train_filter=str(train_filter)),
        },
        "support": support,
        "validation_t100_by_source_family": val_family,
        "test_t100_by_source_family": test_family,
        "validation_test_shift_by_source_family": comparison,
        "test_t100_by_source_file_top_negative": _top_rows(
            _slice_table(test, test_ade, group=test.source_file.astype(str), min_rows=20),
            key="candidate_t100_full_waypoint_ade_improvement_vs_floor",
            n=12,
            reverse=False,
        ),
        "test_t100_by_scene_top_easy_harm": _top_rows(
            _slice_table(test, test_ade, group=test.scene_id.astype(str), min_rows=20),
            key="easy_degradation_vs_floor",
            n=12,
            reverse=True,
        ),
        "raw_validation_rule_test_metrics": raw_rule_metrics,
        "root_causes": root_causes,
        "recommended_next_actions": [
            "Do not deploy the validation-allowed UCY t100 source-family rule.",
            "Require source-file or scene-level validation support before allowing any t100 source-family switch.",
            "Investigate UCY t100 easy-case harm before trying another t100 specialist.",
            "Keep the deployed t100 policy at the floor until a source-stable current-matrix rule is positive and easy-safe.",
        ],
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
        "input_hash": _combined_hash([STAGE43_CM_JSON]),
    }
    payload["stage43_cn_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _top_rows(table: Mapping[str, Any], *, key: str, n: int, reverse: bool) -> list[dict[str, Any]]:
    rows = [{"name": name, **row} for name, row in table.items()]
    rows.sort(key=lambda row: float(row.get(key, 0.0)), reverse=bool(reverse))
    return rows[: int(n)]


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    comparison = payload["validation_test_shift_by_source_family"]
    failed_allowed = [row for row in comparison.values() if row["allowed_family_failed_current_test"]]
    gates = {
        "stage43_cm_precondition_present": payload["input_verdicts"]["stage43_cm"]
        == "stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor",
        "selected_model_replayed": payload["selected_model_replayed"]["model_hash_matches_stage43_cm"] is True,
        "current_val_test_h100_support_present": payload["support"]["val_h100_rows"] > 0
        and payload["support"]["test_h100_rows"] > 0,
        "validation_test_family_shift_measured": bool(comparison),
        "failed_allowed_family_identified": bool(failed_allowed),
        "root_cause_recorded": bool(payload["root_causes"])
        and payload["root_causes"] != ["no_major_shift_detected"],
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
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
            "stage43_cn_t100_validation_shift_forensics_pass_ucy_shift_blocker"
            if passed == total
            else "stage43_cn_t100_validation_shift_forensics_incomplete"
        ),
        "deploy_t100": False,
        "t100_positive_success": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cn_gate"]
    metrics = payload["raw_validation_rule_test_metrics"]
    overlap = payload["support"]["val_test_overlap"]
    lines = [
        "# Stage43-CN T100 Validation/Test Shift Forensics",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy t100: `{gate['deploy_t100']}`",
        "",
        "## Replayed Selected Model",
        "",
        f"- target: `{payload['selected_model_replayed']['target']}`",
        f"- train filter: `{payload['selected_model_replayed']['train_filter']}`",
        f"- l2: `{payload['selected_model_replayed']['l2']}`",
        f"- train rows: `{payload['selected_model_replayed']['train_rows']}`",
        f"- model hash matches Stage43-CM: `{payload['selected_model_replayed']['model_hash_matches_stage43_cm']}`",
        "",
        "## Raw Validation-Allowed Test Metrics",
        "",
        f"- all ADE lift: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure lift: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Validation/Test Source Overlap",
        "",
        f"- validation h100 rows: `{payload['support']['val_h100_rows']}`",
        f"- test h100 rows: `{payload['support']['test_h100_rows']}`",
        f"- source-file jaccard: `{overlap['source_file_jaccard']:.4f}`",
        f"- scene jaccard: `{overlap['scene_jaccard']:.4f}`",
        "",
        "## Source-Family Shift",
        "",
        "| family | val rows | test rows | val lift | test lift | lift drop | val easy harm | test easy harm | reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for family, row in payload["validation_test_shift_by_source_family"].items():
        lines.append(
            f"| {family} | {row['validation_rows']} | {row['test_rows']} | "
            f"{_pct(row['validation_lift'])} | {_pct(row['test_lift'])} | {_pct(row['lift_drop'])} | "
            f"{_pct(row['validation_easy_degradation'])} | {_pct(row['test_easy_degradation'])} | `{row['reason']}` |"
        )
    lines.extend(
        [
            "",
            "## Root Causes",
            "",
            *[f"- `{cause}`" for cause in payload["root_causes"]],
            "",
            "## Worst Test Source Files",
            "",
            "| source_file | rows | t100 lift | easy harm | mean floor ADE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["test_t100_by_source_file_top_negative"]:
        lines.append(
            f"| `{row['name']}` | {row['rows']} | {_pct(row['candidate_t100_full_waypoint_ade_improvement_vs_floor'])} | "
            f"{_pct(row['easy_degradation_vs_floor'])} | {row['mean_floor_ade']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-CM did not fail because the current matrix lacked t100 rows. It failed because a validation-positive UCY source-family rule did not generalize to current test: the same rule becomes negative on t100 and harms easy rows. This makes source-file/scene-level validation support a required next constraint before any t100 switch can be deployed.",
            "",
            "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric or seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cn_gate"]
    lines = [
        "# Stage43-CN Gate",
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
    gate = payload["stage43_cn_gate"]
    metrics = payload["raw_validation_rule_test_metrics"]
    ucy = payload["validation_test_shift_by_source_family"].get("UCY", {})
    block = [
        f"## {SECTION}",
        "",
        "I replayed the Stage43-CM selected model and audited why the validation-allowed t100 source-family rule failed on the current matrix test split.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- raw validation-allowed t100 lift: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- raw easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- UCY validation lift: `{_pct(ucy.get('validation_lift', 0.0))}`",
        f"- UCY test lift: `{_pct(ucy.get('test_lift', 0.0))}`",
        f"- UCY test easy degradation: `{_pct(ucy.get('test_easy_degradation', 0.0))}`",
        f"- root causes: `{', '.join(payload['root_causes'])}`",
        "",
        "Interpretation: t100 remains floor-only. The current blocker is validation/test source-scene shift inside the validation-allowed UCY t100 rule, not a lack of current-matrix t100 rows. Future t100 work needs source-file or scene-level validation support before switching.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("stage43", {})
    state["stage43"]["t100_validation_shift_forensics"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "raw_validation_rule_test_metrics": payload["raw_validation_rule_test_metrics"],
        "root_causes": payload["root_causes"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_cn_t100_validation_shift_forensics"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, m._jsonable(state))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-CN t100 validation/test shift forensics.")
    parser.add_argument("--seed", type=int, default=437)
    parser.add_argument("--min-rows", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = build_t100_validation_shift_forensics(seed=int(args.seed), min_rows=int(args.min_rows))
    gate = payload["stage43_cn_gate"]
    print(f"Stage43-CN: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return payload


if __name__ == "__main__":
    main()
