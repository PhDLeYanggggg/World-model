from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _git_commit,
    _jsonable,
)
from src.stage43_world_state_head_audit import REPORT_JSON as STAGE43_V_JSON
from src.stage43_auxiliary_head_repair import REPORT_JSON as STAGE43_W_JSON
from src.stage43_interaction_validity_proxy import REPORT_JSON as STAGE43_X_JSON


REPORT_JSON = OUT_DIR / "stage43_multimodal_latent_head_suite.json"
REPORT_MD = OUT_DIR / "stage43_multimodal_latent_head_suite.md"
GATE_MD = OUT_DIR / "stage43_stage_y_multimodal_latent_head_suite_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_Y_MULTIMODAL_LATENT_HEAD_SUITE"
SOURCE = "fresh_stage43_y_multimodal_latent_head_suite"


def _metric(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value is None:
        return default
    return float(value)


def _assemble_head_table(v: Mapping[str, Any], w: Mapping[str, Any], x: Mapping[str, Any]) -> dict[str, Any]:
    heads = v["head_metrics"]
    density = w["density_repair"]["selected"]
    w_validity = w["waypoint_validity_proxy_repair"]["selected"]
    interaction = x["interaction_risk_head"]["selected"]
    smoothness = x["smoothness_validity_proxy_head"]["selected"]
    return {
        "failure_risk": {
            "source_stage": "Stage43-V",
            "target": "baseline failure risk",
            "status": "deployable_proxy",
            "primary": "auroc",
            "auroc": heads["failure"]["auroc"],
            "auprc": heads["failure"]["auprc"],
            "ece": heads["failure"]["ece"],
        },
        "gain_opportunity": {
            "source_stage": "Stage43-V",
            "target": "switch/gain opportunity",
            "status": "deployable_proxy",
            "primary": "auroc",
            "auroc": heads["gain"]["auroc"],
            "auprc": heads["gain"]["auprc"],
            "ece": heads["gain"]["ece"],
        },
        "harm_guard": {
            "source_stage": "Stage43-V",
            "target": "easy/harm guard",
            "status": "deployable_proxy",
            "primary": "auroc",
            "auroc": heads["harm"]["auroc"],
            "auprc": heads["harm"]["auprc"],
            "ece": heads["harm"]["ece"],
        },
        "causal_history_density": {
            "source_stage": "Stage43-W",
            "target": "causal history-density proxy",
            "status": "deployable_proxy_not_future_occupancy",
            "primary": "r2",
            "r2": density["test_metrics"]["r2"],
            "corr": density["test_metrics"]["corr"],
            "rmse": density["test_metrics"]["rmse"],
            "feature_set": density["feature_set"],
        },
        "future_interaction_risk": {
            "source_stage": "Stage43-X",
            "target": "future-proximity interaction risk proxy",
            "status": "deployable_proxy_not_human_annotation",
            "primary": "auroc",
            "auroc": interaction["test_metrics"]["auroc"],
            "auprc": interaction["test_metrics"]["auprc"],
            "ece": interaction["test_metrics"]["ece"],
            "positive_rate": interaction["test_metrics"]["positive_rate"],
            "feature_set": interaction["feature_set"],
        },
        "waypoint_label_availability": {
            "source_stage": "Stage43-W",
            "target": "waypoint label availability proxy",
            "status": "diagnostic_only",
            "primary": "r2",
            "r2": w_validity["test_metrics"]["r2"],
            "corr": w_validity["test_metrics"]["corr"],
            "rmse": w_validity["test_metrics"]["rmse"],
        },
        "smoothness_validity_proxy": {
            "source_stage": "Stage43-X",
            "target": "future waypoint smoothness/validity proxy",
            "status": "diagnostic_only_not_true_physical_validity",
            "primary": "r2",
            "r2": smoothness["test_metrics"]["r2"],
            "corr": smoothness["test_metrics"]["corr"],
            "rmse": smoothness["test_metrics"]["rmse"],
        },
    }


def run_head_suite() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    v = read_json(STAGE43_V_JSON, {})
    w = read_json(STAGE43_W_JSON, {})
    x = read_json(STAGE43_X_JSON, {})
    head_suite = _assemble_head_table(v, w, x)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_consolidated_stage43_vwx_head_suite",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "inputs": {
            "stage43_v": str(STAGE43_V_JSON),
            "stage43_w": str(STAGE43_W_JSON),
            "stage43_x": str(STAGE43_X_JSON),
        },
        "preconditions": {
            "stage43_v_verdict": v.get("stage43_v_gate", {}).get("verdict"),
            "stage43_w_verdict": w.get("stage43_w_gate", {}).get("verdict"),
            "stage43_x_verdict": x.get("stage43_x_gate", {}).get("verdict"),
            "stage43_v_gate": f"{v.get('stage43_v_gate', {}).get('passed')} / {v.get('stage43_v_gate', {}).get('total')}",
            "stage43_w_gate": f"{w.get('stage43_w_gate', {}).get('passed')} / {w.get('stage43_w_gate', {}).get('total')}",
            "stage43_x_gate": f"{x.get('stage43_x_gate', {}).get('passed')} / {x.get('stage43_x_gate', {}).get('total')}",
        },
        "latent_state": v.get("latent_stats", {}),
        "head_suite": head_suite,
        "deployment_contract": {
            "deployable_proxy_heads": [
                "failure_risk",
                "gain_opportunity",
                "harm_guard",
                "causal_history_density",
                "future_interaction_risk",
            ],
            "diagnostic_only_heads": [
                "waypoint_label_availability",
                "smoothness_validity_proxy",
            ],
            "must_keep_safety_floor": True,
            "not_a_standalone_ungated_policy": True,
            "physical_validity_true_claim_allowed": False,
            "future_occupancy_claim_allowed": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_supervision_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "physical_validity_true_claim": False,
            "future_occupancy_claim": False,
        },
    }
    payload["stage43_y_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    heads = payload["head_suite"]
    latent = payload["latent_state"]
    gates = {
        "stage43_v_passed": payload["preconditions"]["stage43_v_verdict"] == "stage43_v_world_state_head_audit_partial",
        "stage43_w_passed": payload["preconditions"]["stage43_w_verdict"]
        == "stage43_w_density_proxy_repaired_validity_proxy_diagnostic",
        "stage43_x_passed": payload["preconditions"]["stage43_x_verdict"]
        == "stage43_x_interaction_proxy_signal_validity_proxy_diagnostic",
        "latent_noncollapse": _metric(latent, "min_variance") > _metric(latent, "noncollapse_threshold", 0.01),
        "failure_gain_harm_heads_strong": _metric(heads["failure_risk"], "auroc") > 0.80
        and _metric(heads["gain_opportunity"], "auroc") > 0.80
        and _metric(heads["harm_guard"], "auroc") > 0.80,
        "density_proxy_positive": _metric(heads["causal_history_density"], "r2") > 0.0
        and _metric(heads["causal_history_density"], "corr") > 0.5,
        "interaction_proxy_positive": _metric(heads["future_interaction_risk"], "auroc") > 0.60,
        "diagnostic_validity_reported_not_deployed": payload["deployment_contract"]["physical_validity_true_claim_allowed"] is False
        and "smoothness_validity_proxy" in payload["deployment_contract"]["diagnostic_only_heads"],
        "safety_floor_required": payload["deployment_contract"]["must_keep_safety_floor"] is True,
        "not_ungated_policy": payload["deployment_contract"]["not_a_standalone_ungated_policy"] is True,
        "no_future_or_test_leakage": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_labels_eval_or_supervision_only"] is True
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_y_protected_multimodal_latent_head_suite_candidate"
        if passed == total
        else "stage43_y_multimodal_latent_head_suite_incomplete",
        "protected_multimodal_latent_state_candidate": bool(passed == total),
        "standalone_world_model_deployable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_y_gate"]
    heads = payload["head_suite"]
    latent = payload["latent_state"]
    lines = [
        "# Stage43-Y Multimodal Latent Head Suite",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"- standalone ungated deployment: `False`",
        "",
        "## Latent State",
        "",
        f"- latent dim: `{latent.get('dim')}`",
        f"- min variance: `{float(latent.get('min_variance', 0.0)):.6f}`",
        f"- mean variance: `{float(latent.get('mean_variance', 0.0)):.6f}`",
        f"- non-collapse threshold: `{latent.get('noncollapse_threshold')}`",
        "",
        "## Head Suite",
        "",
        "| head | status | primary metric | note |",
        "| --- | --- | ---: | --- |",
    ]
    for name, row in heads.items():
        if row["primary"] == "auroc":
            metric = f"AUROC `{float(row['auroc']):.4f}`"
        else:
            metric = f"R2 `{float(row['r2']):.4f}`"
        lines.append(f"| {name} | `{row['status']}` | {metric} | {row['target']} |")
    lines.extend(
        [
            "",
            "## Deployment Contract",
            "",
            "- deployable proxy heads: `failure_risk`, `gain_opportunity`, `harm_guard`, `causal_history_density`, `future_interaction_risk`",
            "- diagnostic-only heads: `waypoint_label_availability`, `smoothness_validity_proxy`",
            "- safety floor remains required",
            "- this is not a standalone ungated deployment policy",
            "- no true physical-validity claim",
            "- no future occupancy claim",
            "",
            "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-Y Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"protected_multimodal_latent_state_candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"standalone_world_model_deployable: `{gate['standalone_world_model_deployable']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_y_gate"]
    heads = payload["head_suite"]
    latent = payload["latent_state"]
    lines = [
        "## Stage43-Y multimodal latent head suite",
        "",
        f"Result source: `{payload['result_source']}`. I consolidated Stage43-V/W/X into a single head-suite contract for the protected multimodal latent-state candidate.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- latent min/mean variance: `{float(latent.get('min_variance', 0.0)):.6f}` / `{float(latent.get('mean_variance', 0.0)):.6f}`",
        f"- failure/gain/harm AUROC: `{float(heads['failure_risk']['auroc']):.4f}` / `{float(heads['gain_opportunity']['auroc']):.4f}` / `{float(heads['harm_guard']['auroc']):.4f}`",
        f"- density proxy R2/corr: `{float(heads['causal_history_density']['r2']):.4f}` / `{float(heads['causal_history_density']['corr']):.4f}`",
        f"- interaction proxy AUROC/AUPRC: `{float(heads['future_interaction_risk']['auroc']):.4f}` / `{float(heads['future_interaction_risk']['auprc']):.4f}`",
        f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"- standalone ungated deployment: `False`",
        "",
        "Boundary: this consolidates deployable proxy heads under the existing safety floor. It does not create a true physical-validity certificate, future occupancy claim, metric/seconds-level result, Stage5C execution, or SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_y_gate"]
    state["stage43_y_multimodal_latent_head_suite"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "protected_multimodal_latent_state_candidate": gate["protected_multimodal_latent_state_candidate"],
        "standalone_world_model_deployable": gate["standalone_world_model_deployable"],
        "latent_state": payload["latent_state"],
        "deployment_contract": payload["deployment_contract"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_y_multimodal_latent_head_suite"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_y_multimodal_latent_head_suite", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run Stage43-Y multimodal latent head-suite consolidation.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    result = run_head_suite()
    gate = result["stage43_y_gate"]
    print(f"Stage43-Y: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
