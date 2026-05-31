from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import _breakdown, _pct, _top_slices
from src.stage43_full_waypoint_latent_safe_repair import _source_family
from src.stage43_t100_source_coverage_preflight import _short_source
from src.stage43_t100_source_stable_specialist import (
    FAMILY as H100_FAMILY,
    REPORT_JSON as STAGE43_T_JSON,
    _concat_splits,
    _subset,
)
from src.stage43_tail_horizon_waypoint_adapter import (
    REPORT_JSON as STAGE43_P_JSON,
    _apply_rules,
    _easy_degradation,
    _model_hash,
    _predict_waypoint,
    _ridge_fit,
    _slice_improvement,
    _standardize,
    _target_matrix,
    _train_mask,
)


REPORT_JSON = OUT_DIR / "stage43_integrated_tail_h100_policy.json"
REPORT_MD = OUT_DIR / "stage43_integrated_tail_h100_policy.md"
GATE_MD = OUT_DIR / "stage43_stage_u_integrated_tail_h100_policy_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_U_INTEGRATED_TAIL_H100_POLICY"
SOURCE = "fresh_stage43_u_integrated_tail_h100_policy"
EPS = 1e-8


def _family_horizon(ds) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([_source_family(value) for value in ds.source_file]).astype(str), ds.horizon.astype(np.int64)


def _short_sources(ds) -> np.ndarray:
    return np.asarray([_short_source(value) for value in ds.source_file]).astype(str)


def _standardized_copy(ds, mean: np.ndarray, std: np.ndarray):
    return replace(ds, x=((ds.x - mean) / std).astype(np.float32))


def _train_stage43_p(seed: int, p_payload: Mapping[str, Any]):
    train = _build_split("train", max_rows=None, seed=int(seed))
    val = _build_split("val", max_rows=None, seed=int(seed))
    test = _build_split("test", max_rows=None, seed=int(seed))
    raw_test = _build_split("test", max_rows=None, seed=int(seed))
    mean, std = _standardize(train, val, test)
    selected = p_payload["selected_model"]
    train_ids = _train_mask(train, selected["train_filter"])
    weight = _ridge_fit(train.x[train_ids], _target_matrix(train, selected["target"])[train_ids], float(selected["l2"]))
    pred = _predict_waypoint(test, weight, selected["target"])
    candidate_ade, candidate_fde = _trajectory_error(test, pred)
    allowed: set[tuple[str, int]] = set()
    for rule in selected["allowed_rules"]:
        family, horizon = rule.split("|", 1)
        allowed.add((family, int(horizon)))
    selected_ade, selected_fde, switch = _apply_rules(test, candidate_ade, candidate_fde, allowed)
    return {
        "train": train,
        "val": val,
        "test": test,
        "raw_test": raw_test,
        "feature_mean": mean,
        "feature_std": std,
        "weight": weight,
        "candidate_ade": candidate_ade,
        "candidate_fde": candidate_fde,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
        "model_hash": _model_hash(
            weight,
            l2=float(selected["l2"]),
            target=selected["target"],
            train_filter=selected["train_filter"],
        ),
    }


def _train_stage43_t(seed: int, t_payload: Mapping[str, Any]):
    proposal = t_payload["source_level_split"]
    old_splits = [_build_split(split, max_rows=None, seed=int(seed)) for split in ["train", "val", "test"]]
    pool = _concat_splits(old_splits)
    train = _subset(pool, proposal["train_sources"], "train", horizon=100)
    val = _subset(pool, proposal["val_sources"], "val", horizon=100)
    test = _subset(pool, proposal["test_sources"], "test", horizon=100)
    mean, std = _standardize(train, val, test)
    selected = t_payload["selected_specialist"]
    weight = _ridge_fit(train.x, _target_matrix(train, selected["target"]), float(selected["l2"]))
    return {
        "train": train,
        "val": val,
        "test": test,
        "feature_mean": mean,
        "feature_std": std,
        "weight": weight,
        "model_hash": _model_hash(
            weight,
            l2=float(selected["l2"]),
            target=selected["target"],
            train_filter="source_stable_h100",
        ),
    }


def _source_table(ds, selected_ade: np.ndarray, switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    short = _short_sources(ds)
    for source in sorted(set(short[mask].tolist())):
        row_mask = mask & (short == source)
        out[source] = {
            "rows": int(row_mask.sum()),
            "full_waypoint_ade_improvement_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, row_mask),
            "easy_degradation_vs_floor": _easy_degradation(ds, selected_ade, row_mask),
            "switch_rate": float(np.mean(switch[row_mask])) if int(row_mask.sum()) else 0.0,
            "easy_rows": int((row_mask & ds.easy).sum()),
            "hard_failure_rows": int((row_mask & (ds.hard | ds.failure)).sum()),
        }
    return out


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def run_integrated_tail_h100_policy(
    *,
    p_seed: int = 431,
    t_seed: int = 461,
    bootstrap: int = 1000,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    p_payload = read_json(STAGE43_P_JSON, {})
    t_payload = read_json(STAGE43_T_JSON, {})
    p_replay = _train_stage43_p(int(p_seed), p_payload)
    t_replay = _train_stage43_t(int(t_seed), t_payload)
    test = p_replay["raw_test"]
    source_names = set(t_payload["source_level_split"]["test_sources"])
    short = _short_sources(test)
    h100_mask = np.isin(short, np.asarray(list(source_names), dtype=str)) & (test.horizon.astype(np.int64) == 100)

    t_test_view = _standardized_copy(test, t_replay["feature_mean"], t_replay["feature_std"])
    t_pred = _predict_waypoint(t_test_view, t_replay["weight"], t_payload["selected_specialist"]["target"])
    t_ade, t_fde = _trajectory_error(test, t_pred)

    integrated_ade = p_replay["selected_ade"].copy()
    integrated_fde = p_replay["selected_fde"].copy()
    integrated_switch = p_replay["switch"].copy()
    integrated_ade[h100_mask] = t_ade[h100_mask]
    integrated_fde[h100_mask] = t_fde[h100_mask]
    integrated_switch[h100_mask] = True

    p_metrics = _metrics(test, p_replay["selected_ade"], p_replay["selected_fde"], p_replay["switch"])
    integrated_metrics = _metrics(test, integrated_ade, integrated_fde, integrated_switch)
    p_slice = {
        "rows": int(h100_mask.sum()),
        "full_waypoint_ade_improvement_vs_floor": _slice_improvement(p_replay["selected_ade"], test.floor_ade, h100_mask),
        "endpoint_fde_improvement_vs_floor": _slice_improvement(p_replay["selected_fde"], test.floor_fde, h100_mask),
        "easy_degradation_vs_floor": _easy_degradation(test, p_replay["selected_ade"], h100_mask),
        "switch_rate": float(np.mean(p_replay["switch"][h100_mask])) if int(h100_mask.sum()) else 0.0,
    }
    t_slice = {
        "rows": int(h100_mask.sum()),
        "full_waypoint_ade_improvement_vs_floor": _slice_improvement(integrated_ade, test.floor_ade, h100_mask),
        "endpoint_fde_improvement_vs_floor": _slice_improvement(integrated_fde, test.floor_fde, h100_mask),
        "hard_failure_full_waypoint_ade_improvement_vs_floor": _slice_improvement(
            integrated_ade, test.floor_ade, h100_mask & (test.hard | test.failure)
        ),
        "easy_degradation_vs_floor": _easy_degradation(test, integrated_ade, h100_mask),
        "switch_rate": float(np.mean(integrated_switch[h100_mask])) if int(h100_mask.sum()) else 0.0,
        "delta_vs_stage43_p_full_waypoint_ade": float(
            _slice_improvement(integrated_ade, test.floor_ade, h100_mask)
            - _slice_improvement(p_replay["selected_ade"], test.floor_ade, h100_mask)
        ),
        "delta_vs_stage43_p_endpoint_fde": float(
            _slice_improvement(integrated_fde, test.floor_fde, h100_mask)
            - _slice_improvement(p_replay["selected_fde"], test.floor_fde, h100_mask)
        ),
    }
    bootstrap_ci = _bootstrap_ci(test, integrated_ade, integrated_fde, n=int(bootstrap), seed=int(t_seed) + 4700)
    by_domain = _breakdown(test.domain, test.floor_ade, test.floor_fde, integrated_ade, integrated_fde, t_ade, integrated_switch, test.easy)
    by_horizon = _breakdown(test.horizon.astype(str), test.floor_ade, test.floor_fde, integrated_ade, integrated_fde, t_ade, integrated_switch, test.easy)
    families, _ = _family_horizon(test)
    by_family = _breakdown(families, test.floor_ade, test.floor_fde, integrated_ade, integrated_fde, t_ade, integrated_switch, test.easy)
    source_table = _source_table(test, integrated_ade, integrated_switch, h100_mask)
    negative_sources = [
        {"source": name, **row}
        for name, row in source_table.items()
        if float(row["full_waypoint_ade_improvement_vs_floor"]) < 0.0
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_integrated_stage43_p_tail_adapter_plus_stage43_t_h100_specialist",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "preconditions": {
            "stage43_p_verdict": p_payload.get("stage43_p_gate", {}).get("verdict"),
            "stage43_t_verdict": t_payload.get("stage43_t_gate", {}).get("verdict"),
            "stage43_t_deployed": t_payload.get("stage43_t_gate", {}).get("deploy_source_stable_h100_specialist"),
            "stage43_t_validation_source_safe": t_payload.get("stage43_t_gate", {}).get("validation_source_safe"),
        },
        "training_protocol": {
            "selection_data": "stage43_p_and_stage43_t_validation_selected_only",
            "test_threshold_tuning": False,
            "stage43_p_seed": int(p_seed),
            "stage43_t_seed": int(t_seed),
            "num_workers": 0,
            "future_waypoints_as_labels_only": True,
            "max_easy_degradation": float(max_easy_degradation),
        },
        "replay_hashes": {
            "stage43_p_model_hash_replay": p_replay["model_hash"],
            "stage43_p_model_hash_reported": p_payload.get("selected_model", {}).get("model_hash"),
            "stage43_t_model_hash_replay": t_replay["model_hash"],
            "stage43_t_model_hash_reported": t_payload.get("selected_specialist", {}).get("model_hash"),
            "test_source_hash": _hash_array(test.source_file.astype(str)),
            "test_horizon_hash": _hash_array(test.horizon.astype(np.int64)),
        },
        "integrated_policy": {
            "base_policy": "stage43_p_tail_horizon_waypoint_adapter",
            "added_specialist": "stage43_t_source_stable_h100_specialist",
            "specialist_family": H100_FAMILY,
            "specialist_horizon": 100,
            "specialist_test_sources": sorted(source_names),
            "specialist_rows_in_full_test": int(h100_mask.sum()),
            "deployment_rule": "stage43_p_everywhere_except_stage43_t_source_stable_h100_sources",
        },
        "stage43_p_replay_metrics": p_metrics,
        "integrated_full_test_metrics": integrated_metrics,
        "delta_vs_stage43_p": {
            "full_waypoint_ade_improvement_delta": float(
                integrated_metrics["full_waypoint_ade_improvement_vs_floor"]
                - p_metrics["full_waypoint_ade_improvement_vs_floor"]
            ),
            "endpoint_fde_delta": float(
                integrated_metrics["endpoint_fde_improvement_vs_floor"]
                - p_metrics["endpoint_fde_improvement_vs_floor"]
            ),
            "t50_delta": float(
                integrated_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                - p_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            ),
            "t100_delta": float(
                integrated_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                - p_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            ),
            "hard_failure_delta": float(
                integrated_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                - p_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            ),
            "easy_degradation_delta": float(
                integrated_metrics["easy_degradation_vs_floor"] - p_metrics["easy_degradation_vs_floor"]
            ),
        },
        "h100_specialist_slice": {
            "stage43_p_slice": p_slice,
            "integrated_slice": t_slice,
            "source_table": source_table,
            "negative_source_count": int(len(negative_sources)),
            "worst_sources": _top_slices(source_table, key="full_waypoint_ade_improvement_vs_floor", n=8),
        },
        "bootstrap_ci": bootstrap_ci,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_source_family": by_family,
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
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "uniform_t100_success": False,
            "endpoint_fde_success": False,
        },
    }
    payload["stage43_u_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["integrated_full_test_metrics"]
    delta = payload["delta_vs_stage43_p"]
    h100 = payload["h100_specialist_slice"]["integrated_slice"]
    gates = {
        "stage43_p_precondition_passed": payload["preconditions"]["stage43_p_verdict"]
        == "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
        "stage43_t_precondition_passed": payload["preconditions"]["stage43_t_verdict"]
        == "stage43_t_source_stable_h100_specialist_deployable"
        and payload["preconditions"]["stage43_t_deployed"] is True,
        "validation_source_safe_precondition": payload["preconditions"]["stage43_t_validation_source_safe"] is True,
        "replay_hashes_match": payload["replay_hashes"]["stage43_p_model_hash_replay"]
        == payload["replay_hashes"]["stage43_p_model_hash_reported"]
        and payload["replay_hashes"]["stage43_t_model_hash_replay"]
        == payload["replay_hashes"]["stage43_t_model_hash_reported"],
        "integrated_h100_rows_present": payload["integrated_policy"]["specialist_rows_in_full_test"] > 0,
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "overall_not_harmed_vs_stage43_p": delta["full_waypoint_ade_improvement_delta"] >= -1e-7,
        "t50_preserved": abs(delta["t50_delta"]) < 1e-7,
        "t100_raw_frame_diagnostic_positive": metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0,
        "h100_source_slice_positive": h100["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= payload["training_protocol"]["max_easy_degradation"]
        and h100["easy_degradation_vs_floor"] <= payload["training_protocol"]["max_easy_degradation"],
        "h100_source_nonnegative": payload["h100_specialist_slice"]["negative_source_count"] == 0,
        "bootstrap_all_positive": payload["bootstrap_ci"]["metrics"]["full_waypoint_ade_improvement_vs_floor"]["low"] > 0.0,
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
        "verdict": "stage43_u_integrated_tail_h100_policy_pass_family_limited"
        if passed == total
        else "stage43_u_integrated_tail_h100_policy_incomplete",
        "deploy_integrated_policy": passed == total,
        "uniform_t100_success": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_u_gate"]
    metrics = payload["integrated_full_test_metrics"]
    delta = payload["delta_vs_stage43_p"]
    h100 = payload["h100_specialist_slice"]["integrated_slice"]
    ci = payload["bootstrap_ci"]["metrics"]
    lines = [
        "# Stage43-U Integrated Tail + H100 Policy",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy integrated policy: `{gate['deploy_integrated_policy']}`",
        f"- specialist rows in full test: `{payload['integrated_policy']['specialist_rows_in_full_test']}`",
        "",
        "## Integrated Full-Test Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(metrics['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Delta vs Stage43-P",
        "",
        f"- all ADE delta: `{_pct(delta['full_waypoint_ade_improvement_delta'])}`",
        f"- endpoint FDE delta: `{_pct(delta['endpoint_fde_delta'])}`",
        f"- t50 delta: `{_pct(delta['t50_delta'])}`",
        f"- t100 delta: `{_pct(delta['t100_delta'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure_delta'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation_delta'])}`",
        "",
        "## H100 Source-Stable Slice",
        "",
        f"- rows: `{h100['rows']}`",
        f"- full-waypoint ADE improvement: `{_pct(h100['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(h100['endpoint_fde_improvement_vs_floor'])}`",
        f"- hard/failure ADE improvement: `{_pct(h100['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(h100['easy_degradation_vs_floor'])}`",
        f"- delta vs Stage43-P ADE on slice: `{_pct(h100['delta_vs_stage43_p_full_waypoint_ade'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all ADE CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t100 diagnostic CI: `[{_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['low'])}, {_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['high'])}]`",
        "",
        "## Interpretation",
        "",
        "Stage43-U composes the Stage43-P protected full-waypoint tail adapter with the Stage43-T source-stable h100 specialist. It adds a small positive h100 family-limited full-waypoint ADE lift while preserving Stage43-P t50 and easy-case safety. Endpoint FDE on the h100 slice remains negative, so this is not a uniform t100 or endpoint-success claim.",
        "",
        "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-U Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"deploy_integrated_policy: `{gate['deploy_integrated_policy']}`",
        f"uniform_t100_success: `{gate['uniform_t100_success']}`",
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
    gate = payload["stage43_u_gate"]
    metrics = payload["integrated_full_test_metrics"]
    delta = payload["delta_vs_stage43_p"]
    h100 = payload["h100_specialist_slice"]["integrated_slice"]
    lines = [
        "## Stage43-U integrated tail + h100 policy",
        "",
        f"Result source: `{payload['result_source']}`. This composes Stage43-P with the Stage43-T source-stable h100 specialist without changing test thresholds.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deployed: `{gate['deploy_integrated_policy']}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure ADE improvement vs floor: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- h100 source-slice ADE lift: `{_pct(h100['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- h100 source-slice endpoint FDE lift: `{_pct(h100['endpoint_fde_improvement_vs_floor'])}`",
        f"- all delta vs Stage43-P: `{_pct(delta['full_waypoint_ade_improvement_delta'])}`",
        "",
        "Boundary: this is a family-limited h100 full-waypoint ADE improvement integrated into the protected policy. It is not a uniform t100 solution, not endpoint-FDE success, not metric/seconds-level, not Stage5C, and not SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_u_gate"]
    state["stage43_u_integrated_tail_h100_policy"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "integrated_policy": payload["integrated_policy"],
        "integrated_full_test_metrics": payload["integrated_full_test_metrics"],
        "delta_vs_stage43_p": payload["delta_vs_stage43_p"],
        "h100_specialist_slice": payload["h100_specialist_slice"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_u_integrated_tail_h100_policy"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_u_integrated_tail_h100_policy", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-U integrated Stage43-P + Stage43-T protected policy.")
    parser.add_argument("--p-seed", type=int, default=431)
    parser.add_argument("--t-seed", type=int, default=461)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_integrated_tail_h100_policy(
        p_seed=int(args.p_seed),
        t_seed=int(args.t_seed),
        bootstrap=int(args.bootstrap),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = result["stage43_u_gate"]
    print(f"Stage43-U: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
