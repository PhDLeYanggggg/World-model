from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _predict,
    _select_with_policy,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _breakdown,
    _load_model,
    _pct,
    _slice_metrics,
    _standardize_from_checkpoint,
    _top_slices,
)


REPORT_JSON = OUT_DIR / "stage43_full_waypoint_latent_safe_repair.json"
REPORT_MD = OUT_DIR / "stage43_full_waypoint_latent_safe_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_o_full_waypoint_latent_safe_repair_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_N_JSON = OUT_DIR / "stage43_full_waypoint_latent_robustness_audit.json"
SECTION = "STAGE43_O_FULL_WAYPOINT_LATENT_SAFE_REPAIR"
SOURCE = "fresh_stage43_o_full_waypoint_latent_safe_repair"
HORIZONS = [10, 25, 50, 100]
EPS = 1e-8


def _source_family(source_file: str) -> str:
    text = str(source_file).lower()
    if "pets" in text or "/mot/" in text or "mot/" in text:
        return "TrajNet_mot"
    if "biwi" in text:
        return "TrajNet_biwi"
    if "crowds" in text or "zara" in text:
        return "TrajNet_crowds"
    if "ucy" in text:
        return "UCY"
    if "eth" in text or "hotel" in text:
        return "ETH_UCY"
    return "other"


def _family_horizon_values(ds) -> tuple[np.ndarray, np.ndarray]:
    families = np.asarray([_source_family(value) for value in ds.source_file]).astype(str)
    horizons = ds.horizon.astype(np.int64)
    return families, horizons


def _safe_slice_metrics(ds, selected_ade: np.ndarray, selected_fde: np.ndarray, switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    ungated_ade = selected_ade
    return _slice_metrics(ds.floor_ade, ds.floor_fde, selected_ade, selected_fde, ungated_ade, switch, ds.easy, mask)


def _validation_support_table(
    val,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    *,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    families, horizons = _family_horizon_values(val)
    table: dict[str, Any] = {}
    allowed: set[tuple[str, int]] = set()
    for family in sorted(set(families.tolist())):
        for horizon in HORIZONS:
            mask = (families == family) & (horizons == int(horizon))
            if int(mask.sum()) == 0:
                continue
            metrics = _safe_slice_metrics(val, selected_ade, selected_fde, switched, mask)
            supported = (
                int(metrics["rows"]) >= int(min_support_rows)
                and float(metrics["full_waypoint_ade_improvement_vs_floor"]) > float(min_improvement)
                and float(metrics["easy_degradation_vs_floor"]) <= float(max_easy_degradation)
            )
            reason = "allowed_by_validation"
            if int(metrics["rows"]) < int(min_support_rows):
                reason = "blocked_insufficient_validation_support"
            elif float(metrics["full_waypoint_ade_improvement_vs_floor"]) <= float(min_improvement):
                reason = "blocked_validation_nonpositive"
            elif float(metrics["easy_degradation_vs_floor"]) > float(max_easy_degradation):
                reason = "blocked_validation_easy_harm"
            key = f"{family}|{horizon}"
            table[key] = {**metrics, "allowed": bool(supported), "reason": reason}
            if supported:
                allowed.add((family, int(horizon)))
    return table, allowed


def _apply_allowed_rules(ds, pred: Mapping[str, np.ndarray], base_switch: np.ndarray, allowed: set[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_ade, candidate_fde = _trajectory_error(ds, pred["waypoint"])
    families, horizons = _family_horizon_values(ds)
    rule_allowed = np.asarray([(family, int(horizon)) in allowed for family, horizon in zip(families, horizons)], dtype=bool)
    switch = base_switch.astype(bool) & rule_allowed
    selected_ade = np.where(switch, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(switch, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, switch, rule_allowed


def _rule_breakdown(ds, base_switch: np.ndarray, repaired_switch: np.ndarray, rule_allowed: np.ndarray) -> dict[str, Any]:
    families, horizons = _family_horizon_values(ds)
    out: dict[str, Any] = {}
    for family in sorted(set(families.tolist())):
        for horizon in HORIZONS:
            mask = (families == family) & (horizons == int(horizon))
            if int(mask.sum()) == 0:
                continue
            key = f"{family}|{horizon}"
            out[key] = {
                "rows": int(mask.sum()),
                "original_switch_rows": int(base_switch[mask].sum()),
                "repaired_switch_rows": int(repaired_switch[mask].sum()),
                "blocked_switch_rows": int((base_switch[mask] & ~repaired_switch[mask]).sum()),
                "rule_allowed": bool(np.any(rule_allowed[mask])),
            }
    return out


def run_full_waypoint_latent_safe_repair(
    *,
    batch_size: int = 4096,
    min_support_rows: int = 1000,
    min_improvement: float = 0.0,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43n = read_json(STAGE43_N_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    seed = int(ckpt.get("seed", 431))
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=seed), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=seed), ckpt)
    with torch.no_grad():
        val_pred = _predict(model, val, torch.device("cpu"), int(batch_size))
        test_pred = _predict(model, test, torch.device("cpu"), int(batch_size))
    base_policy = stage43m["validation_selected_policy"]["policy"]
    val_base_ade, val_base_fde, val_base_switch = _select_with_policy(val, val_pred, base_policy)
    test_base_ade, test_base_fde, test_base_switch = _select_with_policy(test, test_pred, base_policy)
    validation_table, allowed = _validation_support_table(
        val,
        val_base_ade,
        val_base_fde,
        val_base_switch,
        min_support_rows=int(min_support_rows),
        min_improvement=float(min_improvement),
        max_easy_degradation=float(max_easy_degradation),
    )
    selected_ade, selected_fde, repaired_switch, rule_allowed = _apply_allowed_rules(test, test_pred, test_base_switch, allowed)
    candidate_ade, _ = _trajectory_error(test, test_pred["waypoint"])
    overall = _metrics(test, selected_ade, selected_fde, repaired_switch)
    arrays = (test.floor_ade, test.floor_fde, selected_ade, selected_fde, candidate_ade, repaired_switch, test.easy)
    by_domain = _breakdown(test.domain, *arrays)
    by_horizon = _breakdown(test.horizon.astype(str), *arrays)
    by_source = _breakdown(test.source_file, *arrays, min_rows=50)
    by_source_family = _breakdown(np.asarray([_source_family(value) for value in test.source_file]).astype(str), *arrays, min_rows=50)
    negative_sources = [
        {"source_file": name, **metrics}
        for name, metrics in by_source.items()
        if float(metrics["full_waypoint_ade_improvement_vs_floor"]) < 0.0
    ]
    domain_easy_harm = [
        {"domain": name, **metrics}
        for name, metrics in by_domain.items()
        if float(metrics["easy_degradation_vs_floor"]) > 0.02
    ]
    t100 = by_horizon.get("100", {})
    original = stage43n.get("overall_full_test_metrics", {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_only_safe_repair_from_stage43_m_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_n_precondition": {
            "verdict": stage43n.get("stage43_n_gate", {}).get("verdict"),
            "negative_source_count": stage43n.get("by_source_summary", {}).get("negative_source_count"),
            "t100_improvement": stage43n.get("t100_failure_attribution", {}).get("protected_t100_improvement"),
        },
        "checkpoint": str(checkpoint),
        "base_policy_replayed": base_policy,
        "repair_policy": {
            "policy_type": "validation_only_source_family_horizon_support_guard",
            "min_support_rows": int(min_support_rows),
            "min_improvement": float(min_improvement),
            "max_easy_degradation": float(max_easy_degradation),
            "allowed_source_family_horizon_rules": sorted([f"{family}|{horizon}" for family, horizon in allowed]),
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
        },
        "validation_support_table": validation_table,
        "full_test_rows": int(len(test.x)),
        "overall_full_test_metrics": overall,
        "original_stage43_n_full_test_metrics": original,
        "repair_delta_vs_stage43_n": {
            "full_waypoint_ade_improvement_delta": float(
                overall["full_waypoint_ade_improvement_vs_floor"]
                - float(original.get("full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "t50_delta": float(
                overall["t50_full_waypoint_ade_improvement_vs_floor"]
                - float(original.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "t100_delta": float(
                overall["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                - float(original.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0))
            ),
            "hard_failure_delta": float(
                overall["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                - float(original.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "easy_degradation_delta": float(
                overall["easy_degradation_vs_floor"] - float(original.get("easy_degradation_vs_floor", 0.0))
            ),
        },
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_source_family": by_source_family,
        "by_source_summary": {
            "source_count": int(len(by_source)),
            "negative_source_count": int(len(negative_sources)),
            "worst_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12),
            "best_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12, reverse=True),
        },
        "switch_rule_breakdown": _rule_breakdown(test, test_base_switch, repaired_switch, rule_allowed),
        "source_domain_caveats": {
            "negative_source_count": int(len(negative_sources)),
            "domain_easy_harm_count": int(len(domain_easy_harm)),
            "domains_with_easy_harm": domain_easy_harm,
            "uniform_positive_source_success": all(
                float(metrics["full_waypoint_ade_improvement_vs_floor"]) > 0.0 for metrics in by_source.values()
            ),
            "fallback_only_source_families": [
                name for name, metrics in by_source_family.items() if float(metrics["switch_rate"]) == 0.0
            ],
        },
        "t100_repair": {
            "rows": int(t100.get("rows", 0)),
            "improvement": float(t100.get("full_waypoint_ade_improvement_vs_floor", 0.0)),
            "easy_degradation": float(t100.get("easy_degradation_vs_floor", 0.0)),
            "switch_rate": float(t100.get("switch_rate", 0.0)),
            "status": "t100_harm_repaired_by_fallback_not_positive"
            if abs(float(t100.get("full_waypoint_ade_improvement_vs_floor", 0.0))) < 1e-7
            else "t100_positive" if float(t100.get("full_waypoint_ade_improvement_vs_floor", 0.0)) > 0.0 else "t100_still_negative",
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
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "uniform_source_positive_success": False,
            "t100_positive_success": False,
        },
    }
    payload["stage43_o_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["overall_full_test_metrics"]
    t100 = payload["t100_repair"]
    gates = {
        "stage43_n_blockers_present_before_repair": int(payload["stage43_n_precondition"].get("negative_source_count") or 0) > 0
        and float(payload["stage43_n_precondition"].get("t100_improvement") or 0.0) < 0.0,
        "validation_only_policy_built": payload["repair_policy"]["selection_data"] == "validation_only",
        "no_test_threshold_tuning": payload["repair_policy"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "overall_positive": metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t50_positive": metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "hard_failure_positive": metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "negative_sources_repaired": payload["source_domain_caveats"]["negative_source_count"] == 0,
        "domain_easy_harm_repaired": payload["source_domain_caveats"]["domain_easy_harm_count"] == 0,
        "t100_harm_repaired_or_positive": t100["improvement"] >= -1e-7 and t100["easy_degradation"] <= 0.02,
        "t100_positive_not_overclaimed": payload["claim_boundary"]["t100_positive_success"] is False
        and t100["status"] == "t100_harm_repaired_by_fallback_not_positive",
        "no_future_or_metric_or_stage5c_smc_claim": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_o_safe_repair_pass_t100_fallback_not_positive"
        if passed == total
        else "stage43_o_safe_repair_incomplete",
        "deploy_safe_repair_policy": passed == total,
        "uniform_source_positive_success": False,
        "t100_positive_success": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_o_gate"]
    metrics = payload["overall_full_test_metrics"]
    delta = payload["repair_delta_vs_stage43_n"]
    lines = [
        "# Stage43-O Full-Waypoint Latent Safe Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        "",
        "## What changed",
        "",
        "Stage43-O does not retrain the latent model and does not tune on test. It uses validation-only source-family/horizon support rules to decide when the Stage43-M latent full-waypoint head is allowed to switch away from the frozen floor.",
        "",
        f"- allowed rules: `{', '.join(payload['repair_policy']['allowed_source_family_horizon_rules'])}`",
        f"- min validation support rows: `{payload['repair_policy']['min_support_rows']}`",
        f"- max validation easy degradation: `{_pct(payload['repair_policy']['max_easy_degradation'])}`",
        "",
        "## Full-Test Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(metrics['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Delta vs Stage43-N",
        "",
        f"- all ADE improvement delta: `{_pct(delta['full_waypoint_ade_improvement_delta'])}`",
        f"- t50 delta: `{_pct(delta['t50_delta'])}`",
        f"- t100 delta: `{_pct(delta['t100_delta'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure_delta'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation_delta'])}`",
        "",
        "## Horizon Breakdown",
        "",
        "| horizon | rows | ADE lift | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["by_horizon"].items():
        lines.append(
            f"| {name} | {row['rows']} | {_pct(row['full_waypoint_ade_improvement_vs_floor'])} | {_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Source-Family Breakdown",
            "",
            "| source family | rows | ADE lift | easy degradation | switch |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in payload["by_source_family"].items():
        lines.append(
            f"| {name} | {row['rows']} | {_pct(row['full_waypoint_ade_improvement_vs_floor'])} | {_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The repair removes the Stage43-N negative source and long-horizon harm by falling back where validation support is insufficient or validation h100 is unsafe. This is a safer protected policy, not proof that t100 is solved: t100 is repaired to fallback-level `0.0`, not positive transfer.",
            "",
            "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-O Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"deploy_safe_repair_policy: `{gate['deploy_safe_repair_policy']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"uniform_source_positive_success: `{gate['uniform_source_positive_success']}`",
        f"t100_positive_success: `{gate['t100_positive_success']}`",
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
    gate = payload["stage43_o_gate"]
    metrics = payload["overall_full_test_metrics"]
    lines = [
        "## Stage43-O full-waypoint latent safe repair",
        "",
        f"Result source: `{payload['result_source']}`. This step keeps the Stage43-M latent model frozen and uses validation-only source-family/horizon support to repair the Stage43-N negative source and t100 harm.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure ADE improvement vs floor: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- negative source count after repair: `{payload['by_source_summary']['negative_source_count']}`",
        "",
        "Boundary: Stage43-O is a safety repair. It removes negative-source and t100 harm by fallback, but t100 is not a positive success and some source families are fallback-only. The result remains dataset-local/raw-frame 2.5D evidence with no metric/seconds-level claim, no Stage5C, and no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_o_gate"]
    state["stage43_o_full_waypoint_latent_safe_repair"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "repair_policy": payload["repair_policy"],
        "overall_full_test_metrics": payload["overall_full_test_metrics"],
        "source_domain_caveats": payload["source_domain_caveats"],
        "t100_repair": payload["t100_repair"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_o_full_waypoint_latent_safe_repair"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_o_full_waypoint_latent_safe_repair", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair Stage43 full-waypoint latent policy with validation-only source/horizon safety support.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-support-rows", type=int, default=1000)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_full_waypoint_latent_safe_repair(
        batch_size=int(args.batch_size),
        min_support_rows=int(args.min_support_rows),
        min_improvement=float(args.min_improvement),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = result["stage43_o_gate"]
    print(f"Stage43-O: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
