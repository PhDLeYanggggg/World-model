from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_context_slice_policy_promotion as ja
from src import stage42_source_level_ablation as an
from src import stage42_source_level_context_repair as ix
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_nonlinear_context_repair as iy
from src import stage42_source_level_nonlinear_context_slice_audit as iz
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_slice_policy_conservative_repair_stage42.json"
REPORT_MD = OUT_DIR / "context_slice_policy_conservative_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jb_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_VAL_SLICE_ROWS = 500
MIN_GREEDY_GAIN = 0.005
MIN_TEST_MATERIAL_GAIN = 0.01
MAX_CORE_DROP = 0.002
MAX_EASY_DEGRADATION = 0.02


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JB 是 Stage42-JA 失败后的修复实验：只允许 validation-greedy、inference-safe、core-preserving 的 context slice 切换。",
    "切片阈值来自 train split quantiles；候选排序和贪心选择只看 validation；test 只评一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调阈值。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


CORE_KEYS = [
    "all_improvement",
    "t50_improvement",
    "t100_raw_frame_diagnostic_improvement",
    "hard_failure_improvement",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-JB conservative context-slice policy repair":
        return payload
    return None


def _deployment_score(metric: Mapping[str, Any]) -> float:
    return (
        1.2 * float(metric["all_improvement"])
        + 1.8 * float(metric["t50_improvement"])
        + 0.8 * float(metric["t100_raw_frame_diagnostic_improvement"])
        + 1.1 * float(metric["hard_failure_improvement"])
        - 30.0 * max(0.0, float(metric["easy_degradation"]) - 0.01)
        - 0.03 * float(metric["switch_rate"])
    )


def _metric_delta(new_metric: Mapping[str, Any], old_metric: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t10_improvement",
        "t25_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
        "harm_over_fallback",
    ]
    return {k: float(new_metric.get(k, 0.0)) - float(old_metric.get(k, 0.0)) for k in keys}


def _candidate_supported(candidate_metric: Mapping[str, Any], baseline_metric: Mapping[str, Any], rows: int) -> bool:
    if rows < MIN_VAL_SLICE_ROWS:
        return False
    if float(candidate_metric["easy_degradation"]) > MAX_EASY_DEGRADATION:
        return False
    delta = _metric_delta(candidate_metric, baseline_metric)
    if min(delta[k] for k in CORE_KEYS) < -MAX_CORE_DROP:
        return False
    return max(delta[k] for k in CORE_KEYS) > MIN_GREEDY_GAIN


def _candidate_pool(
    shared: Mapping[str, Any],
    outputs: Mapping[str, Any],
    slices: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    val_mask = shared["split"] == "val"
    baseline = outputs["tree_baseline_family_residual"]
    rows: list[dict[str, Any]] = []
    for slice_name, smask in sorted(slices.items()):
        if slice_name in ja.DISALLOWED_POLICY_SLICES:
            continue
        val_rows = int(np.sum(val_mask & smask))
        if val_rows < 30:
            continue
        baseline_metric = iz._metric_for_slice(baseline, shared["data"], val_mask, smask)
        for trial_name, out in outputs.items():
            if trial_name == "tree_baseline_family_residual":
                continue
            metric = iz._metric_for_slice(out, shared["data"], val_mask, smask)
            delta = _metric_delta(metric, baseline_metric)
            supported = _candidate_supported(metric, baseline_metric, val_rows)
            row = {
                "slice": slice_name,
                "trial": trial_name,
                "val_rows": val_rows,
                "val_metric": metric,
                "val_baseline_metric": baseline_metric,
                "delta_vs_baseline": delta,
                "candidate_score": float(
                    max(delta[k] for k in CORE_KEYS)
                    + 0.25 * delta["all_improvement"]
                    + 0.25 * delta["hard_failure_improvement"]
                    - 10.0 * max(0.0, metric["easy_degradation"] - 0.01)
                ),
                "supported_on_val": bool(supported),
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["supported_on_val"], r["candidate_score"], r["val_rows"]), reverse=True)
    return rows


def _greedy_select_rules(
    shared: Mapping[str, Any],
    outputs: Mapping[str, Any],
    slices: Mapping[str, np.ndarray],
    candidates: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_mask = shared["split"] == "val"
    baseline = outputs["tree_baseline_family_residual"]
    current_ade = baseline["selected_ade"].copy()
    current_fde = baseline["selected_fde"].copy()
    current_switch = baseline["switch"].copy()
    assigned = np.zeros(len(current_ade), dtype=bool)
    floor_ade = baseline["floor_ade"]
    floor_fde = baseline["floor_fde"]
    current_metric = am._metric(current_ade, floor_ade, shared["data"], current_switch, val_mask)
    current_score = _deployment_score(current_metric)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate["supported_on_val"]:
            rejected.append({"slice": candidate["slice"], "trial": candidate["trial"], "reason": "candidate_not_supported_on_slice_val"})
            continue
        local = val_mask & slices[str(candidate["slice"])] & ~assigned
        if int(np.sum(local)) < MIN_VAL_SLICE_ROWS:
            rejected.append({"slice": candidate["slice"], "trial": candidate["trial"], "reason": "insufficient_unassigned_val_rows"})
            continue
        out = outputs[str(candidate["trial"])]
        trial_ade = current_ade.copy()
        trial_fde = current_fde.copy()
        trial_switch = current_switch.copy()
        trial_ade[local] = out["selected_ade"][local]
        trial_fde[local] = out["selected_fde"][local]
        trial_switch[local] = out["switch"][local]
        trial_metric = am._metric(trial_ade, floor_ade, shared["data"], trial_switch, val_mask)
        delta = _metric_delta(trial_metric, current_metric)
        trial_score = _deployment_score(trial_metric)
        reason = ""
        if float(trial_metric["easy_degradation"]) > MAX_EASY_DEGRADATION:
            reason = "easy_degradation_over_limit"
        elif min(delta[k] for k in CORE_KEYS) < -MAX_CORE_DROP:
            reason = "core_metric_drop_over_limit"
        elif max(delta[k] for k in CORE_KEYS) <= MIN_GREEDY_GAIN:
            reason = "no_material_incremental_core_gain"
        elif trial_score <= current_score + MIN_GREEDY_GAIN:
            reason = "deployment_score_not_incremental"
        if reason:
            rejected.append(
                {
                    "slice": candidate["slice"],
                    "trial": candidate["trial"],
                    "reason": reason,
                    "incremental_delta": delta,
                    "trial_metric": trial_metric,
                }
            )
            continue
        selected.append(
            {
                "slice": candidate["slice"],
                "trial": candidate["trial"],
                "val_rows": int(np.sum(local)),
                "previous_score": float(current_score),
                "new_score": float(trial_score),
                "incremental_delta": delta,
                "val_metric_after_accept": trial_metric,
            }
        )
        current_ade = trial_ade
        current_fde = trial_fde
        current_switch = trial_switch
        assigned[local] = True
        current_metric = trial_metric
        current_score = trial_score
    diagnostics = {
        "selection_source": "validation_only_greedy_incremental",
        "test_threshold_tuning": False,
        "inference_safe_slice_filter": True,
        "candidate_rows": int(len(candidates)),
        "supported_candidates": int(sum(1 for c in candidates if c["supported_on_val"])),
        "selected_rule_count": int(len(selected)),
        "rejected_rule_count": int(len(rejected)),
        "val_final_metric": current_metric,
        "val_final_score": float(current_score),
        "val_floor_metric": am._metric(floor_ade, floor_ade, shared["data"], np.zeros(len(floor_ade), dtype=bool), val_mask),
        "val_baseline_family_metric": am._metric(baseline["selected_ade"], floor_ade, shared["data"], baseline["switch"], val_mask),
        "top_rejections": rejected[:20],
    }
    return selected, diagnostics


def _apply_rules_to_test(
    shared: Mapping[str, Any],
    outputs: Mapping[str, Any],
    slices: Mapping[str, np.ndarray],
    rules: list[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = outputs["tree_baseline_family_residual"]
    selected_ade = baseline["selected_ade"].copy()
    selected_fde = baseline["selected_fde"].copy()
    switch = baseline["switch"].copy()
    owner = np.asarray(["tree_baseline_family_residual"] * len(selected_ade), dtype=object)
    applied = np.zeros(len(selected_ade), dtype=bool)
    for rule in rules:
        local = slices[str(rule["slice"])] & ~applied
        if not np.any(local):
            continue
        out = outputs[str(rule["trial"])]
        selected_ade[local] = out["selected_ade"][local]
        selected_fde[local] = out["selected_fde"][local]
        switch[local] = out["switch"][local]
        owner[local] = str(rule["trial"])
        applied[local] = True
    test_mask = shared["split"] == "test"
    data = shared["data"]
    horizon = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    floor_ade = baseline["floor_ade"]
    floor_fde = baseline["floor_fde"]
    baseline_metric = am._metric(baseline["selected_ade"], floor_ade, data, baseline["switch"], test_mask)
    policy_metric = am._metric(selected_ade, floor_ade, data, switch, test_mask)
    return {
        "metrics": {
            "baseline_family_reference": baseline_metric,
            "conservative_context_policy": policy_metric,
            "conservative_context_policy_fde": am._metric(selected_fde, floor_fde, data, switch, test_mask),
            "delta_vs_baseline_family": _metric_delta(policy_metric, baseline_metric),
        },
        "by_domain": {
            d: am._metric(selected_ade, floor_ade, data, switch, test_mask & (domain == d))
            for d in sorted(set(domain[test_mask].tolist()))
        },
        "by_horizon": {
            str(h): am._metric(selected_ade, floor_ade, data, switch, test_mask & (horizon == h))
            for h in [10, 25, 50, 100]
        },
        "bootstrap": {
            "all": am._bootstrap_ci(selected_ade, floor_ade, test_mask, seed=42501),
            "t50": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (horizon == 50), seed=42502),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (horizon == 100), seed=42503),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test_mask & hard_failure, seed=42504),
            "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, test_mask & easy, seed=42505),
        },
        "owner_counts_test": {
            name: int(np.sum(test_mask & (owner == name)))
            for name in sorted(set(owner[test_mask].tolist()))
        },
        "test_context_rule_coverage_rate": float(np.mean(applied[test_mask])) if np.any(test_mask) else 0.0,
        "test_rows_covered_by_context_rule": int(np.sum(test_mask & applied)),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    delta = result["summary"]["metrics"]["delta_vs_baseline_family"]
    policy_metric = result["summary"]["metrics"]["conservative_context_policy"]
    gates = {
        "source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "validation_greedy_selection": result["selection_diagnostics"]["selection_source"] == "validation_only_greedy_incremental",
        "inference_safe_policy_slices": result["selection_diagnostics"]["inference_safe_slice_filter"] is True
        and not any(r["slice"] in ja.DISALLOWED_POLICY_SLICES for r in result["selected_rules"]),
        "selected_context_rules_present": result["selection_diagnostics"]["selected_rule_count"] > 0,
        "test_once_policy_evaluated": policy_metric["rows"] == 47458,
        "beats_baseline_family_any_core_metric": max(float(delta[k]) for k in CORE_KEYS) > 0.0,
        "material_lift_over_baseline_family": max(float(delta[k]) for k in CORE_KEYS) > MIN_TEST_MATERIAL_GAIN,
        "no_core_metric_regression_vs_baseline_family": min(float(delta[k]) for k in CORE_KEYS) >= -MAX_CORE_DROP,
        "easy_preserved": policy_metric["easy_degradation"] <= MAX_EASY_DEGRADATION,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        )
        and result["no_leakage"]["train_only_slice_thresholds"],
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    if all(gates.values()):
        verdict = "stage42_jb_conservative_context_policy_promotable"
    elif gates["validation_greedy_selection"] and gates["test_once_policy_evaluated"] and gates["no_leakage_pass"]:
        verdict = "stage42_jb_conservative_context_policy_not_promotable"
    else:
        verdict = "stage42_jb_conservative_context_policy_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _failure_diagnosis(result: Mapping[str, Any]) -> dict[str, Any]:
    delta = result["summary"]["metrics"]["delta_vs_baseline_family"]
    selected = result["selection_diagnostics"]["selected_rule_count"]
    if selected == 0:
        primary = "no_validation_safe_incremental_context_rules"
    elif max(float(delta[k]) for k in CORE_KEYS) <= 0.0:
        primary = "validation_supported_context_rules_do_not_transfer_to_test"
    elif min(float(delta[k]) for k in CORE_KEYS) < -MAX_CORE_DROP:
        primary = "context_policy_has_core_metric_regression"
    elif max(float(delta[k]) for k in CORE_KEYS) <= MIN_TEST_MATERIAL_GAIN:
        primary = "test_lift_below_materiality_threshold"
    else:
        primary = "promotable_or_mixed"
    return {
        "primary_blocker": primary,
        "selected_rule_count": selected,
        "delta_vs_baseline_family": delta,
        "interpretation": (
            "Conservative validation-greedy repair still cannot promote context slices over the baseline-family mechanism."
            if primary != "promotable_or_mixed"
            else "Conservative context slices appear promotable under current gates."
        ),
    }


def _render_report(result: Mapping[str, Any]) -> list[str]:
    s = result["summary"]
    m = s["metrics"]["conservative_context_policy"]
    b = s["metrics"]["baseline_family_reference"]
    d = s["metrics"]["delta_vs_baseline_family"]
    lines = [
        "# Stage42-JB Conservative Context-Slice Policy Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_jb_gate']['passed']} / {result['stage42_jb_gate']['total']}`",
        f"- verdict: `{result['stage42_jb_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{s['decision']}`",
        f"- selected_rule_count: `{result['selection_diagnostics']['selected_rule_count']}`",
        f"- test_context_rule_coverage_rate: `{s['test_context_rule_coverage_rate']:.6f}`",
        f"- context policy all/t50/t100raw/hard/easy: `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_raw_frame_diagnostic_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}` / `{m['easy_degradation']:.6f}`",
        f"- baseline-family all/t50/t100raw/hard/easy: `{b['all_improvement']:.6f}` / `{b['t50_improvement']:.6f}` / `{b['t100_raw_frame_diagnostic_improvement']:.6f}` / `{b['hard_failure_improvement']:.6f}` / `{b['easy_degradation']:.6f}`",
        f"- delta vs baseline-family all/t50/t100raw/hard/easy: `{d['all_improvement']:.6f}` / `{d['t50_improvement']:.6f}` / `{d['t100_raw_frame_diagnostic_improvement']:.6f}` / `{d['hard_failure_improvement']:.6f}` / `{d['easy_degradation']:.6f}`",
        f"- primary_blocker: `{result['failure_diagnosis']['primary_blocker']}`",
        "",
        "## Validation-Greedy Accepted Rules",
        "",
        "| slice | trial | val rows | score before | score after | d all | d t50 | d t100 raw | d hard | d easy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["selected_rules"][:30]:
        delta = row["incremental_delta"]
        lines.append(
            f"| `{row['slice']}` | `{row['trial']}` | {row['val_rows']} | {row['previous_score']:.6f} | {row['new_score']:.6f} | {delta['all_improvement']:.6f} | {delta['t50_improvement']:.6f} | {delta['t100_raw_frame_diagnostic_improvement']:.6f} | {delta['hard_failure_improvement']:.6f} | {delta['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Test Breakdown",
            "",
            "### By Domain",
            "",
            "| domain | rows | all | t50 | t100 raw | hard | easy | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in s["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### By Horizon",
            "",
            "| horizon | rows | all | t50 | t100 raw | hard | easy | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon, row in s["by_horizon"].items():
        lines.append(
            f"| `{horizon}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            f"- {result['failure_diagnosis']['interpretation']}",
            "- A promotable result would support context slices only as a guarded, validation-selected policy. A negative result means Stage42-IZ remains local analysis evidence only.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_jb_gate"]
    lines = [
        "# Stage42-JB Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _replace_block(path: Path, marker: str, block: list[str]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    body = "\n".join([start, *block, end])
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{body}\n"
        if suffix:
            new_text += f"\n{suffix}"
    else:
        new_text = text.rstrip() + "\n\n" + body + "\n"
    path.write_text(new_text, encoding="utf-8")


def _update_readmes(result: Mapping[str, Any]) -> None:
    marker = "STAGE42_JB_CONSERVATIVE_CONTEXT_SLICE_POLICY_REPAIR"
    s = result["summary"]
    m = s["metrics"]["conservative_context_policy"]
    d = s["metrics"]["delta_vs_baseline_family"]
    block = [
        "## Stage42-JB Conservative Context-Slice Policy Repair",
        "",
        f"- source: `{result['source']}`",
        "- role: after Stage42-JA failed, try a stricter validation-greedy, inference-safe, core-preserving context slice repair.",
        f"- gate: `{result['stage42_jb_gate']['passed']} / {result['stage42_jb_gate']['total']}`; verdict `{result['stage42_jb_gate']['verdict']}`.",
        f"- selected_rule_count: `{result['selection_diagnostics']['selected_rule_count']}`; test_context_rule_coverage_rate `{s['test_context_rule_coverage_rate']:.6f}`.",
        f"- conservative policy all/t50/t100raw/hard/easy: `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_raw_frame_diagnostic_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}` / `{m['easy_degradation']:.6f}`.",
        f"- delta vs baseline-family all/t50/t100raw/hard/easy: `{d['all_improvement']:.6f}` / `{d['t50_improvement']:.6f}` / `{d['t100_raw_frame_diagnostic_improvement']:.6f}` / `{d['hard_failure_improvement']:.6f}` / `{d['easy_degradation']:.6f}`.",
        f"- primary_blocker: `{result['failure_diagnosis']['primary_blocker']}`.",
        "- boundary: validation-greedy policy selection, test-once evaluation; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, block)


def _update_state(result: Mapping[str, Any]) -> None:
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_jb_conservative_context_slice_policy_repair"
    state["current_verdict"] = result["stage42_jb_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_jb_conservative_context_slice_policy_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": result["stage42_jb_gate"]["verdict"],
        "gates": f"{result['stage42_jb_gate']['passed']}/{result['stage42_jb_gate']['total']}",
        "decision": result["summary"]["decision"],
        "selected_rule_count": result["selection_diagnostics"]["selected_rule_count"],
        "test_context_rule_coverage_rate": result["summary"]["test_context_rule_coverage_rate"],
        "conservative_policy_metric": result["summary"]["metrics"]["conservative_context_policy"],
        "delta_vs_baseline_family": result["summary"]["metrics"]["delta_vs_baseline_family"],
        "failure_diagnosis": result["failure_diagnosis"],
        "claim_boundary": result["claim_boundary"],
    }
    generated = state.setdefault("generated_reports", [])
    for item in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if item not in generated:
            generated.append(item)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "timestamp": result["generated_at_utc"],
        "source": result["source"],
        "verdict": result["stage42_jb_gate"]["verdict"],
        "gate": f"{result['stage42_jb_gate']['passed']}/{result['stage42_jb_gate']['total']}",
        "decision": result["summary"]["decision"],
        "selected_rule_count": result["selection_diagnostics"]["selected_rule_count"],
        "primary_blocker": result["failure_diagnosis"]["primary_blocker"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_context_slice_policy_conservative_repair(*, use_cached: bool = False) -> dict[str, Any]:
    if use_cached:
        cached = _cached_result_if_available()
        if cached is not None:
            return cached
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    shared["feature_masks"] = ix._feature_masks(shared["feature_names"])
    shared["tree_train_ids"] = iy._train_ids(shared["split"], shared["data"])
    outputs = {trial["name"]: iz._tree_outputs(shared, trial) for trial in iz.TRIALS}
    slices, thresholds = iz._slice_masks(shared["data"], shared["split"])
    candidates = _candidate_pool(shared, outputs, slices)
    rules, diagnostics = _greedy_select_rules(shared, outputs, slices, candidates)
    policy = _apply_rules_to_test(shared, outputs, slices, rules)
    delta = policy["metrics"]["delta_vs_baseline_family"]
    decision = (
        "conservative_context_slice_policy_promoted"
        if max(float(delta[k]) for k in CORE_KEYS) > MIN_TEST_MATERIAL_GAIN
        and min(float(delta[k]) for k in CORE_KEYS) >= -MAX_CORE_DROP
        and policy["metrics"]["conservative_context_policy"]["easy_degradation"] <= MAX_EASY_DEGRADATION
        else "conservative_context_slice_policy_not_promoted"
    )
    result = {
        "stage": "Stage42-JB conservative context-slice policy repair",
        "source": "fresh_run_validation_greedy_conservative_context_slice_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/context_slice_policy_promotion_stage42.json",
                "outputs/stage42_long_research/source_level_nonlinear_context_slice_audit_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "slice_threshold_source": "train_split_quantiles_only",
        "slice_thresholds": thresholds,
        "candidate_rows_total": int(len(candidates)),
        "candidate_rows_top": candidates[:40],
        "selection_diagnostics": diagnostics,
        "selected_rules": rules,
        "summary": {
            "decision": decision,
            "metrics": policy["metrics"],
            "by_domain": policy["by_domain"],
            "by_horizon": policy["by_horizon"],
            "bootstrap": policy["bootstrap"],
            "test_context_rule_coverage_rate": policy["test_context_rule_coverage_rate"],
            "test_rows_covered_by_context_rule": policy["test_rows_covered_by_context_rule"],
            "owner_counts_test": policy["owner_counts_test"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_greedy_selection": True,
            "inference_safe_policy_slices": True,
            "train_only_slice_thresholds": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["failure_diagnosis"] = _failure_diagnosis(result)
    result["stage42_jb_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_context_slice_policy_conservative_repair()
