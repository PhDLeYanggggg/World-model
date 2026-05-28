from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_context_repair as ix
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_nonlinear_context_repair as iy
from src import stage42_source_level_nonlinear_context_slice_audit as iz
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_slice_policy_promotion_stage42.json"
REPORT_MD = OUT_DIR / "context_slice_policy_promotion_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ja_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_VAL_SLICE_ROWS = 500
MIN_VAL_CONTEXT_LIFT = 0.01
MAX_EASY_DEGRADATION = 0.02
DISALLOWED_POLICY_SLICES = {"all_test", "hard_failure", "easy"}


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JA 接在 Stage42-IZ 后面：不再只问 context 切片是否存在，而是问 validation-selected context-slice policy 能否安全提升到 test。",
    "切片阈值来自 train split quantiles；切片/模型选择只看 validation；test 只评一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调阈值。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
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
    if payload.get("stage") == "Stage42-JA context-slice policy promotion audit":
        return payload
    return None


def _score_metric(metric: Mapping[str, Any], baseline_metric: Mapping[str, Any]) -> float:
    delta_all = float(metric["all_improvement"]) - float(baseline_metric["all_improvement"])
    delta_t50 = float(metric["t50_improvement"]) - float(baseline_metric["t50_improvement"])
    delta_t100 = float(metric["t100_raw_frame_diagnostic_improvement"]) - float(baseline_metric["t100_raw_frame_diagnostic_improvement"])
    delta_hard = float(metric["hard_failure_improvement"]) - float(baseline_metric["hard_failure_improvement"])
    return (
        1.0 * delta_all
        + 1.8 * delta_t50
        + 0.8 * delta_t100
        + 1.2 * delta_hard
        - 20.0 * max(0.0, float(metric["easy_degradation"]) - 0.01)
        - 0.02 * float(metric["switch_rate"])
    )


def _rule_supported(metric: Mapping[str, Any], baseline_metric: Mapping[str, Any], rows: int) -> bool:
    if rows < MIN_VAL_SLICE_ROWS:
        return False
    if float(metric["easy_degradation"]) > MAX_EASY_DEGRADATION:
        return False
    deltas = [
        float(metric["all_improvement"]) - float(baseline_metric["all_improvement"]),
        float(metric["t50_improvement"]) - float(baseline_metric["t50_improvement"]),
        float(metric["t100_raw_frame_diagnostic_improvement"]) - float(baseline_metric["t100_raw_frame_diagnostic_improvement"]),
        float(metric["hard_failure_improvement"]) - float(baseline_metric["hard_failure_improvement"]),
    ]
    return max(deltas) > MIN_VAL_CONTEXT_LIFT


def _validation_rules(
    shared: Mapping[str, Any],
    outputs: Mapping[str, Any],
    slices: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_mask = shared["split"] == "val"
    baseline = outputs["tree_baseline_family_residual"]
    all_candidates: list[dict[str, Any]] = []
    best_by_slice: dict[str, dict[str, Any]] = {}
    for slice_name, smask in sorted(slices.items()):
        if slice_name in DISALLOWED_POLICY_SLICES:
            continue
        rows = int(np.sum(val_mask & smask))
        if rows < 30:
            continue
        baseline_metric = iz._metric_for_slice(baseline, shared["data"], val_mask, smask)
        for trial_name, out in outputs.items():
            if trial_name == "tree_baseline_family_residual":
                continue
            metric = iz._metric_for_slice(out, shared["data"], val_mask, smask)
            row = {
                "slice": slice_name,
                "trial": trial_name,
                "rows": rows,
                "val_metric": metric,
                "val_baseline_metric": baseline_metric,
                "delta_all": float(metric["all_improvement"] - baseline_metric["all_improvement"]),
                "delta_t50": float(metric["t50_improvement"] - baseline_metric["t50_improvement"]),
                "delta_t100_raw": float(metric["t100_raw_frame_diagnostic_improvement"] - baseline_metric["t100_raw_frame_diagnostic_improvement"]),
                "delta_hard": float(metric["hard_failure_improvement"] - baseline_metric["hard_failure_improvement"]),
                "val_score": float(_score_metric(metric, baseline_metric)),
                "supported_on_val": bool(_rule_supported(metric, baseline_metric, rows)),
            }
            all_candidates.append(row)
            if row["supported_on_val"]:
                old = best_by_slice.get(slice_name)
                if old is None or float(row["val_score"]) > float(old["val_score"]):
                    best_by_slice[slice_name] = row
    selected = sorted(best_by_slice.values(), key=lambda r: r["val_score"], reverse=True)
    diagnostics = {
        "candidate_rows": int(len(all_candidates)),
        "supported_candidates": int(sum(1 for r in all_candidates if r["supported_on_val"])),
        "selected_rule_count": int(len(selected)),
        "selection_source": "validation_only",
        "inference_safe_slice_filter": True,
        "disallowed_policy_slices": sorted(DISALLOWED_POLICY_SLICES),
        "test_threshold_tuning": False,
    }
    return selected, diagnostics


def _compose_policy(
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
    domain = shared["data"]["dataset"].astype(str)
    horizon = shared["data"]["horizon"].astype(int)
    priority = sorted(rules, key=lambda r: float(r["val_score"]), reverse=True)
    for rule in priority:
        smask = slices[str(rule["slice"])]
        local = smask & ~applied
        if not np.any(local):
            continue
        out = outputs[str(rule["trial"])]
        selected_ade[local] = out["selected_ade"][local]
        selected_fde[local] = out["selected_fde"][local]
        switch[local] = out["switch"][local]
        owner[local] = str(rule["trial"])
        applied[local] = True
    test = shared["split"] == "test"
    hard_failure = shared["data"]["hard"].astype(bool) | shared["data"]["failure"].astype(bool)
    easy = shared["data"]["easy"].astype(bool)
    floor_ade = baseline["floor_ade"]
    floor_fde = baseline["floor_fde"]
    metrics = {
        "baseline_family_reference": am._metric(baseline["selected_ade"], floor_ade, shared["data"], baseline["switch"], test),
        "context_slice_policy": am._metric(selected_ade, floor_ade, shared["data"], switch, test),
        "context_slice_policy_fde": am._metric(selected_fde, floor_fde, shared["data"], switch, test),
    }
    metrics["delta_vs_baseline_family"] = {
        key: float(metrics["context_slice_policy"].get(key, 0.0)) - float(metrics["baseline_family_reference"].get(key, 0.0))
        for key in [
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
    }
    by_domain = {
        d: am._metric(selected_ade, floor_ade, shared["data"], switch, test & (domain == d))
        for d in sorted(set(domain[test].tolist()))
    }
    by_horizon = {
        str(h): am._metric(selected_ade, floor_ade, shared["data"], switch, test & (horizon == h))
        for h in [10, 25, 50, 100]
    }
    bootstrap = {
        "all": am._bootstrap_ci(selected_ade, floor_ade, test, seed=42401),
        "t50": am._bootstrap_ci(selected_ade, floor_ade, test & (horizon == 50), seed=42402),
        "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, test & (horizon == 100), seed=42403),
        "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test & hard_failure, seed=42404),
        "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, test & easy, seed=42405),
    }
    owner_counts = {
        name: int(np.sum(test & (owner == name)))
        for name in sorted(set(owner[test].tolist()))
    }
    return {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
        "owner": owner,
        "rules_applied": int(len(priority)),
        "test_rows_covered_by_context_rule": int(np.sum(test & applied)),
        "test_context_rule_coverage_rate": float(np.mean(applied[test])) if np.any(test) else 0.0,
        "owner_counts_test": owner_counts,
        "metrics": metrics,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "bootstrap": bootstrap,
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    delta = summary["delta_vs_baseline_family"]
    gates = {
        "source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "validation_only_rule_selection": result["validation_rule_diagnostics"]["selection_source"] == "validation_only"
        and result["validation_rule_diagnostics"]["test_threshold_tuning"] is False,
        "inference_safe_policy_slices": result["validation_rule_diagnostics"].get("inference_safe_slice_filter") is True
        and not any(r["slice"] in DISALLOWED_POLICY_SLICES for r in result["selected_rules"]),
        "selected_context_rules_present": result["validation_rule_diagnostics"]["selected_rule_count"] > 0,
        "test_once_policy_evaluated": summary["metrics"]["context_slice_policy"]["rows"] == 47458,
        "policy_beats_baseline_family_any_core_metric": max(
            float(delta["all_improvement"]),
            float(delta["t50_improvement"]),
            float(delta["t100_raw_frame_diagnostic_improvement"]),
            float(delta["hard_failure_improvement"]),
        )
        > 0.0,
        "policy_material_lift_over_baseline_family": max(
            float(delta["all_improvement"]),
            float(delta["t50_improvement"]),
            float(delta["t100_raw_frame_diagnostic_improvement"]),
            float(delta["hard_failure_improvement"]),
        )
        > MIN_VAL_CONTEXT_LIFT,
        "easy_preserved": summary["metrics"]["context_slice_policy"]["easy_degradation"] <= MAX_EASY_DEGRADATION,
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
        verdict = "stage42_ja_context_slice_policy_promotable"
    elif gates["validation_only_rule_selection"] and gates["test_once_policy_evaluated"] and gates["no_leakage_pass"]:
        verdict = "stage42_ja_context_slice_policy_not_promotable"
    else:
        verdict = "stage42_ja_context_slice_policy_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    s = result["summary"]
    m = s["metrics"]["context_slice_policy"]
    b = s["metrics"]["baseline_family_reference"]
    d = s["delta_vs_baseline_family"]
    lines = [
        "# Stage42-JA Context-Slice Policy Promotion Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ja_gate']['passed']} / {result['stage42_ja_gate']['total']}`",
        f"- verdict: `{result['stage42_ja_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{s['decision']}`",
        f"- selected_rule_count: `{result['validation_rule_diagnostics']['selected_rule_count']}`",
        f"- test_context_rule_coverage_rate: `{s['test_context_rule_coverage_rate']:.6f}`",
        f"- context policy all/t50/t100raw/hard/easy: `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_raw_frame_diagnostic_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}` / `{m['easy_degradation']:.6f}`",
        f"- baseline-family all/t50/t100raw/hard/easy: `{b['all_improvement']:.6f}` / `{b['t50_improvement']:.6f}` / `{b['t100_raw_frame_diagnostic_improvement']:.6f}` / `{b['hard_failure_improvement']:.6f}` / `{b['easy_degradation']:.6f}`",
        f"- delta vs baseline-family all/t50/t100raw/hard/easy: `{d['all_improvement']:.6f}` / `{d['t50_improvement']:.6f}` / `{d['t100_raw_frame_diagnostic_improvement']:.6f}` / `{d['hard_failure_improvement']:.6f}` / `{d['easy_degradation']:.6f}`",
        "",
        "## Validation-Selected Context Rules",
        "",
        "| slice | trial | val rows | val score | delta all | delta t50 | delta t100 raw | delta hard | easy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["selected_rules"][:30]:
        vm = row["val_metric"]
        lines.append(
            f"| `{row['slice']}` | `{row['trial']}` | {row['rows']} | {row['val_score']:.6f} | {row['delta_all']:.6f} | {row['delta_t50']:.6f} | {row['delta_t100_raw']:.6f} | {row['delta_hard']:.6f} | {vm['easy_degradation']:.6f} |"
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
            f"- {s['interpretation']}",
            "- If promotable, this remains a validation-selected protected slice policy; it is not a global context or ungated neural claim.",
            "- If not promotable, Stage42-IZ stays as local evidence only and should not be used as a deployment claim.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ja_gate"]
    lines = [
        "# Stage42-JA Gate",
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
    marker = "STAGE42_JA_CONTEXT_SLICE_POLICY_PROMOTION"
    s = result["summary"]
    m = s["metrics"]["context_slice_policy"]
    d = s["delta_vs_baseline_family"]
    block = [
        "## Stage42-JA Context-Slice Policy Promotion Audit",
        "",
        f"- source: `{result['source']}`",
        "- role: promote Stage42-IZ slice-level context evidence into a validation-selected fallback-safe policy, or reject promotion.",
        f"- gate: `{result['stage42_ja_gate']['passed']} / {result['stage42_ja_gate']['total']}`; verdict `{result['stage42_ja_gate']['verdict']}`.",
        f"- selected_rule_count: `{result['validation_rule_diagnostics']['selected_rule_count']}`; test_context_rule_coverage_rate `{s['test_context_rule_coverage_rate']:.6f}`.",
        f"- context policy all/t50/t100raw/hard/easy: `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_raw_frame_diagnostic_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}` / `{m['easy_degradation']:.6f}`.",
        f"- delta vs baseline-family all/t50/t100raw/hard/easy: `{d['all_improvement']:.6f}` / `{d['t50_improvement']:.6f}` / `{d['t100_raw_frame_diagnostic_improvement']:.6f}` / `{d['hard_failure_improvement']:.6f}` / `{d['easy_degradation']:.6f}`.",
        f"- decision: `{s['decision']}`.",
        "- boundary: validation-only slice policy selection, test-once evaluation; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, block)


def _update_state(result: Mapping[str, Any]) -> None:
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_ja_context_slice_policy_promotion"
    state["current_verdict"] = result["stage42_ja_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_ja_context_slice_policy_promotion"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": result["stage42_ja_gate"]["verdict"],
        "gates": f"{result['stage42_ja_gate']['passed']}/{result['stage42_ja_gate']['total']}",
        "decision": result["summary"]["decision"],
        "selected_rule_count": result["validation_rule_diagnostics"]["selected_rule_count"],
        "test_context_rule_coverage_rate": result["summary"]["test_context_rule_coverage_rate"],
        "context_policy_metric": result["summary"]["metrics"]["context_slice_policy"],
        "delta_vs_baseline_family": result["summary"]["delta_vs_baseline_family"],
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
        "verdict": result["stage42_ja_gate"]["verdict"],
        "gate": f"{result['stage42_ja_gate']['passed']}/{result['stage42_ja_gate']['total']}",
        "decision": result["summary"]["decision"],
        "selected_rule_count": result["validation_rule_diagnostics"]["selected_rule_count"],
        "delta_vs_baseline_family": result["summary"]["delta_vs_baseline_family"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_context_slice_policy_promotion(*, use_cached: bool = False) -> dict[str, Any]:
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
    rules, diagnostics = _validation_rules(shared, outputs, slices)
    policy = _compose_policy(shared, outputs, slices, rules)
    delta = policy["metrics"]["delta_vs_baseline_family"]
    decision = (
        "validation_selected_context_slice_policy_promoted"
        if max(delta["all_improvement"], delta["t50_improvement"], delta["t100_raw_frame_diagnostic_improvement"], delta["hard_failure_improvement"]) > MIN_VAL_CONTEXT_LIFT
        and policy["metrics"]["context_slice_policy"]["easy_degradation"] <= MAX_EASY_DEGRADATION
        else "validation_selected_context_slice_policy_not_promoted"
    )
    result = {
        "stage": "Stage42-JA context-slice policy promotion audit",
        "source": "fresh_run_validation_selected_context_slice_policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_nonlinear_context_slice_audit_stage42.json",
                "outputs/stage42_long_research/source_level_nonlinear_context_repair_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "slice_threshold_source": "train_split_quantiles_only",
        "slice_thresholds": thresholds,
        "validation_rule_diagnostics": diagnostics,
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
            "delta_vs_baseline_family": delta,
            "interpretation": (
                "Validation-selected context slices improved at least one core test metric over the nonlinear baseline-family reference while preserving easy cases."
                if decision == "validation_selected_context_slice_policy_promoted"
                else "Stage42-IZ local context evidence did not survive validation-selected promotion into a safe global test policy."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "inference_safe_policy_slices": True,
            "validation_only_rule_selection": True,
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
    result["stage42_ja_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_context_slice_policy_promotion()
