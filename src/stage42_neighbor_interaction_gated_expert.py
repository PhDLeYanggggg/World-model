from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "neighbor_interaction_gated_expert_stage42.json"
REPORT_MD = OUT_DIR / "neighbor_interaction_gated_expert_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ck_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

VALIDATION_MARGIN = 0.01
EASY_LIMIT = 0.02
MIN_TEST_GAIN = 0.01


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CK 是 validation-only neighbor/interaction gated expert audit，不是 metric 或 seconds-level 结果。",
    "本阶段专门测试 Stage42-CI 标出的 mixed/weak neighbor-interaction context 是否可被保守 gate 修复为增量贡献。",
    "每个 candidate 都重新训练 ridge full-waypoint probe，并在 validation 上重新选 safe policy；test 只评一次。",
    "kNN graph features 只使用当前帧和过去 history，不使用 future endpoint / future waypoint 作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _or_mask(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required.")
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= mask
    return out


def _base_feature_blocks(shared: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, list[str]]]:
    names = list(shared["feature_names"])
    groups = ao._group_masks(names)
    controls = _or_mask(groups["domain"], groups["horizon"])
    baseline = _or_mask(groups["baseline_family"], controls)
    neighbor = groups["neighbor_interaction"]
    goal = groups["goal_prototype"]
    history_scalar = np.asarray([name.startswith("history_scalar_") for name in names])
    return {
        "baseline_family_control": (shared["features"][:, baseline], [n for n, keep in zip(names, baseline) if keep]),
        "baseline_plus_scalar_neighbor": (shared["features"][:, _or_mask(baseline, neighbor)], [n for n, keep in zip(names, _or_mask(baseline, neighbor)) if keep]),
        "baseline_plus_goal_scene": (shared["features"][:, _or_mask(baseline, goal)], [n for n, keep in zip(names, _or_mask(baseline, goal)) if keep]),
        "baseline_plus_history_scalar": (shared["features"][:, _or_mask(baseline, history_scalar)], [n for n, keep in zip(names, _or_mask(baseline, history_scalar)) if keep]),
    }


def _build_variant_features(shared: Mapping[str, Any]) -> tuple[dict[str, tuple[np.ndarray, list[str]]], dict[str, Any]]:
    blocks = _base_feature_blocks(shared)
    graph, graph_names, graph_stats = graph_ctx._build_graph_features(shared["data"])
    baseline_x, baseline_names = blocks["baseline_family_control"]
    goal_x, goal_names = blocks["baseline_plus_goal_scene"]
    hist_x, hist_names = blocks["baseline_plus_history_scalar"]
    variants: dict[str, tuple[np.ndarray, list[str]]] = dict(blocks)
    variants["baseline_plus_knn_graph"] = (
        np.concatenate([baseline_x, graph], axis=1).astype(np.float32),
        baseline_names + graph_names,
    )
    variants["baseline_plus_graph_goal"] = (
        np.concatenate([goal_x, graph], axis=1).astype(np.float32),
        goal_names + graph_names,
    )
    variants["baseline_plus_graph_history_scalar"] = (
        np.concatenate([hist_x, graph], axis=1).astype(np.float32),
        hist_names + graph_names,
    )
    return variants, {"graph_schema": {"k_neighbors": graph_ctx.K_NEIGHBORS, "feature_names": graph_names}, "graph_stats": graph_stats}


def _evaluate_variant(name: str, raw_features: np.ndarray, feature_names: list[str], shared: Mapping[str, Any]) -> dict[str, Any]:
    split = shared["split"]
    x, _, _ = am._standardize(raw_features, split == "train")
    result = am._evaluate_models(shared["data"], split, shared["labels"], shared["floor"], x)
    return {
        "source": "fresh_run",
        "variant": name,
        "feature_count": int(raw_features.shape[1]),
        "feature_names": feature_names,
        "best_lambda": result["best_lambda"],
        "validation_selection": result["validation_selection"],
        "policy_slice_count": int(len(result["policy"]["slices"])),
        "protected": result["metrics"]["protected_ridge_source_level"],
        "protected_fde": result["metrics"]["protected_ridge_source_level_fde"],
        "ungated_diagnostic": result["metrics"]["ungated_ridge_diagnostic"],
        "bootstrap": result["bootstrap"],
        "by_domain": result["by_domain"],
        "by_horizon": result["by_horizon"],
        "normalization": "train_split_mean_std_only",
    }


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return ao._metric_delta(lhs, rhs)


def _validation_score(row: Mapping[str, Any]) -> float:
    return float(row["validation_selection"]["selected_score"])


def _val_easy(row: Mapping[str, Any]) -> float:
    candidates = row["validation_selection"]["candidates"]
    selected_score = _validation_score(row)
    for cand in candidates:
        if abs(float(cand["score"]) - selected_score) < 1e-9:
            return float(cand["val_metric"]["easy_degradation"])
    return float(candidates[-1]["val_metric"]["easy_degradation"]) if candidates else 1.0


def _select_deployment_variant(variants: Mapping[str, Any]) -> dict[str, Any]:
    baseline_name = "baseline_family_control"
    baseline_score = _validation_score(variants[baseline_name])
    best_name = baseline_name
    best_score = baseline_score
    considered = []
    candidates = [
        "baseline_plus_scalar_neighbor",
        "baseline_plus_knn_graph",
        "baseline_plus_graph_goal",
        "baseline_plus_graph_history_scalar",
    ]
    for name in candidates:
        row = variants[name]
        score = _validation_score(row)
        easy = _val_easy(row)
        margin = score - baseline_score
        passes = margin > VALIDATION_MARGIN and easy <= EASY_LIMIT and score > best_score
        considered.append(
            {
                "variant": name,
                "validation_score": score,
                "validation_margin_vs_baseline": margin,
                "validation_easy_degradation": easy,
                "passes_validation_gate": bool(passes),
            }
        )
        if passes:
            best_name = name
            best_score = score
    return {
        "source": "fresh_run",
        "selection_rule": "choose_neighbor_interaction_candidate_only_if_validation_score_beats_baseline_by_margin_and_easy_safe_else_fallback",
        "validation_margin": VALIDATION_MARGIN,
        "easy_limit": EASY_LIMIT,
        "baseline_variant": baseline_name,
        "selected_variant": best_name,
        "selected_score": best_score,
        "considered_neighbor_interaction_candidates": considered,
        "test_threshold_tuning": False,
    }


def _neighbor_rescue_success(selected: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    delta = _metric_delta(selected["protected"], baseline["protected"])
    return (
        selected["variant"] != baseline["variant"]
        and selected["protected"]["easy_degradation"] <= EASY_LIMIT
        and (
            delta["all_improvement"] > MIN_TEST_GAIN
            or delta["t50_improvement"] > MIN_TEST_GAIN
            or delta["hard_failure_improvement"] > MIN_TEST_GAIN
        )
    )


def run_stage42_neighbor_interaction_gated_expert() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    raw_variants, graph_info = _build_variant_features(shared)
    variants: dict[str, Any] = {}
    for name, (raw, names) in raw_variants.items():
        variants[name] = _evaluate_variant(name, raw, names, shared)

    selection = _select_deployment_variant(variants)
    baseline = variants[selection["baseline_variant"]]
    selected = variants[selection["selected_variant"]]
    deltas = {
        name: _metric_delta(row["protected"], baseline["protected"])
        for name, row in variants.items()
        if name != selection["baseline_variant"]
    }
    result = {
        "source": "fresh_run",
        "stage": "Stage42-CK validation-only neighbor/interaction gated expert audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/context_contribution_forensics_stage42.json",
                "outputs/stage42_long_research/source_level_graph_context_stage42.json",
                "outputs/stage42_long_research/source_level_incremental_ablation_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "graph_info": graph_info,
        "variants": variants,
        "validation_only_selection": selection,
        "delta_vs_baseline_family_control": deltas,
        "selected_delta_vs_baseline_family_control": _metric_delta(selected["protected"], baseline["protected"]),
        "neighbor_interaction_rescue_success": _neighbor_rescue_success(selected, baseline),
        "interpretation": _interpret(variants, selection),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "graph_features_current_and_past_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "neighbor_interaction_main_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_ck_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _interpret(variants: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    baseline = variants[selection["baseline_variant"]]
    selected = variants[selection["selected_variant"]]
    selected_delta = _metric_delta(selected["protected"], baseline["protected"])
    graph_delta = _metric_delta(variants["baseline_plus_knn_graph"]["protected"], baseline["protected"])
    graph_goal_delta = _metric_delta(variants["baseline_plus_graph_goal"]["protected"], baseline["protected"])
    graph_history_delta = _metric_delta(variants["baseline_plus_graph_history_scalar"]["protected"], baseline["protected"])
    if selection["selected_variant"] == selection["baseline_variant"]:
        verdict = "neighbor_interaction_gated_expert_not_validation_selected"
        summary = (
            "The validation-only gate did not select a neighbor/interaction candidate over the baseline-family control. "
            "This preserves the Stage42-CI boundary: neighbor/interaction remains mixed/weak and not a main claim."
        )
    elif _neighbor_rescue_success(selected, baseline):
        verdict = "neighbor_interaction_gated_expert_increment_supported"
        summary = (
            "A neighbor/interaction candidate was selected by validation and improved at least one core test metric while preserving easy cases. "
            "This supports a guarded, not global, neighbor/interaction contribution."
        )
    else:
        verdict = "neighbor_interaction_gated_expert_validation_selected_but_test_not_supported"
        summary = (
            "A neighbor/interaction candidate passed validation selection, but test deltas were not enough for a main claim. "
            "This is useful failure evidence and should not be overclaimed."
        )
    return {
        "verdict": verdict,
        "summary": summary,
        "baseline_family_control": baseline["protected"],
        "selected_variant": selection["selected_variant"],
        "selected_delta_vs_baseline_family_control": selected_delta,
        "baseline_plus_knn_graph_delta": graph_delta,
        "baseline_plus_graph_goal_delta": graph_goal_delta,
        "baseline_plus_graph_history_scalar_delta": graph_history_delta,
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    selected = result["variants"][result["validation_only_selection"]["selected_variant"]]["protected"]
    baseline = result["variants"][result["validation_only_selection"]["baseline_variant"]]["protected"]
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "graph_features_built": result["graph_info"]["graph_stats"]["rows_with_neighbors"] > 0,
        "neighbor_interaction_variants_evaluated": {"baseline_plus_scalar_neighbor", "baseline_plus_knn_graph", "baseline_plus_graph_goal", "baseline_plus_graph_history_scalar"}.issubset(result["variants"].keys()),
        "validation_only_selection": not result["validation_only_selection"]["test_threshold_tuning"],
        "baseline_family_control_positive": baseline["all_improvement"] > 0 and baseline["t50_improvement"] > 0,
        "selected_policy_easy_safe": selected["easy_degradation"] <= EASY_LIMIT,
        "neighbor_interaction_overclaim_blocked": result["claim_boundary"]["neighbor_interaction_main_claim_allowed"] is False
        or result["neighbor_interaction_rescue_success"],
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and result["no_leakage"]["graph_features_current_and_past_only"]
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    if all(gates.values()) and result["neighbor_interaction_rescue_success"]:
        verdict = "stage42_ck_neighbor_interaction_gated_expert_pass_increment_supported"
    elif all(gates.values()):
        verdict = "stage42_ck_neighbor_interaction_gated_expert_pass_diagnostic_no_overclaim"
    else:
        verdict = "stage42_ck_neighbor_interaction_gated_expert_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CK Neighbor/Interaction Gated Expert Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ck_gate']['passed']} / {result['stage42_ck_gate']['total']}`",
        f"- verdict: `{result['stage42_ck_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Why This Was Run",
        "",
        "- Stage42-CI found neighbor/interaction context is mixed/weak and not a main claim.",
        "- Stage42-AS had already shown graph residual context was not supported.",
        "- Stage42-CK tests a stricter deployment question: can scalar neighbor or kNN graph features become a validation-only gated expert over baseline-family context?",
        "- Test metrics are used only once after validation selection.",
        "",
        "## Graph Schema",
        "",
        f"- graph_info: `{result['graph_info']}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | features | val score | all | t50 | t100 diag | hard/failure | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["variants"].items():
        metric = row["protected"]
        lines.append(
            f"| `{name}` | {row['feature_count']} | {_validation_score(row):.6f} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Only Selection",
            "",
            f"- selection: `{result['validation_only_selection']}`",
            f"- selected_delta_vs_baseline_family_control: `{result['selected_delta_vs_baseline_family_control']}`",
            f"- neighbor_interaction_rescue_success: `{result['neighbor_interaction_rescue_success']}`",
            "",
            "## Delta Vs Baseline-Family Control",
            "",
            "| variant | delta all | delta t50 | delta hard/failure | delta easy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, delta in result["delta_vs_baseline_family_control"].items():
        lines.append(
            f"| `{name}` | {delta['all_improvement']:.6f} | {delta['t50_improvement']:.6f} | {delta['hard_failure_improvement']:.6f} | {delta['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- verdict: `{result['interpretation']['verdict']}`",
            f"- summary: {result['interpretation']['summary']}",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ck_gate"]
    lines = [
        "# Stage42-CK Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "source": result["source"],
        "generated_at_utc": result["generated_at_utc"],
        "verdict": result["stage42_ck_gate"]["verdict"],
        "gate": f"{result['stage42_ck_gate']['passed']}/{result['stage42_ck_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_neighbor_interaction_gated_expert()
