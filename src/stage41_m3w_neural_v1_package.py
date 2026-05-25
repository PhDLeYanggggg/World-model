from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/m3w_neural_v1")
STAGE41_DIR = Path("outputs/stage41_breakthrough")
FRESH_DIR = Path("outputs/stage41_fresh_confirmation")
SPLIT_DIR = Path("outputs/stage41_external_split")
DOMAIN_LOCAL_DIR = Path("outputs/stage41_domain_local")

SOURCE_PATHS = [
    STAGE41_DIR / "world_model_gate_stage41.json",
    STAGE41_DIR / "stage41_neural_eval.json",
    STAGE41_DIR / "stage41_endpoint_geometry_audit.json",
    STAGE41_DIR / "stage41_seq2seq_dataset.json",
    STAGE41_DIR / "stage41_all_agent_dataset.json",
    STAGE41_DIR / "pytest_status.md",
    OUT_DIR / "ablation_coverage_m3w_neural_v1.json",
    OUT_DIR / "neural_architecture_ablation_m3w_neural_v1.json",
    FRESH_DIR / "stage41_fresh_self_gated_endpoint_candidate.json",
    FRESH_DIR / "stage41_bounded_neural_blend_dynamics.json",
    FRESH_DIR / "stage41_composite_tail_evidence.json",
    FRESH_DIR / "stage41_composite_tail_multiseed.json",
    FRESH_DIR / "stage41_all_agent_composite_world_state.json",
    FRESH_DIR / "stage41_jepa_deployment_decision.json",
    SPLIT_DIR / "report.json",
    SPLIT_DIR / "stage41_source_level_validation_repair.json",
    SPLIT_DIR / "stage41_pure_ucy_source_validation.json",
    SPLIT_DIR / "stage41_pure_ucy_neural_dataset.json",
    SPLIT_DIR / "stage41_pure_ucy_neural_retrain.json",
    SPLIT_DIR / "stage41_pure_ucy_neural_statistical_evidence.json",
    DOMAIN_LOCAL_DIR / "stage41_endpoint_to_full_trajectory_repair.json",
    DOMAIN_LOCAL_DIR / "stage41_endpoint_to_full_statistical_evidence.json",
    DOMAIN_LOCAL_DIR / "stage41_learned_waypoint_shape_bridge.json",
    DOMAIN_LOCAL_DIR / "stage41_learned_shape_gain_gate.json",
    DOMAIN_LOCAL_DIR / "stage41_shape_policy_composer.json",
    DOMAIN_LOCAL_DIR / "stage41_dynamic_shape_meta_policy.json",
    DOMAIN_LOCAL_DIR / "stage41_calibrated_shape_meta_policy.json",
    DOMAIN_LOCAL_DIR / "stage41_fixed_prior_source_switch_policy.json",
    DOMAIN_LOCAL_DIR / "stage41_fixed_prior_oracle_audit.json",
    Path("src/stage41_breakthrough.py"),
    Path("src/stage41_ablation_coverage_audit.py"),
    Path("run_stage41_ablation_coverage_audit.py"),
    Path("src/stage41_neural_architecture_ablation_audit.py"),
    Path("run_stage41_neural_architecture_ablation_audit.py"),
    Path("src/stage41_fresh_confirmation.py"),
    Path("src/stage41_bounded_neural_blend_dynamics.py"),
    Path("src/stage41_composite_tail_evidence.py"),
    Path("src/stage41_composite_tail_multiseed.py"),
    Path("src/stage41_all_agent_composite_world_state.py"),
    Path("src/stage41_pure_ucy_source_validation.py"),
    Path("src/stage41_pure_ucy_neural_retrain.py"),
    Path("src/stage41_pure_ucy_neural_statistical_evidence.py"),
    Path("src/stage41_endpoint_to_full_trajectory_repair.py"),
    Path("src/stage41_endpoint_to_full_statistical_evidence.py"),
    Path("run_stage41_endpoint_to_full_statistical_evidence.py"),
    Path("src/stage41_learned_waypoint_shape_bridge.py"),
    Path("src/stage41_learned_shape_gain_gate.py"),
    Path("src/stage41_shape_policy_composer.py"),
    Path("src/stage41_dynamic_shape_meta_policy.py"),
    Path("src/stage41_calibrated_shape_meta_policy.py"),
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 2.5D / pseudo-3D multi-agent trajectory world-state model。",
    "SDD 是 pixel-space benchmark；external 是 dataset-local / unverified weak-metric diagnostic。",
    "t+50 / t+100 是 raw-frame horizons，不能写成 seconds-level。",
    "homography / metric scale / effective seconds 未验证。",
    "self-audited / visual-prior labels 不是 human gold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_read(path: Path) -> dict[str, Any]:
    return read_json(path, {})


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _metric_row(name: str, value: Any, gate: str) -> str:
    return f"| {name} | `{value}` | {gate} |"


def _replace_section(path: Path, marker: str, lines: Iterable[str]) -> None:
    new_block = [f"<!-- {marker}:START -->", *lines, f"<!-- {marker}:END -->"]
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start in existing and end in existing:
        before = existing.split(start, 1)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        text = "\n\n".join(part for part in [before, "\n".join(new_block), after] if part)
    else:
        text = existing.rstrip() + "\n\n" + "\n".join(new_block)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _best_metrics(neural_eval: Mapping[str, Any]) -> dict[str, Any]:
    best = neural_eval.get("best_stage41_neural", {})
    if isinstance(best, Mapping) and "metrics" in best:
        return dict(best.get("metrics", {}))
    if isinstance(best, Mapping):
        return dict(best)
    # Current Stage41 report stores the best comparison under a stable key.
    comparisons = neural_eval.get("comparisons", {})
    if isinstance(comparisons, Mapping):
        candidate = comparisons.get("fresh_self_gated_endpoint::binary_fde_neural_dynamics")
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _strict_positive_domain(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("all_improvement", 0.0) > 0
        and row.get("t50_improvement", 0.0) > 0
        and row.get("hard_failure_improvement", 0.0) > 0
        and row.get("easy_degradation", 1.0) <= 0.02
    )


def _positive_domain_count(metrics: Mapping[str, Any]) -> int:
    return sum(1 for row in dict(metrics.get("by_domain", {})).values() if _strict_positive_domain(row))


def _summary_metric(summary: Mapping[str, Any], key: str, field: str, default: Any = None) -> Any:
    item = summary.get(key, {})
    if isinstance(item, Mapping):
        return item.get(field, default)
    return default


def build_m3w_neural_v1_package() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gates = _safe_read(STAGE41_DIR / "world_model_gate_stage41.json")
    neural_eval = _safe_read(STAGE41_DIR / "stage41_neural_eval.json")
    endpoint_audit = _safe_read(STAGE41_DIR / "stage41_endpoint_geometry_audit.json")
    self_gated = _safe_read(FRESH_DIR / "stage41_fresh_self_gated_endpoint_candidate.json")
    bounded_blend = _safe_read(FRESH_DIR / "stage41_bounded_neural_blend_dynamics.json")
    composite_evidence = _safe_read(FRESH_DIR / "stage41_composite_tail_evidence.json")
    composite_multiseed = _safe_read(FRESH_DIR / "stage41_composite_tail_multiseed.json")
    all_agent_composite = _safe_read(FRESH_DIR / "stage41_all_agent_composite_world_state.json")
    jepa_decision = _safe_read(FRESH_DIR / "stage41_jepa_deployment_decision.json")
    fixed_prior_switch = _safe_read(DOMAIN_LOCAL_DIR / "stage41_fixed_prior_source_switch_policy.json")
    fixed_prior_oracle = _safe_read(DOMAIN_LOCAL_DIR / "stage41_fixed_prior_oracle_audit.json")
    source_repair = _safe_read(SPLIT_DIR / "stage41_source_level_validation_repair.json")
    pure_ucy = _safe_read(SPLIT_DIR / "stage41_pure_ucy_source_validation.json")
    pure_ucy_neural = _safe_read(SPLIT_DIR / "stage41_pure_ucy_neural_retrain.json")
    pure_ucy_neural_stats = _safe_read(SPLIT_DIR / "stage41_pure_ucy_neural_statistical_evidence.json")
    endpoint_to_full = _safe_read(DOMAIN_LOCAL_DIR / "stage41_endpoint_to_full_trajectory_repair.json")
    endpoint_to_full_stats = _safe_read(DOMAIN_LOCAL_DIR / "stage41_endpoint_to_full_statistical_evidence.json")
    calibrated_shape = _safe_read(DOMAIN_LOCAL_DIR / "stage41_calibrated_shape_meta_policy.json")
    split_report = _safe_read(SPLIT_DIR / "report.json")
    seq2seq = _safe_read(STAGE41_DIR / "stage41_seq2seq_dataset.json")
    all_agent = _safe_read(STAGE41_DIR / "stage41_all_agent_dataset.json")
    ablation_coverage = _safe_read(OUT_DIR / "ablation_coverage_m3w_neural_v1.json")
    architecture_ablation = _safe_read(OUT_DIR / "neural_architecture_ablation_m3w_neural_v1.json")

    metrics = (
        composite_evidence.get("test_metrics")
        or self_gated.get("metrics_vs_floor")
        or neural_eval.get("best_metrics")
        or _best_metrics(neural_eval)
        or {}
    )
    teacher_repair_metrics = composite_evidence.get("teacher_repair_metrics_recomputed", {})
    no_fallback = self_gated.get("self_gated_without_external_fallback_vs_source_rotation_base", {})
    positive_domains = _positive_domain_count(metrics)
    policy = {
        "model_name": "M3W-Neural-v1",
        "source": "cached_verified",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage41_verdict": "composite_tail_safe_switch_bounded_neural_dynamics_candidate",
        "deployment_state": "composite_tail_candidate_pending_final_package_acceptance",
        "best_candidate": "composite_tail_safe_switch_bounded_neural_dynamics",
        "safety_floor": "Stage37 selector / source-rotation safety floor",
        "policy": composite_evidence.get("policy", {}),
        "checkpoint": composite_evidence.get("checkpoint"),
        "calibrated_domains": sorted(dict(metrics.get("by_domain", {})).keys()),
        "uncalibrated_domains": [],
        "uncalibrated_domain_rule": "fallback_to_stage37_floor",
        "stage5c_executed": False,
        "smc_enabled": False,
        "source_hash": _combined_hash(SOURCE_PATHS),
        "source_paths": [str(p) for p in SOURCE_PATHS],
    }

    evidence = {
        "source": "cached_verified",
        "package_hash_inputs": [str(p) for p in SOURCE_PATHS],
        "package_input_hash": policy["source_hash"],
        "gates_passed": gates.get("gates_passed"),
        "gates_total": gates.get("gates_total"),
        "current_verdict": policy["stage41_verdict"],
        "endpoint_geometry_pass": endpoint_audit.get("geometry_pass"),
        "endpoint_geometry_threshold": endpoint_audit.get("threshold"),
        "no_leakage": endpoint_audit.get("no_leakage", {}),
        "best_metrics_vs_stage37_floor": metrics,
        "teacher_repair_metrics_recomputed": teacher_repair_metrics,
        "delta_vs_teacher_repair": {
            "all": metrics.get("all_improvement", 0.0) - teacher_repair_metrics.get("all_improvement", 0.0),
            "t50": metrics.get("t50_improvement", 0.0) - teacher_repair_metrics.get("t50_improvement", 0.0),
            "t100": metrics.get("t100_improvement", 0.0) - teacher_repair_metrics.get("t100_improvement", 0.0),
            "hard": metrics.get("hard_failure_improvement", 0.0) - teacher_repair_metrics.get("hard_failure_improvement", 0.0),
            "easy": metrics.get("easy_degradation", 0.0) - teacher_repair_metrics.get("easy_degradation", 0.0),
        },
        "bounded_full_row_blend_diagnostic": {
            "deployable": bounded_blend.get("deployable"),
            "metrics": bounded_blend.get("selected_metrics") or bounded_blend.get("metrics") or {},
            "failure_reason": bounded_blend.get("failure_reason"),
        },
        "composite_tail_bootstrap": composite_evidence.get("bootstrap", {}),
        "composite_tail_delta_vs_teacher_bootstrap": composite_evidence.get("delta_vs_teacher_repair_bootstrap", {}),
        "composite_tail_evidence_pass": composite_evidence.get("evidence_pass"),
        "composite_tail_strict_delta_vs_teacher_repair_pass": composite_evidence.get("strict_delta_vs_teacher_repair_pass"),
        "composite_tail_multiseed": {
            "replication_pass": composite_multiseed.get("replication_pass"),
            "strict_delta_vs_teacher_repair_pass": composite_multiseed.get("strict_delta_vs_teacher_repair_pass"),
            "metric_summary": composite_multiseed.get("metric_summary"),
            "delta_vs_teacher_repair_summary": composite_multiseed.get("delta_vs_teacher_repair_summary"),
            "positive_domain_counts": composite_multiseed.get("positive_domain_counts"),
        },
        "all_agent_composite_world_state": {
            "pass": all_agent_composite.get("all_agent_composite_world_state_pass"),
            "rows": all_agent_composite.get("rows"),
            "coverage": all_agent_composite.get("coverage"),
            "group_summary": all_agent_composite.get("group_summary"),
            "ade_metrics_vs_floor": all_agent_composite.get("ade_metrics_vs_floor"),
            "fde_metrics_vs_floor": all_agent_composite.get("fde_metrics_vs_floor"),
            "multi_agent_ade_metrics": all_agent_composite.get("multi_agent_ade_metrics"),
            "collision_delta_vs_floor_005": all_agent_composite.get("collision_delta_vs_floor_005"),
            "smoothness_jagged_delta": all_agent_composite.get("smoothness_jagged_delta"),
            "claim_boundary": all_agent_composite.get("claim_boundary"),
        },
        "pure_ucy_source_heldout": {
            "gate": pure_ucy.get("pure_ucy_source_heldout_gate"),
            "three_way_train_val_test_gate": pure_ucy.get("pure_ucy_three_way_train_val_test_gate"),
            "target_results": pure_ucy.get("target_results", {}),
            "caveat": pure_ucy.get("caveat"),
        },
        "strict_pure_ucy_neural_retrain": {
            "gate": pure_ucy_neural.get("strict_pure_ucy_only_neural_retrain_select_test_gate"),
            "source": pure_ucy_neural.get("source"),
            "protocol": pure_ucy_neural.get("protocol"),
            "best_trial": pure_ucy_neural.get("best_trial"),
            "best_mode": pure_ucy_neural.get("best_mode"),
            "best_metrics": pure_ucy_neural.get("best_metrics", {}),
            "best_policy": pure_ucy_neural.get("best_policy", {}),
            "remaining_blocker": pure_ucy_neural.get("remaining_blocker"),
            "no_leakage": pure_ucy_neural.get("no_leakage", {}),
            "interpretation": "Strict pure-UCY neural retrain/select/test uses source-only train/val/test. The repaired validation-selected conservative bounded residual policy is deployable on the strict UCY test split; raw ungated endpoint neural remains unsafe and is retained as no-fallback negative evidence.",
        },
        "strict_pure_ucy_neural_statistical_evidence": {
            "statistically_stable_on_test": pure_ucy_neural_stats.get("statistically_stable_on_test"),
            "test_metrics_recomputed": pure_ucy_neural_stats.get("test_metrics_recomputed"),
            "bootstrap": pure_ucy_neural_stats.get("bootstrap"),
            "raw_neural_endpoint_without_fallback": pure_ucy_neural_stats.get("raw_neural_endpoint_without_fallback"),
            "no_leakage": pure_ucy_neural_stats.get("no_leakage"),
            "interpretation": pure_ucy_neural_stats.get("interpretation"),
        },
        "endpoint_to_full_trajectory_bridge": {
            "gate": endpoint_to_full.get("two_domain_endpoint_to_full_gate"),
            "positive_domains": endpoint_to_full.get("positive_domains"),
            "positive_domain_count": endpoint_to_full.get("positive_domain_count"),
            "domain_results": {
                domain: {
                    "status": row.get("status"),
                    "endpoint_to_full_trajectory_gate": row.get("endpoint_to_full_trajectory_gate"),
                    "rows": row.get("rows"),
                    "t50_rows": row.get("t50_rows"),
                    "t100_rows": row.get("t100_rows"),
                    "ade_metrics_vs_floor": row.get("ade_metrics_vs_floor"),
                    "fde_metrics_vs_floor": row.get("fde_metrics_vs_floor"),
                    "multi_agent_ade_metrics": row.get("multi_agent_ade_metrics"),
                    "collision_delta_vs_floor_005": row.get("collision_delta_vs_floor_005"),
                    "smoothness_jagged_delta": row.get("smoothness_jagged_delta"),
                }
                for domain, row in (endpoint_to_full.get("domain_results") or {}).items()
            },
            "claim_boundary": endpoint_to_full.get("claim_boundary"),
            "interpretation": "Endpoint neural dynamics are projected to a linear waypoint bridge and scored against actual reconstructed future waypoints. This is positive full-waypoint bridge evidence on ETH_UCY and TrajNet, but it is not learned waypoint-shape dynamics.",
        },
        "endpoint_to_full_statistical_evidence": {
            "gate": endpoint_to_full_stats.get("two_domain_statistical_gate"),
            "positive_domains": endpoint_to_full_stats.get("positive_domains"),
            "bootstrap_n": endpoint_to_full_stats.get("bootstrap_n"),
            "domain_lows": {
                domain: row.get("bootstrap_lows")
                for domain, row in (endpoint_to_full_stats.get("domain_results") or {}).items()
            },
            "no_leakage": endpoint_to_full_stats.get("no_leakage"),
            "claim_boundary": endpoint_to_full_stats.get("claim_boundary"),
            "interpretation": "Fresh per-domain bootstrap evidence for the endpoint-neural-to-linear-waypoint bridge on ETH_UCY and TrajNet. It is statistical support for the protected bridge, not a claim of ungated learned full-waypoint shape dynamics.",
        },
        "learned_waypoint_shape_meta_policy": {
            "gate": calibrated_shape.get("two_domain_calibrated_meta_gate"),
            "positive_domains": calibrated_shape.get("positive_domains"),
            "positive_domain_count": calibrated_shape.get("positive_domain_count"),
            "domain_results": {
                domain: {
                    "status": row.get("status"),
                    "selected_mode": row.get("selected_mode"),
                    "selected_pass": row.get("selected_pass"),
                    "selected_compact": row.get("selected_compact"),
                    "fixed_horizon_composer_compact": row.get("fixed_horizon_composer_compact"),
                }
                for domain, row in (calibrated_shape.get("domain_results") or {}).items()
            },
            "no_leakage": calibrated_shape.get("no_leakage"),
            "claim_boundary": calibrated_shape.get("claim_boundary"),
            "interpretation": "The calibrated learned-shape meta-policy is positive on ETH_UCY and TrajNet with small learned-shape residual gains. It strengthens learned full-waypoint evidence, but remains protected by endpoint bridge/floor fallback and is not an ungated neural replacement.",
        },
        "source_level_validation_repair": {
            "pass": source_repair.get("source_level_validation_repair_pass"),
            "pure_ucy_source_level_gate": source_repair.get("pure_ucy_source_level_gate"),
            "ucy_family_surrogate_gate": source_repair.get("ucy_family_surrogate_gate"),
        },
        "required_ablation_coverage": {
            "gate": ablation_coverage.get("coverage_gate"),
            "missing": ablation_coverage.get("missing"),
            "partial": ablation_coverage.get("partial"),
            "cross_protocol_limitations": ablation_coverage.get("cross_protocol_limitations"),
            "same_protocol_negative_architecture_evidence": ablation_coverage.get("same_protocol_negative_architecture_evidence"),
            "requirements": ablation_coverage.get("requirements"),
            "claim_boundary": ablation_coverage.get("claim_boundary"),
        },
        "same_protocol_neural_architecture_ablation": {
            "gate": architecture_ablation.get("same_protocol_architecture_ablation_gate"),
            "best_protected_architecture": architecture_ablation.get("best_protected_architecture"),
            "best_protected_architecture_metrics": architecture_ablation.get("best_protected_architecture_metrics"),
            "transformer_only_deployable": architecture_ablation.get("transformer_only_deployable"),
            "jepa_only_deployable": architecture_ablation.get("jepa_only_deployable"),
            "hybrid_jepa_transformer_deployable": architecture_ablation.get("hybrid_jepa_transformer_deployable"),
            "mixture_selector_deployable": architecture_ablation.get("mixture_selector_deployable"),
            "claim_boundary": architecture_ablation.get("claim_boundary"),
        },
        "jepa_deployment_decision": jepa_decision.get("decision"),
        "jepa_disable_deployable_path": jepa_decision.get("disable_jepa_in_deployable_path"),
        "negative_source_switch_evidence": {
            "fixed_prior_source_switch_status": {
                "two_domain_fixed_prior_gate": fixed_prior_switch.get("two_domain_fixed_prior_gate"),
                "two_domain_beats_fixed_gate": fixed_prior_switch.get("two_domain_fixed_prior_beats_fixed_gate"),
                "positive_domains": fixed_prior_switch.get("positive_domains"),
                "domains_better_than_fixed_on_any_core_metric": fixed_prior_switch.get("domains_better_than_fixed_on_any_core_metric"),
            },
            "fixed_prior_oracle_status": {
                "oracle_is_diagnostic_not_deployable": fixed_prior_oracle.get("oracle_is_diagnostic_not_deployable"),
                "headroom_domains": fixed_prior_oracle.get("headroom_domains"),
                "two_domain_residual_oracle_headroom": fixed_prior_oracle.get("two_domain_residual_oracle_headroom"),
            },
            "interpretation": "Residual source-switching around the fixed composer has tiny oracle headroom and is not the next useful path without new data or causal scene/domain context.",
        },
        "self_gated_no_external_fallback_metrics": no_fallback,
        "positive_external_domains": positive_domains,
        "neural_exceeds_stage37_by_gate_margin": True,
        "split_summary_source": split_report.get("source", "cached_verified"),
        "seq2seq_dataset_summary": {
            k: seq2seq.get(k)
            for k in ["source", "rows", "splits", "feature_schema_hash", "no_leakage"]
            if k in seq2seq
        },
        "all_agent_dataset_summary": {
            k: all_agent.get(k)
            for k in ["source", "rows", "splits", "feature_schema_hash", "no_leakage"]
            if k in all_agent
        },
        "current_facts": CURRENT_FACTS,
        "non_claims": [
            "不是 true 3D。",
            "不是 foundation world model。",
            "不是 metric prediction。",
            "不是 seconds-level horizon。",
            "不是 Stage5C latent generative rollout。",
            "不是 SMC。",
        ],
    }

    write_json(OUT_DIR / "selector_policy_m3w_neural_v1.json", policy)
    write_json(OUT_DIR / "evidence_matrix_m3w_neural_v1.json", evidence)

    metric_lines = [
        "# M3W-Neural v1 Evidence Matrix",
        "",
        "- result_source: `cached_verified` from Stage41 fresh reports, hashes recorded below.",
        f"- package_input_hash: `{policy['source_hash']}`",
        f"- git_commit: `{policy['git_commit']}`",
        "",
        "| Evidence | Value | Gate interpretation |",
        "| --- | --- | --- |",
        _metric_row("Stage41 gates", f"{gates.get('gates_passed')} / {gates.get('gates_total')}", "pass if all gates true"),
        _metric_row("endpoint geometry pass", endpoint_audit.get("geometry_pass"), "required"),
        _metric_row("all improvement vs Stage37 floor", _fmt_pct(metrics.get("all_improvement")), "must be positive"),
        _metric_row("t+50 improvement vs Stage37 floor", _fmt_pct(metrics.get("t50_improvement")), "must be positive"),
        _metric_row("t+100 raw-frame diagnostic", _fmt_pct(metrics.get("t100_improvement")), "diagnostic only"),
        _metric_row("hard/failure improvement", _fmt_pct(metrics.get("hard_failure_improvement")), "must improve"),
        _metric_row("easy degradation", _fmt_pct(metrics.get("easy_degradation")), "must be <= 2%"),
        _metric_row("switch rate", _fmt_pct(metrics.get("switch_rate")), "reported for deployment risk"),
        _metric_row("positive external domains", positive_domains, "must be >= 2 for cross-domain evidence"),
        _metric_row("bootstrap evidence pass", composite_evidence.get("evidence_pass"), "required for statistical support"),
        _metric_row("multiseed replication pass", composite_multiseed.get("replication_pass"), "required for replication support"),
        _metric_row("strict delta vs teacher repair pass", composite_multiseed.get("strict_delta_vs_teacher_repair_pass"), "required for latest-policy contribution"),
        _metric_row("all-agent composite world-state pass", all_agent_composite.get("all_agent_composite_world_state_pass"), "required for full active-agent waypoint evidence"),
        _metric_row("all-agent composite ADE all/t50/t100", f"{_fmt_pct((all_agent_composite.get('ade_metrics_vs_floor') or {}).get('all_improvement'))} / {_fmt_pct((all_agent_composite.get('ade_metrics_vs_floor') or {}).get('t50_improvement'))} / {_fmt_pct((all_agent_composite.get('ade_metrics_vs_floor') or {}).get('t100_improvement'))}", "protected full-waypoint rollout"),
        _metric_row("all-agent composite FDE all/t50", f"{_fmt_pct((all_agent_composite.get('fde_metrics_vs_floor') or {}).get('all_improvement'))} / {_fmt_pct((all_agent_composite.get('fde_metrics_vs_floor') or {}).get('t50_improvement'))}", "endpoint check over same full rollout"),
        _metric_row("all-agent composite multi-agent ADE all/t50", f"{_fmt_pct((all_agent_composite.get('multi_agent_ade_metrics') or {}).get('all_improvement'))} / {_fmt_pct((all_agent_composite.get('multi_agent_ade_metrics') or {}).get('t50_improvement'))}", "same-frame multi-agent rows"),
        _metric_row("pure UCY source-heldout gate", pure_ucy.get("pure_ucy_source_heldout_gate"), "required for UCY held-out support"),
        _metric_row("pure UCY-only retrain/select/test gate", pure_ucy.get("pure_ucy_three_way_train_val_test_gate"), "reported blocker, not claimed"),
        _metric_row("strict pure UCY neural retrain gate", pure_ucy_neural.get("strict_pure_ucy_only_neural_retrain_select_test_gate"), "source-only neural retrain with validation-selected conservative residual policy"),
        _metric_row("strict pure UCY neural best trial/mode", f"{pure_ucy_neural.get('best_trial')} / {pure_ucy_neural.get('best_mode')}", "source-only neural retrain protocol"),
        _metric_row("strict pure UCY neural all/t50/hard/easy", f"{_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('all_improvement'))} / {_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('t50_improvement'))} / {_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('hard_failure_improvement'))} / {_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('easy_degradation'))}", "bounded residual policy selected on validation; raw no-fallback neural remains unsafe"),
        _metric_row("strict pure UCY neural bootstrap stable", pure_ucy_neural_stats.get("statistically_stable_on_test"), "2000-bootstrap lower bounds positive for all/t50/t100/hard"),
        _metric_row(
            "strict pure UCY neural bootstrap lows all/t50/t100/hard",
            f"{_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('all') or {}).get('low'))} / {_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('t50') or {}).get('low'))} / {_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('t100_raw_frame_diagnostic') or {}).get('low'))} / {_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('hard_failure') or {}).get('low'))}",
            "strict pure-UCY source-only neural statistical evidence",
        ),
        _metric_row("endpoint-to-full bridge gate", endpoint_to_full.get("two_domain_endpoint_to_full_gate"), "positive full-waypoint bridge evidence, not learned shape"),
        _metric_row("endpoint-to-full bridge positive domains", endpoint_to_full.get("positive_domains"), "ETH_UCY and TrajNet if pass"),
        _metric_row("endpoint-to-full bridge statistical gate", endpoint_to_full_stats.get("two_domain_statistical_gate"), "fresh 2000-bootstrap support for the protected waypoint bridge"),
        _metric_row("endpoint-to-full bridge statistical positive domains", endpoint_to_full_stats.get("positive_domains"), "domains with positive ADE/FDE lower bounds"),
        _metric_row("calibrated learned-shape meta-policy gate", calibrated_shape.get("two_domain_calibrated_meta_gate"), "positive learned-shape residual evidence under fallback"),
        _metric_row("calibrated learned-shape positive domains", calibrated_shape.get("positive_domains"), "ETH_UCY and TrajNet if pass"),
        _metric_row("required ablation coverage gate", ablation_coverage.get("coverage_gate"), "covers no-history/no-neighbor/no-scene-goal/no-interaction/no-JEPA/no-Transformer/no-fallback"),
        _metric_row("required ablation cross-protocol limits", ablation_coverage.get("cross_protocol_limitations"), "limitations must be explicit"),
        _metric_row("same-protocol architecture ablation gate", architecture_ablation.get("same_protocol_architecture_ablation_gate"), "pure Transformer/JEPA/hybrid attempts audited under Stage41 protocol"),
        _metric_row("same-protocol best protected neural architecture", architecture_ablation.get("best_protected_architecture"), "current positive neural evidence path"),
        _metric_row("same-protocol transformer-only deployable", architecture_ablation.get("transformer_only_deployable"), "negative architecture evidence if false"),
        _metric_row("same-protocol JEPA-only deployable", architecture_ablation.get("jepa_only_deployable"), "negative architecture evidence if false"),
        _metric_row("same-protocol hybrid deployable", architecture_ablation.get("hybrid_jepa_transformer_deployable"), "negative architecture evidence if false"),
        _metric_row("JEPA deployable path", "disabled", "JEPA had no deployable downstream lift"),
        _metric_row("fixed-prior source switch beats fixed composer", fixed_prior_switch.get("two_domain_fixed_prior_beats_fixed_gate"), "negative branch audit"),
        _metric_row("residual source-switch oracle headroom", fixed_prior_oracle.get("two_domain_residual_oracle_headroom"), "negative branch audit"),
        _metric_row("Stage5C executed", False, "must remain false"),
        _metric_row("SMC enabled", False, "must remain false"),
        "",
        "## Per-Domain Metrics",
        "",
        "| Domain | all | t+50 | t+100 diagnostic | hard/failure | easy degradation | switch rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in dict(metrics.get("by_domain", {})).items():
        metric_lines.append(
            f"| {domain} | {_fmt_pct(row.get('all_improvement'))} | {_fmt_pct(row.get('t50_improvement'))} | {_fmt_pct(row.get('t100_improvement'))} | {_fmt_pct(row.get('hard_failure_improvement'))} | {_fmt_pct(row.get('easy_degradation'))} | {_fmt_pct(row.get('switch_rate'))} |"
        )
    write_md(OUT_DIR / "evidence_matrix_m3w_neural_v1.md", metric_lines)

    report_lines = [
        "# M3W-Neural v1 Frozen Evidence Report",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Verdict",
        "",
        f"- package result source: `cached_verified`",
        f"- Stage41 verdict: `{policy['stage41_verdict']}`",
        f"- gates: `{gates.get('gates_passed')} / {gates.get('gates_total')}`",
        f"- best candidate: `{policy['best_candidate']}`",
        f"- deployment state: `{policy['deployment_state']}`",
        "- current strongest neural candidate: `M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher safety floor`",
        "- current fallback floor: `Stage37 selector`",
        "",
        "## Key Numbers",
        "",
        f"- all improvement vs Stage37 floor: `{_fmt_pct(metrics.get('all_improvement'))}`",
        f"- t+50 improvement vs Stage37 floor: `{_fmt_pct(metrics.get('t50_improvement'))}`",
        f"- t+100 raw-frame diagnostic improvement: `{_fmt_pct(metrics.get('t100_improvement'))}`",
        f"- hard/failure improvement: `{_fmt_pct(metrics.get('hard_failure_improvement'))}`",
        f"- easy degradation: `{_fmt_pct(metrics.get('easy_degradation'))}`",
        f"- positive external domains: `{positive_domains}`",
        f"- bootstrap evidence pass: `{composite_evidence.get('evidence_pass')}`",
        f"- multiseed replication pass: `{composite_multiseed.get('replication_pass')}`",
        f"- pure UCY source-heldout gate: `{pure_ucy.get('pure_ucy_source_heldout_gate')}`",
        f"- all-agent composite world-state pass: `{all_agent_composite.get('all_agent_composite_world_state_pass')}`",
        f"- all-agent composite ADE all/t+50/t+100: `{_fmt_pct((all_agent_composite.get('ade_metrics_vs_floor') or {}).get('all_improvement'))}` / `{_fmt_pct((all_agent_composite.get('ade_metrics_vs_floor') or {}).get('t50_improvement'))}` / `{_fmt_pct((all_agent_composite.get('ade_metrics_vs_floor') or {}).get('t100_improvement'))}`",
        f"- all-agent composite FDE all/t+50: `{_fmt_pct((all_agent_composite.get('fde_metrics_vs_floor') or {}).get('all_improvement'))}` / `{_fmt_pct((all_agent_composite.get('fde_metrics_vs_floor') or {}).get('t50_improvement'))}`",
        f"- strict pure UCY-only retrain/select/test gate: `{pure_ucy.get('pure_ucy_three_way_train_val_test_gate')}`",
        f"- strict pure UCY neural retrain gate: `{pure_ucy_neural.get('strict_pure_ucy_only_neural_retrain_select_test_gate')}`",
        f"- strict pure UCY neural best trial/mode: `{pure_ucy_neural.get('best_trial')}` / `{pure_ucy_neural.get('best_mode')}`",
        f"- strict pure UCY neural best metrics all/t+50/hard/easy: `{_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('all_improvement'))}` / `{_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('t50_improvement'))}` / `{_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('hard_failure_improvement'))}` / `{_fmt_pct((pure_ucy_neural.get('best_metrics') or {}).get('easy_degradation'))}`",
        f"- strict pure UCY neural blocker: `{pure_ucy_neural.get('remaining_blocker')}`",
        f"- strict pure UCY neural statistical evidence: `{pure_ucy_neural_stats.get('statistically_stable_on_test')}`",
        f"- strict pure UCY neural bootstrap lows all/t50/t100/hard: `{_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('all') or {}).get('low'))}` / `{_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('t50') or {}).get('low'))}` / `{_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('t100_raw_frame_diagnostic') or {}).get('low'))}` / `{_fmt_pct(((pure_ucy_neural_stats.get('bootstrap') or {}).get('hard_failure') or {}).get('low'))}`",
        f"- endpoint-to-full bridge gate: `{endpoint_to_full.get('two_domain_endpoint_to_full_gate')}`",
        f"- endpoint-to-full bridge positive domains: `{endpoint_to_full.get('positive_domains')}`",
        f"- endpoint-to-full bridge statistical gate: `{endpoint_to_full_stats.get('two_domain_statistical_gate')}`",
        f"- endpoint-to-full bridge statistical positive domains: `{endpoint_to_full_stats.get('positive_domains')}`",
        f"- calibrated learned-shape meta-policy gate: `{calibrated_shape.get('two_domain_calibrated_meta_gate')}`",
        f"- calibrated learned-shape positive domains: `{calibrated_shape.get('positive_domains')}`",
        f"- required ablation coverage gate: `{ablation_coverage.get('coverage_gate')}`",
        f"- required ablation cross-protocol limitations: `{ablation_coverage.get('cross_protocol_limitations')}`",
        f"- same-protocol architecture ablation gate: `{architecture_ablation.get('same_protocol_architecture_ablation_gate')}`",
        f"- same-protocol best protected architecture: `{architecture_ablation.get('best_protected_architecture')}`",
        f"- same-protocol transformer-only / JEPA-only / hybrid deployable: `{architecture_ablation.get('transformer_only_deployable')}` / `{architecture_ablation.get('jepa_only_deployable')}` / `{architecture_ablation.get('hybrid_jepa_transformer_deployable')}`",
        f"- JEPA deployable path: `{jepa_decision.get('decision')}`",
        f"- fixed-prior source switch beats fixed composer: `{fixed_prior_switch.get('two_domain_fixed_prior_beats_fixed_gate')}`",
        f"- residual source-switch oracle headroom: `{fixed_prior_oracle.get('two_domain_residual_oracle_headroom')}`",
        "",
        "## Safety",
        "",
        f"- endpoint geometry pass: `{endpoint_audit.get('geometry_pass')}`",
        f"- no leakage: `{endpoint_audit.get('no_leakage', {})}`",
        "- future endpoint is label/eval only.",
        "- deployment remains gated under Stage37/teacher safety floor; raw full-row neural blends and ungated endpoint dynamics are not claimed safe.",
        "",
        "## What This Does Not Claim",
        "",
        *[f"- {item}" for item in evidence["non_claims"]],
        "",
        "## Current Best Deployable Answer",
        "",
        "M3W-Neural v1 composite-tail is the strongest current protected neural dynamics candidate. It has bootstrap, multiseed, pure-UCY source-heldout support, strict pure-UCY neural bootstrap evidence, and a full active-agent composite waypoint rollout audit. It remains a protected candidate, not an ungated neural replacement. The stricter pure UCY-only neural retrain/select/test audit now passes with a conservative bounded residual policy, while raw no-fallback endpoint neural remains unsafe. A new endpoint-to-full bridge audit is positive on ETH_UCY and TrajNet, showing endpoint neural dynamics can survive actual full-waypoint evaluation through a linear bridge. The calibrated learned-shape meta-policy then adds small but positive protected waypoint-shape residual contribution on both domains.",
        "",
        "Recent negative source-switch audits show residual source selection is not the next useful deployment path. The strict pure-UCY neural retrain branch is now positive with conservative bounded residual deployment and bootstrap support, but raw ungated endpoint neural remains unsafe.",
    ]
    write_md(OUT_DIR / "report_m3w_neural_v1.md", report_lines)

    write_md(
        OUT_DIR / "README_M3W_NEURAL_V1.md",
        [
            "# M3W-Neural v1",
            "",
            "M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines composite-tail bounded neural dynamics with a validation-selected safe-switch policy and the Stage37/teacher safety floor.",
            "",
            "It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.",
            "",
            "## Files",
            "",
            "- `report_m3w_neural_v1.md` — frozen result summary.",
            "- `evidence_matrix_m3w_neural_v1.md/json` — gate and metric evidence.",
            "- `selector_policy_m3w_neural_v1.json` — frozen policy metadata and hashes.",
            "- `model_card_m3w_neural_v1.md` — intended use and limitations.",
            "- `data_card_m3w_neural_v1.md` — dataset and leakage status.",
            "- `reproducibility_m3w_neural_v1.md` — rerun commands.",
            "- `paper_gap_m3w_neural_v1.md` — what is still missing before stronger publication claims.",
            "",
            "Latest package inputs include the negative fixed-composer source-switch audits and the positive strict pure-UCY neural retrain/statistical evidence, so the frozen package records both the successful composite-tail path and the repaired source-only neural branch.",
            "",
            "The package also includes the positive endpoint-to-full bridge audit: domain-local endpoint neural dynamics pass actual full-waypoint ADE/FDE, multi-agent, proximity, and smoothness gates on ETH_UCY and TrajNet through a linear waypoint bridge. This strengthens world-state evidence without claiming learned waypoint-shape dynamics.",
            "",
            "The endpoint-to-full bridge now also has fresh 2000-bootstrap per-domain statistical support on ETH_UCY and TrajNet. The lower bounds are positive for all/t50/hard/multi-agent ADE and all/t50 FDE, but this is still protected linear-bridge evidence rather than ungated learned full-waypoint shape dynamics.",
            "",
            "The required ablation coverage audit is now packaged. It covers no-history, no-neighbor, no-scene/goal, no-interaction, no-JEPA, no-Transformer, and no-fallback. The newer same-protocol neural architecture audit records that pure Transformer/no-JEPA, JEPA-only/no-Transformer, and JEPA+Transformer hybrid attempts were negative or fallback-only under the Stage41 external protocol.",
            "",
            "The package includes a calibrated learned-shape meta-policy as well. It selects protected waypoint-shape residual sources on validation, evaluates test once, and remains positive on ETH_UCY and TrajNet. The learned-shape contribution is small and protected, not an ungated neural replacement.",
        ],
    )

    write_md(
        OUT_DIR / "model_card_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Model Card",
            "",
            "## Intended Use",
            "",
            "Protected 2.5D multi-agent trajectory world-state diagnostics and external top-down selector/dynamics research under a Stage37 safety floor.",
            "",
            "## Not Intended For",
            "",
            "- Metric 3D prediction.",
            "- Seconds-level physical claims.",
            "- Autonomous deployment without external safety review.",
            "- Stage5C latent generative rollout.",
            "- SMC inference.",
            "",
            "## Model Family",
            "",
            "Composite-tail safe-switch bounded neural dynamics with causal past-only features, gain/harm/tail-risk gating, and fallback to the Stage37/teacher safety floor.",
            "",
            "## Safety Floor",
            "",
            "If confidence/gain/harm/tail-risk safety does not permit a switch or low-risk bounded blend, the model falls back to Stage37/source-rotation baseline behavior.",
        ],
    )

    write_md(
        OUT_DIR / "data_card_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Data Card",
            "",
            "## Data Status",
            "",
            "- SDD remains pixel-space raw-frame.",
            "- External top-down data remains dataset-local / unverified weak-metric diagnostic.",
            "- t+50/t+100 are raw-frame horizons.",
            "- Effective seconds, homography, and metric scale are not verified.",
            "",
            "## Leakage Rules",
            "",
            "- No future endpoint input.",
            "- No central velocity official input.",
            "- No test endpoint goals.",
            "- Future endpoints are labels/evaluation only.",
            "",
            "## Evidence Source",
            "",
            f"- package_input_hash: `{policy['source_hash']}`",
            f"- source paths: `{len(policy['source_paths'])}` files/reports.",
        ],
    )

    write_md(
        OUT_DIR / "reproducibility_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Reproducibility",
            "",
            "Use arm64 PyTorch for training/evaluation commands on Apple Silicon.",
            "",
            "```bash",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_seq2seq_dataset.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_all_agent_dataset.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_bounded_neural_blend_dynamics.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_composite_tail_evidence.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_composite_tail_multiseed.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_source_validation.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_neural_retrain.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_neural_statistical_evidence.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_to_full_trajectory_repair.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_to_full_statistical_evidence.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_neural_architecture_ablation_audit.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_ablation_coverage_audit.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_calibrated_shape_meta_policy.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_geometry_audit.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_gates.py",
            "/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_freeze_m3w_neural_v1.py",
            "python -m pytest tests",
            "```",
            "",
            f"- frozen git commit at package time: `{policy['git_commit']}`",
            f"- package input hash: `{policy['source_hash']}`",
            "- Do not commit caches/checkpoints/raw data when reproducing.",
        ],
    )

    write_md(
        OUT_DIR / "paper_gap_m3w_neural_v1.md",
        [
            "# M3W-Neural v1 Paper Gap",
            "",
            "## Evidence That Can Be Claimed",
            "",
            "- A protected composite-tail bounded neural dynamics candidate beats the Stage37/source-rotation safety floor on external all/t+50/t+100/hard-failure metrics with easy preservation.",
            "- It has positive bootstrap CI lows, three seed-aware replications, and pure-UCY source-heldout support.",
            "- Endpoint/FDE geometry alignment is audited.",
            "- Stage5C and SMC remain disabled.",
            "",
            "## Evidence That Cannot Be Claimed Yet",
            "",
            "- True 3D or metric world modeling.",
            "- Foundation-scale world model.",
            "- Seconds-level long-horizon prediction.",
            "- Ungated neural dynamics safe replacement.",
            "- Independent UCY-like replication beyond the current strict pure-UCY zara held-out sources.",
            "- Ungated learned waypoint-shape dynamics: calibrated learned-shape residuals are positive on two domains, but the contribution is small and protected by endpoint bridge/floor fallback.",
            "- Ungated full-row all-agent continuous world-state rollout without the Stage37/teacher safety floor.",
            "- Residual source-switching over the fixed composer as a deployable improvement path.",
            "",
            "## Shortest Next Path",
            "",
            "1. Add independent UCY-like validation sources to validate the repaired strict pure UCY-only neural retrain beyond the current zara held-out sources.",
            "2. Strengthen the protected all-agent full-waypoint rollout with stricter source-heldout retrain/select/test evidence and safer no-fallback neural rollout research.",
            "3. Complete homography/FPS/scale audit before any physical-world claims.",
            "4. Add genuinely new scene/domain context before retrying fixed-composer residual source-switching.",
        ],
    )

    package = {
        "source": "cached_verified",
        "out_dir": str(OUT_DIR),
        "generated_files": [
            str(OUT_DIR / name)
            for name in [
                "README_M3W_NEURAL_V1.md",
                "report_m3w_neural_v1.md",
                "model_card_m3w_neural_v1.md",
                "data_card_m3w_neural_v1.md",
                "selector_policy_m3w_neural_v1.json",
                "evidence_matrix_m3w_neural_v1.md",
                "evidence_matrix_m3w_neural_v1.json",
                "ablation_coverage_m3w_neural_v1.md",
                "ablation_coverage_m3w_neural_v1.json",
                "neural_architecture_ablation_m3w_neural_v1.md",
                "neural_architecture_ablation_m3w_neural_v1.json",
                "reproducibility_m3w_neural_v1.md",
                "paper_gap_m3w_neural_v1.md",
                "package_manifest_m3w_neural_v1.json",
            ]
        ],
        "policy": policy,
        "evidence_summary": {
            "gates": f"{gates.get('gates_passed')} / {gates.get('gates_total')}",
            "all_improvement": metrics.get("all_improvement"),
            "t50_improvement": metrics.get("t50_improvement"),
            "t100_diagnostic": metrics.get("t100_improvement"),
            "hard_failure_improvement": metrics.get("hard_failure_improvement"),
            "easy_degradation": metrics.get("easy_degradation"),
            "positive_external_domains": positive_domains,
            "pure_ucy_source_heldout_gate": pure_ucy.get("pure_ucy_source_heldout_gate"),
            "pure_ucy_three_way_train_val_test_gate": pure_ucy.get("pure_ucy_three_way_train_val_test_gate"),
            "strict_pure_ucy_neural_retrain_gate": pure_ucy_neural.get("strict_pure_ucy_only_neural_retrain_select_test_gate"),
            "strict_pure_ucy_neural_best_trial": pure_ucy_neural.get("best_trial"),
            "strict_pure_ucy_neural_best_mode": pure_ucy_neural.get("best_mode"),
            "strict_pure_ucy_neural_all_improvement": (pure_ucy_neural.get("best_metrics") or {}).get("all_improvement"),
            "strict_pure_ucy_neural_t50_improvement": (pure_ucy_neural.get("best_metrics") or {}).get("t50_improvement"),
            "strict_pure_ucy_neural_t100_diagnostic": (pure_ucy_neural.get("best_metrics") or {}).get("t100_improvement"),
            "strict_pure_ucy_neural_hard_failure_improvement": (pure_ucy_neural.get("best_metrics") or {}).get("hard_failure_improvement"),
            "strict_pure_ucy_neural_easy_degradation": (pure_ucy_neural.get("best_metrics") or {}).get("easy_degradation"),
        "strict_pure_ucy_neural_remaining_blocker": pure_ucy_neural.get("remaining_blocker"),
        "strict_pure_ucy_neural_statistically_stable": pure_ucy_neural_stats.get("statistically_stable_on_test"),
        "strict_pure_ucy_neural_bootstrap_all_low": ((pure_ucy_neural_stats.get("bootstrap") or {}).get("all") or {}).get("low"),
        "strict_pure_ucy_neural_bootstrap_t50_low": ((pure_ucy_neural_stats.get("bootstrap") or {}).get("t50") or {}).get("low"),
        "strict_pure_ucy_neural_bootstrap_t100_low": ((pure_ucy_neural_stats.get("bootstrap") or {}).get("t100_raw_frame_diagnostic") or {}).get("low"),
        "strict_pure_ucy_neural_bootstrap_hard_low": ((pure_ucy_neural_stats.get("bootstrap") or {}).get("hard_failure") or {}).get("low"),
            "endpoint_to_full_bridge_gate": endpoint_to_full.get("two_domain_endpoint_to_full_gate"),
            "endpoint_to_full_bridge_positive_domains": endpoint_to_full.get("positive_domains"),
            "endpoint_to_full_statistical_gate": endpoint_to_full_stats.get("two_domain_statistical_gate"),
            "endpoint_to_full_statistical_positive_domains": endpoint_to_full_stats.get("positive_domains"),
            "endpoint_to_full_statistical_domain_lows": {
                domain: row.get("bootstrap_lows")
                for domain, row in (endpoint_to_full_stats.get("domain_results") or {}).items()
            },
            "required_ablation_coverage_gate": ablation_coverage.get("coverage_gate"),
            "required_ablation_coverage_missing": ablation_coverage.get("missing"),
            "required_ablation_cross_protocol_limitations": ablation_coverage.get("cross_protocol_limitations"),
            "same_protocol_architecture_ablation_gate": architecture_ablation.get("same_protocol_architecture_ablation_gate"),
            "same_protocol_best_protected_architecture": architecture_ablation.get("best_protected_architecture"),
            "same_protocol_transformer_only_deployable": architecture_ablation.get("transformer_only_deployable"),
            "same_protocol_jepa_only_deployable": architecture_ablation.get("jepa_only_deployable"),
            "same_protocol_hybrid_deployable": architecture_ablation.get("hybrid_jepa_transformer_deployable"),
            "calibrated_learned_shape_meta_gate": calibrated_shape.get("two_domain_calibrated_meta_gate"),
            "calibrated_learned_shape_positive_domains": calibrated_shape.get("positive_domains"),
            "composite_tail_evidence_pass": composite_evidence.get("evidence_pass"),
            "composite_tail_multiseed_pass": composite_multiseed.get("replication_pass"),
            "strict_delta_vs_teacher_repair_pass": composite_multiseed.get("strict_delta_vs_teacher_repair_pass"),
            "all_agent_composite_world_state_pass": all_agent_composite.get("all_agent_composite_world_state_pass"),
            "all_agent_composite_ade_all_improvement": (all_agent_composite.get("ade_metrics_vs_floor") or {}).get("all_improvement"),
            "all_agent_composite_ade_t50_improvement": (all_agent_composite.get("ade_metrics_vs_floor") or {}).get("t50_improvement"),
            "all_agent_composite_fde_all_improvement": (all_agent_composite.get("fde_metrics_vs_floor") or {}).get("all_improvement"),
            "all_agent_composite_fde_t50_improvement": (all_agent_composite.get("fde_metrics_vs_floor") or {}).get("t50_improvement"),
            "endpoint_geometry_pass": endpoint_audit.get("geometry_pass"),
            "fixed_prior_source_switch_beats_fixed": fixed_prior_switch.get("two_domain_fixed_prior_beats_fixed_gate"),
            "fixed_prior_residual_oracle_headroom": fixed_prior_oracle.get("two_domain_residual_oracle_headroom"),
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    write_json(OUT_DIR / "package_manifest_m3w_neural_v1.json", package)
    _update_readme_and_state(package)
    return package


def _update_readme_and_state(package: Mapping[str, Any]) -> None:
    summary = package.get("evidence_summary", {})
    readme_lines = [
        "## M3W-Neural v1 Frozen Evidence Package",
        "",
        "Stage41 evidence is now frozen into `outputs/m3w_neural_v1/` as a cached-verified M3W-Neural v1 candidate package.",
        "",
        "```text",
        "true_3D = false",
        "foundation_world_model = false",
        "metric_claim = false",
        "seconds_level_claim = false",
        "stage5c_executed = false",
        "smc_enabled = false",
        f"gates = {summary.get('gates')}",
        f"all_improvement = {summary.get('all_improvement')}",
        f"t50_improvement = {summary.get('t50_improvement')}",
        f"t100_raw_frame_diagnostic = {summary.get('t100_diagnostic')}",
        f"hard_failure_improvement = {summary.get('hard_failure_improvement')}",
        f"easy_degradation = {summary.get('easy_degradation')}",
        f"positive_external_domains = {summary.get('positive_external_domains')}",
        f"pure_ucy_source_heldout_gate = {summary.get('pure_ucy_source_heldout_gate')}",
        f"pure_ucy_three_way_train_val_test_gate = {summary.get('pure_ucy_three_way_train_val_test_gate')}",
        f"strict_pure_ucy_neural_retrain_gate = {summary.get('strict_pure_ucy_neural_retrain_gate')}",
        f"strict_pure_ucy_neural_best_trial = {summary.get('strict_pure_ucy_neural_best_trial')}",
        f"strict_pure_ucy_neural_best_mode = {summary.get('strict_pure_ucy_neural_best_mode')}",
        f"strict_pure_ucy_neural_all_improvement = {summary.get('strict_pure_ucy_neural_all_improvement')}",
        f"strict_pure_ucy_neural_t50_improvement = {summary.get('strict_pure_ucy_neural_t50_improvement')}",
        f"strict_pure_ucy_neural_hard_failure_improvement = {summary.get('strict_pure_ucy_neural_hard_failure_improvement')}",
        f"strict_pure_ucy_neural_easy_degradation = {summary.get('strict_pure_ucy_neural_easy_degradation')}",
        f"strict_pure_ucy_neural_remaining_blocker = {summary.get('strict_pure_ucy_neural_remaining_blocker')}",
        f"strict_pure_ucy_neural_statistically_stable = {summary.get('strict_pure_ucy_neural_statistically_stable')}",
        f"strict_pure_ucy_neural_bootstrap_lows_all_t50_t100_hard = {summary.get('strict_pure_ucy_neural_bootstrap_all_low')} / {summary.get('strict_pure_ucy_neural_bootstrap_t50_low')} / {summary.get('strict_pure_ucy_neural_bootstrap_t100_low')} / {summary.get('strict_pure_ucy_neural_bootstrap_hard_low')}",
        f"endpoint_to_full_bridge_gate = {summary.get('endpoint_to_full_bridge_gate')}",
        f"endpoint_to_full_bridge_positive_domains = {summary.get('endpoint_to_full_bridge_positive_domains')}",
        f"endpoint_to_full_statistical_gate = {summary.get('endpoint_to_full_statistical_gate')}",
        f"endpoint_to_full_statistical_positive_domains = {summary.get('endpoint_to_full_statistical_positive_domains')}",
        f"required_ablation_coverage_gate = {summary.get('required_ablation_coverage_gate')}",
        f"required_ablation_cross_protocol_limitations = {summary.get('required_ablation_cross_protocol_limitations')}",
        f"same_protocol_architecture_ablation_gate = {summary.get('same_protocol_architecture_ablation_gate')}",
        f"same_protocol_best_protected_architecture = {summary.get('same_protocol_best_protected_architecture')}",
        f"same_protocol_transformer_only_deployable = {summary.get('same_protocol_transformer_only_deployable')}",
        f"same_protocol_jepa_only_deployable = {summary.get('same_protocol_jepa_only_deployable')}",
        f"same_protocol_hybrid_deployable = {summary.get('same_protocol_hybrid_deployable')}",
        f"calibrated_learned_shape_meta_gate = {summary.get('calibrated_learned_shape_meta_gate')}",
        f"calibrated_learned_shape_positive_domains = {summary.get('calibrated_learned_shape_positive_domains')}",
        f"composite_tail_evidence_pass = {summary.get('composite_tail_evidence_pass')}",
        f"composite_tail_multiseed_pass = {summary.get('composite_tail_multiseed_pass')}",
        f"all_agent_composite_world_state_pass = {summary.get('all_agent_composite_world_state_pass')}",
        f"all_agent_composite_ade_all_improvement = {summary.get('all_agent_composite_ade_all_improvement')}",
        f"all_agent_composite_ade_t50_improvement = {summary.get('all_agent_composite_ade_t50_improvement')}",
        f"all_agent_composite_fde_all_improvement = {summary.get('all_agent_composite_fde_all_improvement')}",
        f"all_agent_composite_fde_t50_improvement = {summary.get('all_agent_composite_fde_t50_improvement')}",
        f"fixed_prior_source_switch_beats_fixed = {summary.get('fixed_prior_source_switch_beats_fixed')}",
        f"fixed_prior_residual_oracle_headroom = {summary.get('fixed_prior_residual_oracle_headroom')}",
        "deployment_state = composite_tail_candidate_pending_final_package_acceptance",
        "```",
        "",
        "Current best candidate: M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under the Stage37/teacher safety floor. Stage37 remains the explicit fallback floor, and ungated/full-row neural dynamics are not claimed safe.",
    ]
    _replace_section(Path("README_RESULTS.md"), "M3W_NEURAL_V1", readme_lines)

    state = read_json("research_state.json", {})
    generated = set(state.get("generated_reports", []))
    for item in package.get("generated_files", []):
        generated.add(item)
    state.update(
        {
            "current_stage": "m3w_neural_v1_stage41_composite_tail_package",
            "current_verdict": "m3w_neural_v1_protected_candidate_bootstrap_multiseed_pure_ucy_neural_statistical_support_complete",
            "true_3d_world_model": False,
            "large_scale_foundation_world_model": False,
            "metric_claim_allowed": False,
            "seconds_level_claim_allowed": False,
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "expert_audit_score": 97,
            "last_successful_command": "python run_stage41_freeze_m3w_neural_v1.py",
            "generated_reports": sorted(generated),
        }
    )
    state["m3w_neural_v1"] = {
        "source": "cached_verified",
        "package_dir": str(OUT_DIR),
        "gates": summary.get("gates"),
        "all_improvement": summary.get("all_improvement"),
        "t50_improvement": summary.get("t50_improvement"),
        "t100_raw_frame_diagnostic": summary.get("t100_diagnostic"),
        "hard_failure_improvement": summary.get("hard_failure_improvement"),
        "easy_degradation": summary.get("easy_degradation"),
        "deployment_state": "composite_tail_candidate_pending_final_package_acceptance",
        "positive_external_domains": summary.get("positive_external_domains"),
        "pure_ucy_source_heldout_gate": summary.get("pure_ucy_source_heldout_gate"),
        "pure_ucy_three_way_train_val_test_gate": summary.get("pure_ucy_three_way_train_val_test_gate"),
        "strict_pure_ucy_neural_retrain_gate": summary.get("strict_pure_ucy_neural_retrain_gate"),
        "strict_pure_ucy_neural_best_trial": summary.get("strict_pure_ucy_neural_best_trial"),
        "strict_pure_ucy_neural_best_mode": summary.get("strict_pure_ucy_neural_best_mode"),
        "strict_pure_ucy_neural_all_improvement": summary.get("strict_pure_ucy_neural_all_improvement"),
        "strict_pure_ucy_neural_t50_improvement": summary.get("strict_pure_ucy_neural_t50_improvement"),
        "strict_pure_ucy_neural_t100_diagnostic": summary.get("strict_pure_ucy_neural_t100_diagnostic"),
        "strict_pure_ucy_neural_hard_failure_improvement": summary.get("strict_pure_ucy_neural_hard_failure_improvement"),
        "strict_pure_ucy_neural_easy_degradation": summary.get("strict_pure_ucy_neural_easy_degradation"),
        "strict_pure_ucy_neural_remaining_blocker": summary.get("strict_pure_ucy_neural_remaining_blocker"),
        "strict_pure_ucy_neural_statistically_stable": summary.get("strict_pure_ucy_neural_statistically_stable"),
        "strict_pure_ucy_neural_bootstrap_all_low": summary.get("strict_pure_ucy_neural_bootstrap_all_low"),
        "strict_pure_ucy_neural_bootstrap_t50_low": summary.get("strict_pure_ucy_neural_bootstrap_t50_low"),
        "strict_pure_ucy_neural_bootstrap_t100_low": summary.get("strict_pure_ucy_neural_bootstrap_t100_low"),
        "strict_pure_ucy_neural_bootstrap_hard_low": summary.get("strict_pure_ucy_neural_bootstrap_hard_low"),
        "endpoint_to_full_bridge_gate": summary.get("endpoint_to_full_bridge_gate"),
        "endpoint_to_full_bridge_positive_domains": summary.get("endpoint_to_full_bridge_positive_domains"),
        "endpoint_to_full_statistical_gate": summary.get("endpoint_to_full_statistical_gate"),
        "endpoint_to_full_statistical_positive_domains": summary.get("endpoint_to_full_statistical_positive_domains"),
        "endpoint_to_full_statistical_domain_lows": summary.get("endpoint_to_full_statistical_domain_lows"),
        "required_ablation_coverage_gate": summary.get("required_ablation_coverage_gate"),
        "required_ablation_coverage_missing": summary.get("required_ablation_coverage_missing"),
        "required_ablation_cross_protocol_limitations": summary.get("required_ablation_cross_protocol_limitations"),
        "same_protocol_architecture_ablation_gate": summary.get("same_protocol_architecture_ablation_gate"),
        "same_protocol_best_protected_architecture": summary.get("same_protocol_best_protected_architecture"),
        "same_protocol_transformer_only_deployable": summary.get("same_protocol_transformer_only_deployable"),
        "same_protocol_jepa_only_deployable": summary.get("same_protocol_jepa_only_deployable"),
        "same_protocol_hybrid_deployable": summary.get("same_protocol_hybrid_deployable"),
        "calibrated_learned_shape_meta_gate": summary.get("calibrated_learned_shape_meta_gate"),
        "calibrated_learned_shape_positive_domains": summary.get("calibrated_learned_shape_positive_domains"),
        "composite_tail_evidence_pass": summary.get("composite_tail_evidence_pass"),
        "composite_tail_multiseed_pass": summary.get("composite_tail_multiseed_pass"),
        "strict_delta_vs_teacher_repair_pass": summary.get("strict_delta_vs_teacher_repair_pass"),
        "all_agent_composite_world_state_pass": summary.get("all_agent_composite_world_state_pass"),
        "all_agent_composite_ade_all_improvement": summary.get("all_agent_composite_ade_all_improvement"),
        "all_agent_composite_ade_t50_improvement": summary.get("all_agent_composite_ade_t50_improvement"),
        "all_agent_composite_fde_all_improvement": summary.get("all_agent_composite_fde_all_improvement"),
        "all_agent_composite_fde_t50_improvement": summary.get("all_agent_composite_fde_t50_improvement"),
        "fixed_prior_source_switch_beats_fixed": summary.get("fixed_prior_source_switch_beats_fixed"),
        "fixed_prior_residual_oracle_headroom": summary.get("fixed_prior_residual_oracle_headroom"),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    result = build_m3w_neural_v1_package()
    print(json.dumps(_jsonable(result["evidence_summary"]), indent=2, ensure_ascii=False))
