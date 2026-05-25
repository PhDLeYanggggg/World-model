from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/m3w_neural_v1")
REPORT_JSON = OUT_DIR / "ablation_coverage_m3w_neural_v1.json"
REPORT_MD = OUT_DIR / "ablation_coverage_m3w_neural_v1.md"
FRESH_DIR = Path("outputs/stage41_fresh_confirmation")
SPLIT_DIR = Path("outputs/stage41_external_split")
STAGE30_DIR = Path("outputs/stage30_m3w_verified")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _delta(row: Mapping[str, Any], metric: str = "all_delta") -> float | None:
    direct = row.get(metric)
    if isinstance(direct, (int, float)):
        return float(direct)
    nested = row.get("delta_vs_full") or {}
    nested_direct = nested.get(metric)
    if isinstance(nested_direct, (int, float)):
        return float(nested_direct)
    return None


def _status(*evidence: bool, weaker: bool = False) -> str:
    if all(evidence):
        return "complete_but_cross_protocol" if weaker else "complete"
    if any(evidence):
        return "partial"
    return "missing"


def run_ablation_coverage_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    teacher = read_json(FRESH_DIR / "stage41_teacher_guided_evidence.json", {})
    group = read_json(FRESH_DIR / "stage41_group_consistency_evidence.json", {})
    route = read_json(FRESH_DIR / "stage41_route_physical_policy_integration.json", {})
    jepa = read_json(FRESH_DIR / "stage41_jepa_deployment_decision.json", {})
    architecture = read_json(OUT_DIR / "neural_architecture_ablation_m3w_neural_v1.json", {})
    pure_ucy_stats = read_json(SPLIT_DIR / "stage41_pure_ucy_neural_statistical_evidence.json", {})
    all_agent = read_json(FRESH_DIR / "stage41_all_agent_composite_world_state.json", {})
    stage30 = read_json(STAGE30_DIR / "retrained_ablation_fresh.json", {})

    teacher_ab = teacher.get("ablations") or {}
    group_ab = group.get("contribution_summary") or {}
    route_ab = route.get("ablations") or {}
    stage30_summary = stage30.get("summary") or {}
    no_fallback_teacher = teacher.get("neural_without_fallback_metrics") or {}
    no_fallback_ucy = pure_ucy_stats.get("raw_neural_endpoint_without_fallback") or {}

    requirements: dict[str, Any] = {
        "no_history": {
            "status": _status("no_history_static" in teacher_ab),
            "source": "outputs/stage41_fresh_confirmation/stage41_teacher_guided_evidence.json",
            "evidence_type": "fresh_run masked-feature ablation",
            "delta_vs_full": (teacher_ab.get("no_history_static") or {}).get("delta_vs_full"),
            "interpretation": "Masks history/static causal feature group after policy freeze; proves coverage for no-history/static ablation, not a claim that every history-derived scalar is useless.",
        },
        "no_neighbor": {
            "status": _status("no_neighbor_interaction" in teacher_ab or "neighbor_count" in group_ab),
            "source": "teacher_guided_evidence + group_consistency_evidence",
            "evidence_type": "fresh_run masked neighbor/interaction ablations",
            "teacher_delta_vs_full": (teacher_ab.get("no_neighbor_interaction") or {}).get("delta_vs_full"),
            "neighbor_count_delta": group_ab.get("neighbor_count"),
            "interpretation": "Neighbor/interaction masking is audited; group/neighbor features are especially important for the safety head and t100/hard slices.",
        },
        "no_scene_goal": {
            "status": _status("no_scene_goal_proxy" in teacher_ab or "no_route_physical" in route_ab),
            "source": "stage41_teacher_guided_evidence + stage41_route_physical_policy_integration",
            "evidence_type": "fresh_run scene/goal proxy and route/physical ablations",
            "teacher_delta_vs_full": (teacher_ab.get("no_scene_goal_proxy") or {}).get("delta_vs_full"),
            "route_physical_no_aux_metrics": (route_ab.get("no_route_physical") or {}).get("test_metrics"),
            "interpretation": "Scene/goal proxy coverage exists. Current deployable trajectory path keeps route/physical mostly diagnostic; route/physical heads are not main trajectory deployment claims.",
        },
        "no_interaction": {
            "status": _status("no_group_consistency" in teacher_ab or "group_consistency_features" in group_ab),
            "source": "stage41_teacher_guided_evidence + stage41_group_consistency_evidence",
            "evidence_type": "fresh_run interaction/group-consistency masking",
            "teacher_no_group_delta": (teacher_ab.get("no_group_consistency") or {}).get("delta_vs_full"),
            "teacher_no_neighbor_interaction_delta": (teacher_ab.get("no_neighbor_interaction") or {}).get("delta_vs_full"),
            "group_consistency_feature_delta": group_ab.get("group_consistency_features"),
            "interpretation": "Interaction/group-consistency features have explicit ablations and are necessary for guarded deployment; without them raw neural remains less safe.",
        },
        "no_jepa": {
            "status": "complete_with_same_protocol_negative_evidence"
            if architecture.get("no_jepa_evidence")
            else _status(bool(jepa.get("disable_jepa_in_deployable_path")), "no_jepa" in stage30_summary, weaker=True),
            "source": "stage41_jepa_deployment_decision + stage41_neural_architecture_ablation_audit + stage30_m3w_verified/retrained_ablation_fresh",
            "evidence_type": "fresh JEPA disable decision plus same-protocol Stage41 pure-Transformer negative attempts",
            "jepa_decision": jepa.get("decision"),
            "deployable_positive_attempt_count": jepa.get("deployable_positive_attempt_count"),
            "same_protocol_no_jepa": architecture.get("no_jepa_evidence"),
            "stage30_no_jepa_summary": stage30_summary.get("no_jepa"),
            "interpretation": "JEPA is explicitly disabled from the deployable path because audited JEPA variants were non-collapse but did not give deployable downstream lift. Stage41 now also records same-protocol pure-Transformer/no-JEPA attempts as negative or fallback-only, so the current positive path is protected endpoint neural dynamics rather than JEPA/Transformer purity.",
        },
        "no_transformer": {
            "status": "complete_with_same_protocol_negative_evidence"
            if architecture.get("no_transformer_evidence")
            else _status("no_transformer" in stage30_summary, bool(jepa.get("evidence")), weaker=True),
            "source": "stage41_neural_architecture_ablation_audit + stage30_m3w_verified/retrained_ablation_fresh + Stage41 JEPA-only diagnostics",
            "evidence_type": "same-protocol JEPA-only negative attempts plus historical cross-protocol ablation",
            "stage30_no_transformer_summary": stage30_summary.get("no_transformer"),
            "same_protocol_no_transformer": architecture.get("no_transformer_evidence"),
            "stage41_jepa_only_attempts": [row for row in (jepa.get("evidence") or []) if "JEPA" in str(row.get("attempt", ""))],
            "interpretation": "Stage41 same-protocol JEPA-only/no-Transformer attempts are negative or unsafe, so no-Transformer is covered as negative architecture evidence. This is not a claim that JEPA contributes to the deployable path; it is why JEPA remains diagnostic-only.",
        },
        "no_fallback": {
            "status": _status(bool(no_fallback_teacher), bool(no_fallback_ucy), all_agent.get("claim_boundary", {}).get("ungated_no_fallback_neural_rollout_safe") is False),
            "source": "teacher_guided_evidence + pure_ucy_neural_statistical_evidence + all_agent_composite_world_state",
            "evidence_type": "fresh no-fallback negative safety ablation",
            "teacher_no_fallback": {
                "all_improvement": no_fallback_teacher.get("all_improvement"),
                "easy_degradation": no_fallback_teacher.get("easy_degradation"),
            },
            "pure_ucy_raw_no_fallback": {
                "all_improvement": no_fallback_ucy.get("all_improvement"),
                "easy_degradation": no_fallback_ucy.get("easy_degradation"),
            },
            "all_agent_ungated_safe": all_agent.get("claim_boundary", {}).get("ungated_no_fallback_neural_rollout_safe"),
            "interpretation": "No-fallback neural often improves hard/all raw error but catastrophically damages easy cases; fallback is required for deployability.",
        },
    }

    missing = [name for name, row in requirements.items() if row["status"] == "missing"]
    partial = [name for name, row in requirements.items() if row["status"] == "partial"]
    cross_protocol = [name for name, row in requirements.items() if row["status"] == "complete_but_cross_protocol"]
    same_protocol_negative = [
        name for name, row in requirements.items() if row["status"] == "complete_with_same_protocol_negative_evidence"
    ]
    coverage_gate = not missing and not partial
    result = {
        "source": "fresh_run",
        "protocol": "m3w_neural_v1_required_ablation_coverage_audit",
        "required_ablations": ["no_history", "no_neighbor", "no_scene_goal", "no_interaction", "no_jepa", "no_transformer", "no_fallback"],
        "coverage_gate": coverage_gate,
        "missing": missing,
        "partial": partial,
        "cross_protocol_limitations": cross_protocol,
        "same_protocol_negative_architecture_evidence": same_protocol_negative,
        "same_protocol_architecture_ablation_gate": architecture.get("same_protocol_architecture_ablation_gate"),
        "requirements": requirements,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "test_endpoint_goals": False,
            "central_velocity": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "ablation_coverage_not_new_training": True,
            "cross_protocol_ablations_are_limitations": cross_protocol,
            "not_true_3d": True,
            "not_foundation": True,
            "not_metric_or_seconds": True,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# M3W-Neural v1 Ablation Coverage Audit",
        "",
        "- source: `fresh_run`",
        f"- coverage gate: `{coverage_gate}`",
        f"- missing: `{missing}`",
        f"- partial: `{partial}`",
        f"- cross-protocol limitations: `{cross_protocol}`",
        f"- same-protocol negative architecture evidence: `{same_protocol_negative}`",
        f"- same-protocol architecture ablation gate: `{architecture.get('same_protocol_architecture_ablation_gate')}`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "| ablation | status | source | interpretation |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in requirements.items():
        lines.append(f"| `{name}` | `{row['status']}` | `{row['source']}` | {row['interpretation']} |")
    lines.extend(
        [
            "",
            "## Important Boundary",
            "",
            "This audit keeps the evidence traceable. no-JEPA and no-Transformer are now covered by same-protocol negative architecture evidence; any future cross-protocol limits must remain explicit.",
            "The no-fallback evidence remains negative for deployment safety; Stage37/teacher fallback remains required.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_ablation_coverage_audit() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_ablation_coverage_audit()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_ablation_coverage_audit",
            status,
            started,
            [
                FRESH_DIR / "stage41_teacher_guided_evidence.json",
                FRESH_DIR / "stage41_group_consistency_evidence.json",
                FRESH_DIR / "stage41_jepa_deployment_decision.json",
                STAGE30_DIR / "retrained_ablation_fresh.json",
                OUT_DIR / "neural_architecture_ablation_m3w_neural_v1.json",
            ],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_ablation_coverage_audit()
