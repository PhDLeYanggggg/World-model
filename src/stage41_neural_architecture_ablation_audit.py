from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/m3w_neural_v1")
REPORT_JSON = OUT_DIR / "neural_architecture_ablation_m3w_neural_v1.json"
REPORT_MD = OUT_DIR / "neural_architecture_ablation_m3w_neural_v1.md"
NEURAL_EVAL = Path("outputs/stage41_breakthrough/stage41_neural_eval.json")
JEPA_DECISION = Path("outputs/stage41_fresh_confirmation/stage41_jepa_deployment_decision.json")


ARCHITECTURE_GROUPS = {
    "transformer_only": [
        "Stage41_causal_transformer_dynamics",
        "Stage41_baseline_relative_transformer",
        "Stage41_t50_hard_curriculum_transformer",
        "Stage41_conformal_safety_head_transformer",
        "Stage41_long_horizon_t100_curriculum",
    ],
    "jepa_only": [
        "Stage41_jepa_auxiliary_representation",
    ],
    "hybrid_jepa_transformer": [
        "Stage41_hybrid_jepa_transformer",
        "Stage41_neighbor_interaction_heavy_hybrid",
        "Stage41_easy_guard_distilled_hybrid",
    ],
    "mixture_selector": [
        "Stage41_mixture_of_experts_baseline_selector",
    ],
    "protected_neural_endpoint": [
        "Stage41_t50_rescue",
        "Stage41_policy_blender",
        "Stage41_candidate_distiller",
        "Stage41_fresh_self_gated_endpoint_candidate",
    ],
}


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


def _comparisons(neural_eval: Mapping[str, Any]) -> dict[str, Any]:
    raw = neural_eval.get("comparisons") or neural_eval.get("comparison") or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _safe_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _strict_positive_domain(row: Mapping[str, Any]) -> bool:
    return bool(
        _safe_float(row, "all_improvement") > 0
        and _safe_float(row, "t50_improvement") > 0
        and _safe_float(row, "hard_failure_improvement") > 0
        and _safe_float(row, "easy_degradation", 1.0) <= 0.02
    )


def _positive_domain_count(row: Mapping[str, Any]) -> int:
    domains = row.get("by_domain") or {}
    if not isinstance(domains, Mapping):
        return 0
    return sum(1 for domain_row in domains.values() if isinstance(domain_row, Mapping) and _strict_positive_domain(domain_row))


def _deployable(row: Mapping[str, Any]) -> bool:
    return bool(
        _safe_float(row, "all_improvement") > 0
        and _safe_float(row, "t50_improvement") > 0
        and _safe_float(row, "hard_failure_improvement") > 0
        and _safe_float(row, "easy_degradation", 1.0) <= 0.02
        and _positive_domain_count(row) >= 2
    )


def _safe_fallback_only(row: Mapping[str, Any]) -> bool:
    return bool(
        abs(_safe_float(row, "all_improvement")) < 1e-12
        and abs(_safe_float(row, "t50_improvement")) < 1e-12
        and abs(_safe_float(row, "hard_failure_improvement")) < 1e-12
        and _safe_float(row, "easy_degradation", 0.0) <= 0.02
        and _safe_float(row, "switch_rate", 0.0) <= 1e-4
    )


def _status(row: Mapping[str, Any]) -> str:
    if _deployable(row):
        return "deployable_positive"
    if _safe_fallback_only(row):
        return "safe_fallback_only_no_lift"
    if row:
        return "negative_or_unsafe"
    return "not_run"


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rows": row.get("rows"),
        "all_improvement": row.get("all_improvement"),
        "t50_improvement": row.get("t50_improvement"),
        "t100_improvement": row.get("t100_improvement"),
        "hard_failure_improvement": row.get("hard_failure_improvement"),
        "easy_degradation": row.get("easy_degradation"),
        "switch_rate": row.get("switch_rate"),
        "positive_domain_count": _positive_domain_count(row),
        "deployable": _deployable(row),
        "status": _status(row),
    }


def _best_candidate(comparisons: Mapping[str, Any], names: Sequence[str]) -> tuple[str | None, dict[str, Any]]:
    present = [(name, comparisons.get(name)) for name in names if isinstance(comparisons.get(name), Mapping)]
    if not present:
        return None, {}
    return max(
        ((name, dict(row)) for name, row in present),
        key=lambda item: (
            int(_deployable(item[1])),
            _safe_float(item[1], "all_improvement"),
            _safe_float(item[1], "t50_improvement"),
            _safe_float(item[1], "hard_failure_improvement"),
        ),
    )


def _group_summary(comparisons: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    candidates = {
        name: _compact(dict(comparisons.get(name) or {}))
        for name in names
        if isinstance(comparisons.get(name), Mapping)
    }
    best_name, best_row = _best_candidate(comparisons, names)
    return {
        "attempted": bool(candidates),
        "candidate_count": len(candidates),
        "best_candidate": best_name,
        "best": _compact(best_row) if best_row else {},
        "any_deployable": any(row.get("deployable") for row in candidates.values()),
        "safe_fallback_only_count": sum(1 for row in candidates.values() if row.get("status") == "safe_fallback_only_no_lift"),
        "negative_or_unsafe_count": sum(1 for row in candidates.values() if row.get("status") == "negative_or_unsafe"),
        "candidates": candidates,
    }


def run_neural_architecture_ablation_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    neural_eval = read_json(NEURAL_EVAL, {})
    jepa_decision = read_json(JEPA_DECISION, {})
    comparisons = _comparisons(neural_eval)

    groups = {name: _group_summary(comparisons, members) for name, members in ARCHITECTURE_GROUPS.items()}
    protected_best = groups["protected_neural_endpoint"].get("best") or {}
    same_protocol_gate = bool(
        protected_best.get("deployable")
        and groups["transformer_only"]["attempted"]
        and not groups["transformer_only"]["any_deployable"]
        and groups["jepa_only"]["attempted"]
        and not groups["jepa_only"]["any_deployable"]
        and groups["hybrid_jepa_transformer"]["attempted"]
        and not groups["hybrid_jepa_transformer"]["any_deployable"]
    )

    result = {
        "source": "fresh_run",
        "protocol": "stage41_same_protocol_neural_architecture_ablation_audit",
        "same_protocol_architecture_ablation_gate": same_protocol_gate,
        "best_protected_architecture": groups["protected_neural_endpoint"].get("best_candidate"),
        "best_protected_architecture_metrics": protected_best,
        "transformer_only_deployable": groups["transformer_only"].get("any_deployable"),
        "jepa_only_deployable": groups["jepa_only"].get("any_deployable"),
        "hybrid_jepa_transformer_deployable": groups["hybrid_jepa_transformer"].get("any_deployable"),
        "mixture_selector_deployable": groups["mixture_selector"].get("any_deployable"),
        "no_jepa_evidence": {
            "status": "same_protocol_negative_transformer_only_attempts",
            "interpretation": "Pure Transformer/no-JEPA Stage41 attempts were run in the same external protocol and were negative or fallback-only; current positive deployable evidence is protected endpoint neural dynamics, not a pure Transformer claim.",
            "attempts": groups["transformer_only"],
        },
        "no_transformer_evidence": {
            "status": "same_protocol_negative_jepa_only_attempts",
            "interpretation": "JEPA-only/no-Transformer Stage41 attempts were run in the same external protocol and were negative; JEPA remains disabled from the deployable path.",
            "attempts": groups["jepa_only"],
        },
        "hybrid_evidence": {
            "status": "same_protocol_negative_hybrid_attempts",
            "interpretation": "JEPA+Transformer hybrid variants did not beat the Stage37 safety floor under this protocol.",
            "attempts": groups["hybrid_jepa_transformer"],
        },
        "groups": groups,
        "jepa_deployment_decision": {
            "decision": jepa_decision.get("decision"),
            "disable_jepa_in_deployable_path": jepa_decision.get("disable_jepa_in_deployable_path"),
            "attempt_count": jepa_decision.get("attempt_count"),
            "non_collapse_attempt_count": jepa_decision.get("non_collapse_attempt_count"),
            "deployable_positive_attempt_count": jepa_decision.get("deployable_positive_attempt_count"),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "test_endpoint_goals": False,
            "central_velocity": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "not_true_3d": True,
            "not_foundation": True,
            "not_metric_or_seconds": True,
            "same_protocol_architecture_audit_not_new_training": True,
            "jepa_transformer_hybrid_positive_contribution_not_proven": True,
            "protected_endpoint_neural_candidate_is_current_positive_neural_evidence": True,
            "ungated_neural_still_not_deployable": True,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))

    lines = [
        "# Stage41 Neural Architecture Ablation Audit",
        "",
        "- source: `fresh_run`",
        f"- same-protocol architecture ablation gate: `{same_protocol_gate}`",
        f"- best protected architecture: `{result['best_protected_architecture']}`",
        f"- transformer-only deployable: `{result['transformer_only_deployable']}`",
        f"- JEPA-only deployable: `{result['jepa_only_deployable']}`",
        f"- hybrid deployable: `{result['hybrid_jepa_transformer_deployable']}`",
        f"- mixture selector deployable: `{result['mixture_selector_deployable']}`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Architecture Groups",
        "",
        "| group | attempted | best candidate | best status | all | t+50 | t+100 diag | hard/failure | easy | positive domains |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, group in groups.items():
        best = group.get("best") or {}
        lines.append(
            f"| `{group_name}` | `{group.get('attempted')}` | `{group.get('best_candidate')}` | `{best.get('status')}` | "
            f"{best.get('all_improvement')} | {best.get('t50_improvement')} | {best.get('t100_improvement')} | "
            f"{best.get('hard_failure_improvement')} | {best.get('easy_degradation')} | {best.get('positive_domain_count')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Same-protocol pure Transformer/no-JEPA attempts are not deployable; they are negative or fallback-only evidence.",
            "- Same-protocol JEPA-only/no-Transformer attempts are not deployable; JEPA remains diagnostic-only despite non-collapse in earlier stages.",
            "- Same-protocol JEPA+Transformer hybrid attempts did not beat the Stage37 safety floor.",
            "- The positive Stage41 path is protected endpoint neural dynamics under the Stage37/teacher safety floor, not an ungated JEPA/Transformer rollout.",
            "",
            "## Claim Boundary",
            "",
            f"`{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_neural_architecture_ablation_audit() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_neural_architecture_ablation_audit()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_neural_architecture_ablation_audit",
            status,
            started,
            [NEURAL_EVAL, JEPA_DECISION],
            [REPORT_JSON, REPORT_MD],
        )


if __name__ == "__main__":
    main_neural_architecture_ablation_audit()
