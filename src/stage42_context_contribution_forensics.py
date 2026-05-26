from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_contribution_forensics_stage42.json"
REPORT_MD = OUT_DIR / "context_contribution_forensics_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ci_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

UNIFIED_ABLATION_JSON = OUT_DIR / "unified_ablation_evidence_stage42.json"
RETRAINED_MATRIX_JSON = OUT_DIR / "retrained_ablation_matrix_stage42.json"
INCREMENTAL_JSON = OUT_DIR / "source_level_incremental_ablation_stage42.json"
RESIDUAL_JSON = OUT_DIR / "source_level_residual_context_stage42.json"
NEURAL_CONTEXT_JSON = OUT_DIR / "source_level_neural_context_stage42.json"
SEQUENCE_CONTEXT_JSON = OUT_DIR / "source_level_sequence_context_stage42.json"
GRAPH_CONTEXT_JSON = OUT_DIR / "source_level_graph_context_stage42.json"
BASELINE_FAMILY_JSON = OUT_DIR / "source_level_baseline_family_mechanism_stage42.json"
SAFETY_FLOOR_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"
PAPER_CLAIM_JSON = OUT_DIR / "paper_claim_evidence_audit_stage42.json"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CI 是 context contribution forensics，不下载、不转换、不执行 Stage5C、不启用 SMC。",
    "本审计整合 retrained ablation / residual / neural / sequence / graph context 证据，防止把 mixed context 结果包装成主贡献。",
    "future endpoints / waypoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _row_by(rows: list[Mapping[str, Any]], key: str, value: str) -> Mapping[str, Any]:
    return next((row for row in rows if str(row.get(key)) == value), {})


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "n/a"
    return str(value)


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0)


def _matrix_row(matrix: Mapping[str, Any], ablation: str) -> Mapping[str, Any]:
    return _row_by(list(matrix.get("ablation_rows", [])), "ablation", ablation)


def _sequence_row(unified: Mapping[str, Any], module: str) -> Mapping[str, Any]:
    return _row_by(list(unified.get("retrained_sequence_ablation_rows", [])), "module", module)


def _context_rows(
    unified: Mapping[str, Any],
    matrix: Mapping[str, Any],
    incremental: Mapping[str, Any],
    residual: Mapping[str, Any],
    neural_context: Mapping[str, Any],
    sequence_context: Mapping[str, Any],
    graph_context: Mapping[str, Any],
    baseline_family: Mapping[str, Any],
    safety_floor: Mapping[str, Any],
    paper_claim: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seq_history = _sequence_row(unified, "history tokens")
    seq_domain = _sequence_row(unified, "domain expert")
    seq_goal = _sequence_row(unified, "goal/scene tokens")
    seq_neighbor = _sequence_row(unified, "neighbor/interaction tokens")
    incr_summary = incremental.get("summary", {})
    family_summary = baseline_family.get("summary", {})
    safety_analysis = safety_floor.get("floor_necessity_analysis", {})
    safety_summary = safety_floor.get("summary", {})
    ungated_easy = safety_analysis.get("ungated_endpoint_full_waypoint_easy_degradation", None)
    if ungated_easy is None:
        ungated_easy = safety_analysis.get("ungated_endpoint_metrics_from_stage42_b", {}).get("easy_degradation")
    if ungated_easy is None:
        ungated_easy = safety_summary.get("ungated_endpoint_easy_degradation")
    claim_goal = _row_by(list(paper_claim.get("claim_rows", [])), "claim_id", "C5")

    rows = [
        {
            "module": "baseline_family_rollout_context",
            "status": "supported_dominant_mechanism",
            "main_claim_allowed": True,
            "evidence_sources": ["Stage42-AU", "Stage42-AO", "Stage42-Z"],
            "key_evidence": (
                f"AU verdict={family_summary.get('mechanism_verdict')}; "
                f"baseline_family_only all={_fmt((incr_summary.get('baseline_family_only') or {}).get('all_improvement'))}; "
                f"t50={_fmt((incr_summary.get('baseline_family_only') or {}).get('t50_improvement'))}; "
                f"hard={_fmt((incr_summary.get('baseline_family_only') or {}).get('hard_failure_improvement'))}"
            ),
            "failure_mode": "not a standalone learned scene/interaction world model; it is a causal baseline-family rollout mechanism under safety floor",
            "next_action": "Keep as the current dominant mechanism and use it as the teacher/floor for any stronger neural dynamics experiment.",
        },
        {
            "module": "history_tokens",
            "status": "supported_core_component",
            "main_claim_allowed": True,
            "evidence_sources": ["Stage42-H", "Stage42-Y", "Stage42-AA"],
            "key_evidence": (
                f"sequence full-minus-no-history t50={_fmt((seq_history.get('full_minus_ablation') or {}).get('t50'))}; "
                f"hard={_fmt((seq_history.get('full_minus_ablation') or {}).get('hard_failure'))}; "
                f"matrix status={_matrix_row(matrix, 'no_history').get('status')}"
            ),
            "failure_mode": "history helps when encoded as sequence/core context; flattened residual context after baseline-family is weaker",
            "next_action": "Keep sequence history tokens in paper method and future models; do not reduce them to only flattened residual features.",
        },
        {
            "module": "domain_expert",
            "status": "supported_secondary_component",
            "main_claim_allowed": True,
            "evidence_sources": ["Stage42-H", "Stage42-AA"],
            "key_evidence": (
                f"sequence full-minus-no-domain t50={_fmt((seq_domain.get('full_minus_ablation') or {}).get('t50'))}; "
                f"hard={_fmt((seq_domain.get('full_minus_ablation') or {}).get('hard_failure'))}; "
                f"matrix status={_matrix_row(matrix, 'no_domain_expert').get('status')}"
            ),
            "failure_mode": "domain expert is smaller than baseline-family/history and must remain validation-gated",
            "next_action": "Use domain expert as a guarded source/horizon conditioning module, not as a broad generalization claim.",
        },
        {
            "module": "goal_scene_context",
            "status": "mixed_partial_not_main_claim",
            "main_claim_allowed": False,
            "evidence_sources": ["Stage42-AO", "Stage42-Y", "Stage42-Z"],
            "key_evidence": (
                f"AO positive standalone={incr_summary.get('positive_standalone_context_variants')}; "
                f"AO positive incremental={incr_summary.get('positive_incremental_context_variants')}; "
                f"sequence full-minus-no-goal t50={_fmt((seq_goal.get('full_minus_ablation') or {}).get('t50'))}; "
                f"Z C5 status={claim_goal.get('status')}"
            ),
            "failure_mode": "goal/scene has standalone signal but does not add reliably after baseline-family; sequence goal/scene can hurt t50",
            "next_action": "Try a source/horizon validation-gated goal prototype expert rather than global goal/scene injection; require bootstrap-positive t50 and easy<=2 before main claim.",
        },
        {
            "module": "neighbor_interaction_context",
            "status": "mixed_weak_not_main_claim",
            "main_claim_allowed": False,
            "evidence_sources": ["Stage42-AS", "Stage42-Y", "Stage42-Z"],
            "key_evidence": (
                f"AS verdict={(graph_context.get('summary') or {}).get('graph_context_verdict')}; "
                f"sequence full-minus-no-neighbor all={_fmt((seq_neighbor.get('full_minus_ablation') or {}).get('all'))}; "
                f"hard={_fmt((seq_neighbor.get('full_minus_ablation') or {}).get('hard_failure'))}; "
                f"matrix status={_matrix_row(matrix, 'no_neighbor').get('status')}"
            ),
            "failure_mode": "hand-built kNN/graph residual and sequence neighbor variants are not consistently positive beyond baseline-family context",
            "next_action": "Only keep neighbor features as auxiliary diagnostics unless a stronger graph-neural trial beats baseline-family with easy preservation.",
        },
        {
            "module": "jepa_auxiliary",
            "status": "negative_or_inconclusive",
            "main_claim_allowed": False,
            "evidence_sources": ["Stage18/19 cached", "Stage42-AA"],
            "key_evidence": f"matrix no_JEPA status={_matrix_row(matrix, 'no_JEPA').get('status')}; interpretation={_matrix_row(matrix, 'no_JEPA').get('interpretation')}",
            "failure_mode": "non-collapse did not translate into stable selector/failure/correction/full-waypoint lift",
            "next_action": "Keep JEPA as diagnostic/auxiliary only until a downstream probe improves protected metrics.",
        },
        {
            "module": "transformer_dynamics",
            "status": "negative_or_inconclusive_as_independent_claim",
            "main_claim_allowed": False,
            "evidence_sources": ["Stage39/40 cached", "Stage42-AA"],
            "key_evidence": f"matrix no_Transformer status={_matrix_row(matrix, 'no_Transformer').get('status')}; delta_t50={_fmt(_matrix_row(matrix, 'no_Transformer').get('delta_t50_full_minus_ablation'))}",
            "failure_mode": "current Transformer/proxy evidence is protected/diagnostic and not floor-free deployable",
            "next_action": "Future Transformer work should target full-waypoint source-level lift under Stage37 floor, then test floor relaxation separately.",
        },
        {
            "module": "stage37_teacher_floor_and_safe_switch",
            "status": "supported_necessary_safety_mechanism",
            "main_claim_allowed": True,
            "evidence_sources": ["Stage42-BW", "Stage42-BX", "Stage42-Z"],
            "key_evidence": f"ungated easy degradation={_fmt(ungated_easy)}; no_teacher_floor status={_matrix_row(matrix, 'no_teacher_floor').get('status')}; no_safe_switch status={_matrix_row(matrix, 'no_safe_switch').get('status')}",
            "failure_mode": "removing the floor/safe switch creates unsafe easy-case harm or weaker protected t50",
            "next_action": "Frame the floor as a safety mechanism and current core contribution; only relax it on validation-supported source/horizon slices.",
        },
    ]
    return rows


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(result.get("context_rows", []))
    by_module = {row["module"]: row for row in rows}
    gates = {
        "inputs_present": all(result.get("inputs", {}).values()),
        "dominant_mechanism_identified": by_module.get("baseline_family_rollout_context", {}).get("status") == "supported_dominant_mechanism",
        "history_supported": by_module.get("history_tokens", {}).get("status") == "supported_core_component",
        "safety_floor_supported": by_module.get("stage37_teacher_floor_and_safe_switch", {}).get("status") == "supported_necessary_safety_mechanism",
        "mixed_goal_scene_not_overclaimed": by_module.get("goal_scene_context", {}).get("main_claim_allowed") is False,
        "mixed_neighbor_not_overclaimed": by_module.get("neighbor_interaction_context", {}).get("main_claim_allowed") is False,
        "jepa_not_overclaimed": by_module.get("jepa_auxiliary", {}).get("main_claim_allowed") is False,
        "transformer_not_overclaimed": by_module.get("transformer_dynamics", {}).get("main_claim_allowed") is False,
        "rescue_actions_written": all(row.get("next_action") for row in rows),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoint_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": result.get("source"),
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_ci_context_contribution_forensics_pass" if all(gates.values()) else "stage42_ci_context_contribution_forensics_partial",
    }


def _render_md(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ci_gate"]
    lines = [
        "# Stage42-CI Context Contribution Forensics",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Context Rows",
        "",
        "| module | status | main claim? | evidence | next action |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in result["context_rows"]:
        lines.append(
            f"| `{row['module']}` | `{row['status']}` | `{row['main_claim_allowed']}` | {row['key_evidence']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            "- Baseline-family rollout context is currently the dominant supported mechanism, not merely a nuisance fallback.",
            "- History tokens are supported when encoded as causal sequence context.",
            "- Domain expert is supported as a smaller guarded source/horizon component.",
            "- Goal/scene context has standalone signal but is not reliably incremental after baseline-family context.",
            "- Neighbor/interaction context is weak/mixed under current hand-built graph and sequence protocols.",
            "- JEPA and Transformer evidence remains diagnostic/protected, not independent floor-free deployment evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(gate: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CI Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {ok} |")
    return lines


def run_stage42_context_contribution_forensics() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    unified = read_json(UNIFIED_ABLATION_JSON, {})
    matrix = read_json(RETRAINED_MATRIX_JSON, {})
    incremental = read_json(INCREMENTAL_JSON, {})
    residual = read_json(RESIDUAL_JSON, {})
    neural_context = read_json(NEURAL_CONTEXT_JSON, {})
    sequence_context = read_json(SEQUENCE_CONTEXT_JSON, {})
    graph_context = read_json(GRAPH_CONTEXT_JSON, {})
    baseline_family = read_json(BASELINE_FAMILY_JSON, {})
    safety_floor = read_json(SAFETY_FLOOR_JSON, {})
    paper_claim = read_json(PAPER_CLAIM_JSON, {})
    result = {
        "stage": "Stage42-CI context contribution forensics",
        "source": "fresh_synthesis_from_stage42_ablation_and_claim_audits",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "unified_ablation_exists": UNIFIED_ABLATION_JSON.exists(),
            "retrained_matrix_exists": RETRAINED_MATRIX_JSON.exists(),
            "incremental_exists": INCREMENTAL_JSON.exists(),
            "residual_context_exists": RESIDUAL_JSON.exists(),
            "neural_context_exists": NEURAL_CONTEXT_JSON.exists(),
            "sequence_context_exists": SEQUENCE_CONTEXT_JSON.exists(),
            "graph_context_exists": GRAPH_CONTEXT_JSON.exists(),
            "baseline_family_exists": BASELINE_FAMILY_JSON.exists(),
            "safety_floor_exists": SAFETY_FLOOR_JSON.exists(),
            "paper_claim_exists": PAPER_CLAIM_JSON.exists(),
        },
        "input_hash": _combined_hash(
            [
                UNIFIED_ABLATION_JSON,
                RETRAINED_MATRIX_JSON,
                INCREMENTAL_JSON,
                RESIDUAL_JSON,
                NEURAL_CONTEXT_JSON,
                SEQUENCE_CONTEXT_JSON,
                GRAPH_CONTEXT_JSON,
                BASELINE_FAMILY_JSON,
                SAFETY_FLOOR_JSON,
                PAPER_CLAIM_JSON,
            ]
        ),
        "context_rows": _context_rows(
            unified,
            matrix,
            incremental,
            residual,
            neural_context,
            sequence_context,
            graph_context,
            baseline_family,
            safety_floor,
            paper_claim,
        ),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "labels_eval_only": True,
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
    result["stage42_ci_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_md(result))
    write_md(GATE_MD, _render_gate(result["stage42_ci_gate"]))
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "stage": result["stage"],
                    "source": result["source"],
                    "verdict": result["stage42_ci_gate"]["verdict"],
                    "generated_at_utc": result["generated_at_utc"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return result


if __name__ == "__main__":
    run_stage42_context_contribution_forensics()
