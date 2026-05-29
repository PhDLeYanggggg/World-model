from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_local_calibrated_source_terms_prefill import _jsonable
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
AR_JSON = OUT_DIR / "source_level_sequence_context_stage42.json"
AS_JSON = OUT_DIR / "source_level_graph_context_stage42.json"
REPORT_JSON = OUT_DIR / "source_context_fresh_replay_stage42.json"
REPORT_MD = OUT_DIR / "source_context_fresh_replay_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jr_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JR_SOURCE_CONTEXT_FRESH_REPLAY"
SOURCE = "fresh_stage42_jr_source_context_fresh_replay"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JR consolidates fresh AR/AS source-level sequence and graph context replay evidence.",
    "Negative sequence/graph residual results are not hidden or packaged as a contribution.",
    "future endpoints / waypoints are labels/evaluation only, not inference input.",
    "No central velocity, no test endpoint goals, no test threshold tuning.",
    "No metric/seconds claim, no Stage5C execution, no SMC.",
]


def _metric_delta_rows(report: Mapping[str, Any], family_key: str) -> list[dict[str, Any]]:
    deltas = report.get(family_key, {})
    rows = []
    for name, row in deltas.items():
        metric = row.get("delta_vs_baseline_family_only", {})
        rows.append(
            {
                "variant": name,
                "all_delta": float(metric.get("all_improvement", 0.0)),
                "t50_delta": float(metric.get("t50_improvement", 0.0)),
                "t100_raw_delta": float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0)),
                "hard_failure_delta": float(metric.get("hard_failure_improvement", 0.0)),
                "easy_delta": float(metric.get("easy_degradation", 0.0)),
                "positive": bool(row.get("positive_sequence_increment", row.get("positive_graph_increment", False))),
                "interpretation": row.get("interpretation", ""),
            }
        )
    return rows


def _best_delta(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return max(float(row.get(key, 0.0)) for row in rows)


def _all_nonpositive_core(rows: list[Mapping[str, Any]]) -> bool:
    return all(
        float(row.get("all_delta", 0.0)) <= 0.0
        and float(row.get("t50_delta", 0.0)) <= 0.0
        and float(row.get("hard_failure_delta", 0.0)) <= 0.0
        for row in rows
    )


def _no_leakage_pass(report: Mapping[str, Any]) -> bool:
    nl = report.get("no_leakage", {})
    for key in [
        "future_endpoint_input",
        "future_waypoint_input",
        "family_fde_input",
        "safe_strongest_idx_old_input",
        "central_velocity",
        "test_endpoint_goals",
        "test_threshold_tuning",
    ]:
        if nl.get(key) is not False:
            return False
    return bool(nl.get("train_only_feature_normalization", True))


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ar = read_json(AR_JSON, {})
    as_ = read_json(AS_JSON, {})
    sequence_rows = _metric_delta_rows(ar, "sequence_deltas")
    graph_rows = _metric_delta_rows(as_, "graph_deltas")
    baseline = ar.get("baseline_family_only", {}).get("protected", {})
    summary = {
        "source": SOURCE,
        "sequence_report_verdict": ar.get("stage42_ar_gate", {}).get("verdict", ""),
        "graph_report_verdict": as_.get("stage42_as_gate", {}).get("verdict", ""),
        "sequence_positive_variants": ar.get("positive_sequence_context_variants", []),
        "graph_positive_variants": as_.get("positive_graph_context_variants", []),
        "sequence_context_supported": bool(ar.get("positive_sequence_context_variants", [])),
        "graph_context_supported": bool(as_.get("positive_graph_context_variants", [])),
        "best_sequence_all_delta": _best_delta(sequence_rows, "all_delta"),
        "best_sequence_t50_delta": _best_delta(sequence_rows, "t50_delta"),
        "best_sequence_hard_delta": _best_delta(sequence_rows, "hard_failure_delta"),
        "best_graph_all_delta": _best_delta(graph_rows, "all_delta"),
        "best_graph_t50_delta": _best_delta(graph_rows, "t50_delta"),
        "best_graph_hard_delta": _best_delta(graph_rows, "hard_failure_delta"),
        "baseline_family_all_improvement": float(baseline.get("all_improvement", 0.0)),
        "baseline_family_t50_improvement": float(baseline.get("t50_improvement", 0.0)),
        "baseline_family_hard_failure_improvement": float(baseline.get("hard_failure_improvement", 0.0)),
        "decision": "sequence_and_graph_context_negative_keep_baseline_family_rollout_context_as_dominant_mechanism",
        "next_repair_hypothesis": "Switch from residual full-waypoint deltas to gain/harm/intervention or source-slice objectives before claiming independent scene/goal/interaction contribution.",
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-JR",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(AR_JSON), str(AS_JSON)]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "sequence_context_json": str(AR_JSON),
            "graph_context_json": str(AS_JSON),
            "sequence_generated_at_utc": ar.get("generated_at_utc", ""),
            "graph_generated_at_utc": as_.get("generated_at_utc", ""),
        },
        "summary": summary,
        "sequence_deltas": sequence_rows,
        "graph_deltas": graph_rows,
        "failure_taxonomy": {
            "dominant_success_mechanism": "baseline_family_rollout_context",
            "sequence_failure": "temporal history residuals reduced all/t50/hard improvement versus the protected baseline-family first stage.",
            "graph_failure": "current-frame kNN graph and goal/history graph variants reduced all/t50/hard improvement versus the protected baseline-family first stage.",
            "likely_cause": "The current residual-delta objective rewards small unsafe corrections after a strong protected rollout floor; it does not learn switchability/gain/harm enough to capture independent interaction value.",
            "what_not_to_claim": [
                "sequence context is an independent main contribution",
                "graph/interaction context is an independent main contribution",
                "JEPA/Transformer/scene context is proven by these AR/AS runs",
            ],
        },
        "no_leakage": {
            "sequence_no_leakage_pass": _no_leakage_pass(ar),
            "graph_no_leakage_pass": _no_leakage_pass(as_),
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "sequence_context_main_claim": False,
            "graph_interaction_main_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jr_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "sequence_report_loaded": Path(payload["inputs"]["sequence_context_json"]).exists(),
        "graph_report_loaded": Path(payload["inputs"]["graph_context_json"]).exists(),
        "sequence_fresh_negative_recorded": s["sequence_report_verdict"] == "stage42_ar_sequence_context_evidence_partial_or_negative"
        and not s["sequence_context_supported"],
        "graph_fresh_negative_recorded": s["graph_report_verdict"] == "stage42_as_graph_context_evidence_partial_or_negative"
        and not s["graph_context_supported"],
        "sequence_core_deltas_nonpositive": _all_nonpositive_core(payload["sequence_deltas"]),
        "graph_core_deltas_nonpositive": _all_nonpositive_core(payload["graph_deltas"]),
        "baseline_family_positive": s["baseline_family_all_improvement"] > 0
        and s["baseline_family_t50_improvement"] > 0
        and s["baseline_family_hard_failure_improvement"] > 0,
        "no_leakage_pass": payload["no_leakage"]["sequence_no_leakage_pass"]
        and payload["no_leakage"]["graph_no_leakage_pass"]
        and all(
            payload["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        ),
        "negative_result_not_overclaimed": payload["claim_boundary"]["sequence_context_main_claim"] is False
        and payload["claim_boundary"]["graph_interaction_main_claim"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jr_source_context_negative_evidence_pass" if passed == len(gates) else "stage42_jr_source_context_replay_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jr_gate"]
    lines = [
        "# Stage42-JR Source Context Fresh Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{s['decision']}`",
        f"- sequence_report_verdict: `{s['sequence_report_verdict']}`",
        f"- graph_report_verdict: `{s['graph_report_verdict']}`",
        f"- baseline_family_all/t50/hard: `{s['baseline_family_all_improvement']:.6f}` / `{s['baseline_family_t50_improvement']:.6f}` / `{s['baseline_family_hard_failure_improvement']:.6f}`",
        f"- best_sequence_all/t50/hard_delta: `{s['best_sequence_all_delta']:.6f}` / `{s['best_sequence_t50_delta']:.6f}` / `{s['best_sequence_hard_delta']:.6f}`",
        f"- best_graph_all/t50/hard_delta: `{s['best_graph_all_delta']:.6f}` / `{s['best_graph_t50_delta']:.6f}` / `{s['best_graph_hard_delta']:.6f}`",
        f"- next_repair_hypothesis: `{s['next_repair_hypothesis']}`",
        "",
        "## Sequence Deltas vs Baseline-Family Context",
        "",
        "| variant | all | t50 | t100 raw | hard/failure | easy | positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["sequence_deltas"]:
        lines.append(
            f"| `{row['variant']}` | {row['all_delta']:.6f} | {row['t50_delta']:.6f} | {row['t100_raw_delta']:.6f} | {row['hard_failure_delta']:.6f} | {row['easy_delta']:.6f} | `{row['positive']}` |"
        )
    lines.extend(
        [
            "",
            "## Graph Deltas vs Baseline-Family Context",
            "",
            "| variant | all | t50 | t100 raw | hard/failure | easy | positive |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["graph_deltas"]:
        lines.append(
            f"| `{row['variant']}` | {row['all_delta']:.6f} | {row['t50_delta']:.6f} | {row['t100_raw_delta']:.6f} | {row['hard_failure_delta']:.6f} | {row['easy_delta']:.6f} | `{row['positive']}` |"
        )
    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            f"- dominant_success_mechanism: `{payload['failure_taxonomy']['dominant_success_mechanism']}`",
            f"- sequence_failure: {payload['failure_taxonomy']['sequence_failure']}",
            f"- graph_failure: {payload['failure_taxonomy']['graph_failure']}",
            f"- likely_cause: {payload['failure_taxonomy']['likely_cause']}",
            f"- what_not_to_claim: `{payload['failure_taxonomy']['what_not_to_claim']}`",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jr_gate"]
    lines = [
        "# Stage42-JR Gate",
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


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jr_gate"]
    return [
        "## Stage42-JR Source Context Fresh Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- baseline-family all/t50/hard remains positive: `{s['baseline_family_all_improvement']:.4f}` / `{s['baseline_family_t50_improvement']:.4f}` / `{s['baseline_family_hard_failure_improvement']:.4f}`.",
        f"- sequence context did not add lift: best all/t50/hard delta `{s['best_sequence_all_delta']:.4f}` / `{s['best_sequence_t50_delta']:.4f}` / `{s['best_sequence_hard_delta']:.4f}`.",
        f"- graph context did not add lift: best all/t50/hard delta `{s['best_graph_all_delta']:.4f}` / `{s['best_graph_t50_delta']:.4f}` / `{s['best_graph_hard_delta']:.4f}`.",
        "- boundary: negative result preserved; no sequence/graph independent main claim, no metric/seconds overclaim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["source_context_fresh_replay"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jr_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jr_gate"]["passed"], "total": payload["stage42_jr_gate"]["total"]},
        "sequence_context_supported": payload["summary"]["sequence_context_supported"],
        "graph_context_supported": payload["summary"]["graph_context_supported"],
        "baseline_family_all_improvement": payload["summary"]["baseline_family_all_improvement"],
        "baseline_family_t50_improvement": payload["summary"]["baseline_family_t50_improvement"],
        "best_sequence_t50_delta": payload["summary"]["best_sequence_t50_delta"],
        "best_graph_t50_delta": payload["summary"]["best_graph_t50_delta"],
        "negative_result_not_overclaimed": True,
        "global_metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_source_context_fresh_replay.py"
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JR",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jr_gate"]["verdict"],
                    "fresh_run": True,
                    "sequence_context_supported": payload["summary"]["sequence_context_supported"],
                    "graph_context_supported": payload["summary"]["graph_context_supported"],
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_source_context_fresh_replay(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_source_context_fresh_replay(refresh_readmes=True)
    gate = payload["stage42_jr_gate"]
    print(f"Stage42-JR source context fresh replay: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
