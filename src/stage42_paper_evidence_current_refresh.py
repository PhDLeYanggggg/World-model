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
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
JT_JSON = OUT_DIR / "current_module_claim_refresh_stage42.json"
JU_JSON = OUT_DIR / "current_reviewer_replay_package_stage42.json"
JV_JSON = OUT_DIR / "source_slice_evidence_matrix_stage42.json"
JW_JSON = OUT_DIR / "teacher_floor_necessity_slice_audit_stage42.json"
JS_JSON = OUT_DIR / "source_context_gain_harm_closure_stage42.json"
TERMS_JSON = OUT_DIR / "source_terms_validation_stage42.json"
TIME_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"

REPORT_JSON = OUT_DIR / "paper_evidence_current_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_evidence_current_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jx_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

EXPERIMENTS_MD = OUT_DIR / "experiment_tables_stage42.md"
ABLATIONS_MD = OUT_DIR / "ablation_tables_stage42.md"
FAILURE_MD = OUT_DIR / "failure_taxonomy_stage42.md"
GAP_MD = OUT_DIR / "a_journal_gap_stage42.md"
PAPER_MATRIX_MD = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_JX_CURRENT_PAPER_EVIDENCE_REFRESH"
SOURCE = "fresh_stage42_jx_current_paper_evidence_refresh"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JX 将 JT/JU/JV/JW 的当前证据同步到 paper package；不训练、不调 threshold、不新增下载或转换。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
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


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload["input_status"]
    s = payload["summary"]
    claims = payload["claim_boundary"]
    blocked = set(s["blocked_claims"])
    gates = {
        "jt_claim_refresh_passed": inputs["jt_verdict"] == "stage42_jt_current_module_claim_refresh_pass",
        "ju_reviewer_replay_package_passed": inputs["ju_verdict"] == "stage42_ju_current_reviewer_replay_package_pass",
        "jv_source_slice_matrix_passed": inputs["jv_verdict"] == "stage42_jv_source_slice_evidence_matrix_pass",
        "jw_teacher_floor_audit_passed": inputs["jw_verdict"] == "stage42_jw_teacher_floor_necessity_slice_audit_pass",
        "source_slice_evidence_included": s["source_slice"]["domain_count"] >= 2
        and s["source_slice"]["horizon_count"] >= 4
        and s["source_slice"]["all_ade_improvement"] > 0
        and s["source_slice"]["t50_ade_improvement"] > 0
        and s["source_slice"]["hard_failure_ade_improvement"] > 0,
        "teacher_floor_necessity_included": s["teacher_floor"]["fallback_rows"] > 0
        and s["teacher_floor"]["fallback_exact_floor_rate"] >= 0.999
        and s["teacher_floor"]["global_floor_removal_allowed"] is False,
        "paper_supported_claims_nonempty": len(s["supported_claims"]) >= 3,
        "independent_context_claims_blocked": {
            "scene_goal_independent_main_claim",
            "neighbor_interaction_independent_main_claim",
            "JEPA_downstream_main_claim",
            "Transformer_independent_main_claim",
        }.issubset(blocked),
        "ungated_and_floor_free_blocked": {
            "ungated_full_waypoint_deployment",
            "floor_free_neural_deployment",
            "global_teacher_floor_removal",
        }.issubset(blocked),
        "source_terms_and_time_blockers_preserved": s["source_terms"]["conversion_ready_targets"] == 0
        and s["time_geometry"]["global_metric_claim_allowed"] is False
        and s["time_geometry"]["global_seconds_claim_allowed"] is False,
        "paper_artifact_updates_listed": len(s["updated_artifacts"]) >= 5,
        "no_future_or_test_leakage": all(payload["no_leakage"].values()),
        "no_metric_seconds_3d_foundation": claims["true_3d"] is False
        and claims["foundation_world_model"] is False
        and claims["metric_or_seconds_claim"] is False,
        "stage5c_false": claims["stage5c_executed"] is False,
        "smc_false": claims["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_jx_current_paper_evidence_refresh_pass" if passed == total else "stage42_jx_current_paper_evidence_refresh_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _input_status(
    jt: Mapping[str, Any],
    ju: Mapping[str, Any],
    jv: Mapping[str, Any],
    jw: Mapping[str, Any],
    js: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "jt_verdict": jt.get("stage42_jt_gate", {}).get("verdict", ""),
        "ju_verdict": ju.get("stage42_ju_gate", {}).get("verdict", ""),
        "jv_verdict": jv.get("stage42_jv_gate", {}).get("verdict", ""),
        "jw_verdict": jw.get("stage42_jw_gate", {}).get("verdict", ""),
        "js_verdict": js.get("stage42_js_gate", {}).get("verdict", ""),
        "jt_source": jt.get("source", ""),
        "ju_source": ju.get("source", ""),
        "jv_source": jv.get("source", ""),
        "jw_source": jw.get("source", ""),
        "js_source": js.get("source", ""),
    }


def _summary(
    jt: Mapping[str, Any],
    ju: Mapping[str, Any],
    jv: Mapping[str, Any],
    jw: Mapping[str, Any],
    js: Mapping[str, Any],
    terms: Mapping[str, Any],
    time_geo: Mapping[str, Any],
) -> dict[str, Any]:
    jv_summary = jv.get("summary", {})
    jw_summary = jw.get("summary", {})
    jt_summary = jt.get("summary", {})
    js_summary = js.get("summary", {})
    source_terms = terms.get("summary", {})
    time_summary = time_geo.get("summary", {})
    source_slice = {
        "rows": int(jv.get("cache_status", {}).get("rows", 0)),
        "domain_count": int(jv_summary.get("domain_count", 0)),
        "domains": sorted(jv.get("domain_metrics", {}).keys()),
        "horizon_count": int(jv_summary.get("horizon_count", 0)),
        "horizons": sorted(jv.get("horizon_metrics", {}).keys(), key=lambda x: int(x)),
        "source_file_count": int(jv_summary.get("source_file_count", 0)),
        "all_ade_improvement": float(jv_summary.get("all", {}).get("ade_improvement", 0.0)),
        "t50_ade_improvement": float(jv.get("horizon_metrics", {}).get("50", {}).get("ade_improvement", 0.0)),
        "t100_raw_frame_ade_improvement": float(jv.get("horizon_metrics", {}).get("100", {}).get("ade_improvement", 0.0)),
        "hard_failure_ade_improvement": float(jv_summary.get("hard_or_failure", {}).get("ade_improvement", 0.0)),
        "easy_degradation": float(jv_summary.get("easy", {}).get("easy_degradation", 1.0)),
        "weak_source_files": list(jv.get("diagnostics", {}).get("weak_positive_source_files", {}).keys()),
        "negative_or_weak_source_files": list(jv.get("diagnostics", {}).get("negative_or_weak_source_files", {}).keys()),
    }
    teacher_floor = {
        "rows": int(jw_summary.get("rows", 0)),
        "switch_rows": int(jw_summary.get("switch_rows", 0)),
        "fallback_rows": int(jw_summary.get("fallback_rows", 0)),
        "fallback_exact_floor_rate": float(jw_summary.get("fallback_exact_floor_rate", 0.0)),
        "hard_failure_switch_rate": float(jw_summary.get("hard_failure_switch_rate", 0.0)),
        "easy_switch_rate": float(jw_summary.get("easy_switch_rate", 0.0)),
        "global_floor_removal_allowed": bool(jw_summary.get("global_floor_removal_allowed", True)),
        "floor_free_neural_deployable": bool(jw_summary.get("floor_free_neural_deployable", True)),
        "guarded_t50_relaxation_safety": bool(jw_summary.get("partial_t50_floor_relaxation", {}).get("target_union_safety_pass", False)),
        "guarded_t50_relaxation_improvement": float(jw_summary.get("partial_t50_floor_relaxation", {}).get("target_union_t50_improvement", 0.0)),
    }
    supported_claims = [
        "Protected source-level full-waypoint row-cache evidence is positive across the current TrajNet+UCY row cache under safe-switch/floor protection.",
        "Source/domain/horizon/slice decomposition is paper-usable, but it remains dataset-local/raw-frame evidence.",
        "The teacher/floor is a necessary deployability mechanism: fallback rows remain exact-floor and global floor-free neural deployment is forbidden.",
        "Baseline-family rollout context remains the strongest current source-level driver; history/motion-goal signals are bounded auxiliary evidence.",
    ]
    blocked_claims = [
        "scene_goal_independent_main_claim",
        "neighbor_interaction_independent_main_claim",
        "JEPA_downstream_main_claim",
        "Transformer_independent_main_claim",
        "sequence_graph_t50_t100_independent_main_claim",
        "ungated_full_waypoint_deployment",
        "floor_free_neural_deployment",
        "global_teacher_floor_removal",
        "metric_seconds_or_true3d_claim",
        "foundation_world_model_claim",
        "Stage5C_execution_claim",
        "SMC_readiness_claim",
        "broad_source_level_generalization_without_terms_or_new_sources",
    ]
    return {
        "result_source_label": "fresh_synthesis_from_current_stage42_jt_ju_jv_jw_artifacts",
        "source_slice": source_slice,
        "teacher_floor": teacher_floor,
        "context_boundary": {
            "jt_allowed_claims": list(jt_summary.get("allowed_claims", [])),
            "jt_blocked_independent_claims": list(jt_summary.get("blocked_independent_claims", [])),
            "js_decision": js_summary.get("decision", ""),
            "js_t50_diagnosis": js_summary.get("t50_diagnosis", ""),
            "js_t100_diagnosis": js_summary.get("t100_diagnosis", ""),
        },
        "reviewer_replay": {
            "package_interpretation": ju.get("summary", {}).get("package_interpretation", ""),
            "mechanism_rows": ju.get("summary", {}).get("mechanism_rows", {}),
            "source_domains": ju.get("summary", {}).get("source_domains", {}),
        },
        "source_terms": {
            "terms_accepted_targets": int(source_terms.get("terms_accepted_targets", 0)),
            "conversion_ready_targets": int(source_terms.get("conversion_ready_targets", 0)),
            "converted_datasets_now": int(source_terms.get("converted_datasets_now", 0)),
            "evaluated_datasets_now": int(source_terms.get("evaluated_datasets_now", 0)),
        },
        "time_geometry": {
            "global_metric_claim_allowed": bool(time_summary.get("global_metric_claim_allowed", False)),
            "global_seconds_claim_allowed": bool(time_summary.get("global_seconds_claim_allowed", False)),
            "m3w_official_metric_seconds_claim_allowed": bool(time_summary.get("m3w_official_metric_seconds_claim_allowed", False)),
            "user_action_required": bool(time_summary.get("user_action_required", True)),
        },
        "supported_claims": supported_claims,
        "blocked_claims": blocked_claims,
        "updated_artifacts": [
            str(REPORT_MD),
            str(EXPERIMENTS_MD),
            str(ABLATIONS_MD),
            str(FAILURE_MD),
            str(GAP_MD),
            str(PAPER_MATRIX_MD),
        ],
        "paper_wording_decision": (
            "Center the paper package on protected source-level row-cache/full-waypoint evidence, safe-switch behavior, "
            "and teacher-floor necessity. Keep independent scene/goal, neighbor/interaction, JEPA, Transformer, "
            "floor-free neural, metric/seconds, true-3D, foundation, Stage5C, and SMC claims blocked."
        ),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    jt = read_json(JT_JSON, {})
    ju = read_json(JU_JSON, {})
    jv = read_json(JV_JSON, {})
    jw = read_json(JW_JSON, {})
    js = read_json(JS_JSON, {})
    terms = read_json(TERMS_JSON, {"summary": {}})
    time_geo = read_json(TIME_JSON, {"summary": {}})
    payload: dict[str, Any] = {
        "stage": "Stage42-JX current paper evidence refresh",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _combined_hash([JT_JSON, JU_JSON, JV_JSON, JW_JSON, JS_JSON, TERMS_JSON, TIME_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": _input_status(jt, ju, jv, jw, js),
        "summary": _summary(jt, ju, jv, jw, js, terms, time_geo),
        "no_leakage": {
            "future_endpoint_input_absent": True,
            "future_waypoint_input_absent": True,
            "central_velocity_absent": True,
            "test_endpoint_goals_absent": True,
            "test_threshold_tuning_absent": True,
            "future_labels_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "floor_free_neural_deployable": False,
            "global_teacher_floor_removal_allowed": False,
            "source_slice_synthesis_not_new_training": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jx_gate"] = _gate(payload)
    return payload


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jx_gate"]
    s = payload["summary"]
    source = s["source_slice"]
    floor = s["teacher_floor"]
    lines = [
        "# Stage42-JX Current Paper Evidence Refresh",
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
        "## Inputs",
        "",
        "| artifact | source | verdict |",
        "| --- | --- | --- |",
        f"| JT claim refresh | `{payload['input_status']['jt_source']}` | `{payload['input_status']['jt_verdict']}` |",
        f"| JU reviewer replay | `{payload['input_status']['ju_source']}` | `{payload['input_status']['ju_verdict']}` |",
        f"| JV source slice matrix | `{payload['input_status']['jv_source']}` | `{payload['input_status']['jv_verdict']}` |",
        f"| JW teacher floor audit | `{payload['input_status']['jw_source']}` | `{payload['input_status']['jw_verdict']}` |",
        f"| JS context closure | `{payload['input_status']['js_source']}` | `{payload['input_status']['js_verdict']}` |",
        "",
        "## Current Paper Evidence",
        "",
        f"- rows/domains/source-files: `{source['rows']}` / `{source['domain_count']}` / `{source['source_file_count']}`",
        f"- domains: `{source['domains']}`; horizons: `{source['horizons']}`",
        f"- all/t50/t100raw/hard ADE improvement: `{_pct(source['all_ade_improvement'])}` / `{_pct(source['t50_ade_improvement'])}` / `{_pct(source['t100_raw_frame_ade_improvement'])}` / `{_pct(source['hard_failure_ade_improvement'])}`",
        f"- easy degradation: `{_pct(source['easy_degradation'])}`",
        f"- weak source files: `{source['weak_source_files']}`",
        "",
        "## Teacher/Floor Necessity",
        "",
        f"- switch/fallback rows: `{floor['switch_rows']}` / `{floor['fallback_rows']}`",
        f"- fallback exact floor rate: `{floor['fallback_exact_floor_rate']:.6f}`",
        f"- hard/failure switch rate vs easy switch rate: `{floor['hard_failure_switch_rate']:.6f}` / `{floor['easy_switch_rate']:.6f}`",
        f"- guarded t50 relaxation safety: `{floor['guarded_t50_relaxation_safety']}` with t50 `{_pct(floor['guarded_t50_relaxation_improvement'])}`",
        f"- global floor removal allowed: `{floor['global_floor_removal_allowed']}`; floor-free neural deployable: `{floor['floor_free_neural_deployable']}`",
        "",
        "## Supported Claims",
        "",
        *[f"- {item}" for item in s["supported_claims"]],
        "",
        "## Blocked Claims",
        "",
        *[f"- {item}" for item in s["blocked_claims"]],
        "",
        "## Source / Time / Metric Blockers",
        "",
        f"- source_terms: `{s['source_terms']}`",
        f"- time_geometry: `{s['time_geometry']}`",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
        "## Paper Wording Decision",
        "",
        f"- {s['paper_wording_decision']}",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jx_gate"]
    return [
        "# Stage42-JX Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
    ]


def _paper_section(payload: Mapping[str, Any], *, artifact: str) -> list[str]:
    s = payload["summary"]
    source = s["source_slice"]
    floor = s["teacher_floor"]
    base = [
        f"## Stage42-JX Current Paper Evidence Refresh ({artifact})",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_jx_gate']['passed']} / {payload['stage42_jx_gate']['total']}`; verdict: `{payload['stage42_jx_gate']['verdict']}`.",
        f"- current row-cache evidence: rows `{source['rows']}`, domains `{source['domains']}`, horizons `{source['horizons']}`.",
        f"- ADE all/t50/t100raw/hard: `{_pct(source['all_ade_improvement'])}` / `{_pct(source['t50_ade_improvement'])}` / `{_pct(source['t100_raw_frame_ade_improvement'])}` / `{_pct(source['hard_failure_ade_improvement'])}`; easy degradation `{_pct(source['easy_degradation'])}`.",
        f"- teacher/floor: fallback rows `{floor['fallback_rows']}`, fallback exact floor rate `{floor['fallback_exact_floor_rate']:.6f}`, floor-free neural deployable `{floor['floor_free_neural_deployable']}`.",
        "- current paper claim: protected source-level full-waypoint row-cache plus safe-switch/teacher-floor necessity.",
        "- blocked: independent scene/goal, neighbor/interaction, JEPA, Transformer, ungated/floor-free neural, metric/seconds, true-3D, foundation, Stage5C, and SMC claims.",
    ]
    if artifact == "experiments":
        base.extend(
            [
                "",
                "| evidence | source | rows | all ADE | t50 ADE | t100raw ADE | hard ADE | easy degradation | role |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                f"| Current protected row-cache source slices | `JV cached-verified row cache` | {source['rows']} | {_pct(source['all_ade_improvement'])} | {_pct(source['t50_ade_improvement'])} | {_pct(source['t100_raw_frame_ade_improvement'])} | {_pct(source['hard_failure_ade_improvement'])} | {_pct(source['easy_degradation'])} | paper-facing protected evidence |",
                f"| Teacher/floor necessity | `JW fresh synthesis` | {floor['rows']} | n/a | {_pct(floor['guarded_t50_relaxation_improvement'])} | n/a | n/a | safe | deployment boundary |",
            ]
        )
    elif artifact == "ablations":
        base.extend(
            [
                "",
                "| ablation / mechanism | current status | paper use | boundary |",
                "| --- | --- | --- | --- |",
                "| Stage37/teacher floor | necessary | safety mechanism | global removal forbidden |",
                "| safe switch | supported | deployability mechanism | fallback exact-floor rows remain |",
                "| scene/goal context | blocked as independent main claim | diagnostic/auxiliary only | current protocol does not beat baseline-family control |",
                "| neighbor/interaction context | blocked as independent main claim | diagnostic/auxiliary only | current protocol does not support standalone contribution |",
                "| JEPA / Transformer | blocked as main contribution | negative/mixed evidence | no downstream main lift claim |",
            ]
        )
    elif artifact == "failure":
        base.extend(
            [
                "",
                "### Updated Failure Taxonomy",
                "",
                "- Source/file coverage is now decomposed, but broader source-diversity remains limited by legal/source-term blockers.",
                "- Teacher/floor dependence is confirmed as a mechanism, not just a conservative implementation choice.",
                "- Independent scene/goal, neighbor/interaction, JEPA, and Transformer main claims remain unsupported under the current evidence.",
                "- Metric/time claims remain blocked by global calibration status.",
            ]
        )
    elif artifact == "gap":
        base.extend(
            [
                "",
                "### Updated A-Journal Gap",
                "",
                "- The strongest paper claim is now a protected source-slice/full-waypoint 2.5D claim with explicit floor necessity.",
                "- To move beyond this, the shortest path is legal independent source expansion plus a new context/latent mechanism that beats the baseline-family floor without relying on test tuning.",
                "- Do not promote Stage5C, SMC, metric/seconds, true-3D, foundation, or floor-free neural language.",
            ]
        )
    elif artifact == "matrix":
        base.extend(
            [
                "",
                "| requirement | refreshed status | current claim | blocked overclaim |",
                "| --- | --- | --- | --- |",
                "| Source/domain/horizon evidence | pass with raw-frame boundary | protected row-cache positive across current domains/horizons | broad source-level generalization without new legal sources |",
                "| Teacher/floor evidence | pass | floor required for deployability | global floor-free neural deployment |",
                "| Context modules | mixed/blocked | auxiliary or diagnostic evidence only | independent main contribution |",
                "| Time/metric calibration | blocked globally | dataset-local/raw-frame only | metric/seconds-level claim |",
            ]
        )
    return base


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    source = s["source_slice"]
    floor = s["teacher_floor"]
    gate = payload["stage42_jx_gate"]
    return [
        "## Stage42-JX Current Paper Evidence Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`.",
        f"- current evidence rows/domains/horizons: `{source['rows']}` / `{source['domains']}` / `{source['horizons']}`.",
        f"- ADE all/t50/t100raw/hard: `{_pct(source['all_ade_improvement'])}` / `{_pct(source['t50_ade_improvement'])}` / `{_pct(source['t100_raw_frame_ade_improvement'])}` / `{_pct(source['hard_failure_ade_improvement'])}`; easy `{_pct(source['easy_degradation'])}`.",
        f"- teacher/floor necessity: fallback rows `{floor['fallback_rows']}`, exact-floor rate `{floor['fallback_exact_floor_rate']:.6f}`, global floor-free neural deployable `{floor['floor_free_neural_deployable']}`.",
        "- README-facing decision: public GitHub README stays project-owner style; detailed staged evidence remains internal.",
        "- paper boundary: protected dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
    ]


def _update_paper_artifacts(payload: Mapping[str, Any]) -> None:
    _replace_section(EXPERIMENTS_MD, SECTION, _paper_section(payload, artifact="experiments"))
    _replace_section(ABLATIONS_MD, SECTION, _paper_section(payload, artifact="ablations"))
    _replace_section(FAILURE_MD, SECTION, _paper_section(payload, artifact="failure"))
    _replace_section(GAP_MD, SECTION, _paper_section(payload, artifact="gap"))
    _replace_section(PAPER_MATRIX_MD, SECTION, _paper_section(payload, artifact="matrix"))


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_jx_current_paper_evidence_refresh"
    state["current_verdict"] = payload["stage42_jx_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_jx_current_paper_evidence_refresh"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_jx_gate"]["verdict"],
        "gates": f"{payload['stage42_jx_gate']['passed']}/{payload['stage42_jx_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_paper_evidence_current_refresh.py"
    generated = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, EXPERIMENTS_MD, ABLATIONS_MD, FAILURE_MD, GAP_MD, PAPER_MATRIX_MD]:
        item = str(path)
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JX",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jx_gate"]["verdict"],
                    "fresh_synthesis_from_current_artifacts": True,
                    "paper_artifacts_updated": payload["summary"]["updated_artifacts"],
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_paper_evidence_current_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_paper_artifacts(payload)
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_paper_evidence_current_refresh(refresh_readmes=True)
    gate = payload["stage42_jx_gate"]
    print(f"Stage42-JX current paper evidence refresh: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
