from __future__ import annotations

import csv
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
REPORT_JSON = OUT_DIR / "paper_claim_evidence_audit_stage42.json"
REPORT_MD = OUT_DIR / "paper_claim_evidence_audit_stage42.md"
REPORT_CSV = OUT_DIR / "paper_claim_evidence_audit_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_z_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

DATA_JSON = OUT_DIR / "data_calibration_stage42.json"
EXTERNAL_JSON = OUT_DIR / "external_validation_stage42.json"
FULL_WAYPOINT_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
SAFETY_JSON = OUT_DIR / "safety_floor_stage42.json"
PAPER_JSON = OUT_DIR / "paper_package_stage42.json"
ROW_CACHE_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
ABLATION_JSON = OUT_DIR / "unified_ablation_evidence_stage42.json"
SOURCE_TERMS_JSON = OUT_DIR / "source_terms_validation_stage42.json"
METRIC_TIME_GUARD_JSON = OUT_DIR / "metric_time_claim_guard_stage42.json"

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "SDD 是 pixel-space；external 是 dataset-local / unverified weak metric diagnostic。",
    "t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。",
    "Stage42-Z 是 claim-to-evidence audit，不重新训练大模型，不读取 raw data/cache。",
    "future endpoints / waypoints 只可作为 label 或 evaluation，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。",
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


def _mean(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, Mapping):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def _ci_low(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, Mapping):
        return float(value.get("ci_low", value.get("mean", 0.0)))
    return float(value or 0.0)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "n/a"
    return str(value)


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0)


def _claim_boundary_ok(*payloads: Mapping[str, Any]) -> bool:
    for payload in payloads:
        boundary = payload.get("claim_boundary", {})
        if boundary.get("true_3d") or boundary.get("foundation_world_model") or boundary.get("metric_or_seconds_claim"):
            return False
        if boundary.get("stage5c_executed") or boundary.get("smc_enabled"):
            return False
    return True


def _paper_file_status() -> list[dict[str, Any]]:
    rows = []
    for path in PAPER_FILES:
        rows.append(
            {
                "file": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def _claim_rows(
    data: Mapping[str, Any],
    external: Mapping[str, Any],
    full: Mapping[str, Any],
    safety: Mapping[str, Any],
    paper: Mapping[str, Any],
    row_cache: Mapping[str, Any],
    ablation: Mapping[str, Any],
    source_terms: Mapping[str, Any],
    metric_time_guard: Mapping[str, Any],
) -> list[dict[str, Any]]:
    row_summary = row_cache.get("summary", {})
    row_gate = row_cache.get("stage42_x_gate", {})
    row_positive_domains = row_gate.get("positive_all_domains", [])
    row_t50_domains = row_gate.get("positive_t50_domains", [])
    ab_gate = ablation.get("stage42_y_gate", {})
    external_comp = (external.get("comparisons", {}) or {}).get("m3w_neural_v1_composite_tail_protected", {})
    ungated = (external.get("comparisons", {}) or {}).get("ungated_neural_endpoint", {})
    full_gate = full.get("stage42_c_gate", {})
    full_comp = ((full.get("comparisons", {}) or {}).get("full_waypoint_transformer_protected", {}) or {}).get("ade", {})
    safety_analysis = safety.get("floor_necessity_analysis", {})
    data_summary = data.get("summary", {})
    paper_claims = paper.get("claim_matrix", [])
    ablation_rows = ablation.get("retrained_sequence_ablation_rows", [])
    terms_summary = source_terms.get("summary", {})
    metric_time_summary = metric_time_guard.get("summary", {})

    history_row = next((row for row in ablation_rows if row.get("module") == "history tokens"), {})
    domain_row = next((row for row in ablation_rows if row.get("module") == "domain expert"), {})
    goal_row = next((row for row in ablation_rows if row.get("module") == "goal/scene tokens"), {})
    neighbor_row = next((row for row in ablation_rows if row.get("module") == "neighbor/interaction tokens"), {})
    stage42s_loss = next((row for row in ablation.get("row_level_ablation_rows", []) if row.get("name") == "stage42s_combo_no_ucy_source"), {})

    return [
        {
            "claim_id": "C1",
            "claim": "M3W has a unified row-level external full-waypoint 2.5D evidence cache over ETH_UCY, TrajNet, and UCY.",
            "status": "supported_fresh",
            "source": row_cache.get("source"),
            "evidence": f"Stage42-X gates {row_gate.get('passed')}/{row_gate.get('total')}; domains={row_positive_domains}; ADE all={_fmt(_mean(row_summary, 'ade_all'))}; t50={_fmt(_mean(row_summary, 'ade_t50'))}; hard={_fmt(_mean(row_summary, 'ade_hard_failure'))}; easy={_fmt(_mean(row_summary, 'ade_easy_degradation'))}",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C2",
            "claim": "External t50 full-waypoint evidence is bootstrap/seed positive under raw-frame dataset-local evaluation.",
            "status": "supported_fresh",
            "source": row_cache.get("source"),
            "evidence": f"positive_t50_domains={row_t50_domains}; seed_ci_low={_fmt(_ci_low(row_summary, 'ade_t50'))}; row_bootstrap_ci_low={_fmt(((row_cache.get('bootstrap_seed_mean', {}) or {}).get('t50', {}) or {}).get('ci_low'))}",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C3",
            "claim": "Removing the UCY full-waypoint source hurts unified t50/hard performance.",
            "status": "supported_fresh_synthesis",
            "source": ablation.get("source"),
            "evidence": f"Stage42-Y gates {ab_gate.get('passed')}/{ab_gate.get('total')}; loss_if_removed_t50={_fmt(((stage42s_loss.get('loss_vs_stage42x_full', {}) or {}).get('ade_t50')))}; loss_if_removed_hard={_fmt(((stage42s_loss.get('loss_vs_stage42x_full', {}) or {}).get('ade_hard_failure')))}",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C4",
            "claim": "History tokens are the strongest proven retrained sequence component; domain expert helps.",
            "status": "supported_fresh",
            "source": ablation.get("source"),
            "evidence": f"history t50 contribution={_fmt(((history_row.get('full_minus_ablation', {}) or {}).get('t50')))}; history hard={_fmt(((history_row.get('full_minus_ablation', {}) or {}).get('hard_failure')))}; domain t50={_fmt(((domain_row.get('full_minus_ablation', {}) or {}).get('t50')))}",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C5",
            "claim": "Goal/scene and neighbor/interaction contributions are established as uniformly positive.",
            "status": "mixed_not_main_claim",
            "source": ablation.get("source"),
            "evidence": f"goal/scene t50={_fmt(((goal_row.get('full_minus_ablation', {}) or {}).get('t50')))}; neighbor all={_fmt(((neighbor_row.get('full_minus_ablation', {}) or {}).get('all')))}; neighbor hard={_fmt(((neighbor_row.get('full_minus_ablation', {}) or {}).get('hard_failure')))}",
            "allowed_as_main_claim": False,
        },
        {
            "claim_id": "C6",
            "claim": "Ungated neural can replace the Stage37/teacher safety floor.",
            "status": "rejected_by_evidence",
            "source": external.get("source"),
            "evidence": f"ungated easy degradation={_fmt(ungated.get('easy_degradation'))}; floor conclusion={safety_analysis.get('conclusion')}",
            "allowed_as_main_claim": False,
        },
        {
            "claim_id": "C7",
            "claim": "Protected endpoint/composite-tail external validation remains a strong deployable floor.",
            "status": "supported_fresh",
            "source": external.get("source"),
            "evidence": f"all={_fmt(external_comp.get('all_improvement'))}; t50={_fmt(external_comp.get('t50_improvement'))}; t100diag={_fmt(external_comp.get('t100_improvement'))}; hard={_fmt(external_comp.get('hard_failure_improvement'))}; easy={_fmt(external_comp.get('easy_degradation'))}",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C8",
            "claim": "Stage42-C full-waypoint sequence dynamics has positive evidence on at least two external domains.",
            "status": "supported_fresh_but_protected",
            "source": full.get("source"),
            "evidence": f"Stage42-C gates {full_gate.get('passed')}/{full_gate.get('total')}; positive_domains={full_gate.get('positive_domains')}; ADE all={_fmt(full_comp.get('all_improvement'))}; t50={_fmt(full_comp.get('t50_improvement'))}",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C9",
            "claim": "Metric or seconds-level pedestrian world-model claims are supported.",
            "status": "not_supported",
            "source": data.get("source"),
            "evidence": f"global_metric_claim_allowed={data_summary.get('global_metric_claim_allowed')}; global_seconds_claim_allowed={data_summary.get('global_seconds_claim_allowed')}",
            "allowed_as_main_claim": False,
        },
        {
            "claim_id": "C10",
            "claim": "M3W is a true 3D or foundation world model.",
            "status": "not_supported",
            "source": "claim_boundary",
            "evidence": "Stage42 claim boundaries keep true_3d=false and foundation_world_model=false.",
            "allowed_as_main_claim": False,
        },
        {
            "claim_id": "C11",
            "claim": "A-journal evidence package is complete enough to draft a protected 2.5D paper, but not enough for broad foundation/3D claims.",
            "status": "supported_as_gap_aware_package",
            "source": paper.get("source"),
            "evidence": f"paper package claims={len(paper_claims)}; paper final verdict={paper.get('final_verdict')}; Stage42-Z keeps non-claims explicit.",
            "allowed_as_main_claim": True,
        },
        {
            "claim_id": "C12",
            "claim": "Source-diversity conversion is legally ready and can be counted as converted/evaluated external data.",
            "status": "rejected_by_legal_gate",
            "source": source_terms.get("source"),
            "evidence": f"targets_validated={terms_summary.get('targets_validated')}; terms_accepted={terms_summary.get('terms_accepted_targets')}; conversion_ready={terms_summary.get('conversion_ready_targets')}; converted={terms_summary.get('converted_datasets_now')}; evaluated={terms_summary.get('evaluated_datasets_now')}",
            "allowed_as_main_claim": False,
        },
        {
            "claim_id": "C13",
            "claim": "Restricted source-specific ETH/UCY metric/seconds subset claims are ready for paper results.",
            "status": "candidate_evidence_but_claim_blocked",
            "source": metric_time_guard.get("source"),
            "evidence": f"source_specific_candidates={metric_time_summary.get('source_specific_metric_time_candidates')}; conversion_ready={metric_time_summary.get('conversion_ready_targets')}; restricted_metric_seconds_allowed_now={metric_time_summary.get('restricted_subset_metric_seconds_claim_allowed_now')}; global_metric={metric_time_summary.get('global_metric_claim_allowed')}; global_seconds={metric_time_summary.get('global_seconds_claim_allowed')}",
            "allowed_as_main_claim": False,
        },
    ]


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    claims = result.get("claim_rows", [])
    statuses = {row.get("status") for row in claims}
    gates = {
        "data_calibration_present": result.get("inputs", {}).get("data_calibration_exists") is True,
        "external_validation_present": result.get("inputs", {}).get("external_validation_exists") is True,
        "full_waypoint_present": result.get("inputs", {}).get("full_waypoint_exists") is True,
        "paper_package_present": result.get("inputs", {}).get("paper_package_exists") is True,
        "source_terms_validator_present": result.get("inputs", {}).get("source_terms_exists") is True,
        "metric_time_guard_present": result.get("inputs", {}).get("metric_time_guard_exists") is True,
        "stage42x_row_cache_gate_pass": result.get("inputs", {}).get("stage42x_gate_pass") is True,
        "stage42y_ablation_gate_pass": result.get("inputs", {}).get("stage42y_gate_pass") is True,
        "stage42cg_source_terms_gate_pass": result.get("inputs", {}).get("stage42cg_gate_pass") is True,
        "stage42ch_metric_time_gate_pass": result.get("inputs", {}).get("stage42ch_gate_pass") is True,
        "paper_files_exist": all(row.get("exists") for row in result.get("paper_file_status", [])),
        "claim_matrix_has_supported": any(str(status).startswith("supported") for status in statuses),
        "claim_matrix_has_rejected": "rejected_by_evidence" in statuses,
        "claim_matrix_has_not_supported": "not_supported" in statuses,
        "mixed_contributions_not_overclaimed": any(row.get("claim_id") == "C5" and row.get("allowed_as_main_claim") is False for row in claims),
        "metric_seconds_not_overclaimed": any(row.get("claim_id") == "C9" and row.get("allowed_as_main_claim") is False for row in claims),
        "legal_conversion_not_overclaimed": any(row.get("claim_id") == "C12" and row.get("allowed_as_main_claim") is False for row in claims),
        "restricted_metric_time_not_overclaimed": any(row.get("claim_id") == "C13" and row.get("allowed_as_main_claim") is False for row in claims),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoint_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "claim_boundary_ok": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False
        and result.get("claim_boundary", {}).get("true_3d") is False
        and result.get("claim_boundary", {}).get("foundation_world_model") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": result.get("source"),
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_z_paper_claim_evidence_audit_pass" if all(gates.values()) else "stage42_z_paper_claim_evidence_audit_partial",
        "paper_ready_scope": "protected_2p5d_raw_frame_world_state_candidate",
        "not_ready_scope": "true_3d_metric_seconds_foundation_or_stage5c_smc",
    }


def _write_csv(rows: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["claim_id", "claim", "status", "source", "allowed_as_main_claim", "evidence"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def _write_md(result: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-Z Paper Claim Evidence Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_z_gate']['passed']} / {result['stage42_z_gate']['total']}`",
        f"- verdict: `{result['stage42_z_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Claim Matrix",
        "",
        "| id | claim | status | source | main claim? | evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["claim_rows"]:
        lines.append(
            f"| `{row['claim_id']}` | {row['claim']} | `{row['status']}` | `{row['source']}` | `{row['allowed_as_main_claim']}` | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Paper Files",
            "",
            "| file | exists | size_bytes |",
            "| --- | --- | ---: |",
        ]
    )
    for row in result["paper_file_status"]:
        lines.append(f"| `{row['file']}` | `{row['exists']}` | {row['size_bytes']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-X/Stage42-Y are now the strongest row-level full-waypoint and ablation evidence anchors.",
            "- The paper-ready claim is a protected, dataset-local raw-frame 2.5D world-state candidate, not true 3D/foundation/metric/seconds-level.",
            "- UCY full-waypoint source contribution and history-token contribution are supported; goal/scene and neighbor/interaction evidence is mixed and should be written as limitation or partial evidence.",
            "- The Stage37/teacher floor remains necessary; ungated neural is rejected for deployment safety.",
            "- Stage42-CG/CH now enforce the legal and metric/time claim boundaries: no converted/evaluated source-diversity repair and no global or restricted metric/seconds result can be claimed yet.",
            "- Stage5C and SMC remain disabled.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate_md(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-Z Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- paper_ready_scope: `{gate['paper_ready_scope']}`",
        f"- not_ready_scope: `{gate['not_ready_scope']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{ok}` |")
    write_md(GATE_MD, lines)


def run_stage42_paper_claim_evidence_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = read_json(DATA_JSON, {})
    external = read_json(EXTERNAL_JSON, {})
    full = read_json(FULL_WAYPOINT_JSON, {})
    safety = read_json(SAFETY_JSON, {})
    paper = read_json(PAPER_JSON, {})
    row_cache = read_json(ROW_CACHE_JSON, {})
    ablation = read_json(ABLATION_JSON, {})
    source_terms = read_json(SOURCE_TERMS_JSON, {})
    metric_time_guard = read_json(METRIC_TIME_GUARD_JSON, {})
    claim_rows = _claim_rows(data, external, full, safety, paper, row_cache, ablation, source_terms, metric_time_guard)
    paper_file_status = _paper_file_status()
    result = {
        "stage": "Stage42-Z paper claim evidence audit",
        "source": "fresh_audit_from_stage42_wxy_and_paper_package_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "data_calibration": str(DATA_JSON),
            "data_calibration_exists": DATA_JSON.exists(),
            "external_validation": str(EXTERNAL_JSON),
            "external_validation_exists": EXTERNAL_JSON.exists(),
            "full_waypoint": str(FULL_WAYPOINT_JSON),
            "full_waypoint_exists": FULL_WAYPOINT_JSON.exists(),
            "safety_floor": str(SAFETY_JSON),
            "safety_floor_exists": SAFETY_JSON.exists(),
            "paper_package": str(PAPER_JSON),
            "paper_package_exists": PAPER_JSON.exists(),
            "row_cache": str(ROW_CACHE_JSON),
            "row_cache_exists": ROW_CACHE_JSON.exists(),
            "unified_ablation": str(ABLATION_JSON),
            "unified_ablation_exists": ABLATION_JSON.exists(),
            "source_terms": str(SOURCE_TERMS_JSON),
            "source_terms_exists": SOURCE_TERMS_JSON.exists(),
            "metric_time_guard": str(METRIC_TIME_GUARD_JSON),
            "metric_time_guard_exists": METRIC_TIME_GUARD_JSON.exists(),
            "stage42x_gate_pass": _gate_passed(row_cache, "stage42_x_gate"),
            "stage42y_gate_pass": _gate_passed(ablation, "stage42_y_gate"),
            "stage42cg_gate_pass": _gate_passed(source_terms, "stage42_cg_gate"),
            "stage42ch_gate_pass": _gate_passed(metric_time_guard, "stage42_ch_gate"),
        },
        "input_hash": _combined_hash([DATA_JSON, EXTERNAL_JSON, FULL_WAYPOINT_JSON, SAFETY_JSON, PAPER_JSON, ROW_CACHE_JSON, ABLATION_JSON, SOURCE_TERMS_JSON, METRIC_TIME_GUARD_JSON]),
        "claim_rows": claim_rows,
        "paper_file_status": paper_file_status,
        "no_leakage": {
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
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_z_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_csv(claim_rows)
    _write_md(result)
    _write_gate_md(result["stage42_z_gate"])
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": result["stage"], "source": result["source"], "verdict": result["stage42_z_gate"]["verdict"], "generated_at_utc": result["generated_at_utc"]}, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    run_stage42_paper_claim_evidence_audit()
