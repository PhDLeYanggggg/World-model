from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
DATA_CALIBRATION_JSON = OUT_DIR / "data_calibration_stage42.json"
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"

REPORT_JSON = OUT_DIR / "metric_time_claim_guard_stage42.json"
REPORT_MD = OUT_DIR / "metric_time_claim_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ch_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_metric_time_claim_guard_stage42.md"

SOURCE = "fresh_stage42_ch_metric_time_claim_guard"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CH 是 metric/time claim guard，不下载、不转换、不训练、不评估。",
    "source-specific calibration evidence 不等于 global M3W metric/seconds claim。",
    "legal/source terms readiness 和 metric/time calibration 是两个独立前置条件。",
    "t+50 / t+100 默认仍是 raw-frame horizon，不能写成 seconds-level。",
    "SDD 当前仍是 pixel raw-frame；estimated scale 只能诊断，不能作为 official metric claim。",
    "TGSIM 是 traffic diagnostic，不能包装成 pedestrian top-down world-model success。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_claim_rows(bn: Mapping[str, Any], cg: Mapping[str, Any]) -> list[dict[str, Any]]:
    legal_ready_count = int(cg["summary"]["conversion_ready_targets"])
    rows: list[dict[str, Any]] = []
    for record in bn["source_records"]:
        has_source_metric_time = bool(record["source_specific_metric_time_evidence"])
        rows.append(
            {
                "source_id": record["source_id"],
                "domain": record["domain"],
                "dataset": record["dataset"],
                "source_specific_metric_time_evidence": has_source_metric_time,
                "annotation_fps": record["timing"]["annotation_fps"],
                "annotation_timestep_seconds": record["timing"]["annotation_timestep_seconds"],
                "h50_seconds_if_restricted": record["timing"]["h50_annotation_seconds"],
                "h100_seconds_if_restricted": record["timing"]["h100_annotation_seconds"],
                "homography_parseable": record["homography"]["parseable"],
                "allowed_local_claim": record["allowed_local_claim"],
                "paper_metric_seconds_claim_allowed_now": False,
                "reason": _reason(record, legal_ready_count),
            }
        )
    return rows


def _reason(record: Mapping[str, Any], legal_ready_count: int) -> str:
    if not record["source_specific_metric_time_evidence"]:
        return "source-specific metric/time evidence incomplete"
    if legal_ready_count <= 0:
        return "calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets"
    return "future restricted-subset claim may be possible only after legal conversion, no-leakage, source-CV, and final eval"


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bn = _load_json(BN_JSON)
    data = _load_json(DATA_CALIBRATION_JSON)
    cg = _load_json(CG_JSON)
    source_rows = _source_claim_rows(bn, cg)
    candidate_count = sum(1 for row in source_rows if row["source_specific_metric_time_evidence"])
    summary = {
        "source": SOURCE,
        "bn_verdict": bn["stage42_bn_gate"]["verdict"],
        "cg_verdict": cg["stage42_cg_gate"]["verdict"],
        "datasets_audited": data["summary"]["datasets_audited"],
        "source_records_audited": bn["summary"]["source_records_audited"],
        "source_specific_metric_time_candidates": candidate_count,
        "conversion_ready_targets": cg["summary"]["conversion_ready_targets"],
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
        "restricted_subset_metric_seconds_claim_allowed_now": False,
        "sdd_metric_claim_allowed": False,
        "tgsim_pedestrian_world_model_metric_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-CH Metric/Time Claim Guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BN_JSON), str(DATA_CALIBRATION_JSON), str(CG_JSON)]),
        "current_facts": CURRENT_FACTS,
        "summary": summary,
        "source_claim_rows": source_rows,
        "sdd_claim": {
            "allowed": False,
            "reason": "SDD remains pixel raw-frame; estimated scales are diagnostic and not an official metric claim.",
        },
        "tgsim_claim": {
            "allowed_for_pedestrian_world_model": False,
            "reason": "TGSIM metric evidence is traffic diagnostic only.",
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "restricted_subset_metric_seconds_claim_allowed_now": False,
            "metric_or_seconds_claim_requires_future_restricted_eval": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": [
            "For any metric/seconds-level claim, first supply validated source terms/path/source identity and rerun conversion readiness.",
            "Then run a restricted-subset no-leakage/source-CV/final-test evaluation using only calibrated ETH/UCY sources.",
            "Keep SDD, TrajNet snippets, and global M3W claims raw-frame/dataset-local until separately verified.",
        ],
    }
    payload["stage42_ch_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "bn_input_verified": s["bn_verdict"] == "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
        "cg_input_verified": s["cg_verdict"] == "stage42_cg_source_terms_confirmation_validator_pass",
        "source_candidates_identified": s["source_specific_metric_time_candidates"] >= 1,
        "conversion_ready_zero_blocks_restricted_claim": s["conversion_ready_targets"] == 0
        and s["restricted_subset_metric_seconds_claim_allowed_now"] is False,
        "global_metric_blocked": c["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": c["global_seconds_claim_allowed"] is False,
        "sdd_metric_blocked": s["sdd_metric_claim_allowed"] is False,
        "tgsim_pedestrian_claim_blocked": s["tgsim_pedestrian_world_model_metric_claim_allowed"] is False,
        "user_action_written": bool(payload["user_action_required"]),
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_ch_metric_time_claim_guard_pass" if passed == total else "stage42_ch_metric_time_claim_guard_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CH Metric/Time Claim Guard",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ch_gate']['passed']} / {payload['stage42_ch_gate']['total']}`",
        f"- verdict: `{payload['stage42_ch_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- datasets_audited: `{s['datasets_audited']}`",
        f"- source_records_audited: `{s['source_records_audited']}`",
        f"- source_specific_metric_time_candidates: `{s['source_specific_metric_time_candidates']}`",
        f"- conversion_ready_targets: `{s['conversion_ready_targets']}`",
        f"- global_metric_claim_allowed: `{s['global_metric_claim_allowed']}`",
        f"- global_seconds_claim_allowed: `{s['global_seconds_claim_allowed']}`",
        f"- restricted_subset_metric_seconds_claim_allowed_now: `{s['restricted_subset_metric_seconds_claim_allowed_now']}`",
        "",
        "## Source Claim Guard",
        "",
        "| source | domain | source metric/time evidence | h50 seconds if restricted | h100 seconds if restricted | paper claim allowed now | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["source_claim_rows"]:
        lines.append(
            f"| `{row['source_id']}` | `{row['domain']}` | {row['source_specific_metric_time_evidence']} | "
            f"{row['h50_seconds_if_restricted']} | {row['h100_seconds_if_restricted']} | "
            f"{row['paper_metric_seconds_claim_allowed_now']} | {row['reason']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Source-specific ETH/UCY calibration candidates exist, but they are not yet paper-allowed metric/seconds evaluation claims.",
        "- The current legal/readiness validator has zero conversion-ready targets, so restricted subset metric/time claims stay blocked.",
        "- Global M3W, SDD, TrajNet-snippet, and TGSIM-pedestrian-world-model metric/seconds claims remain forbidden.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ch_gate"]
    lines = [
        "# Stage42-CH Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-CH Metric/Time Claim Guard",
        "",
        *[f"- {item}" for item in payload["user_action_required"]],
        "",
        "No metric/seconds claim is allowed until these actions are completed and verified.",
    ]


def run_stage42_metric_time_claim_guard() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_metric_time_claim_guard()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
