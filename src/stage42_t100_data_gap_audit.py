from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_data_calibration as calib
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_t100_source_cv_repair import EASY_THRESHOLD, MIN_SAFE_POSITIVE_FOLDS


OUT_DIR = Path("outputs/stage42_long_research")
BA_JSON = OUT_DIR / "t100_source_cv_repair_stage42.json"
CALIBRATION_JSON = OUT_DIR / "data_calibration_stage42.json"
REPORT_JSON = OUT_DIR / "t100_data_gap_audit_stage42.json"
REPORT_MD = OUT_DIR / "t100_data_gap_audit_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_t100_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bb_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BB 是 Stage42-BA 后的 t100 数据/标定缺口审计，不重新训练模型。",
    "t100 positive gain 在 Stage42-BA train-only source-CV 下缺少独立 source 支持。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。",
    "metric / seconds-level pedestrian claims 仍被禁止，除非未来完成官方 FPS/stride/homography/scale 验证。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _calibration_payload() -> dict[str, Any]:
    if CALIBRATION_JSON.exists():
        payload = read_json(CALIBRATION_JSON, {})
        if isinstance(payload, dict) and payload.get("datasets"):
            return payload
    registry = calib._load_registry()
    known = calib._read_known_metrics()
    datasets = [calib._audit_dataset(spec, registry, known) for spec in calib.DATASET_SPECS]
    return {
        "source": "fresh_run_embedded_stage42_bb",
        "datasets": datasets,
        "summary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
        },
    }


def _t100_source_gap(domain: str, support: Mapping[str, Any], ba_plan_domain: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status = str(support.get("status", "unknown"))
    fold_count = int(support.get("fold_count", 0) or 0)
    safe_positive = int(support.get("safe_positive_fold_count", 0) or 0)
    supported = bool(support.get("supported_for_t100", False))
    t100_source_count = len((ba_plan_domain or {}).get("t100_groups", []) or [])

    if supported:
        missing_safe_positive_folds = 0
        additional_t100_sources_needed = 0
        blocker_type = "none"
    elif status == "not_run":
        missing_safe_positive_folds = MIN_SAFE_POSITIVE_FOLDS
        additional_t100_sources_needed = max(0, 3 - t100_source_count)
        blocker_type = "insufficient_t100_capable_original_train_sources"
    else:
        missing_safe_positive_folds = max(0, MIN_SAFE_POSITIVE_FOLDS - safe_positive)
        additional_t100_sources_needed = missing_safe_positive_folds
        max_easy = float(support.get("max_easy_degradation", 0.0) or 0.0)
        min_t100 = float(support.get("min_t100_improvement", 0.0) or 0.0)
        if max_easy > EASY_THRESHOLD:
            blocker_type = "t100_easy_safety_not_stable_across_source_cv"
        elif min_t100 <= 0.0:
            blocker_type = "t100_positive_gain_not_stable_across_source_cv"
        else:
            blocker_type = "insufficient_safe_positive_source_cv_folds"

    return {
        "source": "fresh_synthesis_from_stage42_ba",
        "domain": domain,
        "status": status,
        "fold_count": fold_count,
        "t100_capable_original_train_sources": t100_source_count,
        "safe_positive_fold_count": safe_positive,
        "supported_for_t100": supported,
        "missing_safe_positive_folds": int(missing_safe_positive_folds),
        "additional_t100_capable_train_sources_needed": int(additional_t100_sources_needed),
        "blocker_type": blocker_type,
        "support_rule": support.get("support_rule", f">={MIN_SAFE_POSITIVE_FOLDS} safe-positive folds and easy <= {EASY_THRESHOLD}"),
        "original_support": dict(support),
    }


def _dataset_action(dataset: Mapping[str, Any], source_gaps: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    dataset_id = str(dataset["dataset_id"])
    raw_found = bool(dataset.get("raw_path_found", False))
    converted_found = bool(dataset.get("converted_path_found", False))
    calibration_state = str(dataset.get("calibration_state", "not_verified"))
    legal = dataset.get("legal_status", {})
    reasons: list[str] = []

    if dataset_id in {"eth_ucy", "trajnet", "ucy"}:
        domain_key = "ETH_UCY" if dataset_id == "eth_ucy" else "TrajNet" if dataset_id == "trajnet" else "UCY"
        gap = source_gaps.get(domain_key, {})
        if not gap.get("supported_for_t100", False):
            reasons.append(
                f"t100 source-CV blocker: {gap.get('blocker_type', 'unknown')}; "
                f"needs at least {gap.get('additional_t100_capable_train_sources_needed', 0)} additional safe t100-capable train source(s) or source-specific repair."
            )
    if dataset_id == "opentraj":
        reasons.append("Use OpenTraj only through legal underlying dataset terms; do not treat toolkit mirror as license override.")
    if dataset_id == "sdd":
        reasons.append("Can support SDD pixel raw-frame work, but does not repair external t100 source support unless separately converted/aligned.")
    if dataset_id == "tgsim":
        reasons.append("Traffic metric diagnostic only; do not use as pedestrian/drone world-model success.")
    if dataset_id == "aerialmpt":
        reasons.append("Potential long aerial/top-down source, but official terms, raw sequences, FPS/stride and calibration must be verified before claims.")
    if not raw_found and not converted_found:
        reasons.append("local path missing: user must provide legal source path or official download instructions.")
    if calibration_state != "traffic_metric_diagnostic_only":
        reasons.append("keep dataset-local/raw-frame claim until official FPS/stride/homography/scale are verified.")
    if bool(legal.get("requires_manual_terms_acceptance")) or bool(legal.get("requires_login")) or bool(legal.get("requires_application")):
        reasons.append("do not auto-download without user accepting official terms/login/application.")

    next_action = "ready_for_nonmetric_diagnostic_use"
    if reasons:
        next_action = "user_action_or_source_specific_repair_required"
    if dataset_id in {"eth_ucy", "trajnet", "ucy"} and source_gaps.get("ETH_UCY" if dataset_id == "eth_ucy" else "TrajNet" if dataset_id == "trajnet" else "UCY", {}).get("supported_for_t100", False):
        next_action = "t100_supported_by_source_cv"

    return {
        "source": "fresh_synthesis_from_calibration_and_stage42_ba",
        "dataset_id": dataset_id,
        "dataset_name": dataset["dataset_name"],
        "raw_path_found": raw_found,
        "converted_path_found": converted_found,
        "calibration_state": calibration_state,
        "metric_claim_allowed": bool(dataset.get("metric_claim_allowed", False)),
        "seconds_claim_allowed": bool(dataset.get("seconds_claim_allowed", False)),
        "official_hint": dataset.get("official_hint"),
        "legal_status": legal,
        "next_action": next_action,
        "reasons": reasons,
    }


def run_stage42_t100_data_gap_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ba = _load_json(BA_JSON)
    calibration = _calibration_payload()
    ba_plan = ba.get("source_cv_plan", {}).get("domains", {})
    source_gaps = {
        domain: _t100_source_gap(domain, support, ba_plan.get(domain))
        for domain, support in sorted(ba.get("domain_t100_support", {}).items())
    }
    dataset_actions = [_dataset_action(row, source_gaps) for row in calibration.get("datasets", [])]
    unsupported_domains = [d for d, row in source_gaps.items() if not row["supported_for_t100"]]
    additional_sources_needed = {
        d: int(row["additional_t100_capable_train_sources_needed"])
        for d, row in source_gaps.items()
        if not row["supported_for_t100"]
    }
    final_after = ba["final_eval"]["after_cv_guard"]["protected"]
    payload = {
        "source": "fresh_synthesis_from_stage42_ba_and_calibration",
        "stage": "Stage42-BB T100 Data Gap Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BA_JSON), str(CALIBRATION_JSON), "outputs/stage42_long_research/data_calibration_stage42.md"]),
        "ba_verdict": ba.get("stage42_ba_gate", {}).get("verdict"),
        "ba_gate": ba.get("stage42_ba_gate", {}),
        "ba_final_after_cv_guard": final_after,
        "source_gaps": source_gaps,
        "dataset_actions": dataset_actions,
        "summary": {
            "source": "fresh_synthesis_from_stage42_ba_and_calibration",
            "domains_audited": len(source_gaps),
            "unsupported_t100_domains": unsupported_domains,
            "supported_t100_domains": [d for d, row in source_gaps.items() if row["supported_for_t100"]],
            "additional_t100_sources_needed_by_domain": additional_sources_needed,
            "any_t100_domain_supported": any(row["supported_for_t100"] for row in source_gaps.values()),
            "final_all_positive_after_guard": final_after["all_improvement"] > 0.0,
            "final_t50_positive_after_guard": final_after["t50_improvement"] > 0.0,
            "final_hard_positive_after_guard": final_after["hard_failure_improvement"] > 0.0,
            "final_easy_safe_after_guard": final_after["easy_degradation"] <= EASY_THRESHOLD,
            "final_t100_positive_after_guard": final_after["t100_raw_frame_diagnostic_improvement"] > 0.0,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "paper_claim": "t100 remains a data/support blocker. Protected all/t50/hard remain positive after source-CV guard, but t100 positive gain must not be claimed until independent source-level support is added or repaired.",
        },
        "user_action_required": _user_actions(dataset_actions, source_gaps),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "reads_stage42_ba_source_cv_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_bb_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _user_actions(dataset_actions: list[Mapping[str, Any]], source_gaps: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for domain, gap in source_gaps.items():
        if not gap["supported_for_t100"]:
            actions.append(
                {
                    "source": "fresh_synthesis_from_stage42_ba",
                    "priority": "high",
                    "target": domain,
                    "action_type": "provide_more_independent_t100_capable_topdown_sources_or_source_specific_repair",
                    "reason": gap["blocker_type"],
                    "minimum_extra_sources": gap["additional_t100_capable_train_sources_needed"],
                    "notes": "Provide legal raw/source files with train/val/test provenance, long enough t100 tracks, and official FPS/stride/homography/scale if available. Do not use test endpoints for goals or threshold selection.",
                }
            )
    for row in dataset_actions:
        if row["next_action"] == "user_action_or_source_specific_repair_required":
            actions.append(
                {
                    "source": "fresh_synthesis_from_calibration",
                    "priority": "medium" if row["dataset_id"] not in {"eth_ucy", "trajnet", "ucy"} else "high",
                    "target": row["dataset_id"],
                    "action_type": row["next_action"],
                    "official_hint": row["official_hint"],
                    "reasons": row["reasons"],
                }
            )
    return actions


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "ba_input_verified": payload["ba_verdict"] == "stage42_ba_t100_source_cv_repair_pass_with_t100_blocker",
        "source_gap_quantified": bool(payload["source_gaps"]),
        "unsupported_domains_reported": bool(summary["unsupported_t100_domains"]),
        "data_actions_built": bool(payload["dataset_actions"]),
        "user_action_generated": bool(payload["user_action_required"]),
        "all_remains_positive_after_guard": summary["final_all_positive_after_guard"],
        "t50_remains_positive_after_guard": summary["final_t50_positive_after_guard"],
        "hard_remains_positive_after_guard": summary["final_hard_positive_after_guard"],
        "easy_remains_safe_after_guard": summary["final_easy_safe_after_guard"],
        "t100_not_overclaimed": not summary["final_t100_positive_after_guard"] and not summary["any_t100_domain_supported"],
        "no_leakage_pass": all(
            payload["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_metrics_for_threshold"]
        ),
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"] and not payload["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bb_t100_data_gap_audit_pass_with_data_blocker" if passed == total else "stage42_bb_t100_data_gap_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_source_gap_table(source_gaps: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| domain | folds | safe-positive folds | supported | extra sources needed | blocker |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in source_gaps.items():
        lines.append(
            f"| `{domain}` | {row['fold_count']} | {row['safe_positive_fold_count']} | `{row['supported_for_t100']}` | {row['additional_t100_capable_train_sources_needed']} | {row['blocker_type']} |"
        )
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BB T100 Data Gap Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bb_gate']['passed']} / {payload['stage42_bb_gate']['total']}`",
        f"- verdict: `{payload['stage42_bb_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Stage42-BA Dependency",
        "",
        f"- ba_verdict: `{payload['ba_verdict']}`",
        f"- after_source_cv_guard_all: `{payload['ba_final_after_cv_guard']['all_improvement']:.6f}`",
        f"- after_source_cv_guard_t50: `{payload['ba_final_after_cv_guard']['t50_improvement']:.6f}`",
        f"- after_source_cv_guard_t100_raw_frame_diagnostic: `{payload['ba_final_after_cv_guard']['t100_raw_frame_diagnostic_improvement']:.6f}`",
        f"- after_source_cv_guard_hard_failure: `{payload['ba_final_after_cv_guard']['hard_failure_improvement']:.6f}`",
        f"- after_source_cv_guard_easy_degradation: `{payload['ba_final_after_cv_guard']['easy_degradation']:.6f}`",
        "",
        "## T100 Source Support Gaps",
        "",
    ]
    lines.extend(_render_source_gap_table(payload["source_gaps"]))
    lines.extend(
        [
            "",
            "## Dataset Actions",
            "",
            "| dataset | raw | converted | calibration | metric | seconds | next action |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["dataset_actions"]:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['raw_path_found']}` | `{row['converted_path_found']}` | {row['calibration_state']} | `{row['metric_claim_allowed']}` | `{row['seconds_claim_allowed']}` | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- unsupported_t100_domains: `{summary['unsupported_t100_domains']}`",
            f"- supported_t100_domains: `{summary['supported_t100_domains']}`",
            f"- additional_t100_sources_needed_by_domain: `{summary['additional_t100_sources_needed_by_domain']}`",
            f"- final_all_positive_after_guard: `{summary['final_all_positive_after_guard']}`",
            f"- final_t50_positive_after_guard: `{summary['final_t50_positive_after_guard']}`",
            f"- final_hard_positive_after_guard: `{summary['final_hard_positive_after_guard']}`",
            f"- final_easy_safe_after_guard: `{summary['final_easy_safe_after_guard']}`",
            f"- final_t100_positive_after_guard: `{summary['final_t100_positive_after_guard']}`",
            f"- global_metric_claim_allowed: `{summary['global_metric_claim_allowed']}`",
            f"- global_seconds_claim_allowed: `{summary['global_seconds_claim_allowed']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-BB does not train a new model. It turns the Stage42-BA t100 source-CV blocker into an actionable data/calibration gap report.",
            "- Protected all/t50/hard remain positive after the source-CV guard, but t100 positive gain is not supported by enough independent train-only source-CV evidence.",
            "- The correct deployment/paper posture is to keep t100 as raw-frame diagnostic blocker until additional independent t100-capable sources or source-specific support are available.",
            "- Metric and seconds-level claims remain rejected for pedestrian/top-down domains until FPS/stride/homography/scale are verified from official sources.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BB User Action Required For T100",
        "",
        f"- source: `{payload['source']}`",
        "- purpose: list concrete data/calibration actions needed before t100 can be claimed as stable positive transfer.",
        "",
    ]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['target']}",
                "",
                f"- priority: `{action['priority']}`",
                f"- action_type: `{action['action_type']}`",
            ]
        )
        if "reason" in action:
            lines.append(f"- reason: `{action['reason']}`")
        if "minimum_extra_sources" in action:
            lines.append(f"- minimum_extra_sources: `{action['minimum_extra_sources']}`")
        if "official_hint" in action:
            lines.append(f"- official_hint: `{action['official_hint']}`")
        if "reasons" in action:
            lines.append("- reasons:")
            lines.extend([f"  - {reason}" for reason in action["reasons"]])
        if "notes" in action:
            lines.append(f"- notes: {action['notes']}")
        lines.append("")
    lines.extend(
        [
            "## Non-Actionable Non-Claims",
            "",
            "- Do not use TGSIM traffic metric success as pedestrian/top-down t100 success.",
            "- Do not use SDD pixel raw-frame success as external metric/time calibration.",
            "- Do not use test endpoints, future waypoints, or central velocity as inference input.",
            "- Do not write t100 as seconds-level until FPS/stride/effective seconds are verified.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bb_gate"]
    lines = [
        "# Stage42-BB Gate",
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


def _append_ledger(payload: Mapping[str, Any]) -> None:
    row = {
        "stage": payload["stage"],
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_bb_gate"]["verdict"],
        "gate": f"{payload['stage42_bb_gate']['passed']}/{payload['stage42_bb_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_t100_data_gap_audit()
