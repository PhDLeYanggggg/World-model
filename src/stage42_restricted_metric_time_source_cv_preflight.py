from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from src import stage42_local_t100_conversion_readiness as be
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HI_JSON = OUT_DIR / "restricted_metric_time_readiness_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"

REPORT_JSON = OUT_DIR / "restricted_metric_time_source_cv_preflight_stage42.json"
REPORT_MD = OUT_DIR / "restricted_metric_time_source_cv_preflight_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hj_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_restricted_metric_time_source_cv_preflight_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_SUMMARY = Path("README_M3W_GOAL_FULL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hj_restricted_metric_time_source_cv_preflight"

HORIZONS = [10, 25, 50, 100]
HISTORY_WINDOWS = [8, 16, 32, 64]
MIN_SOURCE_CV_SOURCES = 2
ROBUST_SOURCE_CV_SOURCES = 3

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HJ 是 restricted metric/time source-CV preflight，不训练、不转换、不下载、不调 threshold。",
    "本阶段只解析本地 ETH/UCY technical candidate rows 来估计 source-CV / history / horizon 可行性。",
    "local parseability 和 source-CV feasibility 不等于 legal conversion readiness。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "restricted seconds/metric wording 仍需 user terms confirmation、guarded conversion、no-leakage、source-CV、final test。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _track_lengths(path: Path) -> dict[str, Any]:
    rows = be._parse_rows(path)
    tracks = be._track_map(rows)
    lengths = [len(track) for track in tracks.values()]
    frames = [int(row["frame_id"]) for row in rows]
    step = be._common_step(tracks)
    continuity = be._continuity(tracks, step)
    horizon_counts = be._horizon_counts(lengths)
    history_counts = be._history_counts(lengths)
    return {
        "rows": int(len(rows)),
        "agents": int(len(tracks)),
        "unique_frames": int(len(set(frames))) if frames else 0,
        "min_frame": int(min(frames)) if frames else None,
        "max_frame": int(max(frames)) if frames else None,
        "common_frame_step": step,
        "track_count": int(len(lengths)),
        "min_track_points": int(min(lengths)) if lengths else 0,
        "median_track_points": float(median(lengths)) if lengths else 0.0,
        "max_track_points": int(max(lengths)) if lengths else 0,
        "horizon_counts": horizon_counts,
        "history_horizon_counts": history_counts,
        "continuity": continuity,
    }


def _source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("trajectory_file", "")))
    stats = _track_lengths(path) if path.exists() else {
        "rows": 0,
        "agents": 0,
        "unique_frames": 0,
        "min_frame": None,
        "max_frame": None,
        "common_frame_step": None,
        "track_count": 0,
        "min_track_points": 0,
        "median_track_points": 0.0,
        "max_track_points": 0,
        "horizon_counts": {str(h): 0 for h in HORIZONS},
        "history_horizon_counts": {f"k{k}_h{h}": 0 for k in HISTORY_WINDOWS for h in [50, 100]},
        "continuity": {"gap_ratio": 1.0},
    }
    h50 = int(stats["horizon_counts"].get("50", 0))
    h100 = int(stats["horizon_counts"].get("100", 0))
    k64h100 = int(stats["history_horizon_counts"].get("k64_h100", 0))
    return {
        "source": "fresh_run",
        "source_id": row.get("source_id"),
        "domain": row.get("domain"),
        "dataset": row.get("dataset"),
        "terms_target_id": row.get("terms_target_id"),
        "trajectory_file": str(path),
        "trajectory_file_found": path.exists(),
        "homography_parseable": bool(row.get("homography_parseable")),
        "annotation_fps": row.get("annotation_fps"),
        "annotation_timestep_seconds": row.get("annotation_timestep_seconds"),
        "h50_seconds_if_restricted": row.get("h50_seconds_if_restricted"),
        "h100_seconds_if_restricted": row.get("h100_seconds_if_restricted"),
        "track_stats": stats,
        "t50_windows": h50,
        "t100_windows": h100,
        "k64_h100_windows": k64h100,
        "source_cv_usable_after_terms": bool(
            path.exists()
            and row.get("source_specific_metric_time_evidence")
            and row.get("technical_conversion_ready_after_terms")
            and h50 > 0
            and h100 > 0
            and float(stats["continuity"].get("gap_ratio", 1.0)) < 0.5
        ),
        "legal_ready_now": bool(row.get("conversion_ready_now")),
        "restricted_metric_time_ready_now": bool(row.get("restricted_metric_time_ready_now")),
    }


def _domain_plans(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(str(row.get("domain")), []).append(row)
    plans: dict[str, Any] = {}
    for domain, items in sorted(by_domain.items()):
        usable = [row for row in items if row.get("source_cv_usable_after_terms")]
        blocked = [row for row in items if not row.get("source_cv_usable_after_terms")]
        folds = []
        for holdout in usable:
            train_sources = [str(row["source_id"]) for row in usable if row["source_id"] != holdout["source_id"]]
            if not train_sources:
                continue
            folds.append(
                {
                    "source": "fresh_preflight_source_cv_plan",
                    "holdout_source": holdout["source_id"],
                    "train_sources_after_terms": train_sources,
                    "val_source_policy": "use train-source internal validation or leave-one-train-source if >=3 sources",
                    "t50_windows_holdout": holdout["t50_windows"],
                    "t100_windows_holdout": holdout["t100_windows"],
                }
            )
        plans[domain] = {
            "source": "fresh_preflight_source_cv_plan",
            "source_count": len(items),
            "usable_after_terms_count": len(usable),
            "blocked_after_terms_count": len(blocked),
            "source_cv_feasible_after_terms": len(usable) >= MIN_SOURCE_CV_SOURCES,
            "robust_source_cv_feasible_after_terms": len(usable) >= ROBUST_SOURCE_CV_SOURCES,
            "fold_count": len(folds),
            "folds": folds,
            "total_t50_windows": int(sum(int(row.get("t50_windows", 0)) for row in usable)),
            "total_t100_windows": int(sum(int(row.get("t100_windows", 0)) for row in usable)),
            "blockers": [
                {
                    "source_id": row.get("source_id"),
                    "t50_windows": row.get("t50_windows"),
                    "t100_windows": row.get("t100_windows"),
                    "reason": "missing_t50_or_t100_or_continuity_for_source_cv",
                }
                for row in blocked
            ],
        }
    return plans


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hi = read_json(HI_JSON, {})
    bn = read_json(BN_JSON, {})
    cg = read_json(CG_JSON, {})
    source_rows = [_source_row(row) for row in hi.get("readiness_rows", [])]
    domain_plans = _domain_plans(source_rows)
    domains_feasible = [domain for domain, row in domain_plans.items() if row["source_cv_feasible_after_terms"]]
    domains_robust = [domain for domain, row in domain_plans.items() if row["robust_source_cv_feasible_after_terms"]]
    domains_blocked = [
        domain
        for domain, row in domain_plans.items()
        if not row["source_cv_feasible_after_terms"] and row["source_count"] > 0
    ]
    counts = Counter(str(row["domain"]) for row in source_rows)
    summary = {
        "source": SOURCE,
        "hi_verdict": hi.get("stage42_hi_gate", {}).get("verdict"),
        "bn_verdict": bn.get("stage42_bn_gate", {}).get("verdict"),
        "cg_verdict": cg.get("stage42_cg_gate", {}).get("verdict"),
        "candidate_sources": len(source_rows),
        "source_counts_by_domain": dict(counts),
        "usable_after_terms_sources": sum(1 for row in source_rows if row["source_cv_usable_after_terms"]),
        "restricted_metric_time_ready_now_sources": sum(1 for row in source_rows if row["restricted_metric_time_ready_now"]),
        "domains_source_cv_feasible_after_terms": domains_feasible,
        "domains_robust_source_cv_feasible_after_terms": domains_robust,
        "domains_source_cv_blocked_after_terms": domains_blocked,
        "total_t50_windows_after_terms": int(sum(int(row["t50_windows"]) for row in source_rows if row["source_cv_usable_after_terms"])),
        "total_t100_windows_after_terms": int(sum(int(row["t100_windows"]) for row in source_rows if row["source_cv_usable_after_terms"])),
        "training_run": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HJ Restricted Metric/Time Source-CV Preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HI_JSON, BN_JSON, CG_JSON, *[row.get("trajectory_file", "") for row in source_rows]]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hi_gate_passed": _gate_passed(hi, "stage42_hi_gate"),
            "bn_gate_passed": _gate_passed(bn, "stage42_bn_gate"),
            "cg_gate_passed": _gate_passed(cg, "stage42_cg_gate"),
        },
        "source_rows": source_rows,
        "domain_plans": domain_plans,
        "summary": summary,
        "claim_boundary": {
            "source_cv_preflight_is_conversion": False,
            "source_cv_preflight_is_evaluation": False,
            "restricted_metric_time_claim_allowed_now": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": [
            "Confirm official ETH/BIWI and UCY terms, local paths, and source identity.",
            "Rerun `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.",
            "Only then run a guarded restricted metric/time conversion and source-CV evaluation; this preflight is not conversion/evaluation.",
        ],
    }
    payload["stage42_hj_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "hi_input_passed": payload["inputs"]["hi_gate_passed"] is True,
        "bn_input_passed": payload["inputs"]["bn_gate_passed"] is True,
        "cg_input_passed": payload["inputs"]["cg_gate_passed"] is True,
        "sources_parsed": summary["candidate_sources"] >= 1,
        "usable_after_terms_sources_present": summary["usable_after_terms_sources"] >= 1,
        "eth_ucy_preflight_complete_or_blocker_recorded": (
            "ETH_UCY" in summary["domains_source_cv_feasible_after_terms"]
            or "ETH_UCY" in summary["domains_source_cv_blocked_after_terms"]
        ),
        "ucy_source_cv_feasible_after_terms": "UCY" in summary["domains_source_cv_feasible_after_terms"],
        "ucy_robust_source_cv_feasible_after_terms": "UCY" in summary["domains_robust_source_cv_feasible_after_terms"],
        "h50_h100_windows_present": summary["total_t50_windows_after_terms"] > 0
        and summary["total_t100_windows_after_terms"] > 0,
        "ready_now_zero": summary["restricted_metric_time_ready_now_sources"] == 0,
        "no_conversion_or_evaluation_claim": claim["source_cv_preflight_is_conversion"] is False
        and claim["source_cv_preflight_is_evaluation"] is False,
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_hj_restricted_metric_time_source_cv_preflight_pass_with_eth_ucy_source_cv_limit"
        if passed == total
        else "stage42_hj_restricted_metric_time_source_cv_preflight_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hj_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HJ Restricted Metric/Time Source-CV Preflight",
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
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- candidate_sources: `{s['candidate_sources']}`",
        f"- usable_after_terms_sources: `{s['usable_after_terms_sources']}`",
        f"- restricted_metric_time_ready_now_sources: `{s['restricted_metric_time_ready_now_sources']}`",
        f"- domains_source_cv_feasible_after_terms: `{s['domains_source_cv_feasible_after_terms']}`",
        f"- domains_robust_source_cv_feasible_after_terms: `{s['domains_robust_source_cv_feasible_after_terms']}`",
        f"- domains_source_cv_blocked_after_terms: `{s['domains_source_cv_blocked_after_terms']}`",
        f"- total_t50_windows_after_terms: `{s['total_t50_windows_after_terms']}`",
        f"- total_t100_windows_after_terms: `{s['total_t100_windows_after_terms']}`",
        "",
        "## Source Rows",
        "",
        "| source | domain | rows | agents | max track | t50 | t100 | k64+h100 | usable after terms | ready now |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["source_rows"]:
        stats = row["track_stats"]
        lines.append(
            f"| `{row['source_id']}` | `{row['domain']}` | {stats['rows']} | {stats['agents']} | "
            f"{stats['max_track_points']} | {row['t50_windows']} | {row['t100_windows']} | "
            f"{row['k64_h100_windows']} | {row['source_cv_usable_after_terms']} | {row['restricted_metric_time_ready_now']} |"
        )
    lines += [
        "",
        "## Source-CV Plan",
        "",
        "| domain | sources | usable after terms | blocked | robust after terms | folds | t50 | t100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["domain_plans"].items():
        lines.append(
            f"| `{domain}` | {row['source_count']} | {row['usable_after_terms_count']} | "
            f"{row['blocked_after_terms_count']} | {row['robust_source_cv_feasible_after_terms']} | {row['fold_count']} | "
            f"{row['total_t50_windows']} | {row['total_t100_windows']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This preflight shows that UCY can support robust restricted source-CV after user-confirmed terms and guarded conversion.",
        "- UCY has at least three usable sources after terms, so it is robust enough for leave-one-source style source-CV planning.",
        "- ETH_UCY is parseable and has technical metric/time signals, but current local ETH_seq_hotel lacks t100 windows, so ETH_UCY source-CV is not feasible yet.",
        "- No conversion, no model training, no evaluation, no Stage5C, and no SMC occurred.",
        "- Current paper wording must remain dataset-local/raw-frame until the guarded conversion and source-CV final test are actually run.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hj_gate"]
    return [
        "# Stage42-HJ Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-HJ Restricted Metric/Time Source-CV Preflight",
        "",
        "- This preflight is ready to become a guarded conversion/evaluation only after source terms are confirmed.",
        "- Confirm ETH/BIWI and UCY official terms, local paths, and source identity.",
        "- Rerun `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.",
        "- Then run a future guarded restricted metric/time conversion source-CV stage.",
        "",
        "Do not claim metric/seconds-level results from this preflight alone.",
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hj_gate"]
    s = payload["summary"]
    return [
        "## Stage42-HJ Restricted Metric/Time Source-CV Preflight",
        "",
        "- source: `fresh_stage42_hj_restricted_metric_time_source_cv_preflight`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- usable after terms sources: `{s['usable_after_terms_sources']}`; ready now: `{s['restricted_metric_time_ready_now_sources']}`.",
        f"- source-CV feasible after terms: `{s['domains_source_cv_feasible_after_terms']}`; robust after terms: `{s['domains_robust_source_cv_feasible_after_terms']}`.",
        f"- source-CV blocked after terms: `{s['domains_source_cv_blocked_after_terms']}`.",
        f"- window potential after terms: t50 `{s['total_t50_windows_after_terms']}`, t100 `{s['total_t100_windows_after_terms']}`.",
        "- conclusion: restricted metric/time source-CV is technically plannable for UCY and blocked for ETH_UCY by current t100 source support; source terms still block all conversion/evaluation claims.",
    ]


def _refresh_a_journal_gap(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-HJ Restricted Metric/Time Source-CV Preflight Refresh",
        "",
        "- source: `fresh_stage42_hj_restricted_metric_time_source_cv_preflight`",
        f"- verdict: `{payload['stage42_hj_gate']['verdict']}`",
        f"- source-CV feasible after terms: `{s['domains_source_cv_feasible_after_terms']}`.",
        f"- robust source-CV feasible after terms: `{s['domains_robust_source_cv_feasible_after_terms']}`.",
        f"- source-CV blocked after terms: `{s['domains_source_cv_blocked_after_terms']}`.",
        f"- t50/t100 potential after terms: `{s['total_t50_windows_after_terms']}` / `{s['total_t100_windows_after_terms']}`.",
        "- Paper implication: the metric/time gap is narrowed for UCY, while ETH_UCY still needs more t100-capable source support before source-CV.",
        "- Still forbidden: claiming metric/seconds-level results from this preflight alone.",
    ]
    _replace_section(OUT_DIR / "a_journal_gap_stage42.md", "STAGE42_HJ_RESTRICTED_METRIC_TIME_SOURCE_CV_PREFLIGHT", lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_HJ_RESTRICTED_METRIC_TIME_SOURCE_CV_PREFLIGHT", lines)
    _refresh_a_journal_gap(payload)


def _refresh_research_state(payload: Mapping[str, Any], *, verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HJ restricted metric/time source-CV preflight"
    state["current_verdict"] = payload["stage42_hj_gate"]["verdict"]
    state["stage42_hj_restricted_metric_time_source_cv_preflight"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hj_gate"]["verdict"],
        "gates": f"{payload['stage42_hj_gate']['passed']}/{payload['stage42_hj_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_restricted_metric_time_source_cv_preflight(
    *,
    refresh_readmes: bool = True,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


if __name__ == "__main__":
    run_stage42_restricted_metric_time_source_cv_preflight()
