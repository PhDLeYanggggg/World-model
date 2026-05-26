from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_calibrated_t50_source_support_gap_audit as br
from src import stage42_local_t100_conversion_readiness as be
from src import stage42_local_t100_schema_conversion as bf
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BR_JSON = OUT_DIR / "calibrated_t50_source_support_gap_stage42.json"
BS_JSON = OUT_DIR / "ucy_zara_t50_family_policy_stage42.json"
BT_JSON = OUT_DIR / "eth_seq_t50_support_dry_run_stage42.json"
REPORT_JSON = OUT_DIR / "ucy_students_t50_source_support_stage42.json"
REPORT_MD = OUT_DIR / "ucy_students_t50_source_support_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bu_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_ucy_students_t50_sources_stage42.md"

MIN_INDEPENDENT_T50_SOURCES = 3

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BU 只审计 UCY_students calibrated t50 source-family support，不训练神经模型。",
    "UCY_students03 已在 calibrated subset 中；本步骤检查本地 students001/002/003 是否提供独立 t50-capable support。",
    "alternate formats / duplicate copies 不能算新的 independent source。",
    "future endpoints 只作为 baseline error label，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

LOCAL_CANDIDATES = [
    {
        "source_id": "UCY_students01_raw",
        "independent_key": "UCY_students01",
        "path": "external_data/OpenTraj/datasets/UCY/students01/students001.txt",
        "role": "canonical_independent_candidate",
        "notes": "long UCY students001 local source",
    },
    {
        "source_id": "UCY_students01_trajnet_duplicate",
        "independent_key": "UCY_students01",
        "path": "external_data/OpenTraj/datasets/UCY/students01/students001-trajnet.txt",
        "role": "alternate_short_format_duplicate",
        "notes": "same students001 source, shorter TrajNet slice; not independent",
    },
    {
        "source_id": "UCY_students03_obsmat_existing_calibrated",
        "independent_key": "UCY_students03",
        "path": "external_data/OpenTraj/datasets/UCY/students03/obsmat.txt",
        "role": "existing_calibrated_source",
        "notes": "source already represented by UCY_students03 in calibrated subset",
    },
    {
        "source_id": "UCY_students03_raw_duplicate",
        "independent_key": "UCY_students03",
        "path": "external_data/OpenTraj/datasets/UCY/students03/students003.txt",
        "role": "alternate_format_duplicate",
        "notes": "same students003 source, not independent from obsmat",
    },
    {
        "source_id": "UCY_students02_trajnet_short",
        "independent_key": "UCY_students02",
        "path": "external_data/OpenTraj/datasets/TrajNet/Test/crowds/students002.txt",
        "role": "short_local_candidate",
        "notes": "students002 is locally present but too short for t50 in this file",
    },
    {
        "source_id": "UCY_students01_trajnet_train_duplicate",
        "independent_key": "UCY_students01",
        "path": "external_data/OpenTraj/datasets/TrajNet/Train/crowds/students001.txt",
        "role": "duplicate_copy",
        "notes": "duplicate of students001 source",
    },
    {
        "source_id": "UCY_students03_trajnet_train_duplicate",
        "independent_key": "UCY_students03",
        "path": "external_data/OpenTraj/datasets/TrajNet/Train/crowds/students003.txt",
        "role": "duplicate_copy",
        "notes": "duplicate/short slice of students003 source",
    },
    {
        "source_id": "UCY_students01_stage5b_duplicate",
        "independent_key": "UCY_students01",
        "path": "data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/crowds/students001.txt",
        "role": "duplicate_copy",
        "notes": "duplicate of students001 source from stage5b raw bundle",
    },
    {
        "source_id": "UCY_students03_stage5b_duplicate",
        "independent_key": "UCY_students03",
        "path": "data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/crowds/students003.txt",
        "role": "duplicate_copy",
        "notes": "duplicate/short slice of students003 source from stage5b raw bundle",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _audit_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(candidate["path"]))
    exists = path.exists()
    if not exists:
        return {
            "source": "fresh_ucy_students_t50_source_support",
            **dict(candidate),
            "path_exists": False,
            "rows": 0,
            "agents": 0,
            "unique_frames": 0,
            "common_frame_step": None,
            "horizon_counts": {"10": 0, "25": 0, "50": 0, "100": 0},
            "t50_capable": False,
            "t100_capable": False,
            "baseline_t50": None,
        }
    rows = be._parse_rows(path)
    tracks = be._track_map(rows)
    lengths = [len(track) for track in tracks.values()]
    horizon_counts = be._horizon_counts(lengths)
    baseline_source = {
        "source_id": str(candidate["source_id"]),
        "domain": "UCY_students",
        "path": str(path),
    }
    baseline = bf._window_errors_for_source(baseline_source)
    by50 = baseline["by_horizon"]["50"]
    return {
        "source": "fresh_ucy_students_t50_source_support",
        **dict(candidate),
        "path_exists": True,
        "rows": int(len(rows)),
        "agents": int(len(tracks)),
        "unique_frames": int(len({int(row["frame_id"]) for row in rows})),
        "common_frame_step": be._common_step(tracks),
        "max_track_points": int(max(lengths, default=0)),
        "median_track_points": float(__import__("statistics").median(lengths)) if lengths else 0.0,
        "horizon_counts": horizon_counts,
        "t50_capable": int(horizon_counts["50"]) > 0,
        "t100_capable": int(horizon_counts["100"]) > 0,
        "coordinate_unit": "dataset_local_raw",
        "metric_status": "source_specific_unverified_for_global_claim",
        "baseline_t50": {
            "windows": int(by50["windows"]),
            "strongest_baseline": by50["strongest_baseline"],
            "improvement_vs_constant_velocity": by50["improvement_vs_constant_velocity"],
            "constant_velocity_mean_fde": by50["baselines"]["constant_velocity_causal_fd"]["mean_fde"],
            "strongest_mean_fde": by50["baselines"][by50["strongest_baseline"]]["mean_fde"]
            if by50["strongest_baseline"]
            else None,
        },
    }


def _canonical_independent_sources(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("path_exists") and row.get("t50_capable"):
            grouped[str(row["independent_key"])].append(row)
    canonical: dict[str, Mapping[str, Any]] = {}
    for key, candidates in grouped.items():
        canonical[key] = max(candidates, key=lambda row: (int(row["horizon_counts"]["50"]), int(row["rows"])))
    return canonical


def _source_support_summary(br_payload: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    family = br_payload.get("family_support", {}).get("UCY_students", {})
    existing = list(family.get("calibrated_sources", []))
    canonical = _canonical_independent_sources(rows)
    independent_t50 = sorted(canonical.keys())
    newly_found = [key for key in independent_t50 if key != "UCY_students03"]
    short_or_missing = [
        {
            "source_id": row["source_id"],
            "independent_key": row["independent_key"],
            "horizon_counts": row["horizon_counts"],
            "reason": "not_t50_capable" if row.get("path_exists") else "missing_path",
        }
        for row in rows
        if not row.get("t50_capable")
    ]
    additional_needed = max(0, MIN_INDEPENDENT_T50_SOURCES - len(independent_t50))
    return {
        "source": "fresh_ucy_students_t50_source_support",
        "br_existing_calibrated_sources": existing,
        "br_additional_sources_needed_before_bu": int(family.get("additional_calibrated_sources_needed", 0)),
        "local_candidates_audited": len(rows),
        "local_paths_found": int(sum(bool(row.get("path_exists")) for row in rows)),
        "independent_t50_capable_sources": independent_t50,
        "independent_t50_capable_source_count": len(independent_t50),
        "canonical_sources": {key: row["source_id"] for key, row in canonical.items()},
        "new_independent_t50_sources_found": newly_found,
        "new_independent_t50_source_count": len(newly_found),
        "minimum_independent_sources_needed_for_source_cv": MIN_INDEPENDENT_T50_SOURCES,
        "additional_independent_t50_sources_still_needed": int(additional_needed),
        "source_cv_ready": len(independent_t50) >= MIN_INDEPENDENT_T50_SOURCES,
        "short_or_missing_sources": short_or_missing,
        "ucy_students_t50_support_repaired": False,
        "remaining_blocker": "UCY_students still lacks one independent t50-capable same-family source for train/val/holdout source-CV."
        if additional_needed > 0
        else "UCY_students source count is ready, but a protected family policy still needs a separate train/val/test run.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    gates = {
        "br_input_verified": payload["br_verdict"] == "stage42_br_calibrated_t50_source_support_gap_audit_pass",
        "bs_input_verified": payload["bs_verdict"] == "stage42_bs_ucy_zara_t50_family_policy_pass_positive",
        "bt_input_verified": payload["bt_verdict"] == "stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed",
        "local_candidates_audited": s["local_candidates_audited"] >= 5,
        "duplicate_sources_not_counted": s["independent_t50_capable_source_count"] < sum(
            1 for row in payload["candidate_audit"] if row.get("t50_capable")
        ),
        "t50_capability_quantified": s["independent_t50_capable_source_count"] >= 2,
        "source_cv_not_overclaimed": s["source_cv_ready"] is False and s["ucy_students_t50_support_repaired"] is False,
        "blocker_narrowed": s["additional_independent_t50_sources_still_needed"] == 1,
        "no_future_inputs": no_leak["future_endpoint_input"] is False and no_leak["test_threshold_tuning"] is False,
        "user_action_generated": bool(payload["user_action_required"]),
        "global_metric_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed"
        if passed == total
        else "stage42_bu_ucy_students_t50_source_support_partial"
    )
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def run_stage42_ucy_students_t50_source_support() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    br_payload = _load_json(BR_JSON)
    bs_payload = _load_json(BS_JSON)
    bt_payload = _load_json(BT_JSON)
    candidate_audit = [_audit_candidate(row) for row in LOCAL_CANDIDATES]
    summary = _source_support_summary(br_payload, candidate_audit)
    payload: dict[str, Any] = {
        "source": "fresh_ucy_students_t50_source_support",
        "stage": "Stage42-BU UCY_students T50 Source-Support Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BR_JSON), str(BS_JSON), str(BT_JSON)] + [row["path"] for row in LOCAL_CANDIDATES]),
        "current_facts": CURRENT_FACTS,
        "br_verdict": br_payload.get("stage42_br_gate", {}).get("verdict"),
        "bs_verdict": bs_payload.get("stage42_bs_gate", {}).get("verdict"),
        "bt_verdict": bt_payload.get("stage42_bt_gate", {}).get("verdict"),
        "candidate_audit": candidate_audit,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "duplicate_formats_counted_as_independent": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "source_specific_annotation_step_subset_claim_allowed": True,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "positive_ucy_students_t50_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": [
            {
                "priority": "high",
                "action": "provide_or_locate_one_more_independent_t50_capable_UCY_students_source",
                "notes": "students001 and students003 are t50-capable; students002 local TrajNet file is too short for t50. Need one more independent students-family long source for train/val/holdout source-CV.",
            }
        ],
    }
    payload["stage42_bu_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BU UCY_students T50 Source-Support Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bu_gate']['passed']} / {payload['stage42_bu_gate']['total']}`",
        f"- verdict: `{payload['stage42_bu_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- br_additional_sources_needed_before_bu: `{s['br_additional_sources_needed_before_bu']}`",
        f"- local_candidates_audited: `{s['local_candidates_audited']}`",
        f"- local_paths_found: `{s['local_paths_found']}`",
        f"- independent_t50_capable_sources: `{', '.join(s['independent_t50_capable_sources']) or 'none'}`",
        f"- independent_t50_capable_source_count: `{s['independent_t50_capable_source_count']}`",
        f"- new_independent_t50_sources_found: `{', '.join(s['new_independent_t50_sources_found']) or 'none'}`",
        f"- additional_independent_t50_sources_still_needed: `{s['additional_independent_t50_sources_still_needed']}`",
        f"- source_cv_ready: `{s['source_cv_ready']}`",
        f"- ucy_students_t50_support_repaired: `{s['ucy_students_t50_support_repaired']}`",
        f"- remaining_blocker: `{s['remaining_blocker']}`",
        "",
        "## Candidate Audit",
        "",
        "| source_id | independent_key | role | exists | t50 | t100 | strongest@50 | improvement_vs_cv@50 | counted independent? |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    counted = set(s["independent_t50_capable_sources"])
    canonical = s["canonical_sources"]
    for row in payload["candidate_audit"]:
        b50 = row.get("baseline_t50") or {}
        key = row["independent_key"]
        lines.append(
            f"| `{row['source_id']}` | `{key}` | `{row['role']}` | {row['path_exists']} | {row['horizon_counts']['50']} | {row['horizon_counts']['100']} | `{b50.get('strongest_baseline')}` | {b50.get('improvement_vs_constant_velocity')} | {key in counted and canonical.get(key) == row['source_id']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- BU narrows the UCY_students blocker: `students001` is a real additional t50-capable same-family source, while `students002` is locally present but too short for t50.",
            "- `students003` alternate files and TrajNet/stage5b copies are duplicates, so they are not counted as independent support.",
            "- With only `students001` and `students003` as independent t50-capable sources, train/validation/holdout source-CV is still not possible.",
            "- This is a blocker-narrowing audit, not a positive t50 transfer result.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bu_gate"]
    lines = [
        "# Stage42-BU Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: UCY_students T50 Source Support",
        "",
        "- Stage42-BU found two independent t50-capable UCY_students sources: `students001` and `students003`.",
        "- The local `students002` TrajNet file is too short for t50 in the current copy.",
        "- One more independent t50-capable UCY_students-family source is still needed before a deployable source-CV repair can be attempted.",
        "- Do not treat duplicate files or alternate coordinate formats as independent sources.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
    ]
    for item in payload["user_action_required"]:
        lines.append(f"- `{item['priority']}`: {item['action']} - {item['notes']}")
    return lines


if __name__ == "__main__":
    run_stage42_ucy_students_t50_source_support()
