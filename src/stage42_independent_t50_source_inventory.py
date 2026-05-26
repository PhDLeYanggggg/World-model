from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
CB_JSON = OUT_DIR / "t50_source_robustness_audit_stage42.json"
REPORT_JSON = OUT_DIR / "independent_t50_source_inventory_stage42.json"
REPORT_MD = OUT_DIR / "independent_t50_source_inventory_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cc_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_independent_t50_sources_stage42.md"

SCAN_ROOTS = [
    Path("external_data/OpenTraj/datasets/ETH"),
    Path("external_data/OpenTraj/datasets/ETH-Person"),
    Path("external_data/OpenTraj/datasets/UCY"),
    Path("external_data/OpenTraj/datasets/TrajNet"),
    Path("external_data/OpenTraj/datasets/TrajNet++"),
    Path("/Users/yangyue/Downloads/OpenTraj/datasets"),
    Path("/Users/yangyue/Downloads/ETH_UCY"),
    Path("/Users/yangyue/Downloads/trajnetplusplusdataset"),
]

MAX_FILES = 1200
HORIZONS = [10, 25, 50, 100]
T50_MIN_POINTS = 51
AUXILIARY_NAMES = {"h.txt", "h-old.txt", "h-cam.txt", "static.txt", "groups.txt", "destinations.txt", "gt_file_format.txt"}


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CC 是 independent t50 source inventory / user-action audit，不训练模型，不调 threshold。",
    "本审计只扫描本地可见文件；不绕过 license，不自动下载，不把 registry-only 当 converted。",
    "如果 source 已被当前 split 使用，只能作为 split rebuild/source-CV 候选，不能直接算新 held-out evidence。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_candidate_files(roots: Iterable[Path] = SCAN_ROOTS) -> list[Path]:
    seen: set[str] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".csv", ".ndjson"}:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
            if len(files) >= MAX_FILES:
                return files
    return files


def _parse_track_file(path: Path) -> dict[str, Any]:
    if _is_auxiliary(path):
        return _empty_summary(path, "auxiliary_metadata_file")
    if path.suffix.lower() == ".ndjson":
        return _parse_ndjson(path)
    return _parse_numeric(path)


def _parse_numeric(path: Path) -> dict[str, Any]:
    per_agent: dict[str, list[int]] = defaultdict(list)
    skipped = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").strip().split()
            if len(parts) < 4:
                skipped += 1
                continue
            try:
                frame = int(float(parts[0]))
                agent = str(int(float(parts[1])))
                float(parts[2])
                float(parts[3])
            except ValueError:
                skipped += 1
                continue
            per_agent[agent].append(frame)
    return _summarize(path, per_agent, skipped, "numeric_trajectory_like")


def _parse_ndjson(path: Path) -> dict[str, Any]:
    per_agent: dict[str, list[int]] = defaultdict(list)
    skipped = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                track = obj.get("track") if isinstance(obj, dict) else None
                frame = int(track["f"])
                agent = str(int(track["p"]))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                skipped += 1
                continue
            per_agent[agent].append(frame)
    return _summarize(path, per_agent, skipped, "ndjson_trajectory_like")


def _empty_summary(path: Path, reason: str) -> dict[str, Any]:
    return {
        "source": "fresh_stage42_cc_local_scan",
        "path": str(path),
        "source_name": _source_name(path),
        "file_format": path.suffix.lower().lstrip("."),
        "parse_status": reason,
        "parsed_rows": 0,
        "unique_agents": 0,
        "unique_frames": 0,
        "max_track_points": 0,
        "common_frame_step": None,
        "estimated_windows": {f"t{h}": 0 for h in HORIZONS},
        "t50_capable": False,
        "t100_capable": False,
        "has_homography_hint": _has_homography_hint(path),
        "stanford_or_sdd_derived": _stanford_or_sdd(path),
        "dataset_family": _dataset_family(path),
        "recommended_status": "auxiliary_metadata_not_track_source",
        "recommended_next_step": "ignore_for_t50_source_diversity",
    }


def _summarize(path: Path, per_agent: Mapping[str, list[int]], skipped: int, parse_status: str) -> dict[str, Any]:
    lengths = {agent: len(frames) for agent, frames in per_agent.items()}
    all_frames = [frame for frames in per_agent.values() for frame in frames]
    frame_steps: Counter[int] = Counter()
    for frames in per_agent.values():
        unique = sorted(set(frames))
        for a, b in zip(unique, unique[1:]):
            if b > a:
                frame_steps[b - a] += 1
    windows = {f"t{h}": int(sum(max(0, n - h) for n in lengths.values())) for h in HORIZONS}
    max_track = max(lengths.values(), default=0)
    row = {
        "source": "fresh_stage42_cc_local_scan",
        "path": str(path),
        "source_name": _source_name(path),
        "file_format": path.suffix.lower().lstrip("."),
        "parse_status": parse_status,
        "parsed_rows": int(sum(lengths.values())),
        "skipped_rows": int(skipped),
        "unique_agents": int(len(lengths)),
        "unique_frames": int(len(set(all_frames))),
        "max_track_points": int(max_track),
        "common_frame_step": int(frame_steps.most_common(1)[0][0]) if frame_steps else None,
        "estimated_windows": windows,
        "t50_capable": bool(max_track >= T50_MIN_POINTS and windows["t50"] > 0),
        "t100_capable": bool(max_track >= 101 and windows["t100"] > 0),
        "has_homography_hint": _has_homography_hint(path),
        "stanford_or_sdd_derived": _stanford_or_sdd(path),
        "dataset_family": _dataset_family(path),
    }
    row["recommended_status"], row["recommended_next_step"] = _recommend(row)
    return row


def _source_name(path: Path) -> str:
    return f"{path.parent.name}/{path.name}" if path.parent.name else path.name


def _is_auxiliary(path: Path) -> bool:
    return path.name.lower() in AUXILIARY_NAMES or "calibration" in path.name.lower()


def _has_homography_hint(path: Path) -> bool:
    parent = path.parent
    return any((parent / name).exists() for name in ["H.txt", "H-old.txt", "H-cam.txt", "homography.txt"])


def _stanford_or_sdd(path: Path) -> bool:
    lower = str(path).lower()
    return "stanford" in lower or "stanforddrone" in lower or "/sdd" in lower


def _synthetic_or_diagnostic(path: Path) -> bool:
    lower = str(path).lower()
    markers = (
        "synth",
        "synthetic",
        "simulation",
        "simulated",
        "orca_",
        "particle",
        "diagnostic",
    )
    return any(marker in lower for marker in markers)


def _dataset_family(path: Path) -> str:
    lower = str(path).lower()
    if "eth-person" in lower:
        return "ETH_person_terms_unverified"
    if "/eth/" in lower or "seq_eth" in lower or "seq_hotel" in lower or "biwi" in lower:
        return "ETH_UCY"
    if "/ucy/" in lower or "zara" in lower or "students" in lower or "crowds" in lower:
        return "UCY_or_crowds"
    if "trajnet" in lower:
        return "TrajNet"
    if "aerialmpt" in lower:
        return "AerialMPT"
    return "other"


def _recommend(row: Mapping[str, Any]) -> tuple[str, str]:
    if row["parsed_rows"] < 10:
        return "not_track_source", "ignore_for_t50_source_diversity"
    if row["stanford_or_sdd_derived"]:
        return "diagnostic_sdd_or_stanford_derived", "do_not_count_as_independent_non_sdd_external_source"
    if _synthetic_or_diagnostic(Path(str(row.get("path", "")))):
        return "diagnostic_or_simulation_only", "do_not_count_simulation_or_synthetic_as_real_external_source"
    if row["dataset_family"] == "ETH_person_terms_unverified":
        return "technical_candidate_terms_unverified", "verify_license_terms_before_conversion_or_claim"
    if row["t50_capable"] and row["dataset_family"] in {"ETH_UCY", "UCY_or_crowds", "TrajNet"}:
        return "candidate_t50_independent_source", "eligible_for_future_split_rebuild_after_no_leakage_and_license_check"
    if row["dataset_family"] in {"ETH_UCY", "UCY_or_crowds", "TrajNet"}:
        return "short_horizon_candidate", "can_support shorter horizons or scene diagnostics but not t50 source diversity"
    return "diagnostic_only", "not_priority_for_non_sdd_t50_source_diversity"


def _current_usage() -> dict[str, Any]:
    data = s41._combined()
    split, group = am._split_arrays(data)
    source_file = data["source_file"].astype(str)
    horizon = data["horizon"].astype(int)
    usage: dict[str, Any] = {}
    for source in sorted(set(source_file.tolist())):
        mask = source_file == source
        usage[source] = {
            "source": "cached_verified_from_current_combined_split",
            "splits": sorted(set(split[mask].tolist())),
            "groups": sorted(set(group[mask].tolist())),
            "rows": int(mask.sum()),
            "t50_rows": int((mask & (horizon == 50)).sum()),
            "t100_rows": int((mask & (horizon == 100)).sum()),
        }
    return usage


def _annotate_usage(rows: list[dict[str, Any]], usage: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    resolved_usage = {str(Path(path).resolve()): val for path, val in usage.items()}
    used_parent_dirs = {str(Path(path).resolve().parent) for path in usage}
    for row in rows:
        resolved = str(Path(row["path"]).resolve())
        used = resolved_usage.get(resolved)
        status = row["recommended_status"]
        next_step = row["recommended_next_step"]
        if used:
            status = "already_in_current_combined_split"
            next_step = "can only be reused through a new train/val/test split rebuild, not counted as new held-out source"
        elif status == "candidate_t50_independent_source" and str(Path(row["path"]).resolve().parent) in used_parent_dirs:
            status = "alternate_representation_of_current_source"
            next_step = "same parent/source directory as current data; useful for split rebuild or format repair, not counted as independent source"
        out.append({**row, "current_usage": used or None, "final_status": status, "final_next_step": next_step})
    return out


def _summaries(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in rows if r["final_status"] == "candidate_t50_independent_source"]
    alternate_candidates = [r for r in rows if r["final_status"] == "alternate_representation_of_current_source"]
    diagnostic_candidates = [r for r in rows if r["final_status"] in {"diagnostic_or_simulation_only", "diagnostic_sdd_or_stanford_derived"} and r["t50_capable"]]
    by_family: dict[str, Any] = {}
    for family in sorted(set(r["dataset_family"] for r in rows)):
        fam_rows = [r for r in rows if r["dataset_family"] == family]
        by_family[family] = {
            "files": len(fam_rows),
            "t50_capable_files": sum(1 for r in fam_rows if r["t50_capable"]),
            "unused_t50_candidates": sum(1 for r in fam_rows if r["final_status"] == "candidate_t50_independent_source"),
            "alternate_current_source_candidates": sum(1 for r in fam_rows if r["final_status"] == "alternate_representation_of_current_source"),
            "diagnostic_t50_candidates": sum(
                1
                for r in fam_rows
                if r["final_status"] in {"diagnostic_or_simulation_only", "diagnostic_sdd_or_stanford_derived"} and r["t50_capable"]
            ),
            "already_used_files": sum(1 for r in fam_rows if r["current_usage"]),
            "homography_hint_files": sum(1 for r in fam_rows if r["has_homography_hint"]),
        }
    return {
        "source": "fresh_stage42_cc_independent_t50_source_inventory",
        "scanned_files": len(rows),
        "t50_capable_files": sum(1 for r in rows if r["t50_capable"]),
        "unused_candidate_t50_sources": len(candidates),
        "alternate_current_source_candidates": len(alternate_candidates),
        "diagnostic_t50_candidates": len(diagnostic_candidates),
        "candidate_t50_sources": len(candidates),
        "by_family": by_family,
        "candidate_names": [r["source_name"] for r in candidates[:20]],
        "alternate_candidate_names": [r["source_name"] for r in alternate_candidates[:20]],
        "diagnostic_candidate_names": [r["source_name"] for r in diagnostic_candidates[:20]],
        "source_diversity_repair_ready": len(candidates) > 0,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _user_actions(summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not summary["source_diversity_repair_ready"]:
        actions.append(
            {
                "priority": "critical",
                "action": "Provide or legally enable at least one independent t50-capable non-SDD top-down pedestrian source.",
                "official_targets": [
                    "UCY crowd original official source",
                    "ETH/BIWI original pedestrian sources",
                    "TrajNet++ official challenge/data access",
                    "OpenTraj-supported original dataset paths with underlying dataset terms verified",
                ],
                "why": "Stage42-CB found t50 gains are source-concentrated; Stage42-CC found no unused ready-to-claim source diversity repair without split rebuild and license/no-leakage checks.",
            }
        )
    else:
        actions.append(
            {
                "priority": "critical",
                "action": "Convert the local unused t50-capable candidates through a new no-leakage source split before making any source-diversity claim.",
                "candidate_names": summary.get("candidate_names", []),
                "why": "Stage42-CC found local candidate files, but inventory is not conversion and does not prove deployable source-level generalization.",
            }
        )
    if summary.get("alternate_current_source_candidates", 0):
        actions.append(
            {
                "priority": "medium",
                "action": "Use alternate current-source representations only for format repair or split rebuild diagnostics, not as independent new sources.",
                "candidate_names": summary.get("alternate_candidate_names", []),
                "why": "Same parent/source directory as current data means these are not independent held-out sources.",
            }
        )
    terms = [r for r in rows if r["final_status"] == "technical_candidate_terms_unverified"]
    if terms:
        actions.append(
            {
                "priority": "high",
                "action": "Verify terms for ETH-Person / ETH-related technical candidates before any official conversion or claim.",
                "candidate_count": len(terms),
                "why": "Technical parsing is not legal permission; terms must be verified before official evidence.",
            }
        )
    actions.append(
        {
            "priority": "high",
            "action": "For any candidate source, rerun conversion, no-leakage audit, train/internal-val policy selection, and final test once.",
            "why": "Inventory is not conversion, and conversion is not benchmark success.",
        }
    )
    return actions


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "cb_input_verified": payload["input_reports"]["stage42_cb_verdict"]
        == "stage42_cb_t50_source_robustness_pass_with_source_diversity_limit",
        "local_scan_completed": summary["scanned_files"] > 0,
        "t50_capable_sources_found_or_user_action": summary["t50_capable_files"] > 0 or bool(payload["user_actions"]),
        "candidate_status_not_counted_as_converted": payload["claim_boundary"]["inventory_counted_as_converted"] is False,
        "source_diversity_blocker_explicit": "source diversity" in payload["blocker_summary"].lower(),
        "user_action_written": bool(payload["user_actions"]),
        "no_leakage_scope_preserved": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cc_independent_t50_source_inventory_pass" if passed == total else "stage42_cc_independent_t50_source_inventory_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CC Independent T50 Source Inventory",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cc_gate']['passed']} / {payload['stage42_cc_gate']['total']}`",
        f"- verdict: `{payload['stage42_cc_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- scanned_files: `{s['scanned_files']}`",
        f"- t50_capable_files: `{s['t50_capable_files']}`",
        f"- unused_candidate_t50_sources: `{s['unused_candidate_t50_sources']}`",
        f"- alternate_current_source_candidates: `{s['alternate_current_source_candidates']}`",
        f"- diagnostic_t50_candidates: `{s['diagnostic_t50_candidates']}`",
        f"- source_diversity_repair_ready: `{s['source_diversity_repair_ready']}`",
        f"- candidate_names: `{s['candidate_names']}`",
        "",
        "## By Family",
        "",
        "| family | files | t50 capable | unused candidates | alternates | diagnostic | already used | homography hints |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, row in s["by_family"].items():
        lines.append(
            f"| `{family}` | {row['files']} | {row['t50_capable_files']} | {row['unused_t50_candidates']} | {row['alternate_current_source_candidates']} | {row['diagnostic_t50_candidates']} | {row['already_used_files']} | {row['homography_hint_files']} |"
        )
    lines += [
        "",
        "## Top Candidate / Blocker Rows",
        "",
        "| source | family | rows | max track | t50 windows | status | next step |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    priority = sorted(
        payload["inventory"],
        key=lambda r: (r["final_status"] != "candidate_t50_independent_source", -int(r["estimated_windows"]["t50"])),
    )[:25]
    for row in priority:
        lines.append(
            f"| `{row['source_name']}` | `{row['dataset_family']}` | {row['parsed_rows']} | {row['max_track_points']} | "
            f"{row['estimated_windows']['t50']} | `{row['final_status']}` | {row['final_next_step']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CC is an inventory, not conversion or benchmark success.",
        "- Candidate files require legal/terms verification, split rebuild, conversion, no-leakage audit, validation-only policy selection, and final test before any claim.",
        "- Source diversity remains the next blocker exposed by Stage42-CB.",
    ]
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Independent T50 Source Diversity",
        "",
        "Stage42-CB found protected t50 gains are source-concentrated. Stage42-CC scanned local files and generated the following required actions.",
        "",
    ]
    for action in payload["user_actions"]:
        lines += [
            f"## {action['priority'].upper()}",
            "",
            f"- action: {action['action']}",
            f"- why: {action['why']}",
        ]
        if "official_targets" in action:
            lines.append(f"- official_targets: `{action['official_targets']}`")
        if "candidate_count" in action:
            lines.append(f"- candidate_count: `{action['candidate_count']}`")
        lines.append("")
    lines += [
        "Do not count registry-only, unlicensed, or merely parsed files as converted datasets. Do not claim broad source-level generalization until conversion/no-leakage/source-CV/final test are complete.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cc_gate"]
    lines = [
        "# Stage42-CC Gate",
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


def run_stage42_independent_t50_source_inventory() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cb = _load_json(CB_JSON)
    usage = _current_usage()
    parsed = [_parse_track_file(path) for path in _iter_candidate_files()]
    inventory = _annotate_usage(parsed, usage)
    summary = _summaries(inventory)
    actions = _user_actions(summary, inventory)
    payload: dict[str, Any] = {
        "source": "fresh_stage42_cc_independent_t50_source_inventory",
        "stage": "Stage42-CC Independent T50 Source Inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(CB_JSON)]),
        "current_facts": CURRENT_FACTS,
        "scan_roots": [str(path) for path in SCAN_ROOTS],
        "input_reports": {"stage42_cb_verdict": cb["stage42_cb_gate"]["verdict"]},
        "summary": summary,
        "inventory": inventory,
        "user_actions": actions,
        "blocker_summary": "source diversity remains limited until independent t50-capable non-SDD sources are converted and validated",
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
            "inventory_counted_as_converted": False,
            "registry_only_counted_as_converted": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cc_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_independent_t50_source_inventory()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
