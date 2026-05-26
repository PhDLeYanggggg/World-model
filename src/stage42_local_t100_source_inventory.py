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
REPORT_JSON = OUT_DIR / "local_t100_source_inventory_stage42.json"
REPORT_MD = OUT_DIR / "local_t100_source_inventory_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bd_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_t100_conversion_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

DATASETS_ROOT = Path("external_data/OpenTraj/datasets")
SCAN_ROOTS = [
    DATASETS_ROOT / "ETH",
    DATASETS_ROOT / "ETH-Person",
    DATASETS_ROOT / "UCY",
    DATASETS_ROOT / "TrajNet",
    DATASETS_ROOT / "TrajNet++",
]
MAX_FILES = 500
T100_MIN_TRACK_POINTS = 101


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BD 是本机 local t100 source inventory，不训练模型、不下载数据。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "本步骤只识别可转换候选；是否纳入 official evaluation 还需要后续 conversion/no-leakage/source-CV。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _iter_candidate_files(roots: Iterable[Path] = SCAN_ROOTS) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".ndjson"}:
                files.append(path)
                if len(files) >= MAX_FILES:
                    return sorted(files)
    return sorted(files)


def _parse_numeric_trajectory_file(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, str, float, float]] = []
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
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                skipped += 1
                continue
            rows.append((frame, agent, x, y))
    return _summarize_rows(path, rows, skipped)


def _parse_ndjson_trajectory_file(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, str, float, float]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            track = obj.get("track") if isinstance(obj, dict) else None
            if not isinstance(track, dict):
                skipped += 1
                continue
            try:
                frame = int(track["f"])
                agent = str(int(track["p"]))
                x = float(track["x"])
                y = float(track["y"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            rows.append((frame, agent, x, y))
    return _summarize_rows(path, rows, skipped)


def _summarize_rows(path: Path, rows: list[tuple[int, str, float, float]], skipped: int) -> dict[str, Any]:
    per_agent: dict[str, list[int]] = defaultdict(list)
    for frame, agent, _, _ in rows:
        per_agent[agent].append(frame)
    track_lengths = {agent: len(frames) for agent, frames in per_agent.items()}
    max_track = max(track_lengths.values(), default=0)
    t100_window_estimate = int(sum(max(0, n - T100_MIN_TRACK_POINTS + 1) for n in track_lengths.values()))
    frames = [row[0] for row in rows]
    frame_step_counter: Counter[int] = Counter()
    for agent, agent_frames in per_agent.items():
        sorted_frames = sorted(set(agent_frames))
        for a, b in zip(sorted_frames, sorted_frames[1:]):
            step = b - a
            if step > 0:
                frame_step_counter[step] += 1
    common_frame_step = frame_step_counter.most_common(1)[0][0] if frame_step_counter else None
    rel = _safe_rel(path)
    return {
        "source": "fresh_local_parse",
        "path": str(path),
        "relative_path": rel,
        "file_format": path.suffix.lower().lstrip("."),
        "parsed_rows": int(len(rows)),
        "skipped_rows": int(skipped),
        "unique_agents": int(len(per_agent)),
        "unique_frames": int(len(set(frames))),
        "min_frame": int(min(frames)) if frames else None,
        "max_frame": int(max(frames)) if frames else None,
        "common_frame_step": int(common_frame_step) if common_frame_step is not None else None,
        "max_track_points": int(max_track),
        "t100_capable": bool(max_track >= T100_MIN_TRACK_POINTS),
        "estimated_t100_windows": t100_window_estimate,
        "synthetic_or_diagnostic": _synthetic_or_diagnostic(path),
    }


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(DATASETS_ROOT))
    except ValueError:
        return str(path)


def _synthetic_or_diagnostic(path: Path) -> bool:
    lower = str(path).lower()
    return any(token in lower for token in ["synth", "orca", "sim", "kitti", "argoverse", "nuscenes", "waymo"])


def _candidate_groups(path: Path) -> list[str]:
    rel = _safe_rel(path)
    groups: list[str] = []
    if rel.startswith("ETH/") or rel.startswith("UCY/"):
        groups.append(f"ETH_UCY::{rel}")
    if rel.startswith("TrajNet/"):
        groups.append(f"TrajNet::{rel}")
    if rel.startswith("UCY/"):
        groups.append(f"UCY::{rel}")
    if rel.startswith("TrajNet++/"):
        groups.append(f"TrajNet::{rel}")
    return groups


def _current_source_cv_groups() -> dict[str, Any]:
    data = s41._combined()
    split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    groups: dict[str, dict[str, Any]] = {}
    for g in sorted(set(group.tolist())):
        mask = group == g
        d_counts = Counter(domain[mask].tolist())
        d = d_counts.most_common(1)[0][0] if d_counts else "unknown"
        train_t100 = int(((split == "train") & mask & (horizon == 100)).sum())
        all_t100 = int((mask & (horizon == 100)).sum())
        groups[str(g)] = {
            "domain": d,
            "train_t100_rows": train_t100,
            "all_t100_rows": all_t100,
        }
    return groups


def _annotate_inventory(rows: list[Mapping[str, Any]], known_groups: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    known_set = set(known_groups.keys())
    for row in rows:
        path = Path(str(row["path"]))
        candidates = _candidate_groups(path)
        matched = [g for g in candidates if g in known_set]
        rel = str(row["relative_path"])
        if rel.startswith("ETH/"):
            suggested_domain = "ETH_UCY"
        elif rel.startswith("UCY/"):
            suggested_domain = "UCY_or_ETH_UCY"
        elif rel.startswith("TrajNet/") or rel.startswith("TrajNet++/"):
            suggested_domain = "TrajNet"
        else:
            suggested_domain = "diagnostic"
        status = "already_in_current_source_groups" if matched else "local_candidate_not_in_current_source_groups"
        if row["synthetic_or_diagnostic"]:
            status = "diagnostic_or_synthetic_not_official_external_repair"
        out.append(
            {
                **dict(row),
                "candidate_group_names": candidates,
                "matched_current_groups": matched,
                "already_in_current_source_groups": bool(matched),
                "suggested_domain": suggested_domain,
                "conversion_status": status,
                "recommended_next_step": _next_step(row, matched, suggested_domain),
            }
        )
    return out


def _next_step(row: Mapping[str, Any], matched: list[str], suggested_domain: str) -> str:
    if row["synthetic_or_diagnostic"]:
        return "diagnostic_only_or_exclude_from_real_external_t100_claim"
    if matched:
        return "already_used_by_current_stage42_source_cv_or_combined_data"
    if row["t100_capable"] and suggested_domain in {"ETH_UCY", "TrajNet", "UCY_or_ETH_UCY"}:
        return "candidate_for_stage42_be_conversion_and_train_only_source_cv"
    if suggested_domain in {"ETH_UCY", "TrajNet", "UCY_or_ETH_UCY"}:
        return "candidate_for_short_horizon_or_scene_goal_diagnostic_not_t100"
    return "not_priority_for_external_pedestrian_t100"


def run_stage42_local_t100_source_inventory() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    files = _iter_candidate_files()
    parsed: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() == ".ndjson":
            parsed.append(_parse_ndjson_trajectory_file(path))
        else:
            parsed.append(_parse_numeric_trajectory_file(path))
    known_groups = _current_source_cv_groups()
    inventory = _annotate_inventory(parsed, known_groups)
    novel_t100 = [
        row
        for row in inventory
        if row["t100_capable"]
        and not row["already_in_current_source_groups"]
        and not row["synthetic_or_diagnostic"]
        and row["suggested_domain"] in {"ETH_UCY", "TrajNet", "UCY_or_ETH_UCY"}
    ]
    by_domain: dict[str, Any] = {}
    for domain in ["ETH_UCY", "TrajNet", "UCY_or_ETH_UCY"]:
        rows = [row for row in inventory if row["suggested_domain"] == domain]
        by_domain[domain] = {
            "source": "fresh_local_inventory",
            "files": len(rows),
            "t100_capable_files": sum(1 for row in rows if row["t100_capable"]),
            "already_used_t100_files": sum(1 for row in rows if row["t100_capable"] and row["already_in_current_source_groups"]),
            "novel_t100_candidate_files": sum(
                1
                for row in rows
                if row["t100_capable"] and not row["already_in_current_source_groups"] and not row["synthetic_or_diagnostic"]
            ),
            "estimated_novel_t100_windows": int(
                sum(
                    row["estimated_t100_windows"]
                    for row in rows
                    if row["t100_capable"] and not row["already_in_current_source_groups"] and not row["synthetic_or_diagnostic"]
                )
            ),
        }
    payload = {
        "source": "fresh_local_path_inventory",
        "stage": "Stage42-BD Local T100 Source Inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(root) for root in SCAN_ROOTS] + ["data/stage41_world_model/combined_external.npz"]),
        "scan_roots": [str(root) for root in SCAN_ROOTS],
        "known_current_group_count": len(known_groups),
        "inventory": inventory,
        "novel_t100_candidates": novel_t100,
        "summary": {
            "source": "fresh_local_path_inventory",
            "files_scanned": len(files),
            "parseable_files": sum(1 for row in inventory if row["parsed_rows"] > 0),
            "t100_capable_files": sum(1 for row in inventory if row["t100_capable"]),
            "already_used_t100_files": sum(1 for row in inventory if row["t100_capable"] and row["already_in_current_source_groups"]),
            "novel_t100_candidate_files": len(novel_t100),
            "estimated_novel_t100_windows": int(sum(row["estimated_t100_windows"] for row in novel_t100)),
            "by_domain": by_domain,
            "stage42_be_conversion_recommended": bool(novel_t100),
            "stage5c_executed": False,
            "smc_enabled": False,
            "metric_or_seconds_claim": False,
        },
        "user_action_required": _user_actions(novel_t100),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "local_file_inventory_only": True,
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
    payload["stage42_bd_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _user_actions(novel_t100: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if novel_t100:
        actions.append(
            {
                "source": "fresh_local_inventory",
                "priority": "high",
                "action_type": "stage42_be_convert_local_novel_t100_sources",
                "candidate_count": len(novel_t100),
                "top_candidates": [row["relative_path"] for row in novel_t100[:20]],
                "notes": "Convert only after source-specific terms/provenance are verified; then rerun train-only source-CV without test metrics.",
            }
        )
    actions.append(
        {
            "source": "fresh_local_inventory",
            "priority": "medium",
            "action_type": "verify_terms_before_claim",
            "notes": "Local availability does not imply redistribution or metric/seconds-level claim permission.",
        }
    )
    return actions


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "local_roots_scanned": s["files_scanned"] > 0,
        "parseable_sources_found": s["parseable_files"] > 0,
        "known_current_groups_loaded": payload["known_current_group_count"] > 0,
        "t100_capable_sources_identified": s["t100_capable_files"] > 0,
        "novel_candidates_reported_or_explicit_none": s["novel_t100_candidate_files"] >= 0,
        "conversion_recommendation_recorded": "stage42_be_conversion_recommended" in s,
        "no_leakage_pass": all(
            payload["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_metrics_for_threshold"]
        ),
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bd_local_t100_source_inventory_pass" if passed == total else "stage42_bd_local_t100_source_inventory_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BD Local T100 Source Inventory",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bd_gate']['passed']} / {payload['stage42_bd_gate']['total']}`",
        f"- verdict: `{payload['stage42_bd_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- files_scanned: `{s['files_scanned']}`",
        f"- parseable_files: `{s['parseable_files']}`",
        f"- t100_capable_files: `{s['t100_capable_files']}`",
        f"- already_used_t100_files: `{s['already_used_t100_files']}`",
        f"- novel_t100_candidate_files: `{s['novel_t100_candidate_files']}`",
        f"- estimated_novel_t100_windows: `{s['estimated_novel_t100_windows']}`",
        f"- stage42_be_conversion_recommended: `{s['stage42_be_conversion_recommended']}`",
        "",
        "## Domain Summary",
        "",
        "| domain | files | t100 capable | already used | novel candidates | est novel t100 windows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in s["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['files']} | {row['t100_capable_files']} | {row['already_used_t100_files']} | {row['novel_t100_candidate_files']} | {row['estimated_novel_t100_windows']} |"
        )
    lines.extend(
        [
            "",
            "## Top Novel T100 Candidates",
            "",
            "| path | domain | rows | agents | max track | est t100 windows | next step |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["novel_t100_candidates"][:40]:
        lines.append(
            f"| `{row['relative_path']}` | {row['suggested_domain']} | {row['parsed_rows']} | {row['unique_agents']} | {row['max_track_points']} | {row['estimated_t100_windows']} | {row['recommended_next_step']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-BD found local candidate files; this is not yet conversion, training, or official evaluation.",
            "- Any novel t100 candidates must go through Stage42-BE conversion, no-leakage split construction, and train-only source-CV before t100 claims can change.",
            "- Synthetic/diagnostic files are excluded from real external t100 repair claims unless a separate diagnostic-only protocol is written.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BD User Action Required For Local T100 Conversion",
        "",
        f"- source: `{payload['source']}`",
        "",
    ]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['action_type']}",
                "",
                f"- priority: `{action['priority']}`",
            ]
        )
        if "candidate_count" in action:
            lines.append(f"- candidate_count: `{action['candidate_count']}`")
        if "top_candidates" in action:
            lines.append("- top_candidates:")
            lines.extend([f"  - `{path}`" for path in action["top_candidates"]])
        lines.append(f"- notes: {action['notes']}")
        lines.append("")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bd_gate"]
    lines = [
        "# Stage42-BD Gate",
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
        "verdict": payload["stage42_bd_gate"]["verdict"],
        "gate": f"{payload['stage42_bd_gate']['passed']}/{payload['stage42_bd_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_local_t100_source_inventory()
