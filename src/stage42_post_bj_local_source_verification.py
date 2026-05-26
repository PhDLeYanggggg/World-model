from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import stage42_local_t100_source_inventory as bd
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BJ_JSON = OUT_DIR / "post_bi_t100_source_package_stage42.json"
REPORT_JSON = OUT_DIR / "post_bj_local_source_verification_stage42.json"
REPORT_MD = OUT_DIR / "post_bj_local_source_verification_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bk_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_post_bj_local_sources_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

DATASETS_ROOT = Path("external_data/OpenTraj/datasets")
SCAN_ROOTS = [
    DATASETS_ROOT / "ETH",
    DATASETS_ROOT / "ETH-Person",
    DATASETS_ROOT / "UCY",
    DATASETS_ROOT / "TrajNet",
    DATASETS_ROOT / "TrajNet++",
]
SUPPORTED_SUFFIXES = {".txt", ".csv", ".ndjson", ".xml"}
T100_MIN_TRACK_POINTS = 101


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BK 是 post-BJ 本地 source/path verification 和 loader-gap audit，不训练模型。",
    "本步骤检查本地 OpenTraj/ETH/UCY/TrajNet/ETH-Person 文件是否有 t100 conversion potential。",
    "本步骤不会把本地路径存在写成 license 已确认，也不会把 conversion candidate 写成 evaluated source。",
    "future waypoints / endpoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_files(roots: Iterable[Path] = SCAN_ROOTS) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
    return sorted(files)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(DATASETS_ROOT))
    except ValueError:
        return str(path)


def _domain(path: Path) -> str:
    rel = _safe_rel(path)
    if rel.startswith("ETH/") or rel.startswith("ETH-Person/"):
        return "ETH_UCY"
    if rel.startswith("UCY/"):
        return "UCY"
    if rel.startswith("TrajNet/") or rel.startswith("TrajNet++/"):
        return "TrajNet"
    return "diagnostic"


def _independent_key(path: Path) -> str:
    rel = _safe_rel(path)
    parts = rel.split("/")
    domain = _domain(path)
    if rel.startswith("ETH-Person/") and len(parts) >= 3:
        return f"{domain}::{parts[0]}/{parts[1]}/{Path(parts[2]).stem}"
    if len(parts) >= 2:
        return f"{domain}::{parts[0]}/{parts[1]}"
    return f"{domain}::{rel}"


def _is_diagnostic(path: Path) -> bool:
    lower = str(path).lower()
    return any(token in lower for token in ["synth", "orca", "kitti", "argoverse", "nuscenes", "waymo"])


def _summarize_rows(path: Path, rows: list[tuple[int, str, float, float]], skipped_rows: int) -> dict[str, Any]:
    per_agent: dict[str, list[int]] = defaultdict(list)
    for frame, agent, _, _ in rows:
        per_agent[agent].append(frame)
    lengths = {agent: len(frames) for agent, frames in per_agent.items()}
    max_track = max(lengths.values(), default=0)
    t100_windows = int(sum(max(0, n - T100_MIN_TRACK_POINTS + 1) for n in lengths.values()))
    frames = [row[0] for row in rows]
    step_counter: Counter[int] = Counter()
    for agent_frames in per_agent.values():
        uniq = sorted(set(agent_frames))
        for a, b in zip(uniq, uniq[1:]):
            step = b - a
            if step > 0:
                step_counter[step] += 1
    common_step = step_counter.most_common(1)[0][0] if step_counter else None
    return {
        "source": "fresh_post_bj_local_source_parse",
        "path": str(path),
        "relative_path": _safe_rel(path),
        "domain": _domain(path),
        "independent_key": _independent_key(path),
        "file_format": path.suffix.lower().lstrip("."),
        "parsed_rows": int(len(rows)),
        "skipped_rows": int(skipped_rows),
        "unique_agents": int(len(per_agent)),
        "unique_frames": int(len(set(frames))),
        "min_frame": int(min(frames)) if frames else None,
        "max_frame": int(max(frames)) if frames else None,
        "common_frame_step": int(common_step) if common_step is not None else None,
        "max_track_points": int(max_track),
        "t100_capable": bool(max_track >= T100_MIN_TRACK_POINTS),
        "estimated_t100_windows": t100_windows,
        "synthetic_or_diagnostic": _is_diagnostic(path),
    }


def _parse_numeric(path: Path) -> dict[str, Any]:
    return bd._parse_numeric_trajectory_file(path)


def _parse_ndjson(path: Path) -> dict[str, Any]:
    return bd._parse_ndjson_trajectory_file(path)


def _parse_xml(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, str, float, float]] = []
    skipped = 0
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return _summarize_rows(path, [], 1)
    for frame_node in root.findall(".//frame"):
        try:
            frame = int(float(frame_node.attrib.get("number", "")))
        except ValueError:
            skipped += 1
            continue
        for object_node in frame_node.findall(".//object"):
            box = object_node.find("box")
            if box is None:
                skipped += 1
                continue
            try:
                agent = str(int(float(object_node.attrib.get("id", ""))))
                x = float(box.attrib["xc"])
                y = float(box.attrib["yc"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            rows.append((frame, agent, x, y))
    return _summarize_rows(path, rows, skipped)


def _parse_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".txt", ".csv"}:
        row = _parse_numeric(path)
    elif path.suffix.lower() == ".ndjson":
        row = _parse_ndjson(path)
    elif path.suffix.lower() == ".xml":
        row = _parse_xml(path)
    else:
        row = _summarize_rows(path, [], 0)
    return {
        **row,
        "domain": _domain(path),
        "independent_key": _independent_key(path),
        "synthetic_or_diagnostic": _is_diagnostic(path),
        "loader_supported": path.suffix.lower() in SUPPORTED_SUFFIXES,
    }


def _classify(rows: list[Mapping[str, Any]], bj: Mapping[str, Any]) -> dict[str, Any]:
    current_support = bj.get("domain_support", {})
    by_domain: dict[str, dict[str, Any]] = {}
    conversion_candidates: list[dict[str, Any]] = []
    loader_gaps: list[dict[str, Any]] = []
    for domain in ["ETH_UCY", "UCY", "TrajNet"]:
        domain_rows = [row for row in rows if row["domain"] == domain and not row["synthetic_or_diagnostic"]]
        t100_rows = [row for row in domain_rows if row["t100_capable"]]
        groups = sorted({str(row["independent_key"]) for row in t100_rows})
        existing = int(current_support.get(domain, {}).get("independent_sources", 0) or 0)
        potential_new = max(0, len(groups) - existing)
        by_domain[domain] = {
            "source": "fresh_post_bj_local_source_verification",
            "domain": domain,
            "parsed_files": len(domain_rows),
            "t100_capable_files": len(t100_rows),
            "independent_t100_groups": len(groups),
            "independent_t100_group_ids": groups,
            "stage42_bj_independent_sources": existing,
            "potential_new_independent_groups_vs_bj": potential_new,
            "additional_sources_needed_after_bj": current_support.get(domain, {}).get("additional_independent_sources_needed"),
        }
    for row in rows:
        if row["synthetic_or_diagnostic"]:
            continue
        if row["domain"] not in {"ETH_UCY", "UCY", "TrajNet"}:
            continue
        if row["t100_capable"]:
            license_status = "local_path_present_terms_unverified"
            conversion_candidates.append(
                {
                    "source": "fresh_post_bj_conversion_candidate",
                    "relative_path": row["relative_path"],
                    "domain": row["domain"],
                    "independent_key": row["independent_key"],
                    "file_format": row["file_format"],
                    "max_track_points": row["max_track_points"],
                    "estimated_t100_windows": row["estimated_t100_windows"],
                    "license_status": license_status,
                    "conversion_status": "candidate_pending_license_terms_and_source_cv",
                }
            )
        elif row["domain"] == "TrajNet" and row["parsed_rows"] > 0:
            loader_gaps.append(
                {
                    "source": "fresh_post_bj_loader_gap",
                    "relative_path": row["relative_path"],
                    "domain": row["domain"],
                    "max_track_points": row["max_track_points"],
                    "reason": "TrajNet local files parse as fixed short snippets (max <=20), not raw long tracks, so they cannot repair raw-frame t100.",
                }
            )
    conversion_candidates.sort(key=lambda r: (r["domain"], r["independent_key"], r["relative_path"]))
    return {
        "source": "fresh_post_bj_local_source_verification",
        "by_domain": by_domain,
        "conversion_candidates": conversion_candidates,
        "loader_gaps": loader_gaps[:80],
    }


def run_stage42_post_bj_local_source_verification() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bj = _load_json(BJ_JSON)
    files = _iter_files()
    rows = [_parse_file(path) for path in files]
    classified = _classify(rows, bj)
    eth = classified["by_domain"]["ETH_UCY"]
    traj = classified["by_domain"]["TrajNet"]
    payload: dict[str, Any] = {
        "source": "fresh_post_bj_local_source_verification",
        "stage": "Stage42-BK Post-BJ Local Source Verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BJ_JSON)] + [str(path) for path in files[:200]]),
        "current_facts": CURRENT_FACTS,
        "bj_verdict": bj.get("stage42_bj_gate", {}).get("verdict"),
        "scan_roots": [str(root) for root in SCAN_ROOTS],
        "file_count": len(files),
        "parsed_files": rows,
        "classified": classified,
        "summary": {
            "source": "fresh_post_bj_local_source_verification",
            "eth_ucy_t100_capable_files": eth["t100_capable_files"],
            "eth_ucy_independent_t100_groups": eth["independent_t100_groups"],
            "eth_ucy_potential_new_groups_vs_bj": eth["potential_new_independent_groups_vs_bj"],
            "trajnet_t100_capable_files": traj["t100_capable_files"],
            "trajnet_independent_t100_groups": traj["independent_t100_groups"],
            "trajnet_loader_gap_files": len(classified["loader_gaps"]),
            "conversion_candidates": len(classified["conversion_candidates"]),
            "eth_person_xml_candidates": [
                row["relative_path"]
                for row in classified["conversion_candidates"]
                if str(row["relative_path"]).startswith("ETH-Person/")
            ],
            "can_repair_eth_ucy_with_local_candidates_after_license_confirmation": eth["independent_t100_groups"] >= 3,
            "can_repair_trajnet_with_local_candidates": traj["independent_t100_groups"] >= 3,
            "auto_download_executed": False,
            "global_t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "converted_dataset_claim_allowed": False,
            "global_t100_positive_claim_allowed": False,
        },
    }
    payload["stage42_bk_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "bj_input_verified": payload["bj_verdict"] == "stage42_bj_post_bi_t100_source_package_pass",
        "local_roots_scanned": payload["file_count"] > 0,
        "xml_loader_gap_closed": len(summary["eth_person_xml_candidates"]) >= 3,
        "eth_ucy_repair_candidates_found": summary["can_repair_eth_ucy_with_local_candidates_after_license_confirmation"],
        "trajnet_gap_explained": summary["trajnet_t100_capable_files"] == 0 and summary["trajnet_loader_gap_files"] > 0,
        "conversion_candidates_not_counted_as_evaluated": not payload["claim_boundary"]["converted_dataset_claim_allowed"],
        "no_auto_download": not summary["auto_download_executed"],
        "no_global_t100_overclaim": not summary["global_t100_positive_claim_allowed"],
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"] and not payload["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bk_local_source_verification_pass" if passed == total else "stage42_bk_local_source_verification_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BK Post-BJ Local Source Verification",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bk_gate']['passed']} / {payload['stage42_bk_gate']['total']}`",
        f"- verdict: `{payload['stage42_bk_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Domain Summary",
        "",
        "| domain | parsed files | t100 files | independent t100 groups | potential new vs BJ | after-BJ needed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["classified"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['parsed_files']} | {row['t100_capable_files']} | {row['independent_t100_groups']} | {row['potential_new_independent_groups_vs_bj']} | {row['additional_sources_needed_after_bj']} |"
        )
    lines.extend(
        [
            "",
            "## Conversion Candidates",
            "",
            "| relative_path | domain | independent_key | format | max track | t100 windows | status |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["classified"]["conversion_candidates"]:
        lines.append(
            f"| `{row['relative_path']}` | `{row['domain']}` | `{row['independent_key']}` | `{row['file_format']}` | {row['max_track_points']} | {row['estimated_t100_windows']} | `{row['conversion_status']}` |"
        )
    lines.extend(
        [
            "",
            "## TrajNet Loader Gap",
            "",
            f"- trajnet_t100_capable_files: `{summary['trajnet_t100_capable_files']}`",
            f"- trajnet_loader_gap_files_sampled: `{summary['trajnet_loader_gap_files']}`",
            "- Interpretation: local TrajNet files are fixed short challenge snippets, not raw long-track sources; they cannot repair raw-frame t100 without original longer trajectories.",
            "",
            "## Summary",
            "",
            f"- eth_ucy_t100_capable_files: `{summary['eth_ucy_t100_capable_files']}`",
            f"- eth_ucy_independent_t100_groups: `{summary['eth_ucy_independent_t100_groups']}`",
            f"- eth_ucy_potential_new_groups_vs_bj: `{summary['eth_ucy_potential_new_groups_vs_bj']}`",
            f"- eth_person_xml_candidates: `{summary['eth_person_xml_candidates']}`",
            f"- can_repair_eth_ucy_with_local_candidates_after_license_confirmation: `{summary['can_repair_eth_ucy_with_local_candidates_after_license_confirmation']}`",
            f"- can_repair_trajnet_with_local_candidates: `{summary['can_repair_trajnet_with_local_candidates']}`",
            f"- global_t100_positive_claim_allowed: `{summary['global_t100_positive_claim_allowed']}`",
            "",
            "## Interpretation",
            "",
            "- ETH-Person XML files close a loader gap and provide local ETH_UCY t100 conversion candidates, pending license/terms confirmation and conversion/no-leakage/source-CV.",
            "- TrajNet local challenge snippets do not provide raw-frame t100 sources; TrajNet still needs official longer sources or a different non-t100 claim.",
            "- This is source/path verification only. It is not a converted dataset, trained model, evaluation result, metric claim, or seconds-level claim.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BK User Action Required",
        "",
        f"- source: `{payload['source']}`",
        "",
        "## ETH_UCY",
        "",
        "- action: confirm license/terms for local ETH-Person XML files before using them as ETH_UCY t100 source-CV repair candidates.",
        f"- local_candidates: `{summary['eth_person_xml_candidates']}`",
        "- next_step_after_confirmation: convert XML to Stage42 external source rows, run no-leakage, then train-only source-CV.",
        "",
        "## TrajNet",
        "",
        "- action: provide official longer raw TrajNet++ / original trajectory sources with tracks longer than 100 raw frames, or confirm that only 8/20-step snippets are legally available.",
        "- reason: current local TrajNet files parse as fixed short snippets and cannot repair raw-frame t100.",
        "",
        "## Non-Claims",
        "",
        "- Do not count ETH-Person XML candidates as converted/evaluated until conversion and no-leakage/source-CV run.",
        "- Do not count TrajNet snippet files as t100 support.",
        "- Do not report dataset-local/raw-frame horizons as metric or seconds-level.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bk_gate"]
    lines = [
        "# Stage42-BK Gate",
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
        "verdict": payload["stage42_bk_gate"]["verdict"],
        "gate": f"{payload['stage42_bk_gate']['passed']}/{payload['stage42_bk_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_post_bj_local_source_verification()
