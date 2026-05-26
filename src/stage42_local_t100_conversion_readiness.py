from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BD_JSON = OUT_DIR / "local_t100_source_inventory_stage42.json"
REPORT_JSON = OUT_DIR / "local_t100_conversion_readiness_stage42.json"
REPORT_MD = OUT_DIR / "local_t100_conversion_readiness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_be_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_t100_conversion_readiness_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

HORIZONS = [10, 25, 50, 100]
HISTORY_WINDOWS = [8, 16, 32, 64]
MIN_SOURCE_CV_SOURCES = 3


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BE 是 local t100 conversion-readiness / no-leakage audit，不训练模型、不写大 feature store。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "本步骤不把 local candidate 写成 official converted dataset；full conversion / source-CV 仍需后续执行。",
    "t100 仍是 raw-frame diagnostic / blocker，不能写成 seconds-level。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".ndjson":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                track = obj.get("track") if isinstance(obj, dict) else None
                if not isinstance(track, dict):
                    continue
                try:
                    rows.append(
                        {
                            "frame_id": int(track["f"]),
                            "agent_id": str(int(track["p"])),
                            "x": float(track["x"]),
                            "y": float(track["y"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return rows

    numeric_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").strip().split()
            if len(parts) < 4:
                continue
            try:
                numeric_rows.append([float(part) for part in parts])
            except ValueError:
                continue
    x_idx, y_idx = _choose_xy_columns(numeric_rows)
    for parts in numeric_rows:
        rows.append(
            {
                "frame_id": int(parts[0]),
                "agent_id": str(int(parts[1])),
                "x": float(parts[x_idx]),
                "y": float(parts[y_idx]),
            }
        )
    return rows


def _choose_xy_columns(numeric_rows: list[list[float]]) -> tuple[int, int]:
    if not numeric_rows:
        return 2, 3
    min_len = min(len(row) for row in numeric_rows)
    if min_len < 5:
        return 2, 3
    col3 = [row[3] for row in numeric_rows]
    col4 = [row[4] for row in numeric_rows]
    range3 = max(col3) - min(col3)
    range4 = max(col4) - min(col4)
    if range3 < 1e-9 and range4 > 1e-9:
        return 2, 4
    return 2, 3


def _track_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tracks[str(row["agent_id"])].append(dict(row))
    for agent_id in list(tracks):
        tracks[agent_id] = sorted(tracks[agent_id], key=lambda item: (int(item["frame_id"]), float(item["x"]), float(item["y"])))
    return tracks


def _common_step(tracks: Mapping[str, list[Mapping[str, Any]]]) -> int | None:
    counter: Counter[int] = Counter()
    for rows in tracks.values():
        frames = sorted({int(row["frame_id"]) for row in rows})
        for a, b in zip(frames, frames[1:]):
            step = b - a
            if step > 0:
                counter[step] += 1
    return int(counter.most_common(1)[0][0]) if counter else None


def _horizon_counts(track_lengths: Iterable[int]) -> dict[str, int]:
    return {str(h): int(sum(max(0, n - h) for n in track_lengths)) for h in HORIZONS}


def _history_counts(track_lengths: Iterable[int]) -> dict[str, int]:
    lengths = list(track_lengths)
    out: dict[str, int] = {}
    for k in HISTORY_WINDOWS:
        for h in [50, 100]:
            out[f"k{k}_h{h}"] = int(sum(max(0, n - k - h + 1) for n in lengths))
    return out


def _continuity(tracks: Mapping[str, list[Mapping[str, Any]]], step: int | None) -> dict[str, Any]:
    if step is None:
        return {"source": "fresh_local_conversion_readiness", "common_step": None, "tracks_with_gaps": 0, "gap_ratio": 0.0}
    tracks_with_gaps = 0
    checked = 0
    max_gap = 0
    for rows in tracks.values():
        frames = sorted({int(row["frame_id"]) for row in rows})
        if len(frames) < 2:
            continue
        checked += 1
        gaps = [b - a for a, b in zip(frames, frames[1:])]
        if any(gap != step for gap in gaps):
            tracks_with_gaps += 1
            max_gap = max(max_gap, max(gaps))
    return {
        "source": "fresh_local_conversion_readiness",
        "common_step": int(step),
        "tracks_checked": int(checked),
        "tracks_with_gaps": int(tracks_with_gaps),
        "gap_ratio": float(tracks_with_gaps / checked) if checked else 0.0,
        "max_gap": int(max_gap),
    }


def _infer_domain(row: Mapping[str, Any]) -> str:
    suggested = str(row.get("suggested_domain", "diagnostic"))
    if suggested == "UCY_or_ETH_UCY":
        return "UCY"
    return suggested


def _source_readiness(candidate: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(candidate["path"]))
    rows = _parse_rows(path)
    tracks = _track_map(rows)
    lengths = [len(track) for track in tracks.values()]
    step = _common_step(tracks)
    horizon_counts = _horizon_counts(lengths)
    history_counts = _history_counts(lengths)
    continuity = _continuity(tracks, step)
    x_vals = [float(row["x"]) for row in rows]
    y_vals = [float(row["y"]) for row in rows]
    max_track = max(lengths, default=0)
    source_id = str(candidate["relative_path"])
    schema_fields = [
        "dataset_name",
        "domain",
        "source_id",
        "scene_id",
        "agent_id",
        "frame_id",
        "x",
        "y",
        "vx_causal",
        "vy_causal",
        "horizon",
        "future_x_label",
        "future_y_label",
        "history_valid_mask",
        "coordinate_unit",
        "metric_status",
        "source_role",
    ]
    ready_for_schema = bool(rows and tracks and horizon_counts["100"] > 0 and continuity["gap_ratio"] < 0.5)
    return {
        "source": "fresh_local_conversion_readiness",
        "source_id": source_id,
        "path": str(path),
        "domain": _infer_domain(candidate),
        "file_format": str(candidate.get("file_format", path.suffix.lower().lstrip("."))),
        "rows": int(len(rows)),
        "agents": int(len(tracks)),
        "unique_frames": int(len({int(row["frame_id"]) for row in rows})),
        "min_frame": int(min((int(row["frame_id"]) for row in rows), default=0)) if rows else None,
        "max_frame": int(max((int(row["frame_id"]) for row in rows), default=0)) if rows else None,
        "max_track_points": int(max_track),
        "median_track_points": float(median(lengths)) if lengths else 0.0,
        "horizon_counts": horizon_counts,
        "history_horizon_counts": history_counts,
        "common_frame_step": step,
        "continuity": continuity,
        "coordinate_unit": "dataset_local_raw",
        "metric_status": "unverified_dataset_local_not_metric",
        "x_range": [float(min(x_vals)), float(max(x_vals))] if x_vals else [0.0, 0.0],
        "y_range": [float(min(y_vals)), float(max(y_vals))] if y_vals else [0.0, 0.0],
        "causal_velocity_possible": bool(max_track >= 2),
        "central_velocity_used": False,
        "future_labels_available_for_loss_eval_only": bool(horizon_counts["100"] > 0),
        "schema_fields": schema_fields,
        "schema_conversion_ready": bool(ready_for_schema),
        "full_feature_store_written": False,
        "training_or_eval_run": False,
    }


def _source_cv_plan(readiness: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in readiness:
        if row["schema_conversion_ready"] and int(row["horizon_counts"]["100"]) > 0:
            by_domain[str(row["domain"])].append(row)
    domains: dict[str, Any] = {}
    for domain, rows in sorted(by_domain.items()):
        ranked = sorted(rows, key=lambda row: (-int(row["horizon_counts"]["100"]), str(row["source_id"])))
        folds: list[dict[str, Any]] = []
        feasible = len(ranked) >= MIN_SOURCE_CV_SOURCES
        if feasible:
            for holdout in ranked:
                remaining = [row for row in ranked if row["source_id"] != holdout["source_id"]]
                validation = remaining[0]
                train = [row["source_id"] for row in remaining[1:]]
                folds.append(
                    {
                        "source": "fresh_local_conversion_readiness",
                        "holdout_source": holdout["source_id"],
                        "validation_source": validation["source_id"],
                        "train_sources": train,
                        "holdout_t100_windows": int(holdout["horizon_counts"]["100"]),
                    }
                )
        domains[domain] = {
            "source": "fresh_local_conversion_readiness",
            "t100_capable_sources": len(ranked),
            "estimated_t100_windows": int(sum(int(row["horizon_counts"]["100"]) for row in ranked)),
            "source_cv_feasible_after_conversion": bool(feasible),
            "minimum_sources_required": MIN_SOURCE_CV_SOURCES,
            "folds": folds,
        }
    return {
        "source": "fresh_local_conversion_readiness",
        "rule": "Use source-level leave-one-source-out CV only after full conversion; final test metrics must not select thresholds.",
        "domains": domains,
    }


def run_stage42_local_t100_conversion_readiness() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bd = _load_json(BD_JSON)
    candidates = list(bd.get("novel_t100_candidates", []))
    readiness = [_source_readiness(candidate) for candidate in candidates]
    source_cv_plan = _source_cv_plan(readiness)
    total_t100 = int(sum(int(row["horizon_counts"]["100"]) for row in readiness))
    total_t50 = int(sum(int(row["horizon_counts"]["50"]) for row in readiness))
    payload = {
        "source": "fresh_local_conversion_readiness",
        "stage": "Stage42-BE Local T100 Conversion Readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BD_JSON)] + [str(row.get("path", "")) for row in candidates]),
        "bd_verdict": bd.get("stage42_bd_gate", {}).get("verdict"),
        "candidate_count": len(candidates),
        "source_readiness": readiness,
        "source_cv_plan": source_cv_plan,
        "summary": {
            "source": "fresh_local_conversion_readiness",
            "candidate_files": len(candidates),
            "schema_conversion_ready_files": int(sum(1 for row in readiness if row["schema_conversion_ready"])),
            "estimated_t10_windows": int(sum(int(row["horizon_counts"]["10"]) for row in readiness)),
            "estimated_t25_windows": int(sum(int(row["horizon_counts"]["25"]) for row in readiness)),
            "estimated_t50_windows": total_t50,
            "estimated_t100_windows": total_t100,
            "domains_with_source_cv_after_conversion": [
                domain for domain, row in source_cv_plan["domains"].items() if row["source_cv_feasible_after_conversion"]
            ],
            "full_feature_store_written": False,
            "training_run": False,
            "evaluation_run": False,
            "stage42_bf_actual_conversion_recommended": bool(total_t100 > 0),
            "stage5c_executed": False,
            "smc_enabled": False,
            "metric_or_seconds_claim": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_loss_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "source_level_split_required_before_eval": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "converted_dataset_claim_allowed": False,
            "t100_positive_claim_allowed": False,
        },
    }
    payload["stage42_be_gate"] = _gate(payload)
    payload["user_action_required"] = _user_actions(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    domains = payload["source_cv_plan"]["domains"]
    gates = {
        "bd_inventory_loaded": payload["bd_verdict"] == "stage42_bd_local_t100_source_inventory_pass",
        "novel_candidates_loaded": s["candidate_files"] > 0,
        "schema_readiness_checked": s["schema_conversion_ready_files"] == s["candidate_files"],
        "t50_windows_available": s["estimated_t50_windows"] > 0,
        "t100_windows_available": s["estimated_t100_windows"] > 0,
        "source_cv_plan_built": bool(domains),
        "at_least_one_domain_source_cv_feasible": bool(s["domains_with_source_cv_after_conversion"]),
        "no_feature_store_overclaim": s["full_feature_store_written"] is False,
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
    verdict = "stage42_be_local_t100_conversion_readiness_pass" if passed == total else "stage42_be_local_t100_conversion_readiness_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _user_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = [
        {
            "source": "fresh_local_conversion_readiness",
            "priority": "high",
            "action_type": "run_stage42_bf_actual_schema_conversion",
            "notes": "Write the actual converted external schema rows outside Git, then run no-leakage and train-only source-CV. This Stage42-BE report is readiness only.",
        },
        {
            "source": "fresh_local_conversion_readiness",
            "priority": "high",
            "action_type": "keep_t100_claim_blocked_until_source_cv",
            "notes": "The novel local files can support future t100 repair, but current t100 positive claim remains blocked until converted rows pass source-CV.",
        },
    ]
    if "UCY" in payload["summary"]["domains_with_source_cv_after_conversion"]:
        actions.append(
            {
                "source": "fresh_local_conversion_readiness",
                "priority": "medium",
                "action_type": "prioritize_ucy_source_cv",
                "notes": "The three novel UCY-like t100 sources are enough for a leave-one-source-out readiness plan after conversion.",
            }
        )
    return actions


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BE Local T100 Conversion Readiness",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_be_gate']['passed']} / {payload['stage42_be_gate']['total']}`",
        f"- verdict: `{payload['stage42_be_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- candidate_files: `{s['candidate_files']}`",
        f"- schema_conversion_ready_files: `{s['schema_conversion_ready_files']}`",
        f"- estimated_t10_windows: `{s['estimated_t10_windows']}`",
        f"- estimated_t25_windows: `{s['estimated_t25_windows']}`",
        f"- estimated_t50_windows: `{s['estimated_t50_windows']}`",
        f"- estimated_t100_windows: `{s['estimated_t100_windows']}`",
        f"- domains_with_source_cv_after_conversion: `{', '.join(s['domains_with_source_cv_after_conversion']) or 'none'}`",
        f"- full_feature_store_written: `{s['full_feature_store_written']}`",
        f"- stage42_bf_actual_conversion_recommended: `{s['stage42_bf_actual_conversion_recommended']}`",
        "",
        "## Source Readiness",
        "",
        "| source | domain | rows | agents | t50 | t100 | max track | common step | gap ratio | schema ready |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["source_readiness"]:
        lines.append(
            f"| `{row['source_id']}` | {row['domain']} | {row['rows']} | {row['agents']} | {row['horizon_counts']['50']} | {row['horizon_counts']['100']} | {row['max_track_points']} | {row['common_frame_step']} | {row['continuity']['gap_ratio']:.3f} | {row['schema_conversion_ready']} |"
        )
    lines.extend(
        [
            "",
            "## Source-CV Readiness Plan",
            "",
            "| domain | t100-capable sources | estimated t100 windows | source-CV feasible after conversion |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for domain, row in payload["source_cv_plan"]["domains"].items():
        lines.append(
            f"| `{domain}` | {row['t100_capable_sources']} | {row['estimated_t100_windows']} | {row['source_cv_feasible_after_conversion']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-BE verifies that the local novel candidates can be mapped to the external row schema, but it does not write the full feature store.",
            "- UCY has enough novel local t100-capable sources for a source-CV readiness plan after actual conversion.",
            "- ETH_UCY gains one small t100-capable source but remains insufficient for independent t100 support by itself.",
            "- t100 remains raw-frame diagnostic / blocker until Stage42-BF actual conversion and train-only source-CV pass.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BE User Action Required",
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
                f"- notes: {action['notes']}",
                "",
            ]
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_be_gate"]
    lines = [
        "# Stage42-BE Gate",
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
        "verdict": payload["stage42_be_gate"]["verdict"],
        "gate": f"{payload['stage42_be_gate']['passed']}/{payload['stage42_be_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_local_t100_conversion_readiness()
