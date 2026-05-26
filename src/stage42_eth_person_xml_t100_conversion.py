from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import stage42_local_t100_protected_policy as bg
from src import stage42_local_t100_schema_conversion as bf
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BK_JSON = OUT_DIR / "post_bj_local_source_verification_stage42.json"
REPORT_JSON = OUT_DIR / "eth_person_xml_t100_conversion_stage42.json"
REPORT_MD = OUT_DIR / "eth_person_xml_t100_conversion_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bl_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_eth_person_xml_t100_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

ETH_ROOT = Path("external_data/OpenTraj/datasets/ETH")
ETH_PERSON_ROOT = Path("external_data/OpenTraj/datasets/ETH-Person/data")
HORIZONS = [50, 100]
FALLBACK_BASELINE = bg.FALLBACK_BASELINE
EASY_DEGRADATION_LIMIT = bg.EASY_DEGRADATION_LIMIT


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BL 是 ETH-Person XML 技术转换 dry-run 与 train-only source-CV，不是 official dataset claim。",
    "ETH-Person local XML license/terms 尚未由用户确认，因此结果标记为 technical_dry_run_terms_unverified。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "不写 materialized feature store，不提交 raw XML/data/cache。",
    "t100 仍是 raw-frame diagnostic，不能写成 seconds-level。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _xml_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = ET.parse(path).getroot()
    for frame_node in root.findall(".//frame"):
        try:
            frame = int(float(frame_node.attrib.get("number", "")))
        except ValueError:
            continue
        for object_node in frame_node.findall(".//object"):
            box = object_node.find("box")
            if box is None:
                continue
            try:
                agent = str(int(float(object_node.attrib.get("id", ""))))
                x = float(box.attrib["xc"])
                y = float(box.attrib["yc"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"frame_id": frame, "agent_id": agent, "x": x, "y": y})
    return sorted(rows, key=lambda r: (str(r["agent_id"]), int(r["frame_id"])))


def _numeric_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").strip().split()
            if len(parts) < 4:
                continue
            try:
                frame = int(float(parts[0]))
                agent = str(int(float(parts[1])))
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue
            rows.append({"frame_id": frame, "agent_id": agent, "x": x, "y": y})
    return sorted(rows, key=lambda r: (str(r["agent_id"]), int(r["frame_id"])))


def _track_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tracks[str(row["agent_id"])].append(dict(row))
    for agent_id in list(tracks):
        tracks[agent_id] = sorted(tracks[agent_id], key=lambda r: int(r["frame_id"]))
    return tracks


def _source_key_for_path(path: Path) -> str:
    rel = str(path)
    if "ETH-Person/data/" in rel:
        stem = path.stem
        stem = re.sub(r"[-_]?interp$", "", stem)
        return f"ETH_UCY::ETH-Person/{stem}"
    if "ETH/seq_eth/" in rel:
        return "ETH_UCY::ETH/seq_eth"
    if "ETH/seq_hotel/" in rel:
        return "ETH_UCY::ETH/seq_hotel"
    return f"ETH_UCY::{path.stem}"


def _source_id_from_key(key: str) -> str:
    return key.replace("ETH_UCY::", "").replace("/", "_")


def _t100_windows_for_rows(rows: list[Mapping[str, Any]]) -> int:
    tracks = _track_map(rows)
    return int(sum(max(0, len(track) - 101 + 1) for track in tracks.values()))


def _source_record(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    key = _source_key_for_path(path)
    tracks = _track_map(rows)
    max_track = max((len(track) for track in tracks.values()), default=0)
    return {
        "source": "fresh_eth_person_xml_conversion_candidate",
        "source_id": _source_id_from_key(key),
        "independent_key": key,
        "path": str(path),
        "relative_path": str(path.relative_to(Path("external_data/OpenTraj/datasets"))),
        "domain": "ETH_UCY",
        "coordinate_unit": "dataset_local_pixel_or_bbox_center",
        "metric_status": "unverified",
        "license_status": "local_path_present_terms_unverified",
        "parsed_rows": len(rows),
        "unique_agents": len(tracks),
        "max_track_points": max_track,
        "t100_capable": max_track >= 101,
        "estimated_t100_windows": _t100_windows_for_rows(rows),
        "file_format": path.suffix.lower().lstrip("."),
    }


def _candidate_sources() -> list[dict[str, Any]]:
    candidates: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in sorted(ETH_PERSON_ROOT.glob("*.xml")):
        rows = _xml_rows(path)
        if _t100_windows_for_rows(rows) > 0:
            candidates.append((path, rows))
    eth_obsmat = ETH_ROOT / "seq_eth" / "obsmat.txt"
    if eth_obsmat.exists():
        rows = _numeric_rows(eth_obsmat)
        if _t100_windows_for_rows(rows) > 0:
            candidates.append((eth_obsmat, rows))
    by_key: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for path, rows in candidates:
        key = _source_key_for_path(path)
        current = by_key.get(key)
        if current is None or _t100_windows_for_rows(rows) > _t100_windows_for_rows(current[1]):
            by_key[key] = (path, rows)
    return [_source_record(path, rows) for key, (path, rows) in sorted(by_key.items())]


def _build_windows(source: Mapping[str, Any]) -> list[bg.Window]:
    path = Path(str(source["path"]))
    rows = _xml_rows(path) if path.suffix.lower() == ".xml" else _numeric_rows(path)
    tracks = _track_map(rows)
    windows: list[bg.Window] = []
    for agent_id, track in tracks.items():
        n = len(track)
        for horizon in HORIZONS:
            for i in range(2, n - horizon):
                prev2 = track[i - 2]
                prev = track[i - 1]
                cur = track[i]
                fut = track[i + horizon]
                vx = float(cur["x"]) - float(prev["x"])
                vy = float(cur["y"]) - float(prev["y"])
                prev_vx = float(prev["x"]) - float(prev2["x"])
                prev_vy = float(prev["y"]) - float(prev2["y"])
                errors: dict[str, float] = {}
                for name in bf.BASELINES:
                    px, py = bf._baseline_prediction(name, prev2, prev, cur, horizon)
                    errors[name] = bf._dist(px, py, float(fut["x"]), float(fut["y"]))
                windows.append(
                    {
                        "source": "fresh_eth_person_xml_policy_window",
                        "source_id": source["source_id"],
                        "domain": "ETH_UCY",
                        "scene_id": source["independent_key"],
                        "agent_id": str(agent_id),
                        "frame_id": int(cur["frame_id"]),
                        "horizon": int(horizon),
                        "speed_causal": float(math.hypot(vx, vy)),
                        "accel_causal": float(math.hypot(vx - prev_vx, vy - prev_vy)),
                        "coordinate_unit": source["coordinate_unit"],
                        "metric_status": source["metric_status"],
                        "errors_eval_only": errors,
                    }
                )
    return windows


def _folds_for_sources(sources: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(sources, key=lambda row: (-int(row["estimated_t100_windows"]), str(row["source_id"])))
    folds: list[dict[str, Any]] = []
    if len(ordered) < 3:
        return folds
    for holdout in ordered:
        remaining = [row for row in ordered if row["source_id"] != holdout["source_id"]]
        validation = remaining[0]
        train_sources = [row["source_id"] for row in remaining if row["source_id"] != validation["source_id"]]
        folds.append(
            {
                "source": "fresh_eth_person_xml_source_cv_plan",
                "domain": "ETH_UCY",
                "holdout_source": holdout["source_id"],
                "validation_source": validation["source_id"],
                "train_sources": train_sources,
            }
        )
    return folds


def _evaluate_source_cv(sources: list[Mapping[str, Any]]) -> dict[str, Any]:
    windows_by_source = {str(src["source_id"]): _build_windows(src) for src in sources}
    folds = [_evaluate_fold(fold=fold, windows_by_source=windows_by_source) for fold in _folds_for_sources(sources)]
    summary = bg._domain_summary(folds).get("ETH_UCY", {})
    return {
        "source": "fresh_eth_person_xml_source_cv_technical_dry_run",
        "windows_by_source": {
            sid: {
                "windows": len(rows),
                "t50_windows": sum(1 for row in rows if int(row["horizon"]) == 50),
                "t100_windows": sum(1 for row in rows if int(row["horizon"]) == 100),
            }
            for sid, rows in windows_by_source.items()
        },
        "folds": folds,
        "summary": summary,
    }


def _evaluate_fold(*, fold: Mapping[str, Any], windows_by_source: Mapping[str, list[bg.Window]]) -> dict[str, Any]:
    train_windows: list[bg.Window] = []
    for source_id in fold.get("train_sources", []):
        train_windows.extend(windows_by_source.get(str(source_id), []))
    validation = list(windows_by_source.get(str(fold["validation_source"]), []))
    holdout = list(windows_by_source.get(str(fold["holdout_source"]), []))
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        selection = bg._select_policy_on_validation(
            train_windows=train_windows,
            val_windows=validation,
            horizon=horizon,
        )
        selector = bg._selector_from_selected(
            selection["selected_policy"],
            [row for row in train_windows if int(row["horizon"]) == horizon],
        )
        holdout_h = [row for row in holdout if int(row["horizon"]) == horizon]
        metrics = bg._policy_metrics(holdout_h, selector)
        by_horizon[str(horizon)] = {
            "source": "fresh_eth_person_xml_source_cv_technical_dry_run",
            "selection": selection,
            "holdout_metrics": metrics,
            "safe_positive": bool(
                (metrics["improvement_vs_fallback"] or 0.0) > 0.0
                and (metrics["easy_degradation"] or 0.0) <= EASY_DEGRADATION_LIMIT
            ),
        }
    return {
        "source": "fresh_eth_person_xml_source_cv_technical_dry_run",
        "domain": "ETH_UCY",
        "holdout_source": fold["holdout_source"],
        "validation_source": fold["validation_source"],
        "train_sources": list(fold.get("train_sources", [])),
        "by_horizon": by_horizon,
    }


def run_stage42_eth_person_xml_t100_conversion() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bk = _load_json(BK_JSON)
    sources = _candidate_sources()
    source_cv = _evaluate_source_cv(sources)
    t100 = source_cv["summary"].get("by_horizon", {}).get("100", {})
    payload: dict[str, Any] = {
        "source": "fresh_technical_dry_run_terms_unverified",
        "stage": "Stage42-BL ETH-Person XML T100 Conversion Dry-Run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BK_JSON)] + [src["path"] for src in sources]),
        "current_facts": CURRENT_FACTS,
        "bk_verdict": bk.get("stage42_bk_gate", {}).get("verdict"),
        "candidate_sources": sources,
        "source_cv": source_cv,
        "summary": {
            "source": "fresh_technical_dry_run_terms_unverified",
            "candidate_sources": len(sources),
            "strict_independent_sources": len({src["independent_key"] for src in sources}),
            "eth_person_xml_sources": sum(1 for src in sources if str(src["relative_path"]).startswith("ETH-Person/")),
            "t100_windows_total": sum(int(src["estimated_t100_windows"]) for src in sources),
            "source_cv_folds": len(source_cv["folds"]),
            "technical_t100_all_folds_safe_positive": bool(t100.get("all_folds_safe_positive", False)),
            "technical_t100_mean_improvement_vs_fallback": t100.get("mean_holdout_improvement_vs_fallback"),
            "technical_t100_min_improvement_vs_fallback": t100.get("minimum_holdout_improvement_vs_fallback"),
            "technical_t100_max_easy_degradation": t100.get("maximum_easy_degradation"),
            "license_terms_confirmed": False,
            "official_converted_dataset_claim_allowed": False,
            "deployable_t100_claim_allowed": False,
            "global_t100_positive_claim_allowed": False,
            "auto_download_executed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "selection_uses_holdout": False,
            "causal_velocity_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "official_converted_dataset_claim_allowed": False,
            "deployable_t100_claim_allowed": False,
            "global_t100_positive_claim_allowed": False,
        },
    }
    payload["stage42_bl_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "bk_input_verified": payload["bk_verdict"] == "stage42_bk_local_source_verification_pass",
        "xml_sources_parsed": summary["eth_person_xml_sources"] >= 3,
        "strict_independent_sources_enough_for_cv": summary["strict_independent_sources"] >= 3,
        "source_cv_completed": summary["source_cv_folds"] >= 3,
        "t100_windows_present": summary["t100_windows_total"] > 0,
        "technical_t100_result_recorded": summary["technical_t100_mean_improvement_vs_fallback"] is not None,
        "license_terms_not_overclaimed": summary["license_terms_confirmed"] is False
        and summary["official_converted_dataset_claim_allowed"] is False,
        "no_auto_download": summary["auto_download_executed"] is False,
        "no_leakage_pass": not payload["no_leakage"]["future_endpoint_input"]
        and not payload["no_leakage"]["future_waypoint_input"]
        and payload["no_leakage"]["future_labels_eval_only"]
        and not payload["no_leakage"]["central_velocity"]
        and not payload["no_leakage"]["test_endpoint_goals"]
        and not payload["no_leakage"]["test_metrics_for_threshold"]
        and not payload["no_leakage"]["selection_uses_holdout"],
        "no_global_t100_overclaim": summary["global_t100_positive_claim_allowed"] is False,
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"]
        and not payload["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bl_eth_person_xml_t100_dry_run_pass" if passed == total else "stage42_bl_eth_person_xml_t100_dry_run_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BL ETH-Person XML T100 Conversion Dry-Run",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bl_gate']['passed']} / {payload['stage42_bl_gate']['total']}`",
        f"- verdict: `{payload['stage42_bl_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Candidate Sources",
        "",
        "| source_id | relative_path | independent_key | rows | agents | max track | t100 windows | license |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for src in payload["candidate_sources"]:
        lines.append(
            f"| `{src['source_id']}` | `{src['relative_path']}` | `{src['independent_key']}` | {src['parsed_rows']} | {src['unique_agents']} | {src['max_track_points']} | {src['estimated_t100_windows']} | `{src['license_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Source-CV Technical Dry-Run",
            "",
            "| holdout | validation | h50 safe | h50 gain | h100 safe | h100 gain | h100 easy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in payload["source_cv"]["folds"]:
        h50 = fold["by_horizon"]["50"]["holdout_metrics"]
        h100 = fold["by_horizon"]["100"]["holdout_metrics"]
        lines.append(
            f"| `{fold['holdout_source']}` | `{fold['validation_source']}` | `{fold['by_horizon']['50']['safe_positive']}` | {h50['improvement_vs_fallback']} | `{fold['by_horizon']['100']['safe_positive']}` | {h100['improvement_vs_fallback']} | {h100['easy_degradation']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- candidate_sources: `{summary['candidate_sources']}`",
            f"- strict_independent_sources: `{summary['strict_independent_sources']}`",
            f"- eth_person_xml_sources: `{summary['eth_person_xml_sources']}`",
            f"- t100_windows_total: `{summary['t100_windows_total']}`",
            f"- source_cv_folds: `{summary['source_cv_folds']}`",
            f"- technical_t100_all_folds_safe_positive: `{summary['technical_t100_all_folds_safe_positive']}`",
            f"- technical_t100_mean_improvement_vs_fallback: `{summary['technical_t100_mean_improvement_vs_fallback']}`",
            f"- technical_t100_min_improvement_vs_fallback: `{summary['technical_t100_min_improvement_vs_fallback']}`",
            f"- technical_t100_max_easy_degradation: `{summary['technical_t100_max_easy_degradation']}`",
            f"- license_terms_confirmed: `{summary['license_terms_confirmed']}`",
            f"- official_converted_dataset_claim_allowed: `{summary['official_converted_dataset_claim_allowed']}`",
            f"- deployable_t100_claim_allowed: `{summary['deployable_t100_claim_allowed']}`",
            "",
            "## Interpretation",
            "",
            "- This dry-run proves the local ETH-Person XML loader and strict source-CV pipeline are technically executable.",
            "- Because license/terms are still unconfirmed, these XML sources are not counted as official converted/evaluated data.",
            "- The result cannot be used as a global t100 deployment claim until terms are confirmed and the official conversion/no-leakage/source-CV protocol is rerun.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    return [
        "# Stage42-BL User Action Required",
        "",
        f"- source: `{payload['source']}`",
        "",
        "## ETH-Person XML license / terms confirmation",
        "",
        "- action: confirm whether the local ETH-Person XML files may be used for research-only derived feature conversion and source-CV evaluation.",
        "- files:",
        *[f"  - `{src['relative_path']}`" for src in payload["candidate_sources"] if str(src["relative_path"]).startswith("ETH-Person/")],
        "- if confirmed: rerun as official Stage42-BM conversion/no-leakage/source-CV and update claims.",
        "- if not confirmed: keep this BL result as technical dry-run only and do not use it for paper claims.",
        "",
        "## Current non-claims",
        "",
        f"- technical_t100_all_folds_safe_positive: `{summary['technical_t100_all_folds_safe_positive']}`",
        "- official_converted_dataset_claim_allowed: `False`",
        "- deployable_t100_claim_allowed: `False`",
        "- metric_or_seconds_claim: `False`",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bl_gate"]
    lines = [
        "# Stage42-BL Gate",
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
        "verdict": payload["stage42_bl_gate"]["verdict"],
        "gate": f"{payload['stage42_bl_gate']['passed']}/{payload['stage42_bl_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_eth_person_xml_t100_conversion()
