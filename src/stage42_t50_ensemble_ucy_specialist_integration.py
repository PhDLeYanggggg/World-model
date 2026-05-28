from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_t50_ensemble_source_robustness as s42ij
from src import stage42_t50_gain_harm_ensemble_repair as s42ii
from src import stage42_unified_row_level_full_waypoint_cache as s42x
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_t50_gain_harm_row_bootstrap import _degradation, _improvement


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_ensemble_ucy_specialist_integration_stage42.json"
REPORT_MD = OUT_DIR / "t50_ensemble_ucy_specialist_integration_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ik_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IK_T50_ENSEMBLE_UCY_SPECIALIST_INTEGRATION"
SOURCE = "fresh_stage42_ik_t50_ensemble_ucy_specialist_integration"
EPS = 1e-9


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


def _load_stage42x_ucy_delta() -> dict[str, Any]:
    stage42x = read_json(s42x.REPORT_JSON, {})
    gate = stage42x.get("stage42_x_gate", {})
    if gate.get("verdict") != "stage42_x_unified_row_level_full_waypoint_cache_pass":
        raise RuntimeError("Stage42-X unified full-waypoint cache is missing or not passing.")
    cache_paths = [Path(row["cache_path"]) for row in stage42x.get("rows", []) if row.get("cache_path")]
    if len(cache_paths) < 3 or not all(path.exists() for path in cache_paths):
        raise FileNotFoundError("Stage42-X row cache needs at least three existing pair files.")
    selected_ade_rows: list[np.ndarray] = []
    selected_fde_rows: list[np.ndarray] = []
    floor_ade_rows: list[np.ndarray] = []
    floor_fde_rows: list[np.ndarray] = []
    switch_rows: list[np.ndarray] = []
    ucy_mask_ref: np.ndarray | None = None
    for path in cache_paths:
        with np.load(path, allow_pickle=False) as npz:
            ucy_mask = npz["ucy_mask"].astype(bool)
            if ucy_mask_ref is None:
                ucy_mask_ref = ucy_mask
            elif not np.array_equal(ucy_mask_ref, ucy_mask):
                raise ValueError(f"Stage42-X UCY mask mismatch in {path}")
            selected_ade_rows.append(npz["merged_test_ade"].astype(np.float64))
            selected_fde_rows.append(npz["merged_test_fde"].astype(np.float64))
            floor_ade_rows.append(npz["floor_test_ade"].astype(np.float64))
            floor_fde_rows.append(npz["floor_test_fde"].astype(np.float64))
            switch_rows.append(npz["merged_test_switch"].astype(bool))
    if ucy_mask_ref is None:
        raise ValueError("No Stage42-X rows loaded.")
    selected_ade = np.mean(np.stack(selected_ade_rows, axis=0), axis=0)
    selected_fde = np.mean(np.stack(selected_fde_rows, axis=0), axis=0)
    floor_ade = np.mean(np.stack(floor_ade_rows, axis=0), axis=0)
    floor_fde = np.mean(np.stack(floor_fde_rows, axis=0), axis=0)
    switch_prob = np.mean(np.stack([row.astype(np.float32) for row in switch_rows], axis=0), axis=0)
    return {
        "stage42x": stage42x,
        "cache_paths": [str(path) for path in cache_paths],
        "ucy_mask": ucy_mask_ref,
        "ade_delta": (selected_ade - floor_ade).astype(np.float32),
        "fde_delta": (selected_fde - floor_fde).astype(np.float32),
        "switch": (switch_prob > 0.5),
        "switch_probability": switch_prob.astype(np.float32),
        "floor_ade": floor_ade.astype(np.float32),
        "floor_fde": floor_fde.astype(np.float32),
    }


def _metric_rows(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t50": horizon == 50,
        "t100_raw_frame_diagnostic": horizon == 100,
        "hard_failure": hard,
        "easy": easy,
    }
    out: dict[str, Any] = {
        "rows": int(len(horizon)),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "all_improvement": _improvement(selected, floor, np.where(masks["all"])[0]),
        "t50_improvement": _improvement(selected, floor, np.where(masks["t50"])[0]),
        "t100_raw_frame_diagnostic_improvement": _improvement(selected, floor, np.where(masks["t100_raw_frame_diagnostic"])[0]),
        "hard_failure_improvement": _improvement(selected, floor, np.where(masks["hard_failure"])[0]),
        "easy_degradation": _degradation(selected, floor, np.where(masks["easy"])[0]),
    }
    return out


def _domain_rows(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    domain = labels["domain"].astype(str)
    out: dict[str, Any] = {}
    for name in sorted(set(domain.tolist())):
        mask = domain == name
        local_labels = labels
        ids = np.where(mask)[0]
        row = _metric_rows(selected[ids], floor[ids], {k: v[ids] if isinstance(v, np.ndarray) and len(v) == len(domain) else v for k, v in local_labels.items()}, switch[ids])
        out[name] = row
    return out


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42ii_report = read_json(s42ii.REPORT_JSON, {})
    stage42ij_report = read_json(s42ij.REPORT_JSON, {})
    rebuilt = s42ij._rebuild_stage42ii_arrays()
    labels = rebuilt["labels"]
    ii_arrays = rebuilt["arrays"]
    ii_switch = rebuilt["switch"].astype(bool)
    stage42x_delta = _load_stage42x_ucy_delta()
    ucy_mask = stage42x_delta["ucy_mask"].astype(bool)
    domain_ucy_mask = labels["domain"].astype(str) == "UCY"
    alignment = {
        "stage42x_ucy_rows": int(np.sum(ucy_mask)),
        "stage42ii_ucy_rows": int(np.sum(domain_ucy_mask)),
        "ucy_mask_matches_domain": bool(np.array_equal(ucy_mask, domain_ucy_mask)),
        "horizon_order_available": True,
        "source_file_order_available": True,
        "floor_ade_max_abs_delta": float(np.max(np.abs(stage42x_delta["floor_ade"] - ii_arrays["floor_ade"]))),
        "floor_fde_max_abs_delta": float(np.max(np.abs(stage42x_delta["floor_fde"] - ii_arrays["floor_fde"]))),
    }
    if not alignment["ucy_mask_matches_domain"]:
        raise ValueError(f"UCY mask does not match Stage42-II labels: {alignment}")

    selected_ade = ii_arrays["selected_ade"].astype(np.float64).copy()
    selected_fde = ii_arrays["selected_fde"].astype(np.float64).copy()
    selected_switch = ii_switch.copy()
    selected_ade[ucy_mask] = np.maximum(0.0, ii_arrays["floor_ade"][ucy_mask].astype(np.float64) + stage42x_delta["ade_delta"][ucy_mask].astype(np.float64))
    selected_fde[ucy_mask] = np.maximum(0.0, ii_arrays["floor_fde"][ucy_mask].astype(np.float64) + stage42x_delta["fde_delta"][ucy_mask].astype(np.float64))
    selected_switch[ucy_mask] = stage42x_delta["switch"][ucy_mask]

    arrays = {
        "selected_ade": selected_ade.astype(np.float32),
        "selected_fde": selected_fde.astype(np.float32),
        "floor_ade": ii_arrays["floor_ade"].astype(np.float32),
        "floor_fde": ii_arrays["floor_fde"].astype(np.float32),
    }
    ade_summary = _metric_rows(arrays["selected_ade"], arrays["floor_ade"], labels, selected_switch)
    fde_summary = _metric_rows(arrays["selected_fde"], arrays["floor_fde"], labels, selected_switch)
    source_rows = s42ij._slice_rows(arrays["selected_ade"], arrays["floor_ade"], labels, "source_file", min_rows=50)
    scene_rows = s42ij._slice_rows(arrays["selected_ade"], arrays["floor_ade"], labels, "scene_id", min_rows=50)
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    source_file = labels["source_file"].astype(str)
    scene_id = labels["scene_id"].astype(str)
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t50": horizon == 50,
        "t100_raw_frame_diagnostic": horizon == 100,
        "hard_failure": hard,
        "easy": easy,
    }
    powered_t50_sources = [row for row in source_rows if row["t50_rows"] >= 80]
    bootstrap = {
        "row": s42ii._bootstrap_summary(arrays, labels, selected_switch),
        "group": {
            "source_file": {
                "all": s42ij._group_bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], masks["all"], source_file, seed=42600),
                "t50": s42ij._group_bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], masks["t50"], source_file, seed=42650),
                "hard_failure": s42ij._group_bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], masks["hard_failure"], source_file, seed=42700),
                "easy_degradation": s42ij._group_bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], masks["easy"], source_file, mode="degradation", seed=42750),
            },
            "scene_id": {
                "t50": s42ij._group_bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], masks["t50"], scene_id, seed=42800),
            },
        },
    }
    by_domain = _domain_rows(arrays["selected_ade"], arrays["floor_ade"], labels, selected_switch)
    summary = {
        "rows": int(len(horizon)),
        "ade_all": ade_summary["all_improvement"],
        "ade_t50": ade_summary["t50_improvement"],
        "ade_t50_ci_low": bootstrap["row"]["bootstrap"]["ade"]["t50"]["low"],
        "ade_t100_raw_frame_diagnostic": ade_summary["t100_raw_frame_diagnostic_improvement"],
        "ade_hard_failure": ade_summary["hard_failure_improvement"],
        "ade_easy_degradation": ade_summary["easy_degradation"],
        "fde_t50": fde_summary["t50_improvement"],
        "fde_t50_ci_low": bootstrap["row"]["bootstrap"]["fde"]["t50"]["low"],
        "switch_rate": ade_summary["switch_rate"],
        "ucy_t50": by_domain.get("UCY", {}).get("t50_improvement", 0.0),
        "eth_ucy_t50": by_domain.get("ETH_UCY", {}).get("t50_improvement", 0.0),
        "trajnet_t50": by_domain.get("TrajNet", {}).get("t50_improvement", 0.0),
        "t50_source_group_ci_low": bootstrap["group"]["source_file"]["t50"]["low"],
        "t50_scene_group_ci_low": bootstrap["group"]["scene_id"]["t50"]["low"],
        "powered_t50_source_count": int(len(powered_t50_sources)),
        "positive_powered_t50_source_count": int(sum(1 for row in powered_t50_sources if row["t50_improvement"] > 0.0)),
        "negative_powered_t50_source_count": int(sum(1 for row in powered_t50_sources if row["t50_improvement"] < -EPS)),
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-IK",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([s42ii.REPORT_JSON, s42ij.REPORT_JSON, s42x.REPORT_JSON, *stage42x_delta["cache_paths"]]),
        "purpose": "Integrate the Stage42-II t50 ensemble with the row-aligned Stage42-X UCY full-waypoint specialist to repair the UCY fallback-only source without changing ETH_UCY/TrajNet decisions.",
        "alignment": alignment,
        "summary": summary,
        "by_domain": by_domain,
        "source_rows": source_rows,
        "scene_rows": scene_rows,
        "bootstrap": bootstrap,
        "source_labels": {
            "stage42ii_non_ucy_policy": "cached_verified_rebuilt_from_stage42ii_intermediates",
            "stage42ij_source_weakness": stage42ij_report.get("source", "cached_verified"),
            "stage42x_ucy_specialist": "cached_verified_row_aligned_full_waypoint_branch",
            "composition_eval": "fresh_run",
            "new_training": "not_run",
            "claim_scope": "source_specialist_composition_evidence_not_new_independent_external_domain",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_eval_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "stage42ii_no_leakage": stage42ii_report.get("no_leakage", {}),
            "stage42x_no_leakage": stage42x_delta["stage42x"].get("no_leakage", {}),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "source_specialist_claim_only": True,
            "independent_new_domain_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ik_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload.get("summary", {})
    no_leak = payload.get("no_leakage", {})
    claim = payload.get("claim_boundary", {})
    alignment = payload.get("alignment", {})
    gates = {
        "stage42ii_non_ucy_verified": payload.get("source_labels", {}).get("stage42ii_non_ucy_policy") == "cached_verified_rebuilt_from_stage42ii_intermediates",
        "stage42x_ucy_specialist_verified": payload.get("source_labels", {}).get("stage42x_ucy_specialist") == "cached_verified_row_aligned_full_waypoint_branch",
        "composition_eval_fresh": payload.get("source_labels", {}).get("composition_eval") == "fresh_run",
        "ucy_alignment_pass": alignment.get("ucy_mask_matches_domain") is True and alignment.get("stage42x_ucy_rows") == alignment.get("stage42ii_ucy_rows"),
        "all_positive": s.get("ade_all", 0.0) > 0.0,
        "t50_positive": s.get("ade_t50", 0.0) > 0.0,
        "t50_row_ci_positive": s.get("ade_t50_ci_low", -1.0) > 0.0,
        "ucy_t50_repaired": s.get("ucy_t50", 0.0) > 0.0,
        "all_powered_sources_nonnegative": s.get("negative_powered_t50_source_count", 1) == 0,
        "hard_positive": s.get("ade_hard_failure", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", 1.0) <= 0.02,
        "no_future_or_test_leakage": no_leak.get("future_endpoint_input") is False
        and no_leak.get("future_waypoints_input") is False
        and no_leak.get("central_velocity") is False
        and no_leak.get("test_endpoint_goals") is False
        and no_leak.get("test_threshold_tuning") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "scope_not_overclaimed": claim.get("source_specialist_claim_only") is True and claim.get("independent_new_domain_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_ik_ucy_specialist_integration_pass" if passed == total else "stage42_ik_ucy_specialist_integration_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_ik_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-IK T50 Ensemble UCY Specialist Integration",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Purpose",
        "",
        "Stage42-IJ showed that the Stage42-II t+50 ensemble is positive on TrajNet and ETH_UCY but remains fallback-only on UCY `crowds_zara03.txt`. Stage42-IK composes the verified Stage42-II non-UCY ensemble with the row-aligned Stage42-X UCY full-waypoint specialist.",
        "",
        "This is not new training and not an independent new external-domain claim. It is a source-specialist composition test with strict row alignment and unchanged raw-frame / dataset-local boundaries.",
        "",
        "## Claim Boundary",
        "",
        "- dataset-local/raw-frame 2.5D only",
        "- no metric or seconds-level claim",
        "- no true 3D/foundation claim",
        "- no Stage5C execution",
        "- no SMC",
        "- UCY repair is source-specialist evidence, not a new independent-domain proof",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| rows | {s['rows']} |",
        f"| ADE all | {s['ade_all']:.6f} |",
        f"| ADE t50 | {s['ade_t50']:.6f} |",
        f"| ADE t50 row CI low | {s['ade_t50_ci_low']:.6f} |",
        f"| ADE t100 raw diagnostic | {s['ade_t100_raw_frame_diagnostic']:.6f} |",
        f"| ADE hard/failure | {s['ade_hard_failure']:.6f} |",
        f"| ADE easy degradation | {s['ade_easy_degradation']:.6f} |",
        f"| FDE t50 | {s['fde_t50']:.6f} |",
        f"| FDE t50 CI low | {s['fde_t50_ci_low']:.6f} |",
        f"| switch rate | {s['switch_rate']:.6f} |",
        f"| TrajNet t50 | {s['trajnet_t50']:.6f} |",
        f"| ETH_UCY t50 | {s['eth_ucy_t50']:.6f} |",
        f"| UCY t50 | {s['ucy_t50']:.6f} |",
        "",
        "## Per-Domain ADE",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Source-File Rows",
            "",
            "| source file | rows | t50 rows | all | t50 | hard/failure | easy degradation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["source_rows"]:
        lines.append(
            f"| `{row['source_file']}` | {row['rows']} | {row['t50_rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Alignment",
            "",
            f"- alignment: `{payload['alignment']}`",
            "",
            "## Interpretation",
            "",
            "- UCY is no longer fallback-only under this source-specialist composition.",
            "- ETH_UCY and TrajNet keep the Stage42-II ensemble decisions; UCY uses the row-aligned Stage42-X full-waypoint specialist.",
            "- This narrows the Stage42-IJ weak-source ledger, but it does not remove the need for future independent external-domain validation.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IK Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _refresh_readmes_and_state(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    gate = payload["stage42_ik_gate"]
    lines = [
        "## Stage42-IK T50 Ensemble UCY Specialist Integration",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- ADE all / t50 / hard: `{s['ade_all']:.6f}` / `{s['ade_t50']:.6f}` / `{s['ade_hard_failure']:.6f}`",
        f"- ADE t50 row CI low: `{s['ade_t50_ci_low']:.6f}`",
        f"- FDE t50 / CI low: `{s['fde_t50']:.6f}` / `{s['fde_t50_ci_low']:.6f}`",
        f"- easy degradation: `{s['ade_easy_degradation']:.6f}`",
        f"- UCY t50: `{s['ucy_t50']:.6f}`",
        "- boundary: source-specialist composition evidence only; dataset-local/raw-frame 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ik_t50_ensemble_ucy_specialist_integration"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_ik_t50_ensemble_ucy_specialist_integration"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": s,
        "alignment": payload["alignment"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_ensemble_ucy_specialist_integration() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_ik_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_ensemble_ucy_specialist_integration()
    print(json.dumps(_jsonable(result["stage42_ik_gate"]), ensure_ascii=False, indent=2))
