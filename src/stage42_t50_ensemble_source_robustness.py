from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_explicit_gain_harm_selector as s42o
from src import stage42_t50_gain_harm_ensemble_repair as s42ii
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_t50_gain_harm_row_bootstrap import _degradation, _improvement


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_ensemble_source_robustness_stage42.json"
REPORT_MD = OUT_DIR / "t50_ensemble_source_robustness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ij_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IJ_T50_ENSEMBLE_SOURCE_ROBUSTNESS"
SOURCE = "fresh_stage42_ij_t50_ensemble_source_robustness"
BOOTSTRAP_N = 2000
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


def _group_bootstrap_ci(
    selected: np.ndarray,
    floor: np.ndarray,
    row_mask: np.ndarray,
    groups: np.ndarray,
    *,
    mode: str = "improvement",
    seed: int = 42000,
    n: int = BOOTSTRAP_N,
) -> dict[str, Any]:
    ids = np.where(row_mask.astype(bool))[0]
    if len(ids) == 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n_rows": 0, "n_groups": 0, "bootstrap_n": 0}
    local_groups = np.asarray(groups[ids]).astype(str)
    unique = np.asarray(sorted(set(local_groups.tolist())), dtype=object)
    if len(unique) < 2:
        value = _degradation(selected, floor, ids) if mode == "degradation" else _improvement(selected, floor, ids)
        return {"low": value, "mid": value, "high": value, "n_rows": int(len(ids)), "n_groups": int(len(unique)), "bootstrap_n": 0}
    by_group: dict[str, np.ndarray] = {g: ids[local_groups == g] for g in unique}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boot = np.concatenate([by_group[str(g)] for g in sampled])
        vals.append(_degradation(selected, floor, boot) if mode == "degradation" else _improvement(selected, floor, boot))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "low": float(np.percentile(arr, 2.5)),
        "mid": float(np.percentile(arr, 50.0)),
        "high": float(np.percentile(arr, 97.5)),
        "n_rows": int(len(ids)),
        "n_groups": int(len(unique)),
        "bootstrap_n": int(n),
    }


def _slice_rows(
    selected: np.ndarray,
    floor: np.ndarray,
    labels: Mapping[str, np.ndarray],
    group_key: str,
    *,
    min_rows: int = 20,
) -> list[dict[str, Any]]:
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    group = labels[group_key].astype(str)
    rows: list[dict[str, Any]] = []
    for name in sorted(set(group.tolist())):
        mask = group == name
        if int(np.sum(mask)) < min_rows:
            continue
        row = {
            group_key: name,
            "rows": int(np.sum(mask)),
            "t50_rows": int(np.sum(mask & (horizon == 50))),
            "t100_rows": int(np.sum(mask & (horizon == 100))),
            "all_improvement": _improvement(selected, floor, np.where(mask)[0]),
            "t50_improvement": _improvement(selected, floor, np.where(mask & (horizon == 50))[0]),
            "t100_raw_frame_diagnostic_improvement": _improvement(selected, floor, np.where(mask & (horizon == 100))[0]),
            "hard_failure_improvement": _improvement(selected, floor, np.where(mask & hard)[0]),
            "easy_degradation": _degradation(selected, floor, np.where(mask & easy)[0]),
        }
        rows.append(row)
    return rows


def _rebuild_stage42ii_arrays() -> dict[str, Any]:
    stage42ii = read_json(s42ii.REPORT_JSON, {})
    if stage42ii.get("stage42_ii_gate", {}).get("verdict") != "stage42_ii_ensemble_repair_stabilizes_t50":
        raise RuntimeError("Stage42-II report is missing or not passing; run run_stage42_t50_gain_harm_ensemble_repair.py first.")
    data = s42ii._prepare_data()
    splits = data["splits"]
    labels_test = data["labels"]["test"]
    cache_events: list[dict[str, Any]] = []
    base_preds = s42ii._base_predictions(splits, cache_events)
    pred_test = s42ii._prediction_ensemble(base_preds, "test")
    scores_test = s42ii._score_ensemble("test", splits, base_preds, data["vocab"], cache_events)
    switch = s42o._selector_switch(scores_test, labels_test, stage42ii["policy"])
    arrays = s42ii._row_error_bundle(pred_test, labels_test, switch)
    return {"stage42ii": stage42ii, "labels": labels_test, "arrays": arrays, "switch": switch, "cache_events": cache_events}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    rebuilt = _rebuild_stage42ii_arrays()
    labels = rebuilt["labels"]
    arrays = rebuilt["arrays"]
    switch = rebuilt["switch"]
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    source_file = labels["source_file"].astype(str)
    scene_id = labels["scene_id"].astype(str)
    selected = arrays["selected_ade"]
    floor = arrays["floor_ade"]
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t50": horizon == 50,
        "t100_raw_frame_diagnostic": horizon == 100,
        "hard_failure": hard,
        "easy": easy,
    }
    source_rows = _slice_rows(selected, floor, labels, "source_file", min_rows=50)
    scene_rows = _slice_rows(selected, floor, labels, "scene_id", min_rows=50)
    powered_t50_sources = [r for r in source_rows if r["t50_rows"] >= 80]
    powered_t50_scenes = [r for r in scene_rows if r["t50_rows"] >= 80]
    group_bootstrap = {
        "source_file": {
            "all": _group_bootstrap_ci(selected, floor, masks["all"], source_file, seed=42100),
            "t50": _group_bootstrap_ci(selected, floor, masks["t50"], source_file, seed=42150),
            "t100_raw_frame_diagnostic": _group_bootstrap_ci(selected, floor, masks["t100_raw_frame_diagnostic"], source_file, seed=42200),
            "hard_failure": _group_bootstrap_ci(selected, floor, masks["hard_failure"], source_file, seed=42250),
            "easy_degradation": _group_bootstrap_ci(selected, floor, masks["easy"], source_file, mode="degradation", seed=42300),
        },
        "scene_id": {
            "all": _group_bootstrap_ci(selected, floor, masks["all"], scene_id, seed=42400),
            "t50": _group_bootstrap_ci(selected, floor, masks["t50"], scene_id, seed=42450),
            "hard_failure": _group_bootstrap_ci(selected, floor, masks["hard_failure"], scene_id, seed=42500),
            "easy_degradation": _group_bootstrap_ci(selected, floor, masks["easy"], scene_id, mode="degradation", seed=42550),
        },
    }
    summary = {
        "rows": int(len(horizon)),
        "source_count": int(len(set(source_file.tolist()))),
        "scene_count": int(len(set(scene_id.tolist()))),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "all_improvement": _improvement(selected, floor, np.where(masks["all"])[0]),
        "t50_improvement": _improvement(selected, floor, np.where(masks["t50"])[0]),
        "t50_source_group_ci_low": group_bootstrap["source_file"]["t50"]["low"],
        "t50_scene_group_ci_low": group_bootstrap["scene_id"]["t50"]["low"],
        "hard_failure_improvement": _improvement(selected, floor, np.where(masks["hard_failure"])[0]),
        "easy_degradation": _degradation(selected, floor, np.where(masks["easy"])[0]),
        "powered_t50_source_count": int(len(powered_t50_sources)),
        "positive_powered_t50_source_count": int(sum(1 for r in powered_t50_sources if r["t50_improvement"] > 0.0)),
        "negative_powered_t50_source_count": int(sum(1 for r in powered_t50_sources if r["t50_improvement"] < -1e-9)),
        "powered_t50_scene_count": int(len(powered_t50_scenes)),
        "positive_powered_t50_scene_count": int(sum(1 for r in powered_t50_scenes if r["t50_improvement"] > 0.0)),
        "negative_powered_t50_scene_count": int(sum(1 for r in powered_t50_scenes if r["t50_improvement"] < -1e-9)),
    }
    payload = {
        "stage": "Stage42-IJ",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([s42ii.REPORT_JSON, s42ii.REPORT_MD, s42ii.GATE_MD]),
        "summary": summary,
        "group_bootstrap": group_bootstrap,
        "source_rows": source_rows,
        "scene_rows": scene_rows,
        "cache_events": rebuilt["cache_events"],
        "source_labels": {
            "stage42ii_policy": "cached_verified",
            "stage42ii_intermediate_predictions_and_scores": "cached_verified",
            "source_scene_robustness_eval": "fresh_run",
            "new_training": "not_run",
        },
        "no_leakage": rebuilt["stage42ii"].get("no_leakage", {}),
        "claim_boundary": rebuilt["stage42ii"].get("claim_boundary", {}),
    }
    payload["stage42_ij_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload.get("summary", {})
    claim = payload.get("claim_boundary", {})
    no_leakage = payload.get("no_leakage", {})
    gates = {
        "stage42ii_report_verified": payload.get("source_labels", {}).get("stage42ii_policy") == "cached_verified",
        "source_scene_eval_fresh": payload.get("source_labels", {}).get("source_scene_robustness_eval") == "fresh_run",
        "source_count_sufficient": s.get("source_count", 0) >= 3,
        "scene_count_sufficient": s.get("scene_count", 0) >= 3,
        "all_positive": s.get("all_improvement", 0.0) > 0.0,
        "t50_positive": s.get("t50_improvement", 0.0) > 0.0,
        "source_group_t50_nonnegative": s.get("t50_source_group_ci_low", -1.0) >= -1e-9,
        "scene_group_t50_nonnegative": s.get("t50_scene_group_ci_low", -1.0) >= -1e-9,
        "no_negative_powered_t50_source": s.get("negative_powered_t50_source_count", 1) == 0,
        "hard_positive": s.get("hard_failure_improvement", 0.0) > 0.0,
        "easy_preserved": s.get("easy_degradation", 1.0) <= 0.02,
        "no_future_or_test_leakage": no_leakage.get("future_endpoint_input") is False
        and no_leakage.get("future_waypoints_input") is False
        and no_leakage.get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    if passed == total:
        verdict = "stage42_ij_t50_ensemble_source_robustness_pass"
    elif gates["t50_positive"] and not gates["source_group_t50_nonnegative"]:
        verdict = "stage42_ij_t50_ensemble_positive_but_source_ci_weak"
    else:
        verdict = "stage42_ij_t50_ensemble_source_robustness_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    gate = payload["stage42_ij_gate"]
    lines = [
        "# Stage42-IJ T50 Ensemble Source Robustness",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Purpose",
        "",
        "Stage42-II repaired the t+50 seed-instability blocker with a validation-selected score/prediction ensemble. Stage42-IJ checks whether that result remains stable at source-file and scene levels.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| rows | {s['rows']} |",
        f"| source count | {s['source_count']} |",
        f"| scene count | {s['scene_count']} |",
        f"| all improvement | {s['all_improvement']:.6f} |",
        f"| t50 improvement | {s['t50_improvement']:.6f} |",
        f"| t50 source-group CI low | {s['t50_source_group_ci_low']:.6f} |",
        f"| t50 scene-group CI low | {s['t50_scene_group_ci_low']:.6f} |",
        f"| hard/failure improvement | {s['hard_failure_improvement']:.6f} |",
        f"| easy degradation | {s['easy_degradation']:.6f} |",
        f"| powered t50 sources positive / total | {s['positive_powered_t50_source_count']} / {s['powered_t50_source_count']} |",
        f"| powered t50 scenes positive / total | {s['positive_powered_t50_scene_count']} / {s['powered_t50_scene_count']} |",
        "",
        "## Source-File Rows",
        "",
        "| source file | rows | t50 rows | all | t50 | hard/failure | easy degradation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["source_rows"][:40]:
        short = Path(row["source_file"]).name
        lines.append(
            f"| `{short}` | {row['rows']} | {row['t50_rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a fresh source/scene robustness evaluation from cached-verified Stage42-II intermediates.",
            "- No new model training is claimed.",
            "- UCY remains fallback-only in Stage42-II; this audit does not rewrite fallback as positive transfer.",
            "- Results remain dataset-local/raw-frame 2.5D only, with no metric or seconds-level claim.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IJ Gate",
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
    gate = payload["stage42_ij_gate"]
    lines = [
        "## Stage42-IJ T50 Ensemble Source Robustness",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- all / t50 / hard: `{s['all_improvement']:.6f}` / `{s['t50_improvement']:.6f}` / `{s['hard_failure_improvement']:.6f}`",
        f"- source-group t50 CI low: `{s['t50_source_group_ci_low']:.6f}`",
        f"- scene-group t50 CI low: `{s['t50_scene_group_ci_low']:.6f}`",
        f"- powered t50 source positives: `{s['positive_powered_t50_source_count']} / {s['powered_t50_source_count']}`",
        f"- easy degradation: `{s['easy_degradation']:.6f}`",
        "- boundary: cached-verified Stage42-II intermediates plus fresh source/scene eval; dataset-local/raw-frame 2.5D only; no metric/seconds, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ij_t50_ensemble_source_robustness"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_ij_t50_ensemble_source_robustness"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": s,
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_ensemble_source_robustness() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_ij_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_ensemble_source_robustness()
    print(json.dumps(_jsonable(result["stage42_ij_gate"]), ensure_ascii=False, indent=2))
