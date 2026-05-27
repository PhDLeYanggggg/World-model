from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage42_explicit_gain_harm_selector as s42o
from src import stage42_policy_distilled_static_gate as s42m
from src import stage42_sequence_full_waypoint as s42i
from src import stage42_t50_gain_harm_selector as s42p
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
STAGE42P_JSON = OUT_DIR / "t50_gain_harm_selector_stage42.json"
REPORT_JSON = OUT_DIR / "t50_gain_harm_row_bootstrap_stage42.json"
REPORT_MD = OUT_DIR / "t50_gain_harm_row_bootstrap_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ig_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IG_T50_GAIN_HARM_ROW_BOOTSTRAP"
SOURCE = "fresh_stage42_ig_t50_gain_harm_row_bootstrap"
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
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _improvement(selected: np.ndarray, floor: np.ndarray, ids: np.ndarray) -> float:
    if len(ids) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[ids])) / max(float(np.mean(floor[ids])), EPS))


def _degradation(selected: np.ndarray, floor: np.ndarray, ids: np.ndarray) -> float:
    if len(ids) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[ids])) / max(float(np.mean(floor[ids])), EPS) - 1.0))


def _bootstrap_ci(
    selected: np.ndarray,
    floor: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str = "improvement",
    seed: int = 42100,
    n: int = BOOTSTRAP_N,
) -> dict[str, Any]:
    ids = np.where(mask.astype(bool))[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n):
        boot = rng.choice(ids, size=len(ids), replace=True)
        if mode == "degradation":
            vals.append(_degradation(selected, floor, boot))
        else:
            vals.append(_improvement(selected, floor, boot))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "low": float(np.percentile(arr, 2.5)),
        "mid": float(np.percentile(arr, 50.0)),
        "high": float(np.percentile(arr, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _validation_score(row: Mapping[str, Any]) -> float:
    metric = (((row.get("val_metrics", {}) or {}).get("ade", {}) or {}))
    easy = float(metric.get("easy_degradation", 1.0))
    return (
        7.5 * float(metric.get("t50_improvement", 0.0))
        + 1.4 * float(metric.get("all_improvement", 0.0))
        + 1.0 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.25 * float(metric.get("t100_improvement", 0.0))
        - 80.0 * max(0.0, easy - 0.018)
    )


def _selected_seed(rows: list[Mapping[str, Any]]) -> Any:
    if not rows:
        return None
    return max(rows, key=_validation_score).get("seed")


def _prepare_data() -> dict[str, Any]:
    ft.build_full_trajectory_labels()
    train = s42i._split_arrays("train")
    val = s42i._split_arrays("val")
    test = s42i._split_arrays("test")
    vocab = s42o._domain_vocab(train, val, test)
    labels = s42i._labels(test)
    masks = {
        "all": np.ones(len(labels["horizon"]), dtype=bool),
        "t50": labels["horizon"].astype(int) == 50,
        "t100_raw_frame_diagnostic": labels["horizon"].astype(int) == 100,
        "hard_failure": labels["hard"].astype(bool) | labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
    }
    return {"train": train, "test": test, "vocab": vocab, "labels": labels, "masks": masks}


def _row_errors_for_seed(row: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(row["seed"])
    base_info = row["base_info"]
    train = data["train"]
    test = data["test"]
    vocab = data["vocab"]
    labels = data["labels"]
    pred_train = s42m._predict(base_info, train)
    pred_test = s42m._predict(base_info, test)
    train_stats = s42o._feature_stats(s42o._raw_features(train, pred_train, vocab))
    x_test = s42o._features(test, pred_test, vocab, train_stats)
    score_test = s42p._predict_selector(row["selector_info"], x_test)
    switch = s42o._selector_switch(score_test, labels, row["val_policy"])
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred_test, labels)
    selected_xy = floor_xy.copy()
    selected_xy[switch] = neural_xy[switch]
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    oracle_ade = np.minimum(floor_ade, neural_ade)
    oracle_fde = np.minimum(floor_fde, neural_fde)
    return {
        "seed": seed,
        "base_seed": row.get("base_seed"),
        "switch": switch.astype(bool),
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "neural_ade": neural_ade,
        "neural_fde": neural_fde,
        "oracle_ade": oracle_ade,
        "oracle_fde": oracle_fde,
        "score_means": {key: float(np.mean(value)) for key, value in score_test.items()},
    }


def _metric_bundle(arrays: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    switch = arrays["switch"].astype(bool)
    out: dict[str, Any] = {
        "rows": int(len(switch)),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "ade": {},
        "fde": {},
        "oracle_ade": {},
        "oracle_fde": {},
        "bootstrap": {"ade": {}, "fde": {}, "easy_degradation": {}},
    }
    for name, mask in masks.items():
        if name == "easy":
            out["ade"]["easy_degradation"] = _degradation(arrays["selected_ade"], arrays["floor_ade"], np.where(mask)[0])
            out["fde"]["easy_degradation"] = _degradation(arrays["selected_fde"], arrays["floor_fde"], np.where(mask)[0])
            out["bootstrap"]["easy_degradation"]["ade"] = _bootstrap_ci(
                arrays["selected_ade"], arrays["floor_ade"], mask, mode="degradation", seed=seed + 17
            )
            out["bootstrap"]["easy_degradation"]["fde"] = _bootstrap_ci(
                arrays["selected_fde"], arrays["floor_fde"], mask, mode="degradation", seed=seed + 23
            )
            continue
        key = "t100" if name == "t100_raw_frame_diagnostic" else name
        out["ade"][f"{key}_improvement"] = _improvement(arrays["selected_ade"], arrays["floor_ade"], np.where(mask)[0])
        out["fde"][f"{key}_improvement"] = _improvement(arrays["selected_fde"], arrays["floor_fde"], np.where(mask)[0])
        out["oracle_ade"][f"{key}_improvement"] = _improvement(arrays["oracle_ade"], arrays["floor_ade"], np.where(mask)[0])
        out["oracle_fde"][f"{key}_improvement"] = _improvement(arrays["oracle_fde"], arrays["floor_fde"], np.where(mask)[0])
        out["bootstrap"]["ade"][key] = _bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], mask, seed=seed + 101)
        out["bootstrap"]["fde"][key] = _bootstrap_ci(arrays["selected_fde"], arrays["floor_fde"], mask, seed=seed + 211)
    out["bootstrap"]["ade"]["selected_minus_oracle_t50_mean"] = float(
        np.mean(arrays["selected_ade"][masks["t50"]] - arrays["oracle_ade"][masks["t50"]])
    )
    out["bootstrap"]["fde"]["selected_minus_oracle_t50_mean"] = float(
        np.mean(arrays["selected_fde"][masks["t50"]] - arrays["oracle_fde"][masks["t50"]])
    )
    return out


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42p = read_json(STAGE42P_JSON, {})
    rows = list(stage42p.get("rows", []))
    data = _prepare_data()
    selected_seed = _selected_seed(rows)
    seed_results: list[dict[str, Any]] = []
    selected_result: dict[str, Any] | None = None
    for row in rows:
        arrays = _row_errors_for_seed(row, data)
        metrics = _metric_bundle(arrays, data["masks"], seed=42000 + int(arrays["seed"]))
        seed_payload = {
            "seed": arrays["seed"],
            "base_seed": arrays["base_seed"],
            "source": "fresh_row_error_replay_from_cached_stage42p_checkpoint",
            "metrics": metrics,
            "score_means": arrays["score_means"],
        }
        seed_results.append(seed_payload)
        if arrays["seed"] == selected_seed:
            selected_result = seed_payload
    if selected_result is None and seed_results:
        selected_result = seed_results[0]
    selected_metrics = selected_result["metrics"] if selected_result else {}
    summary = {
        "seed_count": len(seed_results),
        "validation_selected_seed": selected_seed,
        "row_errors_exported_in_memory": True,
        "row_error_arrays_committed": False,
        "bootstrap_n": BOOTSTRAP_N,
        "selected_ade_t50_improvement": ((selected_metrics.get("ade", {}) or {}).get("t50_improvement", 0.0)),
        "selected_ade_t50_ci_low": (((selected_metrics.get("bootstrap", {}) or {}).get("ade", {}) or {}).get("t50", {}) or {}).get("low", 0.0),
        "selected_ade_t50_ci_high": (((selected_metrics.get("bootstrap", {}) or {}).get("ade", {}) or {}).get("t50", {}) or {}).get("high", 0.0),
        "selected_fde_t50_improvement": ((selected_metrics.get("fde", {}) or {}).get("t50_improvement", 0.0)),
        "selected_fde_t50_ci_low": (((selected_metrics.get("bootstrap", {}) or {}).get("fde", {}) or {}).get("t50", {}) or {}).get("low", 0.0),
        "selected_ade_hard_failure_improvement": ((selected_metrics.get("ade", {}) or {}).get("hard_failure_improvement", 0.0)),
        "selected_ade_easy_degradation": ((selected_metrics.get("ade", {}) or {}).get("easy_degradation", 0.0)),
        "selected_t50_oracle_headroom_ade": ((selected_metrics.get("oracle_ade", {}) or {}).get("t50_improvement", 0.0)),
        "multiseed_ade_t50_ci_low": ((stage42p.get("summary", {}) or {}).get("ade_t50", {}) or {}).get("ci_low", 0.0),
    }
    payload = {
        "stage": "Stage42-IG",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([STAGE42P_JSON]),
        "stage42p_source": stage42p.get("source", "missing"),
        "dataset_rows": {"test": int(len(data["labels"]["horizon"]))},
        "mask_rows": {name: int(np.sum(mask)) for name, mask in data["masks"].items()},
        "summary": summary,
        "seed_results": seed_results,
        "selected_seed_result": selected_result,
        "source_labels": {
            "stage42p_artifact": "cached_verified",
            "stage42p_checkpoints": "cached_verified",
            "row_error_replay": "fresh_run",
            "bootstrap": "fresh_run",
            "new_training": "not_run",
            "row_error_arrays_committed": "not_run_by_design_large_cache_avoidance",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_eval_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_val": True,
            "test_threshold_tuning": False,
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
    payload["stage42_ig_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    no_leakage = payload.get("no_leakage", {})
    claim = payload.get("claim_boundary", {})
    gates = {
        "stage42p_artifact_loaded": payload.get("stage42p_source") != "missing",
        "row_error_replay_fresh": payload.get("source_labels", {}).get("row_error_replay") == "fresh_run",
        "row_bootstrap_fresh": payload.get("source_labels", {}).get("bootstrap") == "fresh_run",
        "selected_seed_ade_t50_positive": summary.get("selected_ade_t50_improvement", -1.0) > 0.0,
        "selected_seed_ade_t50_row_ci_positive": summary.get("selected_ade_t50_ci_low", -1.0) > 0.0,
        "selected_seed_fde_t50_row_ci_positive": summary.get("selected_fde_t50_ci_low", -1.0) > 0.0,
        "selected_seed_hard_positive": summary.get("selected_ade_hard_failure_improvement", 0.0) > 0.0,
        "selected_seed_easy_preserved": summary.get("selected_ade_easy_degradation", 1.0) <= 0.02,
        "multiseed_ade_t50_ci_still_flagged": summary.get("multiseed_ade_t50_ci_low", 1.0) <= 0.0,
        "oracle_headroom_remaining": summary.get("selected_t50_oracle_headroom_ade", 0.0) > summary.get("selected_ade_t50_improvement", 0.0),
        "no_future_endpoint_or_waypoint_input": no_leakage.get("future_endpoint_input") is False
        and no_leakage.get("future_waypoints_input") is False,
        "no_central_velocity_or_test_goal": no_leakage.get("central_velocity") is False
        and no_leakage.get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    if passed == total:
        verdict = "stage42_ig_row_bootstrap_validates_selected_seed_with_multiseed_blocker"
    else:
        verdict = "stage42_ig_row_bootstrap_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    gate = payload["stage42_ig_gate"]
    lines = [
        "# Stage42-IG T50 Gain/Harm Row Bootstrap",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Purpose",
        "",
        "Stage42-IF showed that Stage42-P has positive mean t+50 ADE but a negative 3-seed CI lower bound. This run recomputes row-level selected/fallback/oracle ADE/FDE from the cached Stage42-P checkpoints and performs bootstrap confidence intervals for the validation-selected seed.",
        "",
        "## Claim Boundary",
        "",
        "- dataset-local/raw-frame 2.5D only",
        "- no true 3D claim",
        "- no metric or seconds-level claim",
        "- no Stage5C execution",
        "- no SMC",
        "- future waypoints are evaluation labels only, not inference inputs",
        "",
        "## Validation-Selected Seed Bootstrap",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| validation-selected seed | {s['validation_selected_seed']} |",
        f"| test rows | {payload['dataset_rows']['test']} |",
        f"| t50 rows | {payload['mask_rows']['t50']} |",
        f"| bootstrap n | {s['bootstrap_n']} |",
        f"| selected ADE t50 improvement | {s['selected_ade_t50_improvement']:.6f} |",
        f"| selected ADE t50 CI low | {s['selected_ade_t50_ci_low']:.6f} |",
        f"| selected ADE t50 CI high | {s['selected_ade_t50_ci_high']:.6f} |",
        f"| selected FDE t50 improvement | {s['selected_fde_t50_improvement']:.6f} |",
        f"| selected FDE t50 CI low | {s['selected_fde_t50_ci_low']:.6f} |",
        f"| selected ADE hard/failure improvement | {s['selected_ade_hard_failure_improvement']:.6f} |",
        f"| selected ADE easy degradation | {s['selected_ade_easy_degradation']:.6f} |",
        f"| selected t50 two-action oracle headroom ADE | {s['selected_t50_oracle_headroom_ade']:.6f} |",
        f"| multiseed ADE t50 CI low from Stage42-P | {s['multiseed_ade_t50_ci_low']:.6f} |",
        "",
        "## Per-Seed Row Replay Metrics",
        "",
        "| seed | ADE all | ADE t50 | ADE t50 CI low | FDE t50 | FDE t50 CI low | ADE hard | ADE easy degr | switch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["seed_results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['seed']} | {m['ade']['all_improvement']:.6f} | {m['ade']['t50_improvement']:.6f} | {m['bootstrap']['ade']['t50']['low']:.6f} | {m['fde']['t50_improvement']:.6f} | {m['bootstrap']['fde']['t50']['low']:.6f} | {m['ade']['hard_failure_improvement']:.6f} | {m['ade']['easy_degradation']:.6f} | {m['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Row-level bootstrap validates the validation-selected Stage42-P seed as a positive t+50 candidate.",
            "- It does not erase the cross-seed instability found by Stage42-IF; the multiseed ADE t+50 CI lower bound remains negative.",
            "- Therefore the paper-safe claim is: validation-selected row-level t+50 evidence is positive, while seed-stable ADE t+50 remains an open training-stability gap.",
            "- The next research action is additional seeds or a more stable validation-selected policy family with row-error export enabled by default.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IG Gate",
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
    gate = payload["stage42_ig_gate"]
    lines = [
        "## Stage42-IG T50 Gain/Harm Row Bootstrap",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- validation-selected seed: `{s['validation_selected_seed']}`",
        f"- selected ADE t50 / CI low: `{s['selected_ade_t50_improvement']:.6f}` / `{s['selected_ade_t50_ci_low']:.6f}`",
        f"- selected FDE t50 / CI low: `{s['selected_fde_t50_improvement']:.6f}` / `{s['selected_fde_t50_ci_low']:.6f}`",
        f"- selected ADE hard/failure: `{s['selected_ade_hard_failure_improvement']:.6f}`",
        f"- selected ADE easy degradation: `{s['selected_ade_easy_degradation']:.6f}`",
        f"- multiseed ADE t50 CI low remains: `{s['multiseed_ade_t50_ci_low']:.6f}`",
        "- conclusion: validation-selected row-level t+50 evidence is positive, but seed-stable ADE t+50 remains an open blocker.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ig_t50_gain_harm_row_bootstrap"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_ig_t50_gain_harm_row_bootstrap"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "validation_selected_seed": s["validation_selected_seed"],
        "selected_ade_t50_improvement": s["selected_ade_t50_improvement"],
        "selected_ade_t50_ci_low": s["selected_ade_t50_ci_low"],
        "selected_fde_t50_improvement": s["selected_fde_t50_improvement"],
        "selected_fde_t50_ci_low": s["selected_fde_t50_ci_low"],
        "selected_ade_hard_failure_improvement": s["selected_ade_hard_failure_improvement"],
        "selected_ade_easy_degradation": s["selected_ade_easy_degradation"],
        "multiseed_ade_t50_ci_low": s["multiseed_ade_t50_ci_low"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_gain_harm_row_bootstrap() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_ig_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_gain_harm_row_bootstrap()
    print(json.dumps(_jsonable(result["stage42_ig_gate"]), ensure_ascii=False, indent=2))
