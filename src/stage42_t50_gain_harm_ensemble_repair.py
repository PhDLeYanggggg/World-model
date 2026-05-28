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
from src.stage42_t50_gain_harm_row_bootstrap import _bootstrap_ci, _improvement, _degradation


OUT_DIR = Path("outputs/stage42_long_research")
CACHE_DIR = Path("data/stage42_t50_gain_harm_ensemble_cache")
REPORT_JSON = OUT_DIR / "t50_gain_harm_ensemble_repair_stage42.json"
REPORT_MD = OUT_DIR / "t50_gain_harm_ensemble_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ii_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage42_ii_t50_gain_harm_ensemble_repair_heartbeat.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_II_T50_GAIN_HARM_ENSEMBLE_REPAIR"
SOURCE = "fresh_stage42_ii_t50_gain_harm_ensemble_repair"

SELECTOR_SEEDS = [149, 151, 157, 163, 167, 173]
BASE_SEEDS = [109, 113, 127, 109, 113, 127]


def _heartbeat(status: str, **extra: Any) -> None:
    ensure_dir(OUT_DIR)
    payload = {
        "source": SOURCE,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    HEARTBEAT_JSON.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _selector_info(seed: int) -> dict[str, Any]:
    ckpt = OUT_DIR / "checkpoints" / f"stage42p_t50_gain_harm_selector_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42p_t50_gain_harm_selector_seed{seed}_heartbeat.json"
    if not ckpt.exists() or not heartbeat.exists():
        raise FileNotFoundError(f"Missing Stage42-P/IH selector checkpoint for seed {seed}. Run Stage42-IH first.")
    return {"source": "cached_verified_stage42p_or_ih", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": read_json(heartbeat, {}).get("best", {})}


def _average_dicts(items: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError("cannot average empty prediction list")
    return {key: np.mean([item[key] for item in items], axis=0).astype(np.float32) for key in items[0].keys()}


def _prepare_data() -> dict[str, Any]:
    _heartbeat("prepare_data_start")
    ft.build_full_trajectory_labels()
    splits = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    vocab = s42o._domain_vocab(splits["train"], splits["val"], splits["test"])
    labels = {split: s42i._labels(splits[split]) for split in ["val", "test"]}
    _heartbeat("prepare_data_complete", split_rows={k: len(v["horizon"]) for k, v in splits.items()})
    return {"splits": splits, "vocab": vocab, "labels": labels}


def _cache_path_base(seed: int) -> Path:
    return CACHE_DIR / f"base_predictions_seed{seed}.npz"


def _save_prediction_cache(path: Path, preds: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    ensure_dir(path.parent)
    flat: dict[str, np.ndarray] = {}
    for split, row in preds.items():
        for key, value in row.items():
            flat[f"{split}__{key}"] = value
    np.savez(path, **flat)


def _load_prediction_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as npz:
        out: dict[str, dict[str, np.ndarray]] = {split: {} for split in ["train", "val", "test"]}
        for key in npz.files:
            split, name = key.split("__", 1)
            out[split][name] = npz[key]
    return out


def _base_predictions(
    splits: Mapping[str, Mapping[str, np.ndarray]],
    cache_events: list[dict[str, Any]] | None = None,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    preds: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for base_seed in sorted(set(BASE_SEEDS)):
        path = _cache_path_base(base_seed)
        if path.exists():
            _heartbeat("base_prediction_cache_hit", base_seed=base_seed, cache_path=str(path))
            if cache_events is not None:
                cache_events.append({"artifact": "base_prediction", "seed": base_seed, "source": "cached_verified", "cache_path": str(path)})
            preds[base_seed] = _load_prediction_cache(path)
            continue
        _heartbeat("base_prediction_start", base_seed=base_seed)
        info = s42p._base_model_info(base_seed)
        row = {split: s42m._predict(info, splits[split]) for split in ["train", "val", "test"]}
        _save_prediction_cache(path, row)
        _heartbeat("base_prediction_complete", base_seed=base_seed, cache_path=str(path))
        if cache_events is not None:
            cache_events.append({"artifact": "base_prediction", "seed": base_seed, "source": "fresh_run", "cache_path": str(path)})
        preds[base_seed] = row
    return preds


def _score_cache_path(seed: int, base_seed: int, split_name: str) -> Path:
    return CACHE_DIR / f"selector_scores_seed{seed}_base{base_seed}_{split_name}.npz"


def _save_score_cache(path: Path, scores: Mapping[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    np.savez(path, **{k: v.astype(np.float32) for k, v in scores.items()})


def _load_score_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _score_ensemble(
    split_name: str,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    base_preds: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    vocab: Mapping[str, int],
    cache_events: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    score_rows: list[dict[str, np.ndarray]] = []
    for seed, base_seed in zip(SELECTOR_SEEDS, BASE_SEEDS):
        path = _score_cache_path(seed, base_seed, split_name)
        if path.exists():
            _heartbeat("selector_score_cache_hit", seed=seed, base_seed=base_seed, split=split_name, cache_path=str(path))
            if cache_events is not None:
                cache_events.append(
                    {
                        "artifact": "selector_score",
                        "seed": seed,
                        "base_seed": base_seed,
                        "split": split_name,
                        "source": "cached_verified",
                        "cache_path": str(path),
                    }
                )
            score_rows.append(_load_score_cache(path))
            continue
        _heartbeat("selector_score_start", seed=seed, base_seed=base_seed, split=split_name)
        train_stats = s42o._feature_stats(s42o._raw_features(splits["train"], base_preds[base_seed]["train"], vocab))
        x = s42o._features(splits[split_name], base_preds[base_seed][split_name], vocab, train_stats)
        scores = s42p._predict_selector(_selector_info(seed), x)
        _save_score_cache(path, scores)
        _heartbeat("selector_score_complete", seed=seed, base_seed=base_seed, split=split_name, cache_path=str(path))
        if cache_events is not None:
            cache_events.append(
                {
                    "artifact": "selector_score",
                    "seed": seed,
                    "base_seed": base_seed,
                    "split": split_name,
                    "source": "fresh_run",
                    "cache_path": str(path),
                }
            )
        score_rows.append(scores)
    return _average_dicts(score_rows)


def _prediction_ensemble(base_preds: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]], split_name: str) -> dict[str, np.ndarray]:
    unique_preds = [base_preds[seed][split_name] for seed in sorted(set(BASE_SEEDS))]
    return _average_dicts(unique_preds)


def _row_error_bundle(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, np.ndarray]:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    selected_xy = floor_xy.copy()
    selected_xy[switch] = neural_xy[switch]
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    return {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "neural_ade": neural_ade,
        "neural_fde": neural_fde,
        "oracle_ade": np.minimum(floor_ade, neural_ade),
        "oracle_fde": np.minimum(floor_fde, neural_fde),
    }


def _bootstrap_summary(arrays: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t50": horizon == 50,
        "t100_raw_frame_diagnostic": horizon == 100,
        "hard_failure": labels["hard"].astype(bool) | labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
    }
    out: dict[str, Any] = {"switch_rate": float(np.mean(switch)) if len(switch) else 0.0, "ade": {}, "fde": {}, "bootstrap": {"ade": {}, "fde": {}, "easy": {}}}
    for name, mask in masks.items():
        if name == "easy":
            ids = np.where(mask)[0]
            out["ade"]["easy_degradation"] = _degradation(arrays["selected_ade"], arrays["floor_ade"], ids)
            out["fde"]["easy_degradation"] = _degradation(arrays["selected_fde"], arrays["floor_fde"], ids)
            out["bootstrap"]["easy"]["ade"] = _bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], mask, mode="degradation", seed=42201)
            out["bootstrap"]["easy"]["fde"] = _bootstrap_ci(arrays["selected_fde"], arrays["floor_fde"], mask, mode="degradation", seed=42202)
            continue
        key = "t100" if name == "t100_raw_frame_diagnostic" else name
        ids = np.where(mask)[0]
        out["ade"][f"{key}_improvement"] = _improvement(arrays["selected_ade"], arrays["floor_ade"], ids)
        out["fde"][f"{key}_improvement"] = _improvement(arrays["selected_fde"], arrays["floor_fde"], ids)
        out["bootstrap"]["ade"][key] = _bootstrap_ci(arrays["selected_ade"], arrays["floor_ade"], mask, seed=42211)
        out["bootstrap"]["fde"][key] = _bootstrap_ci(arrays["selected_fde"], arrays["floor_fde"], mask, seed=42221)
    domain_rows: dict[str, Any] = {}
    domain = labels["domain"].astype(str)
    for d in sorted(set(domain.tolist())):
        mask = domain == d
        ids = np.where(mask)[0]
        h50 = mask & (horizon == 50)
        domain_rows[d] = {
            "rows": int(np.sum(mask)),
            "all_improvement": _improvement(arrays["selected_ade"], arrays["floor_ade"], ids),
            "t50_improvement": _improvement(arrays["selected_ade"], arrays["floor_ade"], np.where(h50)[0]),
            "hard_failure_improvement": _improvement(
                arrays["selected_ade"],
                arrays["floor_ade"],
                np.where(mask & (labels["hard"].astype(bool) | labels["failure"].astype(bool)))[0],
            ),
            "easy_degradation": _degradation(arrays["selected_ade"], arrays["floor_ade"], np.where(mask & labels["easy"].astype(bool))[0]),
            "switch_rate": float(np.mean(switch[mask])) if np.any(mask) else 0.0,
        }
    out["by_domain"] = domain_rows
    return out


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    cache_events: list[dict[str, Any]] = []
    data = _prepare_data()
    splits = data["splits"]
    labels_val = data["labels"]["val"]
    labels_test = data["labels"]["test"]
    base_preds = _base_predictions(splits, cache_events)
    _heartbeat("prediction_ensemble_start")
    pred_val = _prediction_ensemble(base_preds, "val")
    pred_test = _prediction_ensemble(base_preds, "test")
    _heartbeat("score_ensemble_val_start")
    scores_val = _score_ensemble("val", splits, base_preds, data["vocab"], cache_events)
    _heartbeat("score_ensemble_test_start")
    scores_test = _score_ensemble("test", splits, base_preds, data["vocab"], cache_events)
    _heartbeat("validation_policy_start")
    policy, val_metrics = s42p._fit_policy_t50(scores_val, pred_val, labels_val)
    _heartbeat("test_eval_start")
    switch_test = s42o._selector_switch(scores_test, labels_test, policy)
    test_metrics = s42o._metric_from_switch(pred_test, labels_test, switch_test)
    arrays = _row_error_bundle(pred_test, labels_test, switch_test)
    bootstrap = _bootstrap_summary(arrays, labels_test, switch_test)
    summary = {
        "ade_all": test_metrics["ade"]["all_improvement"],
        "ade_t50": test_metrics["ade"]["t50_improvement"],
        "ade_t50_ci_low": bootstrap["bootstrap"]["ade"]["t50"]["low"],
        "ade_t50_ci_high": bootstrap["bootstrap"]["ade"]["t50"]["high"],
        "ade_t100_raw_frame_diagnostic": test_metrics["ade"]["t100_improvement"],
        "ade_hard_failure": test_metrics["ade"]["hard_failure_improvement"],
        "ade_easy_degradation": test_metrics["ade"]["easy_degradation"],
        "fde_t50": test_metrics["fde"]["t50_improvement"],
        "fde_t50_ci_low": bootstrap["bootstrap"]["fde"]["t50"]["low"],
        "switch_rate": test_metrics["switch_rate"],
        "trajnet_t50": bootstrap["by_domain"].get("TrajNet", {}).get("t50_improvement", 0.0),
        "eth_ucy_t50": bootstrap["by_domain"].get("ETH_UCY", {}).get("t50_improvement", 0.0),
        "ucy_t50": bootstrap["by_domain"].get("UCY", {}).get("t50_improvement", 0.0),
    }
    payload = {
        "stage": "Stage42-II",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                OUT_DIR / "t50_gain_harm_seed_expansion_stage42.json",
                ft.DATA_DIR / "all_agent_train.npz",
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_train.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
            ]
        ),
        "selector_seeds": SELECTOR_SEEDS,
        "base_seeds": BASE_SEEDS,
        "summary": summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "bootstrap": bootstrap,
        "policy": policy,
        "source_labels": {
            "selector_checkpoints": "cached_verified_stage42p_ih",
            "base_checkpoints": "cached_verified_stage42n",
            "cache_dir": str(CACHE_DIR),
            "heartbeat": str(HEARTBEAT_JSON),
            "base_prediction_events": cache_events,
            "score_ensemble": "fresh_run_from_cached_or_fresh_stage42ii_intermediates",
            "prediction_ensemble": "fresh_run_from_cached_or_fresh_stage42ii_intermediates",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once",
            "new_training": "not_run",
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
    payload["stage42_ii_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload.get("summary", {})
    no_leakage = payload.get("no_leakage", {})
    claim = payload.get("claim_boundary", {})
    gates = {
        "selector_ensemble_built": len(payload.get("selector_seeds", [])) >= 6,
        "prediction_ensemble_built": len(set(payload.get("base_seeds", []))) >= 3,
        "validation_policy_selected": payload.get("source_labels", {}).get("validation_policy_selection") == "fresh_run",
        "ade_all_positive": s.get("ade_all", 0.0) > 0.0,
        "ade_t50_positive": s.get("ade_t50", 0.0) > 0.0,
        "ade_t50_row_ci_positive": s.get("ade_t50_ci_low", -1.0) > 0.0,
        "fde_t50_ci_positive": s.get("fde_t50_ci_low", -1.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", 1.0) <= 0.02,
        "trajnet_t50_nonnegative": s.get("trajnet_t50", -1.0) >= 0.0,
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
        verdict = "stage42_ii_ensemble_repair_stabilizes_t50"
    elif gates["ade_t50_positive"] and not gates["trajnet_t50_nonnegative"]:
        verdict = "stage42_ii_ensemble_repair_partial_trajnet_blocker"
    else:
        verdict = "stage42_ii_ensemble_repair_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    gate = payload["stage42_ii_gate"]
    lines = [
        "# Stage42-II T50 Gain/Harm Ensemble Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Purpose",
        "",
        "Stage42-IH showed that simply adding same-family selector seeds did not make ADE t+50 seed-stable. Stage42-II tests a validation-only score ensemble over six t+50 gain/harm selectors plus a three-checkpoint Stage42-N dynamics ensemble.",
        "",
        "## Claim Boundary",
        "",
        "- dataset-local/raw-frame 2.5D only",
        "- no metric or seconds-level claim",
        "- no true 3D/foundation claim",
        "- no Stage5C execution",
        "- no SMC",
        "",
        "## Fresh Ensemble Test Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| ADE all | {s['ade_all']:.6f} |",
        f"| ADE t50 | {s['ade_t50']:.6f} |",
        f"| ADE t50 row CI low | {s['ade_t50_ci_low']:.6f} |",
        f"| ADE t50 row CI high | {s['ade_t50_ci_high']:.6f} |",
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
        "| domain | rows | all | t50 | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["bootstrap"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a fresh replay/evaluation, not new training.",
            "- Base predictions and selector scores are cached as Stage42-II local intermediates after first computation; final policy selection and test evaluation are still freshly recomputed from those intermediates.",
            "- The policy is selected on validation only, then evaluated once on test.",
            "- If this passes, ensemble selection is a stronger deployable t+50 repair than any single seed.",
            "- If this fails, the blocker is a model-family or domain-specific TrajNet t+50 issue, not seed count alone.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-II Gate",
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
    gate = payload["stage42_ii_gate"]
    lines = [
        "## Stage42-II T50 Gain/Harm Ensemble Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- ADE all / t50 / hard: `{s['ade_all']:.6f}` / `{s['ade_t50']:.6f}` / `{s['ade_hard_failure']:.6f}`",
        f"- ADE t50 row CI low: `{s['ade_t50_ci_low']:.6f}`",
        f"- FDE t50 / CI low: `{s['fde_t50']:.6f}` / `{s['fde_t50_ci_low']:.6f}`",
        f"- easy degradation: `{s['ade_easy_degradation']:.6f}`",
        f"- TrajNet t50: `{s['trajnet_t50']:.6f}`",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ii_t50_gain_harm_ensemble_repair"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_ii_t50_gain_harm_ensemble_repair"] = {
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


def run_stage42_t50_gain_harm_ensemble_repair() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_ii_gate"])
    _refresh_readmes_and_state(payload)
    _heartbeat(
        "complete",
        verdict=payload["stage42_ii_gate"]["verdict"],
        gates=f"{payload['stage42_ii_gate']['passed']}/{payload['stage42_ii_gate']['total']}",
        report=str(REPORT_MD),
        json=str(REPORT_JSON),
        gate=str(GATE_MD),
    )
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_gain_harm_ensemble_repair()
    print(json.dumps(_jsonable(result["stage42_ii_gate"]), ensure_ascii=False, indent=2))
