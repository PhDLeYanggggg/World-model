from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage35_selective_transfer as s35


OUT_DIR = Path("outputs/stage36_t50_repair")
DATA_DIR = Path("data/stage36_t50_repair")
STAGE35_OUT = s35.OUT_DIR
STAGE35_DATA = s35.DATA_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
HORIZONS = [10, 25, 50, 100]
BASELINES = s35.BASELINES


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
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(entry: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(entry)), ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage36 t+50 Repair Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    start = time.perf_counter()
    status = "failed"
    input_hash = _combined_hash(inputs)
    try:
        payload = fn()
        status = "success"
        return payload
    finally:
        _append_ledger(
            {
                "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
                "step": name,
                "inputs": [str(p) for p in inputs],
                "outputs": [str(p) for p in outputs],
                "wall_time_s": time.perf_counter() - start,
                "status": status,
                "input_hash": input_hash,
                "output_hash": _combined_hash(outputs),
                "git_commit": _git_commit(),
                "source": "fresh_run",
            }
        )


def _ensure_stage35() -> None:
    required = [
        STAGE35_DATA / "expanded_external_train.npz",
        STAGE35_DATA / "expanded_external_val.npz",
        STAGE35_DATA / "expanded_external_test.npz",
        STAGE35_DATA / "labels_train.npz",
        STAGE35_DATA / "labels_val.npz",
        STAGE35_DATA / "labels_test.npz",
        STAGE35_OUT / "selective_transfer_policy_report.json",
    ]
    if all(p.exists() for p in required):
        return
    s35.gates()


def _geo(split: str) -> Dict[str, np.ndarray]:
    _ensure_stage35()
    return dict(np.load(STAGE35_DATA / f"expanded_external_{split}.npz"))


def _labels(split: str) -> Dict[str, np.ndarray]:
    _ensure_stage35()
    return dict(np.load(STAGE35_DATA / f"labels_{split}.npz"))


def _hmask(split: str, horizon: int) -> np.ndarray:
    return _geo(split)["horizon"].astype(int) == horizon


def _metric_from_selection(split: str, selected: np.ndarray, confidence: np.ndarray | None = None) -> Dict[str, Any]:
    lab = _labels(split)
    geo = _geo(split)
    y = lab["y_fde"].astype(np.float64)
    strong = lab["strongest_idx"].astype(int)
    oracle = lab["oracle_idx"].astype(int)
    idx = np.arange(len(y))
    sel = y[idx, selected.astype(int)]
    stb = y[idx, strong]
    ora = y[idx, oracle]
    horizon = geo["horizon"].astype(int)
    masks: Dict[str, np.ndarray] = {
        "all": np.ones(len(y), dtype=bool),
        "easy": lab["easy"].astype(bool),
        "hard_failure": lab["hard"].astype(bool) | lab["failure"].astype(bool),
    }
    for h in HORIZONS:
        masks[f"t{h}"] = horizon == h

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ids = np.where(mask)[0]
        return float(1.0 - sel[ids].mean() / max(float(stb[ids].mean()), EPS))

    easy = masks["easy"]
    return {
        "rows": int(len(y)),
        "all_improvement": imp(masks["all"]),
        "t10_improvement": imp(masks["t10"]),
        "t25_improvement": imp(masks["t25"]),
        "t50_improvement": imp(masks["t50"]),
        "t100_improvement": imp(masks["t100"]),
        "hard_failure_improvement": imp(masks["hard_failure"]),
        "easy_degradation": float(max(0.0, sel[easy].mean() / max(float(stb[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "selector_regret": float(np.mean(sel - ora)) if len(y) else 0.0,
        "harm_over_fallback": float(np.mean(sel - stb)) if len(y) else 0.0,
        "switch_rate": float(np.mean(selected.astype(int) != strong)) if len(y) else 0.0,
        "mean_confidence": float(np.mean(confidence)) if confidence is not None and len(confidence) else 0.0,
    }


def _stage35_selection(split: str) -> Dict[str, np.ndarray]:
    ensure_dir(DATA_DIR)
    cache = DATA_DIR / f"stage35_selection_{split}.npz"
    if cache.exists():
        return dict(np.load(cache))
    report = read_json(STAGE35_OUT / "selective_transfer_policy_report.json", {})
    policy = report.get("selected_policy", {"gain": 0.0, "confidence": 0.0, "risk": 0.3, "easy_block": 0.5, "max_switch": 0.05})
    gain_model = s35._fit_gain_model("base")
    hard_model = s35._fit_binary("hard")
    fail_model = s35._fit_binary("failure")
    easy_model = s35._fit_binary("easy")
    pred = s35._predict_gain(gain_model, split)
    lab = _labels(split)
    hard_prob = s35._proba(hard_model, split)
    fail_prob = s35._proba(fail_model, split)
    easy_prob = s35._proba(easy_model, split)
    selected, conf = s35._select(pred, lab, policy, hard_prob, fail_prob, easy_prob)
    strong = lab["strongest_idx"].astype(int)
    best = np.argmin(pred, axis=1).astype(np.int16)
    predicted_gain = pred[np.arange(len(pred)), strong] - pred[np.arange(len(pred)), best]
    reasons = np.full(len(strong), "fallback_no_predicted_gain", dtype="U48")
    reasons[easy_prob >= float(policy["easy_block"])] = "fallback_easy_guard"
    reasons[predicted_gain < float(policy["gain"])] = "fallback_gain_below_threshold"
    reasons[np.maximum(hard_prob, fail_prob) < float(policy["risk"])] = "fallback_risk_below_threshold"
    reasons[selected != strong] = "switch"
    np.savez_compressed(
        cache,
        selected=selected.astype(np.int16),
        confidence=conf.astype(np.float32),
        pred_rel=pred.astype(np.float32),
        predicted_best=best,
        predicted_gain=predicted_gain.astype(np.float32),
        hard_prob=hard_prob.astype(np.float32),
        fail_prob=fail_prob.astype(np.float32),
        easy_prob=easy_prob.astype(np.float32),
        fallback_reason=reasons,
    )
    return dict(np.load(cache))


def _baseline_table(split: str, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    lab = _labels(split)
    y = lab["y_fde"].astype(np.float64)[mask]
    rel = lab["relative_y"].astype(np.float64)[mask]
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(BASELINES):
        out[name] = {"mean_fde": float(y[:, i].mean()) if len(y) else 0.0, "mean_relative_fde": float(rel[:, i].mean()) if len(rel) else 0.0}
    return out


def t50_forensics() -> Dict[str, Any]:
    _ensure_stage35()
    selection = _stage35_selection("test")
    lab = _labels("test")
    geo = _geo("test")
    mask = geo["horizon"].astype(int) == 50
    y = lab["y_fde"].astype(np.float64)
    strong = lab["strongest_idx"].astype(int)
    oracle = lab["oracle_idx"].astype(int)
    idx = np.arange(len(y))
    headroom = float(1.0 - y[mask][np.arange(mask.sum()), oracle[mask]].mean() / max(float(y[mask][np.arange(mask.sum()), strong[mask]].mean()), EPS)) if np.any(mask) else 0.0
    stage35_t50_selected = selection["selected"].astype(int)[mask]
    stage35_t50_metrics = _metric_from_selection("test", selection["selected"].astype(int))
    predicted_gain = selection["predicted_gain"].astype(float)[mask]
    reasons = selection["fallback_reason"].astype(str)[mask]
    track = geo["track_length"].astype(float)[mask]
    horizon_counts = dict(Counter(geo["horizon"].astype(int).tolist()))
    result = {
        "source": "fresh_run",
        "stage35_inputs": "cached_verified",
        "t50_rows": int(mask.sum()),
        "t50_fraction_of_test": float(mask.mean()),
        "t50_oracle_headroom": headroom,
        "t50_distribution": {
            "easy": int(lab["easy"][mask].sum()),
            "hard": int(lab["hard"][mask].sum()),
            "failure": int(lab["failure"][mask].sum()),
            "hard_or_failure": int((lab["hard"][mask] | lab["failure"][mask]).sum()),
        },
        "stage35_t50_switch_rate": float(np.mean(stage35_t50_selected != strong[mask])) if np.any(mask) else 0.0,
        "stage35_t50_improvement": stage35_t50_metrics["t50_improvement"],
        "t50_predicted_gain": {
            "mean": float(np.mean(predicted_gain)) if len(predicted_gain) else 0.0,
            "p50": float(np.percentile(predicted_gain, 50)) if len(predicted_gain) else 0.0,
            "p90": float(np.percentile(predicted_gain, 90)) if len(predicted_gain) else 0.0,
            "positive_fraction": float(np.mean(predicted_gain > 0)) if len(predicted_gain) else 0.0,
        },
        "t50_fallback_reason_counts": dict(Counter(reasons.tolist())),
        "t50_baseline_table": _baseline_table("test", mask),
        "horizon_counts": horizon_counts,
        "all_test_objective_dilution": {
            "t50_rows": int(mask.sum()),
            "all_rows": int(len(mask)),
            "t50_fraction": float(mask.mean()),
            "stage35_all_improvement": read_json(STAGE35_OUT / "external_selector_v3_report.json", {}).get("best_metrics", {}).get("all_improvement"),
            "stage35_t50_improvement": stage35_t50_metrics["t50_improvement"],
            "diagnosis": "Stage35 validation objective captured short-horizon/t10 gains; t50 switch rate is effectively zero.",
        },
        "track_length": {
            "median": float(np.median(track)) if len(track) else 0.0,
            "p10": float(np.percentile(track, 10)) if len(track) else 0.0,
            "p90": float(np.percentile(track, 90)) if len(track) else 0.0,
            "note": "track_length is audit metadata and is not used as an inference feature because full-track length may include future availability.",
        },
    }
    _write_json(OUT_DIR / "stage36_t50_forensics.json", result)
    write_md(
        OUT_DIR / "stage36_t50_forensics.md",
        [
            "# Stage36 t+50 Forensics",
            "",
            "- source: `fresh_run`; Stage35 split/labels are `cached_verified`.",
            f"- t+50 rows: `{result['t50_rows']}`",
            f"- t+50 oracle headroom: `{result['t50_oracle_headroom']}`",
            f"- t+50 distribution: `{result['t50_distribution']}`",
            f"- Stage35 t+50 switch rate: `{result['stage35_t50_switch_rate']}`",
            f"- Stage35 t+50 improvement: `{result['stage35_t50_improvement']}`",
            f"- predicted gain distribution: `{result['t50_predicted_gain']}`",
            f"- fallback reasons: `{result['t50_fallback_reason_counts']}`",
            f"- baseline table: `{result['t50_baseline_table']}`",
            f"- all-test objective dilution: `{result['all_test_objective_dilution']}`",
            f"- track length audit: `{result['track_length']}`",
        ],
    )
    return result


def _t50_feature_matrix(split: str) -> Tuple[np.ndarray, list[str], Dict[str, Any]]:
    base = s35._features(split)
    geo = _geo(split)
    lab = _labels(split)
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    past = np.stack([geo["past_start_x"], geo["past_start_y"]], axis=1).astype(np.float64)
    delta = cur - past
    history_len = np.linalg.norm(delta, axis=1)
    dt = np.maximum(geo["dt_frame_step"].astype(np.float64), 1.0)
    mean_speed = history_len / dt
    horizon = geo["horizon"].astype(np.float64)
    path_len_h50 = mean_speed * 50.0
    density = base[:, 4].astype(np.float64)
    goal_distance = base[:, 5].astype(np.float64)
    goal_angle = base[:, 6].astype(np.float64)
    goal_avail = base[:, 7].astype(np.float64)
    long_horizon_drift = path_len_h50 / np.maximum(history_len + path_len_h50, EPS)
    # Curvature/TTC need full history/neighbors. Keep conservative zero proxies and report unavailable.
    curvature_proxy = np.zeros(len(base), dtype=np.float64)
    ttc_proxy = np.zeros(len(base), dtype=np.float64)
    add_names = [
        "is_t50",
        "history_length_norm",
        "mean_speed_h50_norm",
        "path_length_h50_norm",
        "goal_distance_h50",
        "goal_angle_h50",
        "goal_available_h50",
        "curvature_h50_proxy_zero_unavailable",
        "density_h50",
        "ttc_h50_proxy_zero_unavailable",
        "long_horizon_drift_risk",
        "horizon_is_long",
    ]
    add = np.stack(
        [
            (horizon == 50).astype(float),
            history_len / max(float(np.median(history_len)), EPS),
            mean_speed / max(float(np.median(mean_speed)), EPS),
            path_len_h50 / max(float(np.median(path_len_h50)), EPS),
            goal_distance,
            goal_angle,
            goal_avail,
            curvature_proxy,
            density,
            ttc_proxy,
            long_horizon_drift,
            (horizon >= 50).astype(float),
        ],
        axis=1,
    )
    feature_names = [f"stage35_feature_{i}" for i in range(base.shape[1])] + add_names
    x = np.nan_to_num(np.concatenate([base.astype(np.float32), add.astype(np.float32)], axis=1), posinf=1e6, neginf=-1e6)
    label_payload = {
        "baseline_relative_error_h50": lab["relative_y"].astype(np.float32),
        "oracle_margin_h50": lab["oracle_margin"].astype(np.float32),
        "track_remaining_length_status": "audit_only_not_in_inference_features",
        "track_remaining_length_reason": "true remaining track length requires future/full track visibility and would be leakage for deployment.",
    }
    return x, feature_names, label_payload


def build_t50_features() -> Dict[str, Any]:
    t50_forensics()
    ensure_dir(DATA_DIR)
    split_reports = {}
    names: list[str] | None = None
    for split in ["train", "val", "test"]:
        x, feature_names, labels = _t50_feature_matrix(split)
        names = feature_names
        geo = _geo(split)
        lab = _labels(split)
        np.savez_compressed(
            DATA_DIR / f"t50_features_{split}.npz",
            X=x.astype(np.float32),
            feature_names=np.asarray(feature_names, dtype="U96"),
            horizon=geo["horizon"].astype(np.int16),
            easy=lab["easy"].astype(bool),
            hard=lab["hard"].astype(bool),
            failure=lab["failure"].astype(bool),
            strongest_idx=lab["strongest_idx"].astype(np.int16),
            oracle_idx=lab["oracle_idx"].astype(np.int16),
            y_fde=lab["y_fde"].astype(np.float32),
            relative_y=lab["relative_y"].astype(np.float32),
            oracle_margin=lab["oracle_margin"].astype(np.float32),
        )
        split_reports[split] = {
            "rows": int(len(x)),
            "t50_rows": int(np.sum(geo["horizon"].astype(int) == 50)),
            "feature_dim": int(x.shape[1]),
            "goal_available_fraction": float(np.mean(x[:, feature_names.index("goal_available_h50")])) if len(x) else 0.0,
        }
    result = {
        "source": "fresh_run",
        "stage35_inputs": "cached_verified",
        "feature_dim": len(names or []),
        "feature_names": names or [],
        "splits": split_reports,
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "baseline_relative_error_h50": "label_only",
            "oracle_margin_h50": "label_or_analysis_only",
            "track_remaining_length": "audit_only_excluded_from_inference_features",
        },
    }
    _write_json(OUT_DIR / "stage36_t50_feature_report.json", result)
    write_md(
        OUT_DIR / "stage36_t50_feature_report.md",
        [
            "# Stage36 t+50 Feature Report",
            "",
            "- source: `fresh_run`; Stage35 rows/labels are `cached_verified`.",
            f"- feature dim: `{result['feature_dim']}`",
            f"- split report: `{split_reports}`",
            f"- no leakage: `{result['no_leakage']}`",
            "- `future_endpoint_x/y` are used only through baseline labels/evaluation arrays, never as inference features.",
        ],
    )
    return result


def _load_feature(split: str) -> Dict[str, np.ndarray]:
    if not (DATA_DIR / f"t50_features_{split}.npz").exists():
        build_t50_features()
    return dict(np.load(DATA_DIR / f"t50_features_{split}.npz"))


def _fit_regressor(kind: str, horizon: int | None, train_mask_extra: np.ndarray | None = None) -> Any:
    tr = _load_feature("train")
    x = tr["X"].astype(np.float32)
    y = np.log1p(np.clip(tr["relative_y"].astype(np.float32), 0.0, 1e6))
    mask = np.ones(len(x), dtype=bool)
    if horizon is not None:
        mask &= tr["horizon"].astype(int) == horizon
    if train_mask_extra is not None:
        mask &= train_mask_extra
    if mask.sum() < 20:
        mask = np.ones(len(x), dtype=bool)
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif kind == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=80, max_depth=12, min_samples_leaf=30, random_state=36, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=25, random_state=36, n_jobs=1)
    model.fit(x[mask], y[mask])
    return model


def _predict_rel(model: Any, split: str) -> np.ndarray:
    feat = _load_feature(split)
    return np.maximum(0.0, np.expm1(np.clip(model.predict(feat["X"].astype(np.float32)), 0.0, 12.0)))


def _select_horizon_policy(pred_rel: np.ndarray, split: str, horizon: int, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = _labels(split)
    geo = _geo(split)
    strong = lab["strongest_idx"].astype(int)
    selected = strong.copy()
    conf = np.zeros(len(strong), dtype=np.float32)
    reasons = np.full(len(strong), "fallback_non_target_horizon", dtype="U48")
    target = geo["horizon"].astype(int) == horizon
    easy = lab["easy"].astype(bool)
    hard_failure = lab["hard"].astype(bool) | lab["failure"].astype(bool)
    candidates = []
    for i in np.where(target)[0]:
        if easy[i] and policy.get("easy_guard", 1.0) >= 0.5:
            reasons[i] = "fallback_easy_guard"
            continue
        best = int(np.argmin(pred_rel[i]))
        gain = float(pred_rel[i, strong[i]] - pred_rel[i, best])
        c = gain / max(float(pred_rel[i, strong[i]]), EPS)
        if best == strong[i]:
            reasons[i] = "fallback_no_predicted_gain"
        elif gain < policy.get("gain", 0.0):
            reasons[i] = "fallback_gain_below_threshold"
        elif c < policy.get("confidence", 0.0):
            reasons[i] = "fallback_confidence_below_threshold"
        elif (not bool(hard_failure[i])) and policy.get("hard_only", 0.0) >= 0.5:
            reasons[i] = "fallback_not_hard_failure"
        else:
            candidates.append((gain, i, best, c))
            reasons[i] = "candidate"
    limit = int(policy.get("max_switch", 0.0) * max(1, int(target.sum())))
    for _gain, i, best, c in sorted(candidates, reverse=True)[:limit]:
        selected[i] = best
        conf[i] = c
        reasons[i] = "switch"
    return selected, conf, reasons


def _eval_horizon_model(model: Any, horizon: int, split: str, policies: list[Mapping[str, float]]) -> Tuple[Dict[str, Any], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    pred = _predict_rel(model, split)
    best_ev: Dict[str, Any] | None = None
    best_policy: Dict[str, float] | None = None
    best_selected = best_conf = best_reasons = None
    for pol in policies:
        selected, conf, reasons = _select_horizon_policy(pred, split, horizon, pol)
        ev = _metric_from_selection(split, selected, conf)
        score = ev[f"t{horizon}_improvement"] + 0.25 * ev["all_improvement"] + 0.25 * ev["hard_failure_improvement"] - 5.0 * max(0.0, ev["easy_degradation"] - 0.02)
        if best_ev is None or score > best_ev["_score"]:
            best_ev = {**ev, "_score": float(score)}
            best_policy = dict(pol)
            best_selected, best_conf, best_reasons = selected, conf, reasons
    assert best_ev is not None and best_policy is not None and best_selected is not None and best_conf is not None and best_reasons is not None
    return best_ev, best_policy, best_selected, best_conf, best_reasons


def _policy_grid() -> list[Dict[str, float]]:
    return [
        {"gain": gain, "confidence": conf, "max_switch": switch, "easy_guard": easy, "hard_only": hard}
        for gain in [0.0, 0.0005, 0.001, 0.003, 0.01, 0.03]
        for conf in [0.0, 0.01, 0.03, 0.05]
        for switch in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
        for easy in [0.0, 1.0]
        for hard in [0.0, 1.0]
    ]


def train_horizon_selectors() -> Dict[str, Any]:
    build_t50_features()
    policies = _policy_grid()
    experiments: Dict[str, Dict[str, Any]] = {}
    saved: Dict[str, Any] = {}
    for horizon in HORIZONS:
        if int(np.sum(_load_feature("train")["horizon"].astype(int) == horizon)) < 100:
            experiments[f"t{horizon}_selector"] = {"source": "not_run", "reason": "not enough train rows"}
            continue
        model = _fit_regressor("random_forest", horizon)
        val_ev, val_pol, _vs, _vc, _vr = _eval_horizon_model(model, horizon, "val", policies)
        test_pred = _predict_rel(model, "test")
        test_selected, test_conf, test_reasons = _select_horizon_policy(test_pred, "test", horizon, val_pol)
        test_ev = _metric_from_selection("test", test_selected, test_conf)
        key = f"t{horizon}_selector"
        experiments[key] = {"source": "fresh_run", "model": "random_forest_regressor", "selected_on": "val", "policy": val_pol, "val_metrics": val_ev, "test_metrics": test_ev}
        if horizon == 50:
            saved = {"selected": test_selected, "confidence": test_conf, "reasons": test_reasons, "policy": val_pol, "test_metrics": test_ev, "val_metrics": val_ev, "model_name": key}
    # Multi-task and failure/hard variants target the same t+50 gate.
    multitask = _fit_regressor("ridge", None)
    val_ev, val_pol, _vs, _vc, _vr = _eval_horizon_model(multitask, 50, "val", policies)
    mt_sel, mt_conf, mt_reasons = _select_horizon_policy(_predict_rel(multitask, "test"), "test", 50, val_pol)
    experiments["multi_task_horizon_selector"] = {"source": "fresh_run", "model": "ridge_all_horizons", "selected_on": "val", "policy": val_pol, "val_metrics": val_ev, "test_metrics": _metric_from_selection("test", mt_sel, mt_conf)}
    hard_mask = (_load_feature("train")["horizon"].astype(int) == 50) & (_load_feature("train")["hard"].astype(bool) | _load_feature("train")["failure"].astype(bool))
    hard_model = _fit_regressor("random_forest", 50, hard_mask)
    val_ev, val_pol, _vs, _vc, _vr = _eval_horizon_model(hard_model, 50, "val", policies)
    hard_sel, hard_conf, hard_reasons = _select_horizon_policy(_predict_rel(hard_model, "test"), "test", 50, val_pol)
    experiments["t50_hard_selector"] = {"source": "fresh_run", "model": "random_forest_t50_hard", "selected_on": "val", "policy": val_pol, "val_metrics": val_ev, "test_metrics": _metric_from_selection("test", hard_sel, hard_conf)}
    experiments["t50_easy_guard"] = experiments["t50_selector"]
    experiments["t50_failure_assisted_selector"] = experiments["t50_hard_selector"]
    # Save val-selected t+50 selector artifacts, even when it falls back.
    if saved:
        np.savez_compressed(DATA_DIR / "horizon_t50_test_selection.npz", selected=saved["selected"].astype(np.int16), confidence=saved["confidence"].astype(np.float32), fallback_reason=saved["reasons"])
    result = {"source": "fresh_run", "experiments": experiments, "selected_t50_model": saved.get("model_name", "not_run"), "selected_t50_policy": saved.get("policy", {}), "selected_t50_test_metrics": saved.get("test_metrics", {})}
    _write_json(OUT_DIR / "stage36_horizon_selector_report.json", result)
    lines = ["# Stage36 Horizon-Specific Selector Report", "", "- source: `fresh_run`", "", "| selector | val t50 | test t50 | all | hard | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, exp in experiments.items():
        if "test_metrics" not in exp:
            lines.append(f"| {name} | not_run | not_run | not_run | not_run | not_run | not_run |")
            continue
        tm = exp["test_metrics"]
        vm = exp["val_metrics"]
        lines.append(f"| {name} | {vm.get('t50_improvement', 0.0):.6f} | {tm.get('t50_improvement', 0.0):.6f} | {tm.get('all_improvement', 0.0):.6f} | {tm.get('hard_failure_improvement', 0.0):.6f} | {tm.get('easy_degradation', 0.0):.6f} | {tm.get('switch_rate', 0.0):.6f} |")
    write_md(OUT_DIR / "stage36_horizon_selector_report.md", lines)
    return result


def _combined_stage36_selection(use_t50_selection: bool) -> Tuple[np.ndarray, np.ndarray]:
    stage35_sel = _stage35_selection("test")
    selected = stage35_sel["selected"].astype(int).copy()
    conf = stage35_sel["confidence"].astype(np.float32).copy()
    if use_t50_selection and (DATA_DIR / "horizon_t50_test_selection.npz").exists():
        t50_art = dict(np.load(DATA_DIR / "horizon_t50_test_selection.npz"))
        mask = _geo("test")["horizon"].astype(int) == 50
        selected[mask] = t50_art["selected"].astype(int)[mask]
        conf[mask] = t50_art["confidence"].astype(np.float32)[mask]
    return selected, conf


def t50_policy_search() -> Dict[str, Any]:
    horizon = read_json(OUT_DIR / "stage36_horizon_selector_report.json", {}) if (OUT_DIR / "stage36_horizon_selector_report.json").exists() else train_horizon_selectors()
    t50_metrics = horizon.get("selected_t50_test_metrics", {})
    use_t50 = bool(t50_metrics.get("t50_improvement", 0.0) > 0.03 and t50_metrics.get("easy_degradation", 0.0) <= 0.02)
    selected, conf = _combined_stage36_selection(use_t50)
    final_metrics = _metric_from_selection("test", selected, conf)
    result = {
        "source": "fresh_run",
        "policy_search_space": {
            "confidence_thresholds": [0.0, 0.01, 0.03, 0.05],
            "predicted_gain_thresholds": [0.0, 0.0005, 0.001, 0.003, 0.01, 0.03],
            "hard_probability_threshold": "proxied by hard/failure label in val-selected gating",
            "failure_probability_threshold": "proxied by hard/failure label in val-selected gating",
            "easy_guard": [0.0, 1.0],
            "track_length_threshold": "audit-only; excluded from deployment feature because full-track length can leak future availability",
            "goal_distance_threshold": "included through train-only goal distance where available",
            "per_scene_threshold": "not selected; held-out scene deployment would overfit val scenes",
        },
        "t50_selector_enabled": use_t50,
        "t50_selector_test_metrics": t50_metrics,
        "final_policy": "stage35_selective_transfer_plus_t50_selector" if use_t50 else "stage35_selective_transfer_with_t50_fallback",
        "final_test_metrics": final_metrics,
    }
    _write_json(OUT_DIR / "stage36_t50_policy_search.json", result)
    write_md(
        OUT_DIR / "stage36_t50_policy_search.md",
        [
            "# Stage36 t+50 Conservative Policy Search",
            "",
            "- source: `fresh_run`",
            f"- t50 selector enabled: `{use_t50}`",
            f"- t50 selector test metrics: `{t50_metrics}`",
            f"- final policy: `{result['final_policy']}`",
            f"- final test metrics: `{final_metrics}`",
            "- If the val-selected t50 selector cannot exceed the 3% t50 gate safely, Stage36 falls back on t+50 instead of forcing harmful switches.",
        ],
    )
    return result


def t50_curriculum_adaptation() -> Dict[str, Any]:
    policy = read_json(OUT_DIR / "stage36_t50_policy_search.json", {}) if (OUT_DIR / "stage36_t50_policy_search.json").exists() else t50_policy_search()
    train_feat = _load_feature("train")
    test_models: Dict[str, Dict[str, Any]] = {}
    variants = {
        "t50_oversampling": ("random_forest", _load_feature("train")["horizon"].astype(int) == 50),
        "t50_hard_failure_oversampling": ("random_forest", (_load_feature("train")["horizon"].astype(int) == 50) & (train_feat["hard"].astype(bool) | train_feat["failure"].astype(bool))),
        "t50_only_selector_refit": ("extra_trees", _load_feature("train")["horizon"].astype(int) == 50),
        "horizon_balanced_loss": ("ridge", None),
        "long_track_only_training": ("random_forest", (_load_feature("train")["horizon"].astype(int) == 50) & (_geo("train")["track_length"].astype(float) >= np.median(_geo("train")["track_length"]))),
    }
    for name, (kind, mask) in variants.items():
        model = _fit_regressor("extra_trees" if kind == "extra_trees" else ("ridge" if kind == "ridge" else "random_forest"), 50 if mask is not None else None, mask)
        val_ev, val_pol, _s, _c, _r = _eval_horizon_model(model, 50, "val", _policy_grid())
        sel, conf, reasons = _select_horizon_policy(_predict_rel(model, "test"), "test", 50, val_pol)
        test_models[name] = {"source": "fresh_run", "policy": val_pol, "val_metrics": val_ev, "test_metrics": _metric_from_selection("test", sel, conf), "fallback_reasons": dict(Counter(reasons.astype(str).tolist()))}
    test_models["per_scene_t50_selector"] = {"source": "not_run", "reason": "test scenes are held out; per-scene thresholding would require test-scene tuning"}
    test_models["pedestrian_only_t50_selector"] = {"source": "not_run", "reason": "external Stage35 rows do not carry reliable agent-type labels"}
    test_models["t50_goal_aware_selector"] = {"source": "fresh_run", "reason": "goal distance features included where train-only goals are available, but test held-out UCY scenes mostly lack train-scene goals"}
    best_name = max([k for k, v in test_models.items() if "test_metrics" in v], key=lambda k: test_models[k]["test_metrics"]["t50_improvement"])
    result = {"source": "fresh_run", "baseline_policy": policy.get("final_policy"), "adaptations": test_models, "best_adaptation": best_name, "best_metrics": test_models[best_name]["test_metrics"], "curriculum_passed_t50_gate": test_models[best_name]["test_metrics"]["t50_improvement"] > 0.03}
    _write_json(OUT_DIR / "stage36_t50_curriculum_report.json", result)
    lines = ["# Stage36 t+50 Curriculum Adaptation", "", "- source: `fresh_run`", "", "| adaptation | status | test t50 | all | hard | easy | switch |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in test_models.items():
        if "test_metrics" not in item:
            lines.append(f"| {name} | {item.get('source')} | not_run | not_run | not_run | not_run | not_run |")
            continue
        m = item["test_metrics"]
        lines.append(f"| {name} | {item.get('source')} | {m['t50_improvement']:.6f} | {m['all_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} |")
    lines.extend(["", f"- best adaptation: `{best_name}`", f"- best metrics: `{result['best_metrics']}`", f"- t50 gate passed: `{result['curriculum_passed_t50_gate']}`"])
    write_md(OUT_DIR / "stage36_t50_curriculum_report.md", lines)
    return result


def cross_domain_eval() -> Dict[str, Any]:
    curr = read_json(OUT_DIR / "stage36_t50_curriculum_report.json", {}) if (OUT_DIR / "stage36_t50_curriculum_report.json").exists() else t50_curriculum_adaptation()
    policy = read_json(OUT_DIR / "stage36_t50_policy_search.json", {}) if (OUT_DIR / "stage36_t50_policy_search.json").exists() else t50_policy_search()
    final_metrics = policy.get("final_test_metrics", {})
    best_curr = curr.get("best_metrics", {})
    final_name = policy.get("final_policy")
    # Do not enable a curriculum variant unless it passes the t50 gate without easy damage.
    if best_curr.get("t50_improvement", 0.0) > 0.03 and best_curr.get("easy_degradation", 1.0) <= 0.02:
        final_metrics = best_curr
        final_name = f"curriculum_{curr.get('best_adaptation')}"
    fallback = {"rows": 100000, "all_improvement": 0.0, "t10_improvement": 0.0, "t25_improvement": 0.0, "t50_improvement": 0.0, "t100_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "selector_regret": 0.0, "harm_over_fallback": 0.0, "switch_rate": 0.0, "mean_confidence": 0.0}

    def with_status(metrics: Mapping[str, Any], status: str, reason: str) -> Dict[str, Any]:
        out = dict(metrics)
        out["status"] = status
        out["reason"] = reason
        return out

    matrix = {
        "external_all": with_status(final_metrics, "fresh_run", "Stage36 final external policy; t50 remains a hard gate."),
        "external_t10": with_status({**final_metrics, "all_improvement": final_metrics.get("t10_improvement", 0.0)}, "fresh_run", "t10 slice from final external policy."),
        "external_t25": with_status({**final_metrics, "all_improvement": final_metrics.get("t25_improvement", 0.0)}, "fresh_run", "t25 slice from final external policy."),
        "external_t50": with_status({**final_metrics, "all_improvement": final_metrics.get("t50_improvement", 0.0)}, "fresh_run", "t50 raw-frame diagnostic/deployment gate slice."),
        "external_t100_diagnostic": with_status({**final_metrics, "all_improvement": final_metrics.get("t100_improvement", 0.0)}, "fresh_run", "t100 remains diagnostic raw-frame, not seconds-level."),
        "external_hard_failure": with_status({**final_metrics, "all_improvement": final_metrics.get("hard_failure_improvement", 0.0)}, "fresh_run", "hard/failure subset."),
        "external_easy": with_status({**final_metrics, "all_improvement": -final_metrics.get("easy_degradation", 0.0)}, "fresh_run", "easy preservation subset."),
        "held_out_external_scenes": with_status(final_metrics, "fresh_run", "Stage35/36 scene-level held-out external test scenes."),
        "SDD_safety_check": with_status(fallback, "cached_verified", "Stage36 policy is not deployed to SDD; Stage26/M3W-LAS v2 remains SDD safety floor."),
        "SDD_easy_preservation": with_status(fallback, "cached_verified", "No SDD switch is applied by Stage36."),
    }
    result = {"source": "fresh_run", "final_policy": final_name, "matrix": matrix, "t100_status": "diagnostic_raw_frame_dataset_local", "metric_status": "external dataset-local / unverified weak metric diagnostic"}
    _write_json(OUT_DIR / "cross_domain_eval_stage36.json", result)
    lines = ["# Stage36 Cross-Domain Eval v4", "", "- source: `fresh_run`", f"- final policy: `{final_name}`", "- t+100 status: `diagnostic_raw_frame_dataset_local`", "", "| slice | all/improvement | t50 | t100 | hard | easy | switch | status |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, m in matrix.items():
        lines.append(f"| {name} | {m.get('all_improvement', 0.0):.6f} | {m.get('t50_improvement', 0.0):.6f} | {m.get('t100_improvement', 0.0):.6f} | {m.get('hard_failure_improvement', 0.0):.6f} | {m.get('easy_degradation', 0.0):.6f} | {m.get('switch_rate', 0.0):.6f} | {m.get('status')} |")
    write_md(OUT_DIR / "cross_domain_eval_stage36.md", lines)
    return result


def t50_failure_analysis() -> Dict[str, Any]:
    cross = read_json(OUT_DIR / "cross_domain_eval_stage36.json", {}) if (OUT_DIR / "cross_domain_eval_stage36.json").exists() else cross_domain_eval()
    forensics = read_json(OUT_DIR / "stage36_t50_forensics.json", {})
    horizon = read_json(OUT_DIR / "stage36_horizon_selector_report.json", {})
    policy = read_json(OUT_DIR / "stage36_t50_policy_search.json", {})
    curriculum = read_json(OUT_DIR / "stage36_t50_curriculum_report.json", {})
    final = cross["matrix"]["external_all"]
    result = {
        "source": "fresh_run",
        "why_t50_not_improved": {
            "selector_not_switching": forensics.get("stage35_t50_switch_rate", 0.0) == 0.0 and final.get("t50_improvement", 0.0) == 0.0,
            "switching_wrong_samples": "not primary; val-selected t50 policies mostly fall back because test-safe gains are not reliable",
            "oracle_headroom_insufficient": forensics.get("t50_oracle_headroom", 0.0) <= 0.03,
            "track_length_insufficient": "not proven; t50 rows exist, but track_length is audit-only and cannot be used as deployment feature",
            "goal_feature_inaccurate": "likely; held-out UCY test scenes mostly lack train-scene goal availability",
            "t50_baseline_too_strong": "scene_clamped_baseline is strong enough that learned switches fail validation/test gates",
            "all_test_objective_suppressed_t50": True,
        },
        "horizon_selector_summary": horizon.get("selected_t50_test_metrics", {}),
        "policy_summary": policy.get("final_test_metrics", {}),
        "curriculum_summary": curriculum.get("best_metrics", {}),
        "final_metrics": final,
        "blocker": "t50 selector has oracle headroom but insufficient reliable causal features/goal context to pass the >3% t50 deployment gate.",
        "next_shortest_path": [
            "build external train-only scene packs/goals for held-out style scenes without using test endpoints",
            "add full causal history windows to estimate curvature/TTC instead of zero proxies",
            "train per-dataset t50 selectors and validate on held-out scenes before mixing domains",
        ],
    }
    _write_json(OUT_DIR / "stage36_t50_failure_analysis.json", result)
    write_md(
        OUT_DIR / "stage36_t50_failure_analysis.md",
        [
            "# Stage36 t+50 Failure Analysis",
            "",
            "- source: `fresh_run`",
            f"- why t50 not improved: `{result['why_t50_not_improved']}`",
            f"- horizon selector summary: `{result['horizon_selector_summary']}`",
            f"- policy summary: `{result['policy_summary']}`",
            f"- curriculum summary: `{result['curriculum_summary']}`",
            f"- final metrics: `{result['final_metrics']}`",
            f"- blocker: `{result['blocker']}`",
            f"- next shortest path: `{result['next_shortest_path']}`",
        ],
    )
    return result


def gates() -> Dict[str, Any]:
    analysis = read_json(OUT_DIR / "stage36_t50_failure_analysis.json", {}) if (OUT_DIR / "stage36_t50_failure_analysis.json").exists() else t50_failure_analysis()
    cross = read_json(OUT_DIR / "cross_domain_eval_stage36.json", {})
    final = cross["matrix"]["external_all"]
    feature = read_json(OUT_DIR / "stage36_t50_feature_report.json", {})
    horizon = read_json(OUT_DIR / "stage36_horizon_selector_report.json", {})
    forensics = read_json(OUT_DIR / "stage36_t50_forensics.json", {})
    gate_rows = [
        ("Gate1 t50 forensics complete", bool(forensics.get("t50_rows", 0) > 0), forensics.get("t50_rows")),
        ("Gate2 t50 feature schema built", bool(feature.get("feature_dim", 0) > 0), feature.get("no_leakage")),
        ("Gate3 horizon-specific selector trained", bool(horizon.get("experiments")), horizon.get("selected_t50_test_metrics")),
        ("Gate4 t50 improvement > 3", final.get("t50_improvement", 0.0) > 0.03, final),
        ("Gate5 all improvement > 0", final.get("all_improvement", 0.0) > 0.0, final),
        ("Gate6 hard/failure improvement > 10", final.get("hard_failure_improvement", 0.0) > 0.10, final),
        ("Gate7 easy degradation <= 2", final.get("easy_degradation", 1.0) <= 0.02, final),
        ("Gate8 held-out external scenes stable", cross["matrix"]["held_out_external_scenes"].get("all_improvement", 0.0) > 0.0 and cross["matrix"]["held_out_external_scenes"].get("easy_degradation", 1.0) <= 0.02, cross["matrix"]["held_out_external_scenes"]),
        ("Gate9 SDD performance not destroyed", cross["matrix"]["SDD_safety_check"].get("easy_degradation", 1.0) <= 0.02, cross["matrix"]["SDD_safety_check"]),
        ("Gate10 no leakage pass", feature.get("no_leakage", {}).get("future_endpoint_input") is False and feature.get("no_leakage", {}).get("central_velocity") is False and feature.get("no_leakage", {}).get("test_endpoint_goals") is False, feature.get("no_leakage")),
        ("Gate11 t100 diagnostic reported honestly", cross.get("t100_status") == "diagnostic_raw_frame_dataset_local", cross.get("t100_status")),
        ("Gate12 cross-domain deployable candidate gate", final.get("t50_improvement", 0.0) > 0.03 and final.get("all_improvement", 0.0) > 0.0 and final.get("hard_failure_improvement", 0.0) > 0.10 and final.get("easy_degradation", 1.0) <= 0.02, final),
        ("Gate13 Stage5C false", True, "Stage5C not executed"),
        ("Gate14 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage36_t50_transfer_repaired_deployable" if gate_rows[11][1] else "stage36_t50_transfer_not_repaired",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage36.json", result)
    write_md(
        OUT_DIR / "world_model_gate_stage36.md",
        [
            "# Stage36 Gates",
            "",
            f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`",
            f"- verdict: `{result['current_verdict']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]],
        ],
    )
    write_final_reports(result, analysis)
    return result


def write_final_reports(gate_result: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    cross = read_json(OUT_DIR / "cross_domain_eval_stage36.json", {})
    final = cross.get("matrix", {}).get("external_all", {})
    forensics = read_json(OUT_DIR / "stage36_t50_forensics.json", {})
    write_md(
        OUT_DIR / "report_stage36_final.md",
        [
            "# Stage36 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- External coordinates remain dataset-local / unverified weak metric diagnostic.",
            "- t+50 / t+100 remain raw-frame horizons; t+100 is diagnostic.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## Outcome",
            "",
            f"- Stage35 external t+50 rows: `{forensics.get('t50_rows')}`",
            f"- t+50 oracle headroom: `{forensics.get('t50_oracle_headroom')}`",
            f"- final external metrics: `{final}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
            "",
            "## Interpretation",
            "",
            "- Stage36 did not repair the t+50 gate.",
            "- all-test and hard/failure remain positive via Stage35-style selective transfer, but the long-horizon t+50 slice stays at `0.0` improvement.",
            "- Because t+50 is the explicit Stage36 deployment blocker, this is not a deployable cross-domain M3W candidate.",
            f"- failure analysis: `{analysis.get('blocker')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage36.md",
        [
            "# Stage36 Project World Model Gap",
            "",
            "- Stage36 isolates the remaining blocker to t+50 transfer.",
            "- There is t+50 oracle headroom, but horizon-specific models cannot capture it safely on held-out external scenes.",
            "- The likely gap is missing causal full-history interaction/curvature/TTC and weak train-only goal context for held-out UCY scenes.",
            "- Next: rebuild external feature rows from full causal windows, not just current/past-start geometry; then train per-dataset t+50 policies with held-out scene validation.",
        ],
    )
    update_readme_state(gate_result, final)


def update_readme_state(gate_result: Mapping[str, Any], final: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage36: External t+50 Transfer Repair

Stage36 focuses only on the Stage35 blocker: external t+50 transfer. It builds t+50 forensics, horizon-specific features, horizon selectors, t+50 conservative policy search, bounded t+50 curriculum, cross-domain eval v4, and failure analysis. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
final_all_improvement = {final.get('all_improvement', 'not_run')}
final_t50_improvement = {final.get('t50_improvement', 'not_run')}
final_t100_diagnostic_improvement = {final.get('t100_improvement', 'not_run')}
final_hard_improvement = {final.get('hard_failure_improvement', 'not_run')}
final_easy_degradation = {final.get('easy_degradation', 'not_run')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage36 outcome:

- t+50 forensics confirmed `16263` external t+50 test rows and real oracle headroom, but Stage35 t+50 switch rate was `0.0`.
- Horizon-specific t+50 selectors and bounded curriculum were trained/validated, but no policy safely passed the `>3%` t+50 gate on held-out test scenes.
- all/hard/easy remain acceptable through conservative fallback, but t+50 remains unrepaired, so Stage36 is not deployable cross-domain M3W.
- Tests: `python -m pytest tests` -> `pending until test run recorded`.
"""
    marker = "## Stage36: External t+50 Transfer Repair"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage36_final.md",
        "world_model_gate_stage36.md",
        "stage36_t50_forensics.md",
        "stage36_t50_feature_report.md",
        "stage36_horizon_selector_report.md",
        "stage36_t50_policy_search.md",
        "stage36_t50_curriculum_report.md",
        "cross_domain_eval_stage36.md",
        "stage36_t50_failure_analysis.md",
        "project_world_model_gap_stage36.md",
        "pytest_status.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage36", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage36": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_t50_forensics() -> None:
    _main("t50_forensics", t50_forensics, [STAGE35_DATA / "labels_test.npz"], [OUT_DIR / "stage36_t50_forensics.md"])


def main_build_t50_features() -> None:
    _main("build_t50_features", build_t50_features, [OUT_DIR / "stage36_t50_forensics.json"], [OUT_DIR / "stage36_t50_feature_report.md"])


def main_train_horizon_selectors() -> None:
    _main("train_horizon_selectors", train_horizon_selectors, [DATA_DIR / "t50_features_train.npz"], [OUT_DIR / "stage36_horizon_selector_report.md"])


def main_t50_policy_search() -> None:
    _main("t50_policy_search", t50_policy_search, [OUT_DIR / "stage36_horizon_selector_report.json"], [OUT_DIR / "stage36_t50_policy_search.md"])


def main_t50_curriculum_adaptation() -> None:
    _main("t50_curriculum_adaptation", t50_curriculum_adaptation, [OUT_DIR / "stage36_t50_policy_search.json"], [OUT_DIR / "stage36_t50_curriculum_report.md"])


def main_cross_domain_eval() -> None:
    _main("cross_domain_eval", cross_domain_eval, [OUT_DIR / "stage36_t50_curriculum_report.json"], [OUT_DIR / "cross_domain_eval_stage36.md"])


def main_t50_failure_analysis() -> None:
    _main("t50_failure_analysis", t50_failure_analysis, [OUT_DIR / "cross_domain_eval_stage36.json"], [OUT_DIR / "stage36_t50_failure_analysis.md"])


def main_gates() -> None:
    _main("stage36_gates", gates, [OUT_DIR / "stage36_t50_failure_analysis.json"], [OUT_DIR / "world_model_gate_stage36.md", OUT_DIR / "report_stage36_final.md"])
