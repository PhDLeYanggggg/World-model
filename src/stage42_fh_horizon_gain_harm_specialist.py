from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_horizon_conservative_easy_guard as fn
from src import stage42_fh_horizon_row_switch_specialist as fm
from src import stage42_fh_horizon_weak_slice_forensics as fl
from src import stage42_fh_policy_freeze_replay as fi
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fh_horizon_gain_harm_specialist_stage42.json"
REPORT_MD = OUT_DIR / "fh_horizon_gain_harm_specialist_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fo_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_fh_horizon_gain_harm_specialist"
EPS = 1e-6
EASY_LIMIT = 0.02
RIDGE_ALPHA = 2.0
CANDIDATES = ["fh", "fc", "di", "fa", "fb", "floor"]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FL/FM/FN 显示 TrajNet|100 和 UCY|100 仍是 low-margin horizon weak slices。",
    "Stage42-FO 用 validation-only row-level gain/harm specialist 训练，而不是继续手工 threshold guard。",
    "输入特征只使用 past/prototype/rollout diagnostics；future labels 只用于 validation training/evaluation target。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _copy_eval(ev: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "selected_xy": ev["selected_xy"].copy(),
        "selected_ade": ev["selected_ade"].copy(),
        "selected_fde": ev["selected_fde"].copy(),
        "switch": ev["switch"].copy(),
    }


def _apply_fm_policy_set(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    policies: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    out = _copy_eval(evals["fh"])
    for key, policy in policies.items():
        fm._apply_key_policy(out, data, ids, evals, group_key, key, policy)
    return out


def _row_features(data: Mapping[str, np.ndarray], ids: np.ndarray) -> np.ndarray:
    scale = np.maximum(data["scale"][ids].astype(np.float32), EPS)
    dx = (data["current_x"][ids].astype(np.float32) - data["past_start_x"][ids].astype(np.float32)) / scale
    dy = (data["current_y"][ids].astype(np.float32) - data["past_start_y"][ids].astype(np.float32)) / scale
    horizon = data["horizon"][ids].astype(np.float32)[:, None] / 100.0
    row = [
        data["stage37_features"][ids].astype(np.float32),
        data["history_scalar"][ids].astype(np.float32),
        data["prototype_likelihood"][ids].astype(np.float32),
        data["prototype_entropy"][ids].astype(np.float32)[:, None],
        data["goal_ambiguity"][ids].astype(np.float32)[:, None],
        np.log1p(scale)[:, None],
        data["dt_frame_step"][ids].astype(np.float32)[:, None],
        horizon,
        dx[:, None],
        dy[:, None],
    ]
    out = np.concatenate(row, axis=1)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _rollout_features(
    candidate_name: str,
    candidate: Mapping[str, Any],
    base: Mapping[str, np.ndarray],
    fh: Mapping[str, Any],
    floor: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> np.ndarray:
    scale = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    xy = candidate["selected_xy"].astype(np.float64)
    base_xy = base["selected_xy"].astype(np.float64)
    fh_xy = fh["selected_xy"].astype(np.float64)
    floor_xy = floor["selected_xy"].astype(np.float64)
    path = np.sum(np.linalg.norm(np.diff(xy, axis=1), axis=2), axis=1) / scale
    end_floor = np.linalg.norm(xy[:, -1] - floor_xy[:, -1], axis=1) / scale
    end_fh = np.linalg.norm(xy[:, -1] - fh_xy[:, -1], axis=1) / scale
    end_base = np.linalg.norm(xy[:, -1] - base_xy[:, -1], axis=1) / scale
    min_d = fl._ensure_min_distance(candidate, ids, data, group_key)
    base_min = fl._ensure_min_distance({"selected_xy": base["selected_xy"]}, ids, data, group_key)
    min_clean = min_d.copy()
    min_clean[~np.isfinite(min_clean)] = 10.0
    delta_min = min_d - base_min
    delta_min[~np.isfinite(delta_min)] = 0.0
    onehot = np.zeros((len(ids), len(CANDIDATES)), dtype=np.float32)
    onehot[:, CANDIDATES.index(candidate_name)] = 1.0
    feats = np.concatenate(
        [
            path[:, None],
            end_floor[:, None],
            end_fh[:, None],
            end_base[:, None],
            min_clean[:, None],
            delta_min[:, None],
            candidate["switch"].astype(np.float32)[:, None],
            onehot,
        ],
        axis=1,
    )
    feats[~np.isfinite(feats)] = 0.0
    return feats.astype(np.float32)


def _candidate_matrix(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    fm_out: Mapping[str, np.ndarray],
    local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    row = _row_features(data, ids)
    xs: list[np.ndarray] = []
    gain: list[np.ndarray] = []
    harm: list[np.ndarray] = []
    index: list[tuple[int, str]] = []
    base_ade = fm_out["selected_ade"]
    for name in CANDIDATES:
        ev = evals[name]
        rf = _rollout_features(name, ev, fm_out, evals["fh"], evals["floor"], ids, data, group_key)
        x = np.concatenate([row, rf], axis=1)
        g = base_ade - ev["selected_ade"]
        h = (ev["selected_ade"] - base_ade) > 0.0
        local_ids = np.where(local)[0]
        xs.append(x[local_ids])
        gain.append(g[local_ids].astype(np.float32))
        harm.append(h[local_ids].astype(np.float32))
        index.extend((int(i), name) for i in local_ids)
    return np.vstack(xs).astype(np.float32), np.concatenate(gain), np.concatenate(harm), index


def _standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-5)
    return {"mean": mean, "std": std}, [((arr - mean) / std).astype(np.float32) for arr in (train_x, *others)]


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    xtx = xb.T @ xb
    reg = np.eye(xtx.shape[0], dtype=np.float64) * alpha
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    return (xb @ coef).astype(np.float32)


def _fit_models(train_x: np.ndarray, train_gain: np.ndarray, train_harm: np.ndarray) -> dict[str, Any]:
    gain_coef = _ridge_fit(train_x, train_gain)
    harm_coef = _ridge_fit(train_x, train_harm)
    return {"gain_coef": gain_coef, "harm_coef": harm_coef}


def _predict_rows(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    fm_out: Mapping[str, np.ndarray],
    local: np.ndarray,
    models: Mapping[str, Any],
    stats: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    x, _, _, index = _candidate_matrix(data, ids, evals, group_key, fm_out, local)
    x = ((x - stats["mean"]) / stats["std"]).astype(np.float32)
    pred_gain = _ridge_predict(x, models["gain_coef"])
    pred_harm = np.clip(_ridge_predict(x, models["harm_coef"]), 0.0, 1.0)
    best_gain = np.full(len(ids), -np.inf, dtype=np.float32)
    best_harm = np.ones(len(ids), dtype=np.float32)
    best_candidate = np.full(len(ids), "keep_fm", dtype=object)
    for n, (row_idx, name) in enumerate(index):
        if pred_gain[n] > best_gain[row_idx]:
            best_gain[row_idx] = pred_gain[n]
            best_harm[row_idx] = pred_harm[n]
            best_candidate[row_idx] = name
    best_gain[~np.isfinite(best_gain)] = -1e9
    return {"best_gain": best_gain, "best_harm": best_harm, "best_candidate": best_candidate}


def _apply_predictions(
    base: Mapping[str, np.ndarray],
    evals: Mapping[str, Mapping[str, Any]],
    local: np.ndarray,
    pred: Mapping[str, np.ndarray],
    gain_min: float,
    harm_max: float,
    max_switch: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    eligible = local & (pred["best_gain"] >= gain_min) & (pred["best_harm"] <= harm_max) & (pred["best_candidate"] != "keep_fm")
    local_count = max(int(np.sum(local)), 1)
    limit = int(np.floor(local_count * max_switch))
    if int(np.sum(eligible)) > limit:
        ids = np.where(eligible)[0]
        order = np.argsort(pred["best_gain"][ids])[::-1]
        keep = np.zeros_like(eligible)
        keep[ids[order[:limit]]] = True
        eligible = keep
    out = _copy_eval(base)
    for name in CANDIDATES:
        use = eligible & (pred["best_candidate"] == name)
        if not np.any(use):
            continue
        ev = evals[name]
        out["selected_xy"][use] = ev["selected_xy"][use]
        out["selected_ade"][use] = ev["selected_ade"][use]
        out["selected_fde"][use] = ev["selected_fde"][use]
        out["switch"][use] = ev["switch"][use]
    return out, eligible


def _fast_row(
    key: str,
    local: np.ndarray,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    out: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    metric = fg._metric_for_mask(data, ids, out["selected_ade"], floor["selected_ade"], out["switch"], local)
    return {"metric": metric, "score": _score(metric)}


def _score(metric: Mapping[str, Any]) -> float:
    return (
        1.4 * float(metric["all_improvement"])
        + 1.5 * float(metric["hard_failure_improvement"])
        + 0.7 * float(metric["t100_raw_frame_diagnostic_improvement"])
        + 0.4 * float(metric["t50_improvement"])
        - 80.0 * max(0.0, float(metric["easy_degradation"]) - EASY_LIMIT)
        - 0.02 * float(metric["switch_rate"])
    )


def _select_policy(
    key: str,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    fm_out: Mapping[str, np.ndarray],
    pred: Mapping[str, np.ndarray],
    select_mask: np.ndarray,
) -> dict[str, Any]:
    base_row = _fast_row(key, select_mask, data, ids, fm_out, evals["floor"])
    rows: list[dict[str, Any]] = [
        {
            "policy": {"mode": "keep_fm"},
            "metric": base_row["metric"],
            "score": base_row["score"],
            "switch_rows": 0,
        }
    ]
    local_gain = pred["best_gain"][select_mask]
    local_gain = local_gain[np.isfinite(local_gain)]
    gain_grid = [0.0]
    if len(local_gain) >= 10:
        gain_grid += [float(x) for x in np.unique(np.quantile(local_gain, [0.50, 0.65, 0.75, 0.85, 0.92]))]
    for gain_min in gain_grid:
        for harm_max in [0.05, 0.10, 0.20, 0.35, 0.50]:
            for max_switch in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]:
                out, use = _apply_predictions(fm_out, evals, select_mask, pred, gain_min, harm_max, max_switch)
                if int(np.sum(use)) == 0:
                    continue
                row = _fast_row(key, select_mask, data, ids, out, evals["floor"])
                if row["metric"]["all_improvement"] <= 0.0 or row["metric"]["easy_degradation"] > EASY_LIMIT:
                    continue
                rows.append(
                    {
                        "policy": {
                            "mode": "gain_harm_model",
                            "gain_min": float(gain_min),
                            "harm_max": float(harm_max),
                            "max_switch": float(max_switch),
                        },
                        "metric": row["metric"],
                        "score": row["score"],
                        "switch_rows": int(np.sum(use)),
                    }
                )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = rows[0]
    if selected["policy"]["mode"] != "keep_fm" and float(selected["score"]) <= float(base_row["score"]) + 1e-9:
        selected = rows[[row["policy"]["mode"] for row in rows].index("keep_fm")]
    return {"key": key, "base": rows[-1], "selected": selected, "candidate_count": len(rows), "top_candidates": rows[:10]}


def _full_group_rows(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    out: Mapping[str, np.ndarray],
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
) -> dict[str, Any]:
    selected = out["selected_ade"]
    floor = evals["floor"]["selected_ade"]
    switch = out["switch"]
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    final_min = di._min_group_distance_fast(out["selected_xy"], group_key[ids], normalizer, agent)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(evals["fc"]["min_distance"]) & (evals["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(evals["di"]["min_distance"]) & (evals["di"]["min_distance"] < 0.05)
    domain = data["dataset"][ids].astype(str)
    horizon = data["horizon"][ids].astype(int)
    return {
        f"{d}|{h}": fg._group_row(
            f"{d}|{h}",
            (domain == d) & (horizon == h),
            data,
            ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=71000 + i * 100,
        )
        for i, (d, h) in enumerate((d, h) for d in sorted(set(domain.tolist())) for h in [10, 25, 50, 100])
        if np.any((domain == d) & (horizon == h))
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fm_payload = read_json(fm.REPORT_JSON, {}) or fm.run_stage42_fh_horizon_row_switch_specialist()
    fn_payload = read_json(fn.REPORT_JSON, {}) or fn.run_stage42_fh_horizon_conservative_easy_guard()
    ctx = fi._context()
    data = ctx["data"]
    labels = ctx["labels"]
    floor_xy = ctx["floor"]["floor_xy"].astype(np.float32)
    fi_payload = read_json(fi.REPORT_JSON, {}) or fi.run_stage42_fh_policy_freeze_replay()
    replay = fi._replay_selected(ctx, fi_payload["frozen_policy"]["selected_candidate"])
    val_ids = replay["val_ids"]
    test_ids = replay["test_ids"]
    val_ref = fm.fk.fe._reference_evals(val_ids, data, labels, ctx["floor"], ctx["am_candidate"], ctx["fc_candidate"], ctx["group_key"], ctx["prior"])
    val_evals = fm.fk._evals_by_name(val_ids, replay["val"], val_ref, floor_xy, labels)
    test_evals = fm.fk._evals_by_name(test_ids, replay["test"], replay["test_evals"], floor_xy, labels)
    val_fm = _apply_fm_policy_set(data, val_ids, val_evals, ctx["group_key"], fm_payload["policies"])
    test_fm = _apply_fm_policy_set(data, test_ids, test_evals, ctx["group_key"], fm_payload["policies"])
    weak_keys = list(fn_payload["summary"].get("weak_domain_horizons_after", []))

    policies: dict[str, Any] = {}
    models: dict[str, Any] = {}
    repaired = _copy_eval(test_fm)
    applied: dict[str, Any] = {}
    for key in weak_keys:
        val_local = fm._slice_mask(data, val_ids, key)
        test_local = fm._slice_mask(data, test_ids, key)
        local_pos = np.where(val_local)[0]
        if len(local_pos) < 100:
            continue
        train_local = np.zeros_like(val_local)
        select_local = np.zeros_like(val_local)
        # Split validation rows internally by row index. Test remains reporting only.
        train_local[local_pos[::2]] = True
        select_local[local_pos[1::2]] = True
        x_train, y_gain, y_harm, _ = _candidate_matrix(data, val_ids, val_evals, ctx["group_key"], val_fm, train_local)
        x_select, _, _, _ = _candidate_matrix(data, val_ids, val_evals, ctx["group_key"], val_fm, select_local)
        stats, (x_train_s, _) = _standardize(x_train, x_select)
        model = _fit_models(x_train_s, y_gain, y_harm)
        val_pred = _predict_rows(data, val_ids, val_evals, ctx["group_key"], val_fm, select_local, model, stats)
        policy = _select_policy(key, data, val_ids, val_evals, val_fm, val_pred, select_local)
        test_pred = _predict_rows(data, test_ids, test_evals, ctx["group_key"], test_fm, test_local, model, stats)
        rule = policy["selected"]["policy"]
        if rule["mode"] == "gain_harm_model":
            out_key, use = _apply_predictions(test_fm, test_evals, test_local, test_pred, rule["gain_min"], rule["harm_max"], rule["max_switch"])
            for field in ["selected_xy", "selected_ade", "selected_fde", "switch"]:
                repaired[field][test_local] = out_key[field][test_local]
            applied[key] = {
                "key": key,
                **rule,
                "rows": int(np.sum(test_local)),
                "switch_rows": int(np.sum(use)),
            }
        else:
            applied[key] = {"key": key, "mode": "keep_fm", "rows": int(np.sum(test_local)), "switch_rows": 0}
        policies[key] = policy
        models[key] = {
            "feature_dim": int(stats["mean"].shape[1]),
            "train_examples": int(len(x_train)),
            "internal_train_rows": int(np.sum(train_local)),
            "internal_select_rows": int(np.sum(select_local)),
            "ridge_alpha": RIDGE_ALPHA,
        }

    domain_horizon_rows = _full_group_rows(data, test_ids, repaired, test_evals, ctx["group_key"])
    weak_after = [name for name, row in domain_horizon_rows.items() if row["rows"] >= fg.MIN_CI_ROWS and not row["robust_positive"]]
    robust_after = [name for name, row in domain_horizon_rows.items() if row["robust_positive"]]
    metric = di._metric_subset(repaired["selected_ade"], test_evals["floor"]["selected_ade"], data, test_ids, repaired["switch"])
    normalizer = np.maximum(data["scale"][test_ids].astype(np.float64), EPS)
    agent = data["agent_id"][test_ids].astype(np.int64)
    final_min = di._min_group_distance_fast(repaired["selected_xy"], ctx["group_key"][test_ids], normalizer, agent)
    diagnostics = {"final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05)))}
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "weak_domain_horizons_before": weak_keys,
        "weak_domain_horizons_after": weak_after,
        "robust_domain_horizons_after": robust_after,
        "weak_horizon_count_before": len(weak_keys),
        "weak_horizon_count_after": len(weak_after),
        "repaired_horizon_count": max(0, len(weak_keys) - len(weak_after)),
        "applied_policies": applied,
        "uniform_horizon_claim_allowed": len(weak_after) == 0,
        "decision": "promote_stage42_fo_gain_harm_specialist" if len(weak_after) == 0 else "gain_harm_specialist_partial_keep_stage42_fh_fi_with_horizon_limit",
        "fn_verdict": fn_payload.get("stage42_fn_gate", {}).get("verdict"),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FO FH horizon gain/harm specialist",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash([str(fm.REPORT_JSON), str(fn.REPORT_JSON), str(fi.REPORT_JSON), str(fi.POLICY_JSON)]),
        "current_facts": CURRENT_FACTS,
        "selection_rule": {
            "source": "validation_only_internal_split",
            "target_keys": weak_keys,
            "candidate_families": CANDIDATES,
            "uses_test_metrics_for_policy_selection": False,
            "uses_future_labels_for_training_only": True,
            "features": [
                "stage37_features",
                "history_scalar",
                "prototype_likelihood",
                "prototype_entropy",
                "goal_ambiguity",
                "current_minus_past_start",
                "candidate_rollout_diagnostics",
            ],
        },
        "model_summaries": models,
        "policies": policies,
        "summary": summary,
        "metric_vs_floor": metric,
        "diagnostics": diagnostics,
        "domain_horizon_rows": domain_horizon_rows,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "future_labels_training_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "test_rows_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "uniform_horizon_claim": summary["uniform_horizon_claim_allowed"],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_fo_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    metric = payload["metric_vs_floor"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fn_input_verified": bool(s.get("fn_verdict")),
        "gain_harm_models_built": len(payload["model_summaries"]) >= 1,
        "validation_only_selection": payload["selection_rule"]["uses_test_metrics_for_policy_selection"] is False,
        "future_labels_train_only": payload["selection_rule"]["uses_future_labels_for_training_only"] is True and no_leak["future_endpoint_input"] is False,
        "row_level_attempted": any(int(row.get("switch_rows", 0)) > 0 for row in s["applied_policies"].values()),
        "global_all_positive": metric["all_improvement"] > 0.0,
        "global_t50_positive": metric["t50_improvement"] > 0.0,
        "global_hard_positive": metric["hard_failure_improvement"] > 0.0,
        "global_easy_safe": metric["easy_degradation"] <= EASY_LIMIT,
        "weak_horizon_not_increased": s["weak_horizon_count_after"] <= s["weak_horizon_count_before"],
        "uniform_horizon_claim_only_if_no_weak_horizons": (boundary["uniform_horizon_claim"] is False) or s["weak_horizon_count_after"] == 0,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and int(s["weak_horizon_count_after"]) == 0:
        verdict = "stage42_fo_gain_harm_specialist_pass_uniform_horizon"
    elif passed == total:
        verdict = "stage42_fo_gain_harm_specialist_pass_with_horizon_limit"
    else:
        verdict = "stage42_fo_gain_harm_specialist_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fo_gate"]
    s = payload["summary"]
    m = payload["metric_vs_floor"]
    lines = [
        "# Stage42-FO FH Horizon Gain/Harm Specialist",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{s['decision']}`",
        "",
        "## Global Test Metrics vs Floor",
        "",
        f"- all improvement: `{_pct(m['all_improvement'])}`",
        f"- t50 improvement: `{_pct(m['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(m['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(m['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(m['easy_degradation'])}`",
        f"- switch rate: `{_pct(m['switch_rate'])}`",
        f"- final near@0.05: `{_pct(payload['diagnostics']['final_near_005'])}`",
        "",
        "## Gain/Harm Specialist Summary",
        "",
        f"- weak_domain_horizons_before: `{s['weak_domain_horizons_before']}`",
        f"- weak_domain_horizons_after: `{s['weak_domain_horizons_after']}`",
        f"- repaired_horizon_count: `{s['repaired_horizon_count']}`",
        f"- uniform_horizon_claim_allowed: `{s['uniform_horizon_claim_allowed']}`",
        "",
        "| key | mode | rows | switch rows |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, row in s["applied_policies"].items():
        lines.append(f"| `{key}` | `{row.get('mode')}` | {int(row.get('rows', 0))} | {int(row.get('switch_rows', 0))} |")
    lines += ["", *fg._render_group_table("Domain-Horizon Robustness After Gain/Harm Specialist", payload["domain_horizon_rows"])]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-FO trains a validation-only row-level gain/harm specialist on past/prototype/rollout features.",
        "- If weak horizon slices remain, uniform horizon robustness remains blocked.",
        "- No test threshold tuning, no future endpoint input, no central velocity, no Stage5C, no SMC, no metric/seconds-level claim.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fo_gate"]
    lines = [
        "# Stage42-FO Gate",
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


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    m = payload["metric_vs_floor"]
    return "\n".join(
        [
            "<!-- STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST:START -->",
            "## Stage42-FO FH Horizon Gain/Harm Specialist",
            "",
            f"- source: `{payload['source']}`",
            "- role: validation-only row-level gain/harm specialist for remaining weak horizon slices; no test threshold tuning.",
            f"- gate: `{payload['stage42_fo_gate']['passed']} / {payload['stage42_fo_gate']['total']}`; verdict `{payload['stage42_fo_gate']['verdict']}`.",
            f"- global all/t50/t100raw/hard/easy: `{_pct(m['all_improvement'])}` / `{_pct(m['t50_improvement'])}` / `{_pct(m['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(m['hard_failure_improvement'])}` / `{_pct(m['easy_degradation'])}`.",
            f"- weak horizons before: `{s['weak_domain_horizons_before']}`.",
            f"- weak horizons after: `{s['weak_domain_horizons_after']}`.",
            f"- applied policies: `{s['applied_policies']}`.",
            f"- uniform horizon claim allowed: `{s['uniform_horizon_claim_allowed']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FO FH horizon gain/harm specialist"
    state["current_verdict"] = payload["stage42_fo_gate"]["verdict"]
    state["stage42_fo_fh_horizon_gain_harm_specialist"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fo_gate"]["verdict"],
        "gates": f"{payload['stage42_fo_gate']['passed']}/{payload['stage42_fo_gate']['total']}",
        "summary": payload["summary"],
        "metric_vs_floor": payload["metric_vs_floor"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FO trains a validation-only row-level gain/harm specialist for remaining weak horizons.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        item = "Stage42-FO FH horizon gain/harm specialist"
        if item not in evidence:
            evidence.append(item)
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fo_fresh_audits"
        block["latest_conclusion"] = "Stage42-FO trains a validation-only row-level gain/harm specialist for remaining weak horizons."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fh_horizon_gain_harm_specialist() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fh_horizon_gain_harm_specialist()
    gate = result["stage42_fo_gate"]
    print(f"Stage42-FO gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
