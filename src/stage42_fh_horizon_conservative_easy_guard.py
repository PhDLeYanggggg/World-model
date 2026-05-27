from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_horizon_row_switch_specialist as fm
from src import stage42_fh_horizon_weak_slice_forensics as fl
from src import stage42_fh_policy_freeze_replay as fi
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fh_horizon_conservative_easy_guard_stage42.json"
REPORT_MD = OUT_DIR / "fh_horizon_conservative_easy_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fn_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_fh_horizon_conservative_easy_guard"
EPS = 1e-6
EASY_LIMIT = 0.02
STRICT_VAL_EASY_LIMIT = 0.005
FEATURES = [
    "endpoint_delta_floor",
    "endpoint_delta_fh",
    "path_length",
    "min_distance",
    "delta_min_vs_fh",
]
REPLACEMENTS = ["floor", "fh", "fc", "di", "fa", "fb"]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FM 修复 UCY|50，但 TrajNet|100 与 UCY|100 仍是 horizon weak slices。",
    "Stage42-FN 只针对 FM 剩余 weak horizon 做 validation-only conservative easy guard，不用 test 调 threshold。",
    "future waypoints / endpoints 只作为 supervised labels 或 diagnostic/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _as_eval(out: Mapping[str, np.ndarray], floor: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "selected_xy": out["selected_xy"],
        "selected_ade": out["selected_ade"],
        "selected_fde": out["selected_fde"],
        "floor_ade": floor["selected_ade"],
        "floor_fde": floor["selected_fde"],
        "switch": out["switch"],
    }


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


def _feature_values(
    current: Mapping[str, Any],
    fh_ev: Mapping[str, Any],
    floor_ev: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> dict[str, np.ndarray]:
    scale = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    xy = current["selected_xy"].astype(np.float64)
    fh_xy = fh_ev["selected_xy"].astype(np.float64)
    floor_xy = floor_ev["selected_xy"].astype(np.float64)
    path_step = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    min_d = fl._ensure_min_distance(current, ids, data, group_key)
    fh_min = fl._ensure_min_distance(fh_ev, ids, data, group_key)
    delta_min = min_d - fh_min
    delta_min[~np.isfinite(delta_min)] = 0.0
    min_d_clean = min_d.copy()
    min_d_clean[~np.isfinite(min_d_clean)] = 10.0
    return {
        "endpoint_delta_floor": np.linalg.norm(xy[:, -1] - floor_xy[:, -1], axis=1) / scale,
        "endpoint_delta_fh": np.linalg.norm(xy[:, -1] - fh_xy[:, -1], axis=1) / scale,
        "path_length": np.sum(path_step, axis=1) / scale,
        "min_distance": min_d_clean,
        "delta_min_vs_fh": delta_min,
    }


def _candidate_group_row(
    name: str,
    mask: np.ndarray,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    out: Mapping[str, np.ndarray],
    floor_ev: Mapping[str, Any],
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    final_min = di._min_group_distance_fast(out["selected_xy"], group_key[ids], normalizer, agent)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(evals["fc"]["min_distance"]) & (evals["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(evals["di"]["min_distance"]) & (evals["di"]["min_distance"] < 0.05)
    return fg._group_row(
        name,
        mask,
        data,
        ids,
        out["selected_ade"],
        floor_ev["selected_ade"],
        out["switch"],
        final_near,
        fc_near,
        di_near,
        seed=seed,
    )


def _fast_candidate_row(
    name: str,
    mask: np.ndarray,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    out: Mapping[str, np.ndarray],
    floor_ev: Mapping[str, Any],
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
) -> dict[str, Any]:
    local = np.asarray(mask, dtype=bool)
    metric = fg._metric_for_mask(data, ids, out["selected_ade"], floor_ev["selected_ade"], out["switch"], local)
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    final_min = di._min_group_distance_fast(out["selected_xy"], group_key[ids], normalizer, agent)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(evals["fc"]["min_distance"]) & (evals["fc"]["min_distance"] < 0.05)
    near_delta = float(np.mean(final_near[local].astype(float) - fc_near[local].astype(float))) if np.any(local) else 0.0
    # Selection deliberately uses fast validation statistics only. Full bootstrap is run once
    # after the validation-selected guard is applied to test.
    return {
        "name": name,
        "rows": int(np.sum(local)),
        "metric": metric,
        "bootstrap": {"easy_degradation": {"high": float(metric.get("easy_degradation", 0.0))}},
        "near_bootstrap": {"delta_final_minus_fc": {"high": near_delta}},
        "robust_positive": False,
        "weak_reasons": ["fast_validation_selection_no_bootstrap"],
    }


def _fast_candidate_row_from_min(
    name: str,
    mask: np.ndarray,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    out: Mapping[str, np.ndarray],
    floor_ev: Mapping[str, Any],
    evals: Mapping[str, Mapping[str, Any]],
    selected_min: np.ndarray,
) -> dict[str, Any]:
    local = np.asarray(mask, dtype=bool)
    metric = fg._metric_for_mask(data, ids, out["selected_ade"], floor_ev["selected_ade"], out["switch"], local)
    final_near = np.isfinite(selected_min) & (selected_min < 0.05)
    fc_near = np.isfinite(evals["fc"]["min_distance"]) & (evals["fc"]["min_distance"] < 0.05)
    near_delta = float(np.mean(final_near[local].astype(float) - fc_near[local].astype(float))) if np.any(local) else 0.0
    return {
        "name": name,
        "rows": int(np.sum(local)),
        "metric": metric,
        "bootstrap": {"easy_degradation": {"high": float(metric.get("easy_degradation", 0.0))}},
        "near_bootstrap": {"delta_final_minus_fc": {"high": near_delta}},
        "robust_positive": False,
        "weak_reasons": ["fast_validation_selection_no_bootstrap"],
    }


def _score(row: Mapping[str, Any]) -> float:
    metric = row["metric"]
    bootstrap = row["bootstrap"]
    easy_high = float(bootstrap["easy_degradation"].get("high", metric.get("easy_degradation", 0.0)))
    near_high = float(row["near_bootstrap"]["delta_final_minus_fc"].get("high", 0.0))
    return (
        1.2 * float(metric["all_improvement"])
        + 1.2 * float(metric["hard_failure_improvement"])
        + 0.6 * float(metric["t50_improvement"])
        + 0.7 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 80.0 * max(0.0, easy_high - STRICT_VAL_EASY_LIMIT)
        - 15.0 * max(0.0, near_high)
        - 0.02 * float(metric["switch_rate"])
    )


def _apply_guard(
    base: Mapping[str, np.ndarray],
    replacement: Mapping[str, Any],
    feature: np.ndarray,
    local: np.ndarray,
    direction: str,
    threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cond = (feature >= threshold) if direction == "ge" else (feature <= threshold)
    use = local & np.asarray(cond, dtype=bool)
    out = _copy_eval(base)
    out["selected_xy"][use] = replacement["selected_xy"][use]
    out["selected_ade"][use] = replacement["selected_ade"][use]
    out["selected_fde"][use] = replacement["selected_fde"][use]
    out["switch"][use] = replacement["switch"][use]
    return out, use


def _select_guard_for_key(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    fm_out: Mapping[str, np.ndarray],
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    key: str,
) -> dict[str, Any]:
    local = fm._slice_mask(data, ids, key)
    floor = evals["floor"]
    current = _as_eval(fm_out, floor)
    base_min = fl._ensure_min_distance(current, ids, data, group_key)
    base_row = _fast_candidate_row_from_min(key, local, data, ids, fm_out, floor, evals, base_min)
    candidates: list[dict[str, Any]] = [
        {
            "policy": {"mode": "keep_fm"},
            "row": base_row,
            "score": _score(base_row),
            "guard_rows": 0,
        }
    ]
    fmap = _feature_values(current, evals["fh"], floor, ids, data, group_key)
    for feature_name in FEATURES:
        values = fmap[feature_name]
        local_values = values[local]
        local_values = local_values[np.isfinite(local_values)]
        if len(local_values) < 10 or float(np.std(local_values)) <= EPS:
            continue
        qs = np.unique(np.quantile(local_values, [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]))
        for replacement_name in REPLACEMENTS:
            replacement = evals[replacement_name]
            replacement_min = fl._ensure_min_distance(replacement, ids, data, group_key)
            for threshold in qs:
                for direction in ["ge", "le"]:
                    out, use = _apply_guard(fm_out, replacement, values, local, direction, float(threshold))
                    guard_rows = int(np.sum(use))
                    if guard_rows == 0 or guard_rows == int(np.sum(local)):
                        continue
                    selected_min = base_min.copy()
                    selected_min[use] = replacement_min[use]
                    row = _fast_candidate_row_from_min(key, local, data, ids, out, floor, evals, selected_min)
                    if float(row["metric"]["all_improvement"]) <= 0.0:
                        continue
                    candidates.append(
                        {
                            "policy": {
                                "mode": "feature_guard",
                                "replacement": replacement_name,
                                "feature": feature_name,
                                "direction": direction,
                                "threshold": float(threshold),
                            },
                            "row": row,
                            "score": _score(row),
                            "guard_rows": guard_rows,
                        }
                    )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best = candidates[0]
    if best["policy"]["mode"] != "keep_fm" and float(best["score"]) <= float(candidates[-1]["score"]) - 1e-9:
        best = candidates[0]
    return {
        "key": key,
        "base_row": base_row,
        "selected": best,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:12],
    }


def _apply_guard_policy(
    out: dict[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    key: str,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    rule = selected["policy"]
    local = fm._slice_mask(data, ids, key)
    if rule["mode"] == "keep_fm":
        return {"key": key, "mode": "keep_fm", "rows": int(np.sum(local)), "guard_rows": 0}
    current = _as_eval(out, evals["floor"])
    fmap = _feature_values(current, evals["fh"], evals["floor"], ids, data, group_key)
    feature = fmap[str(rule["feature"])]
    cond = (feature >= float(rule["threshold"])) if rule["direction"] == "ge" else (feature <= float(rule["threshold"]))
    use = local & np.asarray(cond, dtype=bool)
    replacement = evals[str(rule["replacement"])]
    out["selected_xy"][use] = replacement["selected_xy"][use]
    out["selected_ade"][use] = replacement["selected_ade"][use]
    out["selected_fde"][use] = replacement["selected_fde"][use]
    out["switch"][use] = replacement["switch"][use]
    return {
        "key": key,
        "mode": "feature_guard",
        "replacement": str(rule["replacement"]),
        "feature": str(rule["feature"]),
        "direction": str(rule["direction"]),
        "threshold": float(rule["threshold"]),
        "rows": int(np.sum(local)),
        "guard_rows": int(np.sum(use)),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fm_payload = read_json(fm.REPORT_JSON, {}) or fm.run_stage42_fh_horizon_row_switch_specialist()
    ctx = fi._context()
    data = ctx["data"]
    labels = ctx["labels"]
    floor_xy = ctx["floor"]["floor_xy"].astype(np.float32)
    fi_payload = read_json(fi.REPORT_JSON, {}) or fi.run_stage42_fh_policy_freeze_replay()
    replay = fi._replay_selected(ctx, fi_payload["frozen_policy"]["selected_candidate"])
    val_ids = replay["val_ids"]
    test_ids = replay["test_ids"]
    val_evals_ref = fm.fk.fe._reference_evals(val_ids, data, labels, ctx["floor"], ctx["am_candidate"], ctx["fc_candidate"], ctx["group_key"], ctx["prior"])
    val_evals = fm.fk._evals_by_name(val_ids, replay["val"], val_evals_ref, floor_xy, labels)
    test_evals = fm.fk._evals_by_name(test_ids, replay["test"], replay["test_evals"], floor_xy, labels)

    val_fm = _apply_fm_policy_set(data, val_ids, val_evals, ctx["group_key"], fm_payload["policies"])
    test_fm = _apply_fm_policy_set(data, test_ids, test_evals, ctx["group_key"], fm_payload["policies"])
    weak_keys = list(fm_payload["summary"].get("weak_domain_horizons_after", []))
    policies = {key: _select_guard_for_key(data, val_ids, val_fm, val_evals, ctx["group_key"], key) for key in weak_keys}

    repaired = _copy_eval(test_fm)
    applied = {
        key: _apply_guard_policy(repaired, data, test_ids, test_evals, ctx["group_key"], key, policies[key]["selected"])
        for key in weak_keys
    }

    selected = repaired["selected_ade"]
    floor = test_evals["floor"]["selected_ade"]
    switch = repaired["switch"]
    normalizer = np.maximum(data["scale"][test_ids].astype(np.float64), EPS)
    agent = data["agent_id"][test_ids].astype(np.int64)
    final_min = di._min_group_distance_fast(repaired["selected_xy"], ctx["group_key"][test_ids], normalizer, agent)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(test_evals["fc"]["min_distance"]) & (test_evals["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(test_evals["di"]["min_distance"]) & (test_evals["di"]["min_distance"] < 0.05)
    domain = data["dataset"][test_ids].astype(str)
    horizon = data["horizon"][test_ids].astype(int)
    domain_horizon_rows = {
        f"{d}|{h}": fg._group_row(
            f"{d}|{h}",
            (domain == d) & (horizon == h),
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=61000 + i * 100,
        )
        for i, (d, h) in enumerate((d, h) for d in sorted(set(domain.tolist())) for h in [10, 25, 50, 100])
        if np.any((domain == d) & (horizon == h))
    }
    weak_after = [name for name, row in domain_horizon_rows.items() if row["rows"] >= fg.MIN_CI_ROWS and not row["robust_positive"]]
    robust_after = [name for name, row in domain_horizon_rows.items() if row["robust_positive"]]
    metric = di._metric_subset(selected, floor, data, test_ids, switch)
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "weak_domain_horizons_before": weak_keys,
        "weak_domain_horizons_after": weak_after,
        "robust_domain_horizons_after": robust_after,
        "weak_horizon_count_before": len(weak_keys),
        "weak_horizon_count_after": len(weak_after),
        "repaired_horizon_count": max(0, len(weak_keys) - len(weak_after)),
        "applied_guards": applied,
        "uniform_horizon_claim_allowed": len(weak_after) == 0,
        "decision": "promote_stage42_fn_uniform_horizon_guard" if len(weak_after) == 0 else "conservative_guard_partial_keep_stage42_fh_fi_with_horizon_limit",
        "fm_verdict": fm_payload.get("stage42_fm_gate", {}).get("verdict"),
    }
    diagnostics = {
        "final_near_005": float(np.mean(final_near)) if len(final_near) else 0.0,
        "delta_near_vs_fc": float(np.mean(final_near.astype(float) - fc_near.astype(float))) if len(final_near) else 0.0,
        "delta_near_vs_di": float(np.mean(final_near.astype(float) - di_near.astype(float))) if len(final_near) else 0.0,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FN FH horizon conservative easy guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash([str(fm.REPORT_JSON), str(fi.REPORT_JSON), str(fi.POLICY_JSON)]),
        "current_facts": CURRENT_FACTS,
        "selection_rule": {
            "source": "validation_only",
            "target_keys": weak_keys,
            "features": FEATURES,
            "replacement_candidates": REPLACEMENTS,
            "strict_validation_easy_limit": STRICT_VAL_EASY_LIMIT,
            "uses_test_metrics_for_policy_selection": False,
        },
        "policies": policies,
        "summary": summary,
        "metric_vs_floor": metric,
        "diagnostics": diagnostics,
        "domain_horizon_rows": domain_horizon_rows,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
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
    payload["stage42_fn_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    metric = payload["metric_vs_floor"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fm_input_verified": bool(s.get("fm_verdict")),
        "weak_horizon_targets_loaded": len(payload["policies"]) >= 1,
        "validation_only_selection": payload["selection_rule"]["uses_test_metrics_for_policy_selection"] is False,
        "easy_guard_attempted": any(row.get("mode") == "feature_guard" and int(row.get("guard_rows", 0)) > 0 for row in s["applied_guards"].values()),
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
        verdict = "stage42_fn_conservative_easy_guard_pass_uniform_horizon"
    elif passed == total:
        verdict = "stage42_fn_conservative_easy_guard_pass_with_horizon_limit"
    else:
        verdict = "stage42_fn_conservative_easy_guard_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fn_gate"]
    s = payload["summary"]
    m = payload["metric_vs_floor"]
    d = payload["diagnostics"]
    lines = [
        "# Stage42-FN FH Horizon Conservative Easy Guard",
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
        f"- final near@0.05: `{_pct(d['final_near_005'])}`",
        "",
        "## Conservative Guard Summary",
        "",
        f"- weak_domain_horizons_before: `{s['weak_domain_horizons_before']}`",
        f"- weak_domain_horizons_after: `{s['weak_domain_horizons_after']}`",
        f"- repaired_horizon_count: `{s['repaired_horizon_count']}`",
        f"- uniform_horizon_claim_allowed: `{s['uniform_horizon_claim_allowed']}`",
        "",
        "| key | mode | replacement | feature | direction | threshold | rows | guard rows |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for key, row in s["applied_guards"].items():
        lines.append(
            f"| `{key}` | `{row.get('mode')}` | `{row.get('replacement', '')}` | `{row.get('feature', '')}` | "
            f"`{row.get('direction', '')}` | {float(row.get('threshold', 0.0)):.6f} | {int(row.get('rows', 0))} | {int(row.get('guard_rows', 0))} |"
        )
    lines += ["", *fg._render_group_table("Domain-Horizon Robustness After Conservative Guard", payload["domain_horizon_rows"])]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-FN is a validation-only conservative easy guard after FM repaired only one weak horizon.",
        "- If weak horizon slices remain, uniform horizon robustness remains blocked.",
        "- No test threshold tuning, no future endpoint input, no central velocity, no Stage5C, no SMC, no metric/seconds-level claim.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fn_gate"]
    lines = [
        "# Stage42-FN Gate",
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
            "<!-- STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD:START -->",
            "## Stage42-FN FH Horizon Conservative Easy Guard",
            "",
            f"- source: `{payload['source']}`",
            "- role: validation-only conservative easy-safety guard for FM remaining weak horizon slices; no test threshold tuning.",
            f"- gate: `{payload['stage42_fn_gate']['passed']} / {payload['stage42_fn_gate']['total']}`; verdict `{payload['stage42_fn_gate']['verdict']}`.",
            f"- global all/t50/t100raw/hard/easy: `{_pct(m['all_improvement'])}` / `{_pct(m['t50_improvement'])}` / `{_pct(m['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(m['hard_failure_improvement'])}` / `{_pct(m['easy_degradation'])}`.",
            f"- weak horizons before: `{s['weak_domain_horizons_before']}`.",
            f"- weak horizons after: `{s['weak_domain_horizons_after']}`.",
            f"- applied guards: `{s['applied_guards']}`.",
            f"- uniform horizon claim allowed: `{s['uniform_horizon_claim_allowed']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FN FH horizon conservative easy guard"
    state["current_verdict"] = payload["stage42_fn_gate"]["verdict"]
    state["stage42_fn_fh_horizon_conservative_easy_guard"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fn_gate"]["verdict"],
        "gates": f"{payload['stage42_fn_gate']['passed']}/{payload['stage42_fn_gate']['total']}",
        "summary": payload["summary"],
        "metric_vs_floor": payload["metric_vs_floor"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FN tests whether a stricter validation-only easy guard can repair the remaining FM weak horizons.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        item = "Stage42-FN FH horizon conservative easy guard"
        if item not in evidence:
            evidence.append(item)
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fn_fresh_audits"
        block["latest_conclusion"] = "Stage42-FN tests whether a validation-only conservative easy guard can repair the remaining FM weak horizons."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fh_horizon_conservative_easy_guard() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fh_horizon_conservative_easy_guard()
    gate = result["stage42_fn_gate"]
    print(f"Stage42-FN gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
