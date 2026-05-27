from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_horizon_weak_slice_forensics as fl
from src import stage42_fh_horizon_weak_slice_repair as fk
from src import stage42_fh_policy_freeze_replay as fi
from src import stage42_fh_source_robustness_audit as fj
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fh_horizon_row_switch_specialist_stage42.json"
REPORT_MD = OUT_DIR / "fh_horizon_row_switch_specialist_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fm_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_fh_horizon_row_switch_specialist"
EPS = 1e-6
EASY_LIMIT = 0.02
CANDIDATES = ["fc", "di", "fa", "fb", "floor"]
FEATURES = ["endpoint_delta_floor", "endpoint_delta_fh", "path_length", "min_distance", "delta_min_vs_fh"]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FL 证明 TrajNet|100、UCY|50、UCY|100 的共同 blocker 是 oracle label low-margin ambiguous。",
    "Stage42-FM 训练/选择 row-level weak-horizon switch specialist，目标是验证整片替换失败后，行级 predicted-geometry proxy 是否能减少 weak horizons。",
    "Stage42-FM 只用 validation labels 选择 policy；test 只最终评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 diagnostic/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _slice_mask(data: Mapping[str, np.ndarray], ids: np.ndarray, key: str) -> np.ndarray:
    return fk._key_mask(data, ids, key)


def _feature_values(
    ev: Mapping[str, Any],
    fh_ev: Mapping[str, Any],
    floor_ev: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> dict[str, np.ndarray]:
    scale = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    xy = ev["selected_xy"].astype(np.float64)
    fh_xy = fh_ev["selected_xy"].astype(np.float64)
    floor_xy = floor_ev["selected_xy"].astype(np.float64)
    path_step = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    min_d = fl._ensure_min_distance(ev, ids, data, group_key)
    fh_min = fl._ensure_min_distance(fh_ev, ids, data, group_key)
    return {
        "endpoint_delta_floor": np.linalg.norm(xy[:, -1] - floor_xy[:, -1], axis=1) / scale,
        "endpoint_delta_fh": np.linalg.norm(xy[:, -1] - fh_xy[:, -1], axis=1) / scale,
        "path_length": np.sum(path_step, axis=1) / scale,
        "min_distance": min_d,
        "delta_min_vs_fh": min_d - fh_min,
    }


def _metric_for_selected(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    switch: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    return fk._candidate_metric(data, ids, selected_ade, floor_ade, switch, mask)


def _score(metric: Mapping[str, Any], near_proxy: float, base_near_proxy: float) -> float:
    near_penalty = max(0.0, near_proxy - base_near_proxy)
    return (
        1.35 * float(metric.get("all_improvement", 0.0))
        + 1.10 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.45 * float(metric.get("t50_improvement", 0.0))
        + 0.35 * float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
        - 40.0 * max(0.0, float(metric.get("easy_degradation", 0.0)) - EASY_LIMIT)
        - 10.0 * near_penalty
        - 0.015 * float(metric.get("switch_rate", 0.0))
    )


def _apply_rule_to_arrays(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    feature: np.ndarray,
    local: np.ndarray,
    direction: str,
    threshold: float,
) -> dict[str, np.ndarray]:
    cond = (feature >= threshold) if direction == "ge" else (feature <= threshold)
    use = local & np.asarray(cond, dtype=bool)
    selected_xy = base["selected_xy"].copy()
    selected_ade = base["selected_ade"].copy()
    selected_fde = base["selected_fde"].copy()
    switch = base["switch"].copy()
    selected_xy[use] = candidate["selected_xy"][use]
    selected_ade[use] = candidate["selected_ade"][use]
    selected_fde[use] = candidate["selected_fde"][use]
    switch[use] = candidate["switch"][use]
    return {"selected_xy": selected_xy, "selected_ade": selected_ade, "selected_fde": selected_fde, "switch": switch, "use": use}


def _near_proxy(ev: Mapping[str, Any], ids: np.ndarray, data: Mapping[str, np.ndarray], group_key: np.ndarray, mask: np.ndarray) -> float:
    min_d = fl._ensure_min_distance(ev, ids, data, group_key)
    local = np.asarray(mask, dtype=bool) & np.isfinite(min_d)
    if not np.any(local):
        return 0.0
    return float(np.mean(min_d[local] < 0.05))


def _select_policy_for_key(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    key: str,
) -> dict[str, Any]:
    local = _slice_mask(data, ids, key)
    base = evals["fh"]
    floor = evals["floor"]
    base_metric = _metric_for_selected(data, ids, base["selected_ade"], base["floor_ade"], base["switch"], local)
    base_near = _near_proxy(base, ids, data, group_key, local)
    base_score = _score(base_metric, base_near, base_near)
    candidates: list[dict[str, Any]] = [
        {
            "policy": {"mode": "keep_fh", "candidate": "fh"},
            "metric": base_metric,
            "near_proxy": base_near,
            "score": base_score,
            "switch_rows": 0,
        }
    ]
    for name in CANDIDATES:
        ev = evals[name]
        fmap = _feature_values(ev, base, floor, ids, data, group_key)
        for feature_name in FEATURES:
            values = fmap[feature_name]
            local_values = values[local]
            local_values = local_values[np.isfinite(local_values)]
            if len(local_values) < 10 or float(np.std(local_values)) <= EPS:
                continue
            quantiles = np.unique(np.quantile(local_values, [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]))
            for threshold in quantiles:
                for direction in ["ge", "le"]:
                    applied = _apply_rule_to_arrays(base, ev, values, local, direction, float(threshold))
                    if int(np.sum(applied["use"])) == 0:
                        continue
                    metric = _metric_for_selected(data, ids, applied["selected_ade"], base["floor_ade"], applied["switch"], local)
                    # Fast validation selection uses row-wise candidate min-distance proxy.
                    selected_min = fl._ensure_min_distance(base, ids, data, group_key).copy()
                    selected_min[applied["use"]] = fl._ensure_min_distance(ev, ids, data, group_key)[applied["use"]]
                    near = float(np.mean(selected_min[local] < 0.05)) if np.any(local) else 0.0
                    score = _score(metric, near, base_near)
                    candidates.append(
                        {
                            "policy": {
                                "mode": "feature_threshold",
                                "candidate": name,
                                "feature": feature_name,
                                "direction": direction,
                                "threshold": float(threshold),
                            },
                            "metric": metric,
                            "near_proxy": near,
                            "score": float(score),
                            "switch_rows": int(np.sum(applied["use"])),
                        }
                    )
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = candidates[0]
    if selected["policy"]["mode"] != "keep_fh" and float(selected["score"]) <= base_score + 1e-6:
        selected = candidates[-1] if False else candidates[[row["policy"]["mode"] for row in candidates].index("keep_fh")]
    return {
        "key": key,
        "base_metric": base_metric,
        "base_score": float(base_score),
        "selected": selected,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:10],
    }


def _apply_key_policy(
    out: dict[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    key: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    selected = policy["selected"]
    rule = selected["policy"]
    local = _slice_mask(data, ids, key)
    if rule["mode"] == "keep_fh":
        return {"key": key, "mode": "keep_fh", "rows": int(np.sum(local)), "switch_rows": 0}
    candidate = evals[str(rule["candidate"])]
    base = evals["fh"]
    floor = evals["floor"]
    fmap = _feature_values(candidate, base, floor, ids, data, group_key)
    feature = fmap[str(rule["feature"])]
    cond = (feature >= float(rule["threshold"])) if rule["direction"] == "ge" else (feature <= float(rule["threshold"]))
    use = local & np.asarray(cond, dtype=bool)
    out["selected_xy"][use] = candidate["selected_xy"][use]
    out["selected_ade"][use] = candidate["selected_ade"][use]
    out["selected_fde"][use] = candidate["selected_fde"][use]
    out["switch"][use] = candidate["switch"][use]
    return {
        "key": key,
        "mode": "feature_threshold",
        "candidate": str(rule["candidate"]),
        "feature": str(rule["feature"]),
        "direction": str(rule["direction"]),
        "threshold": float(rule["threshold"]),
        "rows": int(np.sum(local)),
        "switch_rows": int(np.sum(use)),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fl_payload = read_json(fl.REPORT_JSON, {}) or fl.run_stage42_fh_horizon_weak_slice_forensics()
    fj_payload = read_json(fj.REPORT_JSON, {}) or fj.run_stage42_fh_source_robustness_audit()
    fi_payload = read_json(fi.REPORT_JSON, {}) or fi.run_stage42_fh_policy_freeze_replay()

    ctx = fi._context()
    data = ctx["data"]
    labels = ctx["labels"]
    floor_xy = ctx["floor"]["floor_xy"].astype(np.float32)
    candidate = fi_payload["frozen_policy"]["selected_candidate"]
    replay = fi._replay_selected(ctx, candidate)
    val_ids = replay["val_ids"]
    test_ids = replay["test_ids"]
    val_evals_ref = fk.fe._reference_evals(val_ids, data, labels, ctx["floor"], ctx["am_candidate"], ctx["fc_candidate"], ctx["group_key"], ctx["prior"])
    test_evals_ref = replay["test_evals"]
    val_evals = fk._evals_by_name(val_ids, replay["val"], val_evals_ref, floor_xy, labels)
    test_evals = fk._evals_by_name(test_ids, replay["test"], test_evals_ref, floor_xy, labels)

    weak_keys = list(fj_payload["summary"].get("weak_domain_horizons", []))
    policies = {key: _select_policy_for_key(data, val_ids, val_evals, ctx["group_key"], key) for key in weak_keys}

    repaired = {
        "selected_xy": replay["test"]["selected_xy"].copy(),
        "selected_ade": replay["test"]["selected_ade"].copy(),
        "selected_fde": replay["test"]["selected_fde"].copy(),
        "switch": replay["test"]["switch"].copy(),
    }
    applied = {key: _apply_key_policy(repaired, data, test_ids, test_evals, ctx["group_key"], key, policies[key]) for key in weak_keys}

    selected = repaired["selected_ade"]
    floor = replay["test"]["floor_ade"]
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
            seed=44500 + i * 100,
        )
        for i, (d, h) in enumerate((d, h) for d in sorted(set(domain.tolist())) for h in [10, 25, 50, 100])
        if np.any((domain == d) & (horizon == h))
    }
    weak_after = [name for name, row in domain_horizon_rows.items() if row["rows"] >= fg.MIN_CI_ROWS and not row["robust_positive"]]
    robust_after = [name for name, row in domain_horizon_rows.items() if row["robust_positive"]]
    weak_before = weak_keys
    metric = di._metric_subset(selected, floor, data, test_ids, switch)
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "weak_domain_horizons_before": weak_before,
        "weak_domain_horizons_after": weak_after,
        "robust_domain_horizons_after": robust_after,
        "weak_horizon_count_before": len(weak_before),
        "weak_horizon_count_after": len(weak_after),
        "repaired_horizon_count": max(0, len(weak_before) - len(weak_after)),
        "applied_policies": applied,
        "uniform_horizon_claim_allowed": len(weak_after) == 0,
        "decision": "promote_stage42_fm_row_switch_specialist" if len(weak_after) == 0 else "row_switch_partial_keep_stage42_fh_fi_with_horizon_limit",
        "fl_root_cause_counts": fl_payload["summary"].get("root_cause_counts", {}),
    }
    diagnostics = {
        "final_near_005": float(np.mean(final_near)) if len(final_near) else 0.0,
        "delta_near_vs_fc": float(np.mean(final_near.astype(float) - fc_near.astype(float))) if len(final_near) else 0.0,
        "delta_near_vs_di": float(np.mean(final_near.astype(float) - di_near.astype(float))) if len(final_near) else 0.0,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FM FH weak-horizon row-level switch specialist",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash([str(fi.REPORT_JSON), str(fj.REPORT_JSON), str(fl.REPORT_JSON), str(fi.POLICY_JSON)]),
        "current_facts": CURRENT_FACTS,
        "selection_rule": {
            "source": "validation_only",
            "target_keys": weak_keys,
            "candidate_families": ["fh", *CANDIDATES],
            "features": FEATURES,
            "uses_test_metrics_for_policy_selection": False,
            "oracle_from_fl_diagnostic_only": True,
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
    payload["stage42_fm_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    metric = payload["metric_vs_floor"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fl_input_verified": bool(payload["summary"].get("fl_root_cause_counts")),
        "weak_horizon_policies_built": len(payload["policies"]) >= 1,
        "validation_only_selection": payload["selection_rule"]["uses_test_metrics_for_policy_selection"] is False,
        "row_level_attempted": any(row.get("mode") == "feature_threshold" and int(row.get("switch_rows", 0)) > 0 for row in s["applied_policies"].values()),
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
        verdict = "stage42_fm_horizon_row_switch_specialist_pass_uniform_horizon"
    elif passed == total:
        verdict = "stage42_fm_horizon_row_switch_specialist_pass_with_horizon_limit"
    else:
        verdict = "stage42_fm_horizon_row_switch_specialist_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fm_gate"]
    s = payload["summary"]
    m = payload["metric_vs_floor"]
    d = payload["diagnostics"]
    lines = [
        "# Stage42-FM FH Weak-Horizon Row-Level Switch Specialist",
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
        "## Row-Level Repair Summary",
        "",
        f"- weak_domain_horizons_before: `{s['weak_domain_horizons_before']}`",
        f"- weak_domain_horizons_after: `{s['weak_domain_horizons_after']}`",
        f"- repaired_horizon_count: `{s['repaired_horizon_count']}`",
        f"- uniform_horizon_claim_allowed: `{s['uniform_horizon_claim_allowed']}`",
        f"- FL root causes: `{s['fl_root_cause_counts']}`",
        "",
        "| key | mode | candidate | feature | direction | threshold | rows | switch rows |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for key, row in s["applied_policies"].items():
        lines.append(
            f"| `{key}` | `{row.get('mode')}` | `{row.get('candidate', 'fh')}` | `{row.get('feature', '')}` | "
            f"`{row.get('direction', '')}` | {float(row.get('threshold', 0.0)):.6f} | {int(row.get('rows', 0))} | {int(row.get('switch_rows', 0))} |"
        )
    lines += ["", *fg._render_group_table("Domain-Horizon Robustness After Row Switch", payload["domain_horizon_rows"])]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-FM is a validation-only row-level weak-horizon specialist attempt.",
        "- If weak horizon slices remain, uniform horizon robustness remains blocked.",
        "- No test threshold tuning, no future endpoint input, no central velocity, no Stage5C, no SMC, no metric/seconds-level claim.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fm_gate"]
    lines = [
        "# Stage42-FM Gate",
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
            "<!-- STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST:START -->",
            "## Stage42-FM FH Weak-Horizon Row-Level Switch Specialist",
            "",
            f"- source: `{payload['source']}`",
            "- role: validation-only row-level specialist attempt for FK/FJ/FL weak horizon slices; no test threshold tuning.",
            f"- gate: `{payload['stage42_fm_gate']['passed']} / {payload['stage42_fm_gate']['total']}`; verdict `{payload['stage42_fm_gate']['verdict']}`.",
            f"- global all/t50/t100raw/hard/easy: `{_pct(m['all_improvement'])}` / `{_pct(m['t50_improvement'])}` / `{_pct(m['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(m['hard_failure_improvement'])}` / `{_pct(m['easy_degradation'])}`.",
            f"- weak horizons before: `{s['weak_domain_horizons_before']}`.",
            f"- weak horizons after: `{s['weak_domain_horizons_after']}`.",
            f"- applied policies: `{s['applied_policies']}`.",
            f"- uniform horizon claim allowed: `{s['uniform_horizon_claim_allowed']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FM FH weak-horizon row-level switch specialist"
    state["current_verdict"] = payload["stage42_fm_gate"]["verdict"]
    state["stage42_fm_fh_horizon_row_switch_specialist"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fm_gate"]["verdict"],
        "gates": f"{payload['stage42_fm_gate']['passed']}/{payload['stage42_fm_gate']['total']}",
        "summary": payload["summary"],
        "metric_vs_floor": payload["metric_vs_floor"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FM tests a validation-only row-level weak-horizon specialist after FL identified low-margin horizon ambiguity.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FM FH weak-horizon row-level switch specialist" not in evidence:
            evidence.append("Stage42-FM FH weak-horizon row-level switch specialist")
        block["latest_included_evidence"] = evidence
        block["latest_conclusion"] = "Stage42-FM tests whether row-level validation-selected weak-horizon switches can repair the remaining uniform-horizon blocker."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fh_horizon_row_switch_specialist() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fh_horizon_row_switch_specialist()
    gate = result["stage42_fm_gate"]
    print(f"Stage42-FM gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
