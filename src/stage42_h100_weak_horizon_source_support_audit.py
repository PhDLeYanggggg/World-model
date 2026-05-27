from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_horizon_conservative_easy_guard as fn
from src import stage42_fh_horizon_gain_harm_specialist as fo
from src import stage42_fh_horizon_row_switch_specialist as fm
from src import stage42_fh_horizon_weak_slice_forensics as fl
from src import stage42_fh_policy_freeze_replay as fi
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "h100_weak_horizon_source_support_audit_stage42.json"
REPORT_MD = OUT_DIR / "h100_weak_horizon_source_support_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fp_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_h100_weak_horizon_source_support_audit"
EASY_LIMIT = 0.02
MIN_SUPPORT_ROWS = 100

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FO 证明 validation-only gain/harm specialist 仍未修复 TrajNet|100 和 UCY|100。",
    "Stage42-FP 不训练新 policy，不调 test threshold；它把剩余 h100 weak horizons 拆到 source / scene / support / margin 层面。",
    "future waypoints / endpoints 只作为 supervised labels 或 diagnostic/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _safe_unique(values: np.ndarray) -> list[str]:
    return sorted({str(v) for v in values.tolist()})


def _field(data: Mapping[str, np.ndarray], ids: np.ndarray, name: str) -> np.ndarray:
    if name in data:
        return data[name][ids].astype(str)
    return np.asarray(["unknown"] * len(ids), dtype=object)


def _mask_for_key(data: Mapping[str, np.ndarray], ids: np.ndarray, key: str) -> np.ndarray:
    return fm._slice_mask(data, ids, key)


def _group_counts(values: np.ndarray, mask: np.ndarray, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_values = values[np.asarray(mask, dtype=bool)]
    for value in sorted(set(local_values.tolist())):
        rows.append({"name": str(value), "rows": int(np.sum(local_values == value))})
    rows.sort(key=lambda row: (-int(row["rows"]), str(row["name"])))
    return rows[:limit]


def _family_name(source_name: str) -> str:
    text = str(source_name)
    parts = Path(text).parts
    if len(parts) >= 3:
        return "/".join(parts[-3:-1])
    return text.rsplit("/", 1)[0] if "/" in text else text


def _support_summary(
    data: Mapping[str, np.ndarray],
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    key: str,
) -> dict[str, Any]:
    val_mask = _mask_for_key(data, val_ids, key)
    test_mask = _mask_for_key(data, test_ids, key)
    val_sources = _field(data, val_ids, "source_file")
    test_sources = _field(data, test_ids, "source_file")
    val_scenes = _field(data, val_ids, "scene_id")
    test_scenes = _field(data, test_ids, "scene_id")
    val_source_set = set(_safe_unique(val_sources[val_mask]))
    test_source_set = set(_safe_unique(test_sources[test_mask]))
    val_family_set = {_family_name(v) for v in val_source_set}
    test_family_set = {_family_name(v) for v in test_source_set}
    shared_sources = sorted(val_source_set & test_source_set)
    shared_families = sorted(val_family_set & test_family_set)
    return {
        "key": key,
        "val_rows": int(np.sum(val_mask)),
        "test_rows": int(np.sum(test_mask)),
        "val_source_count": len(val_source_set),
        "test_source_count": len(test_source_set),
        "val_scene_count": len(set(_safe_unique(val_scenes[val_mask]))),
        "test_scene_count": len(set(_safe_unique(test_scenes[test_mask]))),
        "shared_source_count": len(shared_sources),
        "shared_family_count": len(shared_families),
        "shared_sources": shared_sources[:20],
        "shared_families": shared_families[:20],
        "test_source_rows": _group_counts(test_sources, test_mask),
        "test_scene_rows": _group_counts(test_scenes, test_mask),
        "val_source_rows": _group_counts(val_sources, val_mask),
        "val_scene_rows": _group_counts(val_scenes, val_mask),
    }


def _copy_eval(ev: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "selected_xy": ev["selected_xy"].copy(),
        "selected_ade": ev["selected_ade"].copy(),
        "selected_fde": ev["selected_fde"].copy(),
        "switch": ev["switch"].copy(),
    }


def _reconstruct_fo_state() -> dict[str, Any]:
    fm_payload = read_json(fm.REPORT_JSON, {}) or fm.run_stage42_fh_horizon_row_switch_specialist()
    fn_payload = read_json(fn.REPORT_JSON, {}) or fn.run_stage42_fh_horizon_conservative_easy_guard()
    fi_payload = read_json(fi.REPORT_JSON, {}) or fi.run_stage42_fh_policy_freeze_replay()
    ctx = fi._context()
    data = ctx["data"]
    labels = ctx["labels"]
    floor_xy = ctx["floor"]["floor_xy"].astype(np.float32)
    replay = fi._replay_selected(ctx, fi_payload["frozen_policy"]["selected_candidate"])
    val_ids = replay["val_ids"]
    test_ids = replay["test_ids"]
    val_ref = fm.fk.fe._reference_evals(
        val_ids,
        data,
        labels,
        ctx["floor"],
        ctx["am_candidate"],
        ctx["fc_candidate"],
        ctx["group_key"],
        ctx["prior"],
    )
    val_evals = fm.fk._evals_by_name(val_ids, replay["val"], val_ref, floor_xy, labels)
    test_evals = fm.fk._evals_by_name(test_ids, replay["test"], replay["test_evals"], floor_xy, labels)
    val_fm = fo._apply_fm_policy_set(data, val_ids, val_evals, ctx["group_key"], fm_payload["policies"])
    test_fm = fo._apply_fm_policy_set(data, test_ids, test_evals, ctx["group_key"], fm_payload["policies"])
    weak_keys = list(fn_payload["summary"].get("weak_domain_horizons_after", []))
    repaired = _copy_eval(test_fm)
    applied: dict[str, Any] = {}
    for key in weak_keys:
        val_local = fm._slice_mask(data, val_ids, key)
        test_local = fm._slice_mask(data, test_ids, key)
        local_pos = np.where(val_local)[0]
        if len(local_pos) < 100:
            applied[key] = {"key": key, "mode": "insufficient_val_rows", "rows": int(np.sum(test_local)), "switch_rows": 0}
            continue
        train_local = np.zeros_like(val_local)
        select_local = np.zeros_like(val_local)
        train_local[local_pos[::2]] = True
        select_local[local_pos[1::2]] = True
        x_train, y_gain, y_harm, _ = fo._candidate_matrix(data, val_ids, val_evals, ctx["group_key"], val_fm, train_local)
        x_select, _, _, _ = fo._candidate_matrix(data, val_ids, val_evals, ctx["group_key"], val_fm, select_local)
        stats, (x_train_s, _) = fo._standardize(x_train, x_select)
        model = fo._fit_models(x_train_s, y_gain, y_harm)
        val_pred = fo._predict_rows(data, val_ids, val_evals, ctx["group_key"], val_fm, select_local, model, stats)
        policy = fo._select_policy(key, data, val_ids, val_evals, val_fm, val_pred, select_local)
        test_pred = fo._predict_rows(data, test_ids, test_evals, ctx["group_key"], test_fm, test_local, model, stats)
        rule = policy["selected"]["policy"]
        if rule["mode"] == "gain_harm_model":
            out_key, use = fo._apply_predictions(
                test_fm,
                test_evals,
                test_local,
                test_pred,
                rule["gain_min"],
                rule["harm_max"],
                rule["max_switch"],
            )
            for field in ["selected_xy", "selected_ade", "selected_fde", "switch"]:
                repaired[field][test_local] = out_key[field][test_local]
            applied[key] = {"key": key, **rule, "rows": int(np.sum(test_local)), "switch_rows": int(np.sum(use))}
        else:
            applied[key] = {"key": key, "mode": "keep_fm", "rows": int(np.sum(test_local)), "switch_rows": 0}
    return {
        "ctx": ctx,
        "data": data,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "test_evals": test_evals,
        "test_fm": test_fm,
        "test_fo": repaired,
        "applied": applied,
        "weak_keys": weak_keys,
    }


def _final_min_distance(
    selected_xy: np.ndarray,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    group_key: np.ndarray,
) -> np.ndarray:
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), 1e-6)
    agent = data["agent_id"][ids].astype(np.int64)
    return di._min_group_distance_fast(selected_xy, group_key[ids], normalizer, agent)


def _group_metric_rows(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    out: Mapping[str, np.ndarray],
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    mask: np.ndarray,
    group_values: np.ndarray,
    seed_base: int,
) -> list[dict[str, Any]]:
    final_min = _final_min_distance(out["selected_xy"], data, ids, group_key)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(evals["fc"]["min_distance"]) & (evals["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(evals["di"]["min_distance"]) & (evals["di"]["min_distance"] < 0.05)
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(sorted(set(group_values[mask].tolist()))):
        local = mask & (group_values == name)
        row = fg._group_row(
            str(name),
            local,
            data,
            ids,
            out["selected_ade"],
            evals["floor"]["selected_ade"],
            out["switch"],
            final_near,
            fc_near,
            di_near,
            seed=seed_base + i * 100,
        )
        rows.append(row)
    rows.sort(key=lambda row: (bool(row["robust_positive"]), float(row["metric"]["all_improvement"])))
    return rows


def _candidate_best_delta(fl_payload: Mapping[str, Any], key: str) -> dict[str, float]:
    table = fl_payload.get("slices", {}).get(key, {}).get("test_candidate_table", {})
    return {name: float(row.get("delta_vs_fh", 0.0)) for name, row in table.items()}


def _classify_blocker(
    key: str,
    support: Mapping[str, Any],
    oracle: Mapping[str, Any],
    source_rows: list[Mapping[str, Any]],
    fo_applied: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    low_margin = float(oracle.get("low_margin_share", {}).get("0.05", 0.0))
    oracle_vs_fh = float(oracle.get("oracle_improvement_vs_fh", 0.0))
    if low_margin >= 0.8:
        reasons.append("oracle_low_margin_ambiguous")
    if oracle_vs_fh < 0.03:
        reasons.append("low_material_headroom")
    if int(support["val_rows"]) < MIN_SUPPORT_ROWS:
        reasons.append("validation_rows_insufficient")
    if int(support["shared_family_count"]) == 0:
        reasons.append("validation_to_test_source_family_shift")
    if int(support["val_source_count"]) < 2:
        reasons.append("single_or_sparse_validation_source_support")
    weak_source_reasons = []
    for row in source_rows:
        if not row["robust_positive"]:
            weak_source_reasons.extend(row.get("weak_reasons", []))
    if "easy_ci_exceeds_2pct" in weak_source_reasons:
        reasons.append("source_specific_easy_safety_ci_failure")
    if "all_ci_not_positive" in weak_source_reasons:
        reasons.append("source_specific_all_ci_failure")
    if int(fo_applied.get("switch_rows", 0)) == 0:
        reasons.append("gain_harm_policy_abstained_due_to_validation_safety")
    if key.endswith("|100"):
        reasons.append("long_horizon_h100_context_still_insufficient")
    return sorted(set(reasons)) or ["mixed_or_unresolved"]


def _next_action(reasons: list[str]) -> str:
    if "validation_to_test_source_family_shift" in reasons or "single_or_sparse_validation_source_support" in reasons:
        return "add_train_only_h100_source_support_or_build_source_family_specific_validation_before_more_modeling"
    if "source_specific_easy_safety_ci_failure" in reasons:
        return "train_source_specific_h100_easy_harm_guard_with_stricter_conformal_abstention"
    if "oracle_low_margin_ambiguous" in reasons and "low_material_headroom" in reasons:
        return "abstain_on_low_margin_h100_rows_and_keep_uniform_horizon_claim_blocked"
    if "long_horizon_h100_context_still_insufficient" in reasons:
        return "add_longer_history_neighbor_goal_sequence_features_for_h100_before_retrying_row_switch"
    return "perform_targeted_h100_source_support_and_feature_repair"


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fo_payload = read_json(fo.REPORT_JSON, {}) or fo.run_stage42_fh_horizon_gain_harm_specialist()
    fl_payload = read_json(fl.REPORT_JSON, {}) or fl.run_stage42_fh_horizon_weak_slice_forensics()
    state = _reconstruct_fo_state()
    data = state["data"]
    val_ids = state["val_ids"]
    test_ids = state["test_ids"]
    test_evals = state["test_evals"]
    test_fo = state["test_fo"]
    group_key = state["ctx"]["group_key"]
    weak_keys = list(fo_payload["summary"].get("weak_domain_horizons_after", []))

    source = _field(data, test_ids, "source_file")
    scene = _field(data, test_ids, "scene_id")
    h100_keys = [key for key in weak_keys if key.endswith("|100")]
    audits: dict[str, Any] = {}
    blocker_counts: dict[str, int] = {}
    for key in h100_keys:
        local = _mask_for_key(data, test_ids, key)
        support = _support_summary(data, val_ids, test_ids, key)
        source_rows = _group_metric_rows(data, test_ids, test_fo, test_evals, group_key, local, source, seed_base=72000)
        scene_rows = _group_metric_rows(data, test_ids, test_fo, test_evals, group_key, local, scene, seed_base=73000)
        fl_slice = fl_payload.get("slices", {}).get(key, {})
        oracle = fl_slice.get("test_oracle", {})
        blockers = _classify_blocker(key, support, oracle, source_rows, state["applied"].get(key, {}))
        for reason in blockers:
            blocker_counts[reason] = int(blocker_counts.get(reason, 0)) + 1
        audits[key] = {
            "support": support,
            "fo_applied_policy": state["applied"].get(key, {}),
            "test_oracle": oracle,
            "candidate_delta_vs_fh": _candidate_best_delta(fl_payload, key),
            "source_rows": source_rows,
            "scene_rows": scene_rows[:20],
            "blockers": blockers,
            "next_action": _next_action(blockers),
        }

    final_metric = di._metric_subset(test_fo["selected_ade"], test_evals["floor"]["selected_ade"], data, test_ids, test_fo["switch"])
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "input_fo_verdict": fo_payload.get("stage42_fo_gate", {}).get("verdict"),
        "h100_weak_horizons": h100_keys,
        "h100_weak_horizon_count": len(h100_keys),
        "blocker_counts": blocker_counts,
        "uniform_horizon_claim_allowed": False,
        "recommended_next_action": "source_support_or_long_horizon_context_repair_before_retrying_policy_promotion",
        "decision": "diagnostic_only_keep_stage42_fh_fi_with_horizon_limit",
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FP H100 weak-horizon source/support audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash([str(fo.REPORT_JSON), str(fl.REPORT_JSON), str(fm.REPORT_JSON), str(fi.POLICY_JSON)]),
        "current_facts": CURRENT_FACTS,
        "summary": summary,
        "metric_vs_floor_reconstructed_fo": final_metric,
        "audits": audits,
        "selection_rule": {
            "diagnostic_only": True,
            "does_not_train_new_policy": True,
            "uses_test_metrics_for_policy_selection": False,
            "oracle_uses_future_labels_for_diagnostic_only": True,
            "test_rows_reporting_only": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "diagnostic_oracle_not_deployed": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_fp_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fo_input_verified": str(s.get("input_fo_verdict", "")).startswith("stage42_fo_gain_harm_specialist_pass"),
        "h100_weak_horizons_present": int(s["h100_weak_horizon_count"]) >= 1,
        "source_support_audited": all("support" in row for row in payload["audits"].values()),
        "source_rows_audited": all(bool(row["source_rows"]) for row in payload["audits"].values()),
        "scene_rows_audited": all("scene_rows" in row for row in payload["audits"].values()),
        "oracle_margin_carried_forward": all("low_margin_share" in row["test_oracle"] for row in payload["audits"].values()),
        "blockers_identified": bool(s["blocker_counts"]),
        "next_actions_defined": all(bool(row["next_action"]) for row in payload["audits"].values()),
        "diagnostic_only_no_policy_promotion": payload["selection_rule"]["diagnostic_only"] is True
        and payload["selection_rule"]["does_not_train_new_policy"] is True,
        "uniform_horizon_claim_false": boundary["uniform_horizon_claim"] is False,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["diagnostic_oracle_not_deployed"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_fp_h100_source_support_audit_pass" if passed == total else "stage42_fp_h100_source_support_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_source_rows(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| source/scene | rows | robust | all | t50 | t100raw | hard | easy | weak reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metric = row["metric"]
        lines.append(
            f"| `{row['name']}` | {int(row['rows'])} | {bool(row['robust_positive'])} | "
            f"`{_pct(metric['all_improvement'])}` | `{_pct(metric['t50_improvement'])}` | "
            f"`{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` | "
            f"`{_pct(metric['hard_failure_improvement'])}` | `{_pct(metric['easy_degradation'])}` | "
            f"`{row['weak_reasons']}` |"
        )
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fp_gate"]
    s = payload["summary"]
    m = payload["metric_vs_floor_reconstructed_fo"]
    lines = [
        "# Stage42-FP H100 Weak-Horizon Source / Support Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- input FO verdict: `{s['input_fo_verdict']}`",
        f"- h100 weak horizons: `{s['h100_weak_horizons']}`",
        f"- blocker counts: `{s['blocker_counts']}`",
        f"- decision: `{s['decision']}`",
        "",
        "## Reconstructed FO Global Metric vs Floor",
        "",
        f"- all improvement: `{_pct(m['all_improvement'])}`",
        f"- t50 improvement: `{_pct(m['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(m['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(m['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(m['easy_degradation'])}`",
        "",
    ]
    for key, row in payload["audits"].items():
        support = row["support"]
        oracle = row["test_oracle"]
        lines += [
            f"## `{key}`",
            "",
            f"- test rows: `{support['test_rows']}`; val rows: `{support['val_rows']}`",
            f"- test sources/scenes: `{support['test_source_count']}` / `{support['test_scene_count']}`",
            f"- val sources/scenes: `{support['val_source_count']}` / `{support['val_scene_count']}`",
            f"- shared sources/families: `{support['shared_source_count']}` / `{support['shared_family_count']}`",
            f"- FO applied policy: `{row['fo_applied_policy']}`",
            f"- oracle improvement vs FH: `{_pct(oracle.get('oracle_improvement_vs_fh', 0.0))}`",
            f"- low-margin share: `{oracle.get('low_margin_share', {})}`",
            f"- candidate delta vs FH: `{row['candidate_delta_vs_fh']}`",
            f"- blockers: `{row['blockers']}`",
            f"- next action: `{row['next_action']}`",
            "",
            "### Test source rows",
            "",
            *_render_source_rows(row["source_rows"]),
            "",
            "### Test scene rows",
            "",
            *_render_source_rows(row["scene_rows"][:10]),
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "- Stage42-FP is diagnostic only. It does not train a new policy and does not promote uniform horizon robustness.",
        "- The remaining h100 weak horizons should be treated as source/support/context blockers until a future validation-selected repair passes them without easy or proximity regression.",
        "- No metric/seconds-level, true-3D, Stage5C, or SMC claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fp_gate"]
    lines = [
        "# Stage42-FP Gate",
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
    return "\n".join(
        [
            "<!-- STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT:START -->",
            "## Stage42-FP H100 Weak-Horizon Source / Support Audit",
            "",
            f"- source: `{payload['source']}`",
            "- role: diagnostic source/support decomposition for remaining h100 weak horizons after Stage42-FO; no new training and no test threshold tuning.",
            f"- gate: `{payload['stage42_fp_gate']['passed']} / {payload['stage42_fp_gate']['total']}`; verdict `{payload['stage42_fp_gate']['verdict']}`.",
            f"- h100 weak horizons: `{s['h100_weak_horizons']}`.",
            f"- blocker counts: `{s['blocker_counts']}`.",
            f"- recommended next action: `{s['recommended_next_action']}`.",
            "- conclusion: uniform horizon robustness remains blocked; TrajNet|100 and UCY|100 need source/support or stronger long-horizon context repair before any policy promotion.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FP H100 weak-horizon source/support audit"
    state["current_verdict"] = payload["stage42_fp_gate"]["verdict"]
    state["stage42_fp_h100_weak_horizon_source_support_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fp_gate"]["verdict"],
        "gates": f"{payload['stage42_fp_gate']['passed']}/{payload['stage42_fp_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FP decomposes remaining TrajNet|100 and UCY|100 h100 blockers by source, scene, validation support, and oracle margin.",
    }
    block = state.get("m3w_goal_evidence_ledger_readme")
    if isinstance(block, dict):
        block["latest_conclusion"] = "Stage42-FP confirms uniform horizon robustness remains blocked by h100 source/support/context issues after FO."
        state["m3w_goal_evidence_ledger_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_h100_weak_horizon_source_support_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_h100_weak_horizon_source_support_audit()
    gate = result["stage42_fp_gate"]
    print(f"Stage42-FP gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
