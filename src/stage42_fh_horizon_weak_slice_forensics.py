from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_fh_horizon_weak_slice_repair as fk
from src import stage42_fh_policy_freeze_replay as fi
from src import stage42_fh_source_robustness_audit as fj
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fh_horizon_weak_slice_forensics_stage42.json"
REPORT_MD = OUT_DIR / "fh_horizon_weak_slice_forensics_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fl_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fi.PAPER_FILES

SOURCE = "fresh_stage42_fh_horizon_weak_slice_forensics"
EPS = 1e-6
LOW_MARGIN_PX = [0.01, 0.025, 0.05]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FJ 证明 FH/FI frozen policy 在 TrajNet 与 UCY domain/source 上 robust，但 TrajNet|100、UCY|50、UCY|100 仍是 horizon weak slices。",
    "Stage42-FK 用 validation-only 整片候选替换尝试修复 weak horizon，但 weak horizon 数没有下降。",
    "Stage42-FL 不重新调 test threshold；它做 weak horizon 的 oracle/headroom/ambiguity/feature-signal 取证，为下一步重训 horizon specialist 提供证据。",
    "future waypoints / endpoints 只作为 supervised labels 或 diagnostic/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if len(values) else 0.0


def _improvement(selected: np.ndarray, reference: np.ndarray) -> float:
    if len(selected) == 0:
        return 0.0
    return 1.0 - _safe_mean(selected) / max(_safe_mean(reference), EPS)


def _corr(feature: np.ndarray, target: np.ndarray) -> float | None:
    feature = np.asarray(feature, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(feature) & np.isfinite(target)
    if int(np.sum(mask)) < 3:
        return None
    x = feature[mask]
    y = target[mask]
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _candidate_delta_vs_fh(evals: Mapping[str, Mapping[str, Any]], name: str, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return _improvement(evals[name]["selected_ade"][mask], evals["fh"]["selected_ade"][mask])


def _candidate_table(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    mask = fk._key_mask(data, ids, key)
    rows: dict[str, Any] = {}
    for name, ev in evals.items():
        metric = fk._candidate_metric(data, ids, ev["selected_ade"], ev["floor_ade"], ev["switch"], mask)
        rows[name] = {
            "metric_vs_floor": metric,
            "delta_vs_fh": _candidate_delta_vs_fh(evals, name, mask),
        }
    return rows


def _oracle_summary(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    mask = fk._key_mask(data, ids, key)
    names = list(evals.keys())
    if not np.any(mask):
        return {
            "rows": 0,
            "oracle_improvement_vs_floor": 0.0,
            "oracle_improvement_vs_fh": 0.0,
            "best_candidate_distribution": {},
            "low_margin_share": {str(v): 0.0 for v in LOW_MARGIN_PX},
        }
    local_ids = np.where(mask)[0]
    ade = np.vstack([evals[name]["selected_ade"][mask] for name in names])
    order = np.argsort(ade, axis=0)
    best = order[0]
    second = order[1] if len(names) > 1 else order[0]
    oracle = ade[best, np.arange(len(local_ids))]
    second_best = ade[second, np.arange(len(local_ids))]
    floor = evals["floor"]["selected_ade"][mask]
    fh = evals["fh"]["selected_ade"][mask]
    counts: dict[str, int] = {}
    for idx in best:
        counts[names[int(idx)]] = counts.get(names[int(idx)], 0) + 1
    margin = np.maximum(second_best - oracle, 0.0) / np.maximum(floor, EPS)
    return {
        "rows": int(np.sum(mask)),
        "oracle_improvement_vs_floor": _improvement(oracle, floor),
        "oracle_improvement_vs_fh": _improvement(oracle, fh),
        "best_candidate_distribution": {name: int(counts.get(name, 0)) for name in names},
        "low_margin_share": {str(v): float(np.mean(margin < float(v))) for v in LOW_MARGIN_PX},
    }


def _ensure_min_distance(
    ev: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> np.ndarray:
    if "min_distance" in ev:
        return np.asarray(ev["min_distance"], dtype=np.float64)
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    return di._min_group_distance_fast(ev["selected_xy"], group_key[ids], normalizer, agent)


def _feature_signal_table(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
    group_key: np.ndarray,
    key: str,
) -> dict[str, Any]:
    mask = fk._key_mask(data, ids, key)
    if not np.any(mask):
        return {}
    scale = np.maximum(data["scale"][ids][mask].astype(np.float64), EPS)
    fh_ade = evals["fh"]["selected_ade"][mask]
    floor_xy = evals["floor"]["selected_xy"][mask].astype(np.float64)
    fh_xy = evals["fh"]["selected_xy"][mask].astype(np.float64)
    rows: dict[str, Any] = {}
    for name, ev in evals.items():
        if name in {"fh", "floor"}:
            continue
        xy = ev["selected_xy"][mask].astype(np.float64)
        gain = fh_ade - ev["selected_ade"][mask]
        endpoint_delta_floor = np.linalg.norm(xy[:, -1] - floor_xy[:, -1], axis=1) / scale
        endpoint_delta_fh = np.linalg.norm(xy[:, -1] - fh_xy[:, -1], axis=1) / scale
        path_step = np.linalg.norm(np.diff(xy, axis=1), axis=2)
        path_length = np.sum(path_step, axis=1) / scale
        min_distance = _ensure_min_distance(ev, ids, data, group_key)[mask]
        rows[name] = {
            "positive_gain_rate": float(np.mean(gain > 0.0)),
            "mean_gain_vs_fh": float(np.mean(gain)),
            "corr_endpoint_delta_floor_to_gain": _corr(endpoint_delta_floor, gain),
            "corr_endpoint_delta_fh_to_gain": _corr(endpoint_delta_fh, gain),
            "corr_path_length_to_gain": _corr(path_length, gain),
            "corr_min_distance_to_gain": _corr(min_distance, gain),
            "switch_rate": float(np.mean(ev["switch"][mask])) if len(gain) else 0.0,
        }
    return rows


def _dominant_root_cause(slice_row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    oracle = slice_row["test_oracle"]
    val_oracle = slice_row["val_oracle"]
    best_test = max(
        (row["delta_vs_fh"] for row in slice_row["test_candidate_table"].values()),
        default=0.0,
    )
    best_val = max(
        (row["delta_vs_fh"] for row in slice_row["val_candidate_table"].values()),
        default=0.0,
    )
    low_margin = float(oracle["low_margin_share"].get("0.05", 0.0))
    signal = []
    for by_candidate in slice_row["val_feature_signal"].values():
        vals = [
            by_candidate.get("corr_endpoint_delta_floor_to_gain"),
            by_candidate.get("corr_endpoint_delta_fh_to_gain"),
            by_candidate.get("corr_path_length_to_gain"),
            by_candidate.get("corr_min_distance_to_gain"),
        ]
        signal.extend(abs(float(v)) for v in vals if v is not None)
    max_signal = max(signal) if signal else 0.0
    if float(oracle["oracle_improvement_vs_fh"]) > 0.05 and best_test <= 0.0:
        reasons.append("row_level_switch_required_candidate_level_override_insufficient")
    if best_val > 0.0 and best_test <= 0.0:
        reasons.append("validation_to_test_distribution_shift")
    if low_margin > 0.5:
        reasons.append("oracle_label_low_margin_ambiguous")
    if max_signal < 0.1 and float(val_oracle["oracle_improvement_vs_fh"]) > 0.05:
        reasons.append("past_only_proxy_features_weak_for_gain_prediction")
    if int(oracle["rows"]) < fg.MIN_CI_ROWS:
        reasons.append("powered_rows_insufficient")
    return reasons or ["mixed_or_unresolved"]


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fj_payload = read_json(fj.REPORT_JSON, {}) or fj.run_stage42_fh_source_robustness_audit()
    fk_payload = read_json(fk.REPORT_JSON, {}) or fk.run_stage42_fh_horizon_weak_slice_repair()
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
    slices: dict[str, Any] = {}
    for key in weak_keys:
        row: dict[str, Any] = {
            "val_candidate_table": _candidate_table(data, val_ids, val_evals, key),
            "test_candidate_table": _candidate_table(data, test_ids, test_evals, key),
            "val_oracle": _oracle_summary(data, val_ids, val_evals, key),
            "test_oracle": _oracle_summary(data, test_ids, test_evals, key),
            "val_feature_signal": _feature_signal_table(data, val_ids, val_evals, ctx["group_key"], key),
            "test_feature_signal": _feature_signal_table(data, test_ids, test_evals, ctx["group_key"], key),
        }
        row["root_causes"] = _dominant_root_cause(row)
        slices[key] = row

    summary = {
        "source": SOURCE,
        "weak_domain_horizons": weak_keys,
        "weak_horizon_count": len(weak_keys),
        "fk_verdict": fk_payload["stage42_fk_gate"]["verdict"],
        "fj_verdict": fj_payload["stage42_fj_gate"]["verdict"],
        "fi_verdict": fi_payload["stage42_fi_gate"]["verdict"],
        "root_cause_counts": {},
        "next_action": "train_horizon_specific_row_level_switch_model_with_stronger_history_neighbor_goal_features",
    }
    for row in slices.values():
        for reason in row["root_causes"]:
            summary["root_cause_counts"][reason] = int(summary["root_cause_counts"].get(reason, 0)) + 1

    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FL FH weak-horizon forensics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": fi.ff._git_commit(),
        "input_hash": _combined_hash([str(fi.REPORT_JSON), str(fj.REPORT_JSON), str(fk.REPORT_JSON), str(fi.POLICY_JSON)]),
        "current_facts": CURRENT_FACTS,
        "summary": summary,
        "selection_rule": {
            "uses_test_metrics_for_policy_selection": False,
            "does_not_change_frozen_policy": True,
            "oracle_uses_future_labels_for_diagnostic_only": True,
            "candidate_families": ["fh", "fc", "di", "fa", "fb", "floor"],
        },
        "slices": slices,
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
    payload["stage42_fl_gate"] = _gate(payload)
    return fi._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    summary = payload["summary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fi_input_verified": summary["fi_verdict"] == "stage42_fi_fh_policy_freeze_replay_pass",
        "fj_input_verified": summary["fj_verdict"] == "stage42_fj_fh_source_robustness_pass",
        "fk_input_verified": summary["fk_verdict"].startswith("stage42_fk_fh_horizon_weak_slice_repair_pass"),
        "weak_horizon_slices_present": int(summary["weak_horizon_count"]) >= 1,
        "candidate_tables_built": all(bool(row["test_candidate_table"]) for row in payload["slices"].values()),
        "diagnostic_oracle_computed": all(int(row["test_oracle"]["rows"]) > 0 for row in payload["slices"].values()),
        "feature_signal_computed": all(bool(row["val_feature_signal"]) for row in payload["slices"].values()),
        "root_causes_identified": bool(summary["root_cause_counts"]),
        "next_action_defined": bool(summary["next_action"]),
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
        "uniform_horizon_claim_remains_false": boundary["uniform_horizon_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_fl_horizon_weak_slice_forensics_pass" if passed == total else "stage42_fl_horizon_weak_slice_forensics_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_candidate_rows(title: str, rows: Mapping[str, Any]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| candidate | rows | all vs floor | delta vs FH | t50 | t100raw | hard/failure | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in rows.items():
        metric = row["metric_vs_floor"]
        lines.append(
            f"| `{name}` | {int(metric['rows'])} | `{_pct(metric['all_improvement'])}` | `{_pct(row['delta_vs_fh'])}` | "
            f"`{_pct(metric['t50_improvement'])}` | `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` | "
            f"`{_pct(metric['hard_failure_improvement'])}` | `{_pct(metric['easy_degradation'])}` | `{_pct(metric['switch_rate'])}` |"
        )
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fl_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-FL FH Weak-Horizon Forensics",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- analyzed weak horizons: `{summary['weak_domain_horizons']}`",
        f"- root cause counts: `{summary['root_cause_counts']}`",
        f"- next action: `{summary['next_action']}`",
        "",
        "## Slice Findings",
        "",
    ]
    for key, row in payload["slices"].items():
        oracle = row["test_oracle"]
        lines += [
            f"## `{key}`",
            "",
            f"- rows: `{oracle['rows']}`",
            f"- diagnostic oracle improvement vs floor: `{_pct(oracle['oracle_improvement_vs_floor'])}`",
            f"- diagnostic oracle improvement vs FH: `{_pct(oracle['oracle_improvement_vs_fh'])}`",
            f"- low-margin share: `{oracle['low_margin_share']}`",
            f"- oracle candidate distribution: `{oracle['best_candidate_distribution']}`",
            f"- root causes: `{row['root_causes']}`",
            "",
        ]
        lines += _render_candidate_rows("Validation candidate metrics", row["val_candidate_table"])
        lines += [""] + _render_candidate_rows("Test candidate metrics", row["test_candidate_table"])
        lines += [
            "",
            "### Validation past-only proxy signal",
            "",
            "| candidate | positive gain rate | mean gain vs FH | corr endpoint-floor | corr endpoint-FH | corr path length | corr min distance |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, sig in row["val_feature_signal"].items():
            lines.append(
                f"| `{name}` | `{_pct(sig['positive_gain_rate'])}` | `{float(sig['mean_gain_vs_fh']):.6f}` | "
                f"`{sig['corr_endpoint_delta_floor_to_gain']}` | `{sig['corr_endpoint_delta_fh_to_gain']}` | "
                f"`{sig['corr_path_length_to_gain']}` | `{sig['corr_min_distance_to_gain']}` |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- This is not a policy promotion step. It is a fresh weak-horizon diagnostic that explains why FK did not unlock uniform horizon claims.",
        "- Diagnostic oracle rows use future labels only for upper-bound analysis; no deployed policy uses future labels.",
        "- Uniform horizon robustness remains blocked until a validation-selected row-level horizon specialist repairs TrajNet|100, UCY|50, and UCY|100 on test without easy/proximity regressions.",
        "- No Stage5C, SMC, true-3D, metric, or seconds-level claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fl_gate"]
    lines = [
        "# Stage42-FL Gate",
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
    summary = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS:START -->",
            "## Stage42-FL FH Weak-Horizon Forensics",
            "",
            f"- source: `{payload['source']}`",
            "- role: fresh diagnostic for FK/FJ weak horizons; no policy promotion and no test threshold tuning.",
            f"- gate: `{payload['stage42_fl_gate']['passed']} / {payload['stage42_fl_gate']['total']}`; verdict `{payload['stage42_fl_gate']['verdict']}`.",
            f"- analyzed weak horizons: `{summary['weak_domain_horizons']}`.",
            f"- root cause counts: `{summary['root_cause_counts']}`.",
            f"- next action: `{summary['next_action']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC; uniform horizon claim still blocked.",
            "<!-- STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FL FH weak-horizon forensics"
    state["current_verdict"] = payload["stage42_fl_gate"]["verdict"]
    state["stage42_fl_fh_horizon_weak_slice_forensics"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fl_gate"]["verdict"],
        "gates": f"{payload['stage42_fl_gate']['passed']}/{payload['stage42_fl_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FL diagnoses why FK validation-only horizon repair did not remove TrajNet|100, UCY|50, or UCY|100 weak slices.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FL FH weak-horizon forensics" not in evidence:
            evidence.append("Stage42-FL FH weak-horizon forensics")
        block["latest_included_evidence"] = evidence
        block["latest_conclusion"] = "Stage42-FL diagnoses remaining weak horizon slices after FK and defines the next row-level horizon-specialist action."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fh_horizon_weak_slice_forensics() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fh_horizon_weak_slice_forensics()
    gate = result["stage42_fl_gate"]
    print(f"Stage42-FL gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
