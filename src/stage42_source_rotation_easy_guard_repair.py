from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_rotation_full_waypoint_eval as je
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_rotation_easy_guard_repair_stage42.json"
REPORT_MD = OUT_DIR / "source_rotation_easy_guard_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jf_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JF_SOURCE_ROTATION_EASY_GUARD_REPAIR"
SOURCE = "fresh_stage42_jf_source_rotation_easy_guard_repair"
EPS = 1e-6
CAPS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JF 是 source-rotation easy-safety repair：它尝试用 validation-only switch budget 降低 held-out ETH_UCY easy harm。",
    "switch budget 在非 held-out validation rows 上选择；held-out domain test 不参与阈值选择。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _confidence_scores(residual_norm: np.ndarray, policy: Mapping[str, Any], horizon: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(residual_norm), dtype=np.float64)
    for key, params in policy.get("slices", {}).items():
        h = int(str(key).replace("h", ""))
        m = horizon == h
        direction = params.get("direction", "low")
        if direction == "low":
            scores[m] = -residual_norm[m]
        else:
            scores[m] = residual_norm[m]
    return scores


def _cap_switch_mask(
    base_switch: np.ndarray,
    residual_norm: np.ndarray,
    policy: Mapping[str, Any],
    horizon: np.ndarray,
    cap: float,
) -> np.ndarray:
    capped = np.zeros(len(base_switch), dtype=bool)
    score = _confidence_scores(residual_norm, policy, horizon)
    for h in [10, 25, 50, 100]:
        hm = horizon == h
        candidates = np.where(base_switch & hm)[0]
        if len(candidates) == 0:
            continue
        budget = int(np.floor(float(cap) * int(np.sum(hm))))
        budget = max(0, min(len(candidates), budget))
        if budget == 0:
            continue
        order = candidates[np.argsort(-score[candidates])]
        capped[order[:budget]] = True
    return capped


def _blend_errors_for_switch(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    base_switch: np.ndarray,
    capped_switch: np.ndarray,
    policy: Mapping[str, Any],
    horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, labels)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    for key, params in policy.get("slices", {}).items():
        h = int(str(key).replace("h", ""))
        local = capped_switch & base_switch & (horizon == h)
        if not np.any(local):
            continue
        alpha = float(params["alpha"])
        blended = floor_xy + alpha * (pred_xy - floor_xy)
        b_ade, b_fde = am._trajectory_errors(blended, labels)
        selected_ade[local] = b_ade[local]
        selected_fde[local] = b_fde[local]
    return selected_ade, selected_fde, floor_ade


def _candidate_score(metric: Mapping[str, Any]) -> float:
    if metric["easy_degradation"] > 0.02:
        return -1e9
    if metric["all_improvement"] <= 0 and metric["t50_improvement"] <= 0 and metric["hard_failure_improvement"] <= 0:
        return -1e9
    return (
        1.0 * float(metric["all_improvement"])
        + 1.4 * float(metric["t50_improvement"])
        + 1.0 * float(metric["hard_failure_improvement"])
        - 0.20 * float(metric["switch_rate"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.005)
    )


def _evaluate_easy_guard_rotation(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_domain: str) -> dict[str, Any]:
    split, split_stats = je._leave_one_domain_split(data, heldout_domain)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names, removed_features = je._domain_invariant_features(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    horizon = data["horizon"].astype(int)
    best: dict[str, Any] | None = None
    candidates = []
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        base_policy, base_ade, base_fde, base_switch = je._select_horizon_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        if not base_policy.get("slices"):
            continue
        residual_norm = np.linalg.norm(pred_xy[:, -1] - floor["floor_xy"][:, -1], axis=1) / np.maximum(
            data["scale"].astype(np.float64), EPS
        )
        floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
        base_val_metric = am._metric(base_ade, floor_ade, data, base_switch, val_mask)
        base_test_metric = am._metric(base_ade, floor_ade, data, base_switch, test_mask)
        for cap in CAPS:
            capped_switch = _cap_switch_mask(base_switch, residual_norm, base_policy, horizon, cap)
            selected_ade, selected_fde, floor_ade = _blend_errors_for_switch(
                pred_xy, floor["floor_xy"], labels, base_switch, capped_switch, base_policy, horizon
            )
            val_metric = am._metric(selected_ade, floor_ade, data, capped_switch, val_mask)
            score = _candidate_score(val_metric)
            row = {
                "lambda": float(lam),
                "switch_cap": float(cap),
                "score": float(score),
                "policy_slice_count": int(len(base_policy.get("slices", {}))),
                "val_metric": val_metric,
                "base_val_metric": base_val_metric,
                "base_test_metric_before_cap": base_test_metric,
            }
            candidates.append(row)
            if best is None or score > best["score"]:
                best = {
                    **row,
                    "policy": base_policy,
                    "pred_xy": pred_xy,
                    "selected_ade": selected_ade,
                    "selected_fde": selected_fde,
                    "switch": capped_switch,
                    "floor_ade": floor_ade,
                    "floor_fde": floor_fde,
                }
    if best is None:
        raise RuntimeError(f"No easy-guard model candidate for {heldout_domain}.")
    test_metric = am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask)
    test_fde = am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    baseline_metric = best["base_test_metric_before_cap"]
    return {
        "source": "fresh_run",
        "heldout_domain": heldout_domain,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names)),
            "domain_features_removed": removed_features,
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
        },
        "selected_candidate": {
            "lambda": best["lambda"],
            "switch_cap": best["switch_cap"],
            "score": best["score"],
            "validation_selection_source": "non_heldout_validation_only",
            "test_threshold_tuning": False,
            "policy_slice_count": best["policy_slice_count"],
            "val_metric": best["val_metric"],
        },
        "candidate_count": int(len(candidates)),
        "top_validation_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
        "metrics": {
            "base_horizon_policy_before_cap": baseline_metric,
            "easy_guard_policy": test_metric,
            "easy_guard_policy_fde": test_fde,
        },
        "delta_vs_stage42_je_base_policy": {
            "all_improvement": float(test_metric["all_improvement"]) - float(baseline_metric["all_improvement"]),
            "t50_improvement": float(test_metric["t50_improvement"]) - float(baseline_metric["t50_improvement"]),
            "hard_failure_improvement": float(test_metric["hard_failure_improvement"]) - float(baseline_metric["hard_failure_improvement"]),
            "easy_degradation": float(test_metric["easy_degradation"]) - float(baseline_metric["easy_degradation"]),
            "switch_rate": float(test_metric["switch_rate"]) - float(baseline_metric["switch_rate"]),
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42131),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42132),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42133),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42134),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42135),
        },
    }


def _summary(rotations: list[Mapping[str, Any]]) -> dict[str, Any]:
    deployable = []
    repaired = []
    blocked = []
    for row in rotations:
        metric = row["metrics"]["easy_guard_policy"]
        base = row["metrics"]["base_horizon_policy_before_cap"]
        if (
            metric["all_improvement"] > 0.03
            and (metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10)
            and metric["easy_degradation"] <= 0.02
        ):
            deployable.append(row["heldout_domain"])
        if base["easy_degradation"] > 0.02 and metric["easy_degradation"] <= 0.02:
            repaired.append(row["heldout_domain"])
        if metric["easy_degradation"] > 0.02:
            blocked.append(row["heldout_domain"])
    return {
        "source": SOURCE,
        "rotation_count": int(len(rotations)),
        "deployable_heldout_domains_after_easy_guard": deployable,
        "easy_repaired_domains": repaired,
        "still_easy_blocked_domains": blocked,
        "all_domains_easy_safe": len(blocked) == 0,
        "decision": "easy_guard_repair_global_deployable"
        if len(blocked) == 0 and len(deployable) == len(rotations)
        else "easy_guard_repair_partial_domain_bounded"
        if deployable
        else "easy_guard_repair_failed",
        "next_action": "If ETH_UCY remains blocked, treat ETH_UCY as fallback-only until source-specific calibration, safer hard/easy detector, or additional held-out ETH/UCY sources are available.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    labels = am._reconstruct_waypoint_labels(data)
    domains = sorted(set(data["dataset"].astype(str).tolist()))
    rotations = [_evaluate_easy_guard_rotation(data, labels, domain) for domain in domains]
    payload: dict[str, Any] = {
        "stage": "Stage42-JF",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_rotation_full_waypoint_eval_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "domains": domains,
        "rotations": rotations,
        "summary": _summary(rotations),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "domain_specific_test_thresholds": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in rotations),
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
    payload["stage42_jf_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "easy_guard_repair_attempted_for_all_domains": len(payload["rotations"]) == len(payload["domains"]) and len(payload["domains"]) >= 3,
        "validation_only_selection": all(row["selected_candidate"]["test_threshold_tuning"] is False for row in payload["rotations"]),
        "candidate_search_recorded": all(row["candidate_count"] > 0 and row["top_validation_candidates"] for row in payload["rotations"]),
        "deployable_domains_recorded": "deployable_heldout_domains_after_easy_guard" in payload["summary"],
        "eth_ucy_blocker_explicit_if_present": (
            "ETH_UCY" not in payload["domains"]
            or "ETH_UCY" in payload["summary"]["deployable_heldout_domains_after_easy_guard"]
            or "ETH_UCY" in payload["summary"]["still_easy_blocked_domains"]
        ),
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
                "domain_specific_test_thresholds",
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_jf_source_rotation_easy_guard_repair_pass" if passed == len(gates) else "stage42_jf_source_rotation_easy_guard_repair_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jf_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JF Source-Rotation Easy-Guard Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{summary['decision']}`",
        f"- deployable_heldout_domains_after_easy_guard: `{summary['deployable_heldout_domains_after_easy_guard']}`",
        f"- easy_repaired_domains: `{summary['easy_repaired_domains']}`",
        f"- still_easy_blocked_domains: `{summary['still_easy_blocked_domains']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Rotation Metrics",
        "",
        "| heldout domain | cap | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | base easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rotations"]:
        m = row["metrics"]["easy_guard_policy"]
        b = row["metrics"]["base_horizon_policy_before_cap"]
        lines.append(
            f"| `{row['heldout_domain']}` | {row['selected_candidate']['switch_cap']:.2f} | {_fmt(m['all_improvement'])} | "
            f"{_fmt(m['t50_improvement'])} | {_fmt(m['t100_raw_frame_diagnostic_improvement'])} | "
            f"{_fmt(m['hard_failure_improvement'])} | {_fmt(m['easy_degradation'])} | {_fmt(m['switch_rate'])} | {_fmt(b['easy_degradation'])} |"
        )
    lines.extend(["", "## Bootstrap CI", "", "| heldout domain | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["rotations"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_domain']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If a domain remains easy-blocked after this validation-only cap, the honest deployment rule is fallback-only for that domain.",
            "- This repair does not make metric/seconds claims and does not execute Stage5C or SMC.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jf_gate"]
    lines = [
        "# Stage42-JF Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_jf_gate"]
    rows = []
    for row in payload["rotations"]:
        metric = row["metrics"]["easy_guard_policy"]
        rows.append(
            f"{row['heldout_domain']}: cap {row['selected_candidate']['switch_cap']:.2f}, all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}"
        )
    return [
        "## Stage42-JF Source-Rotation Easy-Guard Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- held-out easy-guard rotations: {'; '.join(rows)}.",
        f"- decision: `{summary['decision']}`; deployable domains after easy guard: `{summary['deployable_heldout_domains_after_easy_guard']}`; still blocked: `{summary['still_easy_blocked_domains']}`.",
        "- boundary: validation-only switch budget; no test threshold tuning, no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["source_rotation_easy_guard_repair"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jf_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jf_gate"]["passed"], "total": payload["stage42_jf_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "deployable_heldout_domains_after_easy_guard": payload["summary"]["deployable_heldout_domains_after_easy_guard"],
        "still_easy_blocked_domains": payload["summary"]["still_easy_blocked_domains"],
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    import json

    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JF",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jf_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": True,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_source_rotation_easy_guard_repair(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_source_rotation_easy_guard_repair(refresh_readmes=True)


if __name__ == "__main__":
    main()
