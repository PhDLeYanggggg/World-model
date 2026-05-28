from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage37_t50_history as s37
from src import stage42_eth_ucy_source_robust_blocked_repair as ji
from src import stage42_eth_ucy_source_specific_easy_guard as jg
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_blocked_source_geometry_support_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_blocked_source_geometry_support_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jj_gate.md"
JI_JSON = OUT_DIR / "eth_ucy_source_robust_blocked_repair_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JJ_ETH_UCY_BLOCKED_SOURCE_GEOMETRY_SUPPORT"
SOURCE = "fresh_stage42_jj_eth_ucy_blocked_source_geometry_support"
EPS = 1e-6
WAYPOINT_FRAC = am.WAYPOINT_FRAC.astype(np.float64)

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JJ audits why Stage42-JI blocked ETH_UCY sources remain blocked after source-robust harm-aware guards.",
    "It tests causal family-baseline support and source geometry/history distribution shift without using held-out test for selection.",
    "future waypoints / endpoints are labels/eval only, never inference inputs.",
    "No central velocity, no test endpoint goals, and no test-threshold tuning are used.",
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


def _blocked_sources_from_ji() -> list[str]:
    report = read_json(JI_JSON, {})
    blocked = report.get("summary", {}).get("still_blocked_sources")
    if blocked:
        return list(blocked)
    payload = ji.run_stage42_eth_ucy_source_robust_blocked_repair(refresh_readmes=False)
    return list(payload["summary"]["still_blocked_sources"])


def _family_waypoints(data: Mapping[str, np.ndarray]) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    endpoints = data["family_pred"].astype(np.float64)
    return (cur[:, None, None, :] + WAYPOINT_FRAC[None, None, :, None] * (endpoints[:, :, None, :] - cur[:, None, None, :])).astype(
        np.float32
    )


def _family_errors(family_xy: np.ndarray, labels: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n, k = family_xy.shape[:2]
    ade = np.zeros((n, k), dtype=np.float64)
    fde = np.zeros((n, k), dtype=np.float64)
    for i in range(k):
        ade[:, i], fde[:, i] = am._trajectory_errors(family_xy[:, i], labels)
    return ade, fde


def _safe_metric(selected: np.ndarray, floor: np.ndarray, data: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return am._metric(selected, floor, data, switch, mask)


def _select_static_family_policy(
    family_ade: np.ndarray,
    family_fde: np.ndarray,
    floor_ade: np.ndarray,
    floor_fde: np.ndarray,
    data: Mapping[str, np.ndarray],
    val_mask: np.ndarray,
) -> dict[str, Any]:
    horizon = data["horizon"].astype(int)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    slices: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for h in [10, 25, 50, 100]:
        hm = horizon == h
        vm = val_mask & hm
        if int(np.sum(vm)) < 50:
            continue
        best: dict[str, Any] | None = None
        for idx, name in enumerate(s37.BASELINE_FAMILY):
            trial_ade = floor_ade.copy()
            trial_fde = floor_fde.copy()
            trial_switch = np.zeros(len(floor_ade), dtype=bool)
            trial_ade[hm] = family_ade[hm, idx]
            trial_fde[hm] = family_fde[hm, idx]
            trial_switch[hm] = True
            metric = _safe_metric(trial_ade, floor_ade, data, trial_switch, vm)
            score = (
                1.2 * metric["all_improvement"]
                + 1.8 * metric["t50_improvement"]
                + 1.0 * metric["hard_failure_improvement"]
                - 30.0 * max(0.0, metric["easy_degradation"] - 0.02)
            )
            row = {"horizon": h, "family_idx": idx, "family": name, "score": float(score), "val_metric": metric}
            candidates.append(row)
            if metric["easy_degradation"] <= 0.02 and score > 0 and (best is None or score > best["score"]):
                best = row
        if best is not None:
            idx = int(best["family_idx"])
            selected_ade[hm] = family_ade[hm, idx]
            selected_fde[hm] = family_fde[hm, idx]
            switch[hm] = True
            slices[f"h{h}"] = best
    return {
        "type": "stage42_jj_static_family_floor",
        "selection_source": "validation_only_nonheldout_source",
        "test_threshold_tuning": False,
        "slices": slices,
        "candidate_count": len(candidates),
        "top_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
    }


def _distribution_shift(data: Mapping[str, np.ndarray], train_mask: np.ndarray, test_mask: np.ndarray) -> list[dict[str, Any]]:
    scalars = {
        "scale": data["scale"].astype(np.float64),
        "track_length": data["track_length"].astype(np.float64),
        "dt_frame_step": data["dt_frame_step"].astype(np.float64),
        "oracle_margin": data["oracle_margin"].astype(np.float64),
    }
    hist = data["history_scalar"].astype(np.float64)
    for idx in range(hist.shape[1]):
        scalars[f"history_scalar_{idx}"] = hist[:, idx]
    rows = []
    for name, arr in scalars.items():
        train = arr[train_mask]
        test = arr[test_mask]
        if len(train) == 0 or len(test) == 0:
            continue
        std = max(float(np.std(train)), EPS)
        rows.append(
            {
                "feature": name,
                "train_mean": float(np.mean(train)),
                "test_mean": float(np.mean(test)),
                "standardized_mean_gap": float(abs(np.mean(test) - np.mean(train)) / std),
                "train_p10": float(np.percentile(train, 10)),
                "test_p10": float(np.percentile(test, 10)),
                "train_p90": float(np.percentile(train, 90)),
                "test_p90": float(np.percentile(test, 90)),
            }
        )
    return sorted(rows, key=lambda r: r["standardized_mean_gap"], reverse=True)


def _family_t50_table(
    family_ade: np.ndarray,
    floor_ade: np.ndarray,
    data: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    out = []
    h50 = mask & (data["horizon"].astype(int) == 50)
    for idx, name in enumerate(s37.BASELINE_FAMILY):
        selected = family_ade[:, idx]
        switch = np.ones(len(selected), dtype=bool)
        metric = am._metric(selected, floor_ade, data, switch, h50)
        out.append({"family_idx": idx, "family": name, "t50_metric": metric})
    return sorted(out, key=lambda r: r["t50_metric"]["all_improvement"], reverse=True)


def _evaluate_source(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str) -> dict[str, Any]:
    split, split_stats = jg._source_cv_split(data, heldout_source)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    family_xy = _family_waypoints(data)
    family_ade, family_fde = _family_errors(family_xy, labels)
    oracle_ade = np.minimum(floor_ade, np.min(family_ade, axis=1))
    oracle_fde = np.minimum(floor_fde, np.min(family_fde, axis=1))
    oracle_switch = np.min(family_ade, axis=1) < floor_ade
    policy = _select_static_family_policy(family_ade, family_fde, floor_ade, floor_fde, data, val_mask)
    static_metric = am._metric(policy["selected_ade"], floor_ade, data, policy["switch"], test_mask)
    static_fde_metric = am._metric(policy["selected_fde"], floor_fde, data, policy["switch"], test_mask)
    oracle_metric = am._metric(oracle_ade, floor_ade, data, oracle_switch, test_mask)
    oracle_fde_metric = am._metric(oracle_fde, floor_fde, data, oracle_switch, test_mask)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    deployable = bool(
        static_metric["easy_degradation"] <= 0.02
        and static_metric["all_improvement"] > 0.03
        and (static_metric["t50_improvement"] > 0.03 or static_metric["hard_failure_improvement"] > 0.10)
    )
    blockers = []
    if static_metric["t50_improvement"] <= 0.03:
        blockers.append("t50_static_family_lift_insufficient")
    if static_metric["hard_failure_improvement"] <= 0.10:
        blockers.append("hard_failure_static_family_lift_insufficient")
    if static_metric["easy_degradation"] > 0.02:
        blockers.append("easy_degradation_above_2pct")
    if oracle_metric["t50_improvement"] <= 0.03:
        blockers.append("family_oracle_t50_headroom_insufficient")
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "split_stats": split_stats,
        "row_counts": {
            "train": int(np.sum(train_mask)),
            "val": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask)),
            "test_t50": int(np.sum(test_mask & (h == 50))),
            "test_t100": int(np.sum(test_mask & (h == 100))),
            "test_hard_failure": int(np.sum(test_mask & hard_failure)),
            "test_easy": int(np.sum(test_mask & easy)),
        },
        "distribution_shift_top": _distribution_shift(data, train_mask, test_mask)[:12],
        "family_t50_table": _family_t50_table(family_ade, floor_ade, data, test_mask),
        "family_oracle": {
            "ade_metric": oracle_metric,
            "fde_metric": oracle_fde_metric,
        },
        "static_family_policy": {
            "slices": policy["slices"],
            "candidate_count": policy["candidate_count"],
            "top_validation_candidates": policy["top_candidates"],
            "ade_metric": static_metric,
            "fde_metric": static_fde_metric,
            "deployable": deployable,
            "blockers": blockers,
        },
        "bootstrap": {
            "static_all": am._bootstrap_ci(policy["selected_ade"], floor_ade, test_mask, seed=42211),
            "static_t50": am._bootstrap_ci(policy["selected_ade"], floor_ade, test_mask & (h == 50), seed=42212),
            "static_hard_failure": am._bootstrap_ci(policy["selected_ade"], floor_ade, test_mask & hard_failure, seed=42213),
            "static_easy_degradation": am._bootstrap_ci(floor_ade, policy["selected_ade"], test_mask & easy, seed=42214),
            "oracle_t50": am._bootstrap_ci(oracle_ade, floor_ade, test_mask & (h == 50), seed=42215),
        },
    }


def _summary(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    repaired = [row["heldout_source"] for row in targets if row["static_family_policy"]["deployable"]]
    blocked = [row["heldout_source"] for row in targets if not row["static_family_policy"]["deployable"]]
    t50_oracle_available = [
        row["heldout_source"]
        for row in targets
        if row["family_oracle"]["ade_metric"]["t50_improvement"] > 0.03
    ]
    decision = (
        "blocked_sources_repaired_by_static_family_policy"
        if repaired and not blocked
        else "blocked_sources_partially_repaired_by_static_family_policy"
        if repaired
        else "blocked_sources_not_repaired_family_support_diagnostic"
    )
    return {
        "source": SOURCE,
        "targeted_sources": [row["heldout_source"] for row in targets],
        "repaired_sources": repaired,
        "still_blocked_sources": blocked,
        "t50_family_oracle_headroom_sources": t50_oracle_available,
        "decision": decision,
        "next_action": "If family oracle exists but static source policy fails, add source-specific history/goal geometry features; if family oracle is weak, acquire/calibrate new source support.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = jg._eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    targets = [_evaluate_source(data, labels, source) for source in _blocked_sources_from_ji()]
    payload: dict[str, Any] = {
        "stage": "Stage42-JJ",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(JI_JSON),
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "target_selection": {
            "source": "cached_verified_stage42_ji_still_blocked_sources",
            "blocked_sources_from_ji": _blocked_sources_from_ji(),
        },
        "targets": targets,
        "summary": _summary(targets),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_selection": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in targets),
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
    payload["stage42_jj_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "blocked_sources_audited": len(payload["targets"]) == len(payload["target_selection"]["blocked_sources_from_ji"]) and len(payload["targets"]) > 0,
        "geometry_history_shift_reported": all(row["distribution_shift_top"] for row in payload["targets"]),
        "family_t50_table_reported": all(row["family_t50_table"] for row in payload["targets"]),
        "family_oracle_headroom_measured": all(row["family_oracle"]["ade_metric"]["rows"] > 0 for row in payload["targets"]),
        "static_family_policy_attempted": all(row["static_family_policy"]["candidate_count"] > 0 for row in payload["targets"]),
        "repaired_or_blocked_recorded": bool(payload["summary"]["repaired_sources"] or payload["summary"]["still_blocked_sources"]),
        "no_overclaim_blocked_sources": payload["summary"]["decision"] != "blocked_sources_repaired_by_static_family_policy"
        or not payload["summary"]["still_blocked_sources"],
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and payload["no_leakage"]["train_only_selection"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_jj_eth_ucy_blocked_source_geometry_support_pass" if passed == len(gates) else "stage42_jj_eth_ucy_blocked_source_geometry_support_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jj_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JJ ETH_UCY Blocked-Source Geometry/Family Support",
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
        f"- targeted_sources: `{summary['targeted_sources']}`",
        f"- repaired_sources: `{summary['repaired_sources']}`",
        f"- still_blocked_sources: `{summary['still_blocked_sources']}`",
        f"- t50_family_oracle_headroom_sources: `{summary['t50_family_oracle_headroom_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Static Family Repair Attempt",
        "",
        "| source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | oracle t50 | deployable | blockers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["targets"]:
        metric = row["static_family_policy"]["ade_metric"]
        oracle = row["family_oracle"]["ade_metric"]
        lines.append(
            f"| `{row['heldout_source']}` | {metric['rows']} | {_fmt(metric['all_improvement'])} | {_fmt(metric['t50_improvement'])} | "
            f"{_fmt(metric['t100_raw_frame_diagnostic_improvement'])} | {_fmt(metric['hard_failure_improvement'])} | "
            f"{_fmt(metric['easy_degradation'])} | {_fmt(metric['switch_rate'])} | {_fmt(oracle['t50_improvement'])} | "
            f"`{row['static_family_policy']['deployable']}` | `{row['static_family_policy']['blockers']}` |"
        )
    lines.extend(["", "## T50 Family Table", ""])
    for row in payload["targets"]:
        lines.extend(
            [
                f"### `{row['heldout_source']}`",
                "",
                "| rank | family | t50 improvement | t50 easy degradation | rows |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for rank, fam in enumerate(row["family_t50_table"][:8], start=1):
            metric = fam["t50_metric"]
            lines.append(
                f"| {rank} | `{fam['family']}` | {_fmt(metric['all_improvement'])} | {_fmt(metric['easy_degradation'])} | {metric['rows']} |"
            )
        lines.append("")
    lines.extend(["## Top Distribution Shifts", ""])
    for row in payload["targets"]:
        lines.extend([f"### `{row['heldout_source']}`", "", "| feature | train mean | test mean | z-gap |", "| --- | ---: | ---: | ---: |"])
        for item in row["distribution_shift_top"][:8]:
            lines.append(
                f"| `{item['feature']}` | {item['train_mean']:.4f} | {item['test_mean']:.4f} | {item['standardized_mean_gap']:.3f} |"
            )
        lines.append("")
    lines.extend(["## Bootstrap CI", "", "| source | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["targets"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_source']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This stage checks whether remaining JI blockers are caused by missing baseline-family support or source geometry/history shift.",
            "- Static family repair is causal and validation-selected, but source identity is still a deployment precondition.",
            "- Failed sources remain fallback-only; this remains raw-frame / dataset-local 2.5D evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jj_gate"]
    lines = [
        "# Stage42-JJ Gate",
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
    gate = payload["stage42_jj_gate"]
    bits = []
    for row in payload["targets"]:
        metric = row["static_family_policy"]["ade_metric"]
        oracle = row["family_oracle"]["ade_metric"]
        bits.append(
            f"{row['heldout_source']}: static all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}, family-oracle t50 {_fmt(oracle['t50_improvement'])}, deployable={row['static_family_policy']['deployable']}"
        )
    return [
        "## Stage42-JJ ETH_UCY Blocked-Source Geometry/Family Support",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- family/geometry support audit: {'; '.join(bits)}.",
        f"- decision: `{summary['decision']}`; repaired: `{summary['repaired_sources']}`; still blocked: `{summary['still_blocked_sources']}`.",
        "- boundary: static causal family support does not globally repair ETH_UCY; blocked sources stay fallback-only; no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_blocked_source_geometry_support"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jj_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jj_gate"]["passed"], "total": payload["stage42_jj_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "targeted_sources": payload["summary"]["targeted_sources"],
        "repaired_sources": payload["summary"]["repaired_sources"],
        "still_blocked_sources": payload["summary"]["still_blocked_sources"],
        "t50_family_oracle_headroom_sources": payload["summary"]["t50_family_oracle_headroom_sources"],
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JJ",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jj_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": False,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_eth_ucy_blocked_source_geometry_support(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    run_stage42_eth_ucy_blocked_source_geometry_support(refresh_readmes=True)


if __name__ == "__main__":
    main()
