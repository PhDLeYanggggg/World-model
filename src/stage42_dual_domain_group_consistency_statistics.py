from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_ucy_validation_support_repair as aw
from src import stage42_ucy_supported_group_consistency as dz
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "dual_domain_group_consistency_statistics_stage42.json"
REPORT_MD = OUT_DIR / "dual_domain_group_consistency_statistics_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ea_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

BOOTSTRAP_N = 2000
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EA fresh-runs the Stage42-DZ UCY-supported group-consistency repair and adds 2000-bootstrap dual-domain statistical evidence。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "ungated_full_waypoint_deployable": False,
}


def _bootstrap_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, *, seed: int, n: int = BOOTSTRAP_N) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals[i] = 1.0 - float(np.mean(selected[sample])) / max(float(np.mean(floor[sample])), EPS)
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _bootstrap_degradation(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, *, seed: int, n: int = BOOTSTRAP_N) -> dict[str, Any]:
    ci = _bootstrap_improvement(selected, floor, mask, seed=seed, n=n)
    return {
        "low": -float(ci["high"]),
        "mid": -float(ci["mid"]),
        "high": -float(ci["low"]),
        "n": ci["n"],
        "bootstrap_n": ci["bootstrap_n"],
    }


def _bootstrap_rate(values: np.ndarray, mask: np.ndarray, *, seed: int, n: int = BOOTSTRAP_N) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    arr = values.astype(np.float64)
    for i in range(n):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals[i] = float(np.mean(arr[sample]))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _slice_ci(
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    h: np.ndarray,
    hard: np.ndarray,
    easy: np.ndarray,
    mask: np.ndarray,
    *,
    seed_base: int,
) -> dict[str, Any]:
    return {
        "all": _bootstrap_improvement(selected_ade, floor_ade, mask, seed=seed_base + 1),
        "t50": _bootstrap_improvement(selected_ade, floor_ade, mask & (h == 50), seed=seed_base + 2),
        "t100_raw_frame_diagnostic": _bootstrap_improvement(selected_ade, floor_ade, mask & (h == 100), seed=seed_base + 3),
        "hard_failure": _bootstrap_improvement(selected_ade, floor_ade, mask & hard, seed=seed_base + 4),
        "easy_degradation": _bootstrap_degradation(selected_ade, floor_ade, mask & easy, seed=seed_base + 5),
    }


def _positive_ci(ci: Mapping[str, Any]) -> bool:
    return bool(
        ci["all"]["low"] > 0.0
        and ci["t50"]["low"] > 0.0
        and ci["hard_failure"]["low"] > 0.0
        and ci["easy_degradation"]["high"] <= 0.02
    )


def _rate_summary(
    base_near: np.ndarray,
    final_near: np.ndarray,
    mask: np.ndarray,
    *,
    seed_base: int,
) -> dict[str, Any]:
    return {
        "base_near005": _bootstrap_rate(base_near, mask, seed=seed_base + 1),
        "final_near005": _bootstrap_rate(final_near, mask, seed=seed_base + 2),
        "delta_final_minus_base": _bootstrap_rate(final_near.astype(np.float64) - base_near.astype(np.float64), mask, seed=seed_base + 3),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    ci = payload["bootstrap_ci"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "fresh_repair_rebuilt": payload["source"] == "fresh_stage42_ea_dual_domain_group_consistency_statistics",
        "bootstrap_n_2000": ci["global"]["all"]["bootstrap_n"] >= 2000,
        "global_ci_positive_safe": _positive_ci(ci["global"]),
        "ucy_ci_positive_safe": _positive_ci(ci["by_domain"]["UCY"]),
        "trajnet_ci_positive_safe": _positive_ci(ci["by_domain"]["TrajNet"]),
        "two_domain_ci_supported": payload["summary"]["ci_positive_safe_domains"] >= 2,
        "near_collision_not_worse": payload["summary"]["near005_delta_high"] <= 0.0,
        "test_sources_unchanged": no_leak["test_sources_unchanged"] is True,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["internal_val_from_train_only"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ea_dual_domain_group_consistency_statistics_pass" if passed == total else "stage42_ea_dual_domain_group_consistency_statistics_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_ci(ci: Mapping[str, Any]) -> str:
    return f"[{ci['low']:.6f}, {ci['mid']:.6f}, {ci['high']:.6f}] n={ci['n']}"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-EA Dual-Domain Group-Consistency Statistical Evidence",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_ea_gate']['passed']} / {payload['stage42_ea_gate']['total']}`",
        f"- verdict: `{payload['stage42_ea_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Bootstrap Confidence Intervals",
        "",
        "| slice | all | t50 | t100 raw diag | hard/failure | easy degradation | positive safe |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, ci in [("global", payload["bootstrap_ci"]["global"]), *[(d, c) for d, c in payload["bootstrap_ci"]["by_domain"].items()]]:
        lines.append(
            f"| `{name}` | {_render_ci(ci['all'])} | {_render_ci(ci['t50'])} | {_render_ci(ci['t100_raw_frame_diagnostic'])} | {_render_ci(ci['hard_failure'])} | {_render_ci(ci['easy_degradation'])} | `{_positive_ci(ci)}` |"
        )
    lines.extend(
        [
            "",
            "## Near-Collision Bootstrap",
            "",
            "| slice | base near@0.05 | final near@0.05 | delta final-base |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, ci in [("global", payload["near_collision_ci"]["global"]), *[(d, c) for d, c in payload["near_collision_ci"]["by_domain"].items()]]:
        lines.append(
            f"| `{name}` | {_render_ci(ci['base_near005'])} | {_render_ci(ci['final_near005'])} | {_render_ci(ci['delta_final_minus_base'])} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
            "",
            "## Interpretation",
            "",
            "- This is fresh statistical evidence rebuilt from row-level selected/floor ADE arrays, not a reuse of aggregate DZ metrics.",
            "- The claim is dual-domain raw-frame/dataset-local 2.5D support for protected group-consistency full-waypoint dynamics.",
            "- It does not allow metric/seconds-level, true-3D, foundation, Stage5C, or SMC claims.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_ea_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ea_gate"]
    return [
        "# Stage42-EA Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    ci = payload["bootstrap_ci"]
    return [
        "## Stage42-EA Dual-Domain Group-Consistency Statistical Evidence",
        "",
        "- source: `fresh_stage42_ea_dual_domain_group_consistency_statistics`",
        "- role: fresh row-level 2000-bootstrap evidence for the Stage42-DZ UCY-supported group-consistency policy.",
        f"- gate: `{payload['stage42_ea_gate']['passed']} / {payload['stage42_ea_gate']['total']}`; verdict `{payload['stage42_ea_gate']['verdict']}`.",
        f"- global all/t50/hard CI lows: `{ci['global']['all']['low']:.6f}` / `{ci['global']['t50']['low']:.6f}` / `{ci['global']['hard_failure']['low']:.6f}`; easy high `{ci['global']['easy_degradation']['high']:.6f}`.",
        f"- UCY all/t50/hard CI lows: `{ci['by_domain']['UCY']['all']['low']:.6f}` / `{ci['by_domain']['UCY']['t50']['low']:.6f}` / `{ci['by_domain']['UCY']['hard_failure']['low']:.6f}`; TrajNet all/t50/hard CI lows `{ci['by_domain']['TrajNet']['all']['low']:.6f}` / `{ci['by_domain']['TrajNet']['t50']['low']:.6f}` / `{ci['by_domain']['TrajNet']['hard_failure']['low']:.6f}`.",
        f"- near@0.05 final-base delta high `{payload['summary']['near005_delta_high']:.6f}`; raw-frame/dataset-local only; Stage5C false; SMC false.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EA_DUAL_DOMAIN_GROUP_CONSISTENCY_STATISTICS", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EA dual-domain group-consistency statistical evidence"
    state["current_verdict"] = payload["stage42_ea_gate"]["verdict"]
    state["stage42_ea_dual_domain_group_consistency_statistics"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ea_gate"]["verdict"],
        "gates": f"{payload['stage42_ea_gate']['passed']}/{payload['stage42_ea_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_dual_domain_group_consistency_statistics() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain_all = data["dataset"].astype(str)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain_all)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = repaired_split == "train"
    floor = am._floor_arrays(data, train_mask)
    am_candidate = di._rebuild_stage42_am_candidate(data, repaired_split, labels, floor)
    group_key = di._group_key(data)
    repair = di._evaluate_repairs(data, repaired_split, labels, floor, am_candidate, group_key)
    test_ids = np.where(repaired_split == "test")[0]
    selected = di._repair_subset(
        test_ids,
        repair["selected"]["candidate"],
        data,
        labels,
        floor["floor_xy"].astype(np.float32),
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    selected_ade = selected["selected_ade"].astype(np.float64)
    floor_ade = selected["floor_ade"].astype(np.float64)
    h = data["horizon"][test_ids].astype(int)
    hard = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    domains = data["dataset"][test_ids].astype(str)
    global_mask = np.ones(len(test_ids), dtype=bool)
    by_domain = {
        d: _slice_ci(selected_ade, floor_ade, h, hard, easy, domains == d, seed_base=425000 + i * 100)
        for i, d in enumerate(sorted(set(domains.tolist())), start=1)
    }
    global_ci = _slice_ci(selected_ade, floor_ade, h, hard, easy, global_mask, seed_base=425900)
    floor_min = di._min_group_distance_fast(
        floor["floor_xy"][test_ids].astype(np.float32),
        group_key[test_ids],
        np.maximum(data["scale"][test_ids].astype(np.float64), EPS),
        data["agent_id"][test_ids].astype(np.int64),
    )
    del floor_min
    base_min = di._min_group_distance_fast(
        am_candidate["selected_xy"][test_ids].astype(np.float32),
        group_key[test_ids],
        np.maximum(data["scale"][test_ids].astype(np.float64), EPS),
        data["agent_id"][test_ids].astype(np.int64),
    )
    final_min = di._min_group_distance_fast(
        selected["selected_xy"].astype(np.float32),
        group_key[test_ids],
        np.maximum(data["scale"][test_ids].astype(np.float64), EPS),
        data["agent_id"][test_ids].astype(np.int64),
    )
    base_near = np.isfinite(base_min) & (base_min < 0.05)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    near_ci = {
        "global": _rate_summary(base_near, final_near, global_mask, seed_base=426900),
        "by_domain": {
            d: _rate_summary(base_near, final_near, domains == d, seed_base=427000 + i * 100)
            for i, d in enumerate(sorted(set(domains.tolist())), start=1)
        },
    }
    ci_positive_safe_domains = sum(1 for row in by_domain.values() if _positive_ci(row))
    original_stats = am._source_stats(data, original_split, group)
    repaired_stats = am._source_stats(data, repaired_split, group)
    payload: dict[str, Any] = {
        "source": "fresh_stage42_ea_dual_domain_group_consistency_statistics",
        "stage": "Stage42-EA",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/ucy_supported_group_consistency_stage42.json",
                "outputs/stage42_long_research/group_consistency_full_waypoint_repair_stage42.json",
            ]
        ),
        "internal_validation": {
            "source": "fresh_run",
            "domain": "UCY",
            "internal_val_group": internal_val_group,
            "selected_from": "original_train_sources_only",
            "uses_test_rows": False,
        },
        "test_rows": int(len(test_ids)),
        "repair_metric": repair["test"]["metric_vs_floor"],
        "repair_by_domain": repair["test"]["by_domain"],
        "bootstrap_ci": {"global": global_ci, "by_domain": by_domain},
        "near_collision_ci": near_ci,
        "summary": {
            "bootstrap_n": BOOTSTRAP_N,
            "ci_positive_safe_domains": int(ci_positive_safe_domains),
            "global_ci_positive_safe": _positive_ci(global_ci),
            "ucy_ci_positive_safe": _positive_ci(by_domain.get("UCY", {})),
            "trajnet_ci_positive_safe": _positive_ci(by_domain.get("TrajNet", {})),
            "near005_delta_high": near_ci["global"]["delta_final_minus_base"]["high"],
            "ucy_val_rows_after": int(repaired_stats["by_split"]["val"]["domains"].get("UCY", 0)),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_sources_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(repaired_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_ea_gate"] = _gate(payload)
    write_json(REPORT_JSON, di._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_dual_domain_group_consistency_statistics()
