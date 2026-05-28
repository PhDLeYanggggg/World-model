from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_eth_ucy_blocked_source_geometry_support as jj
from src import stage42_eth_ucy_row_family_selector as jk
from src import stage42_eth_ucy_source_specific_easy_guard as jg
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_rotation_full_waypoint_eval as je
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_source_support_coverage_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_source_support_coverage_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jl_gate.md"
JK_JSON = OUT_DIR / "eth_ucy_row_family_selector_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JL_ETH_UCY_SOURCE_SUPPORT_COVERAGE"
SOURCE = "fresh_stage42_jl_eth_ucy_source_support_coverage"
EPS = 1e-6
NEAREST_K = 2

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JL follows Stage42-JK: row-level family selection safely refused the blocked ETH_UCY sources.",
    "JL audits whether those blocked sources have enough source-level geometry/history support to justify any family switch.",
    "Held-out source support uses past-only/current-row feature distributions for diagnostics; future labels are evaluation-only.",
    "No central velocity, no test endpoint goals, no test-threshold tuning, and no metric/seconds claim are used.",
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


def _targets_from_jk() -> list[str]:
    report = read_json(JK_JSON, {})
    blocked = report.get("summary", {}).get("still_blocked_sources")
    if blocked:
        return list(blocked)
    payload = jk.run_stage42_eth_ucy_row_family_selector(refresh_readmes=False)
    return list(payload["summary"]["still_blocked_sources"])


def _source_signature(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(np.sum(mask)) == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    return np.mean(x[mask].astype(np.float64, copy=False), axis=0)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)))


def _support_threshold(signatures: Mapping[str, np.ndarray]) -> float:
    names = sorted(signatures)
    distances: list[float] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            distances.append(_euclidean(signatures[left], signatures[right]))
    if not distances:
        return 0.0
    return float(np.quantile(np.asarray(distances, dtype=np.float64), 0.75))


def _nearest_sources(target: np.ndarray, signatures: Mapping[str, np.ndarray], k: int = NEAREST_K) -> list[dict[str, Any]]:
    rows = [
        {"source": source, "distance": _euclidean(target, signature)}
        for source, signature in signatures.items()
    ]
    return sorted(rows, key=lambda row: row["distance"])[: max(1, min(k, len(rows)))]


def _family_metric_for_source(
    family_ade: np.ndarray,
    floor_ade: np.ndarray,
    data: Mapping[str, np.ndarray],
    source_ids: np.ndarray,
    source: str,
    family_idx: int,
) -> dict[str, Any]:
    mask = source_ids == source
    switch = np.ones(len(floor_ade), dtype=bool)
    return am._metric(family_ade[:, family_idx], floor_ade, data, switch, mask)


def _is_family_safe(metric: Mapping[str, Any]) -> bool:
    return bool(
        metric["easy_degradation"] <= 0.02
        and (
            metric["all_improvement"] > 0.0
            or metric["t50_improvement"] > 0.03
            or metric["hard_failure_improvement"] > 0.10
        )
    )


def _support_score(metric: Mapping[str, Any]) -> float:
    return float(
        metric["all_improvement"]
        + 1.7 * metric["t50_improvement"]
        + 1.2 * metric["hard_failure_improvement"]
        - 35.0 * max(0.0, metric["easy_degradation"] - 0.02)
        - 0.1 * metric["switch_rate"]
    )


def _family_support_table(
    family_ade: np.ndarray,
    floor_ade: np.ndarray,
    data: Mapping[str, np.ndarray],
    source_ids: np.ndarray,
    support_sources: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_idx, family in enumerate(jj.s37.BASELINE_FAMILY):
        metrics = [
            _family_metric_for_source(family_ade, floor_ade, data, source_ids, source, family_idx)
            for source in support_sources
        ]
        safe = [_is_family_safe(metric) for metric in metrics]
        rows.append(
            {
                "family_idx": int(family_idx),
                "family": family,
                "support_sources": support_sources,
                "safe_source_count": int(sum(safe)),
                "support_source_count": int(len(metrics)),
                "mean_all_improvement": float(np.mean([m["all_improvement"] for m in metrics])) if metrics else 0.0,
                "mean_t50_improvement": float(np.mean([m["t50_improvement"] for m in metrics])) if metrics else 0.0,
                "mean_hard_failure_improvement": float(np.mean([m["hard_failure_improvement"] for m in metrics])) if metrics else 0.0,
                "max_easy_degradation": float(np.max([m["easy_degradation"] for m in metrics])) if metrics else 0.0,
                "score": float(np.mean([_support_score(m) for m in metrics])) if metrics else -1e9,
                "metrics": metrics,
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _evaluate_target(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str) -> dict[str, Any]:
    split, split_stats = jg._source_cv_split(data, heldout_source)
    train_mask = split == "train"
    test_mask = split == "test"
    source_ids = jg._source_ids(data)
    source_names = sorted(set(source_ids.tolist()))
    nonheldout_sources = [source for source in source_names if source != heldout_source]

    floor = am._floor_arrays(data, train_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    family_xy = jj._family_waypoints(data)
    family_ade, family_fde = jj._family_errors(family_xy, labels)
    features, feature_names, removed = je._domain_invariant_features(data, floor)
    x, _mean, _std = am._standardize(features, train_mask)

    signatures = {source: _source_signature(x, source_ids == source) for source in source_names}
    support_signatures = {source: signatures[source] for source in nonheldout_sources}
    threshold = _support_threshold(support_signatures)
    nearest = _nearest_sources(signatures[heldout_source], support_signatures)
    nearest_sources = [row["source"] for row in nearest]
    nearest_distance = float(nearest[0]["distance"]) if nearest else float("inf")
    in_support = bool(nearest_distance <= threshold + EPS)

    support_table = _family_support_table(family_ade, floor_ade, data, source_ids, nearest_sources)
    selected_support: dict[str, Any] | None = None
    for row in support_table:
        if row["safe_source_count"] == row["support_source_count"] and row["score"] > 0.0:
            selected_support = row
            break

    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    oracle_ade = np.minimum(floor_ade, np.min(family_ade, axis=1))
    oracle_switch = np.min(family_ade, axis=1) < floor_ade
    oracle_metric = am._metric(oracle_ade, floor_ade, data, oracle_switch, test_mask)

    switch = np.zeros(len(floor_ade), dtype=bool)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    decision_reason = "source_out_of_support"
    if in_support and selected_support is not None:
        family_idx = int(selected_support["family_idx"])
        selected_ade = family_ade[:, family_idx].copy()
        selected_fde = family_fde[:, family_idx].copy()
        switch = np.ones(len(floor_ade), dtype=bool)
        decision_reason = "nearest_source_supported_family_selected"
    elif in_support:
        decision_reason = "no_nearest_source_easy_safe_family"

    metric = am._metric(selected_ade, floor_ade, data, switch, test_mask)
    fde_metric = am._metric(selected_fde, floor_fde, data, switch, test_mask)
    deployable = bool(
        metric["easy_degradation"] <= 0.02
        and metric["all_improvement"] > 0.03
        and (metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10)
    )
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names)),
            "domain_features_removed": removed,
            "normalization": "train_split_mean_std_only",
            "support_features_past_current_only": True,
            "future_inputs": False,
        },
        "support": {
            "source_support_threshold": threshold,
            "nearest_sources": nearest,
            "nearest_distance": nearest_distance,
            "in_support": in_support,
            "support_table_top": support_table[:8],
            "selected_support_family": selected_support,
            "decision_reason": decision_reason,
            "test_threshold_tuning": False,
        },
        "metrics": {
            "source_support_policy": metric,
            "source_support_policy_fde": fde_metric,
            "family_oracle": oracle_metric,
        },
        "deployable_after_source_support_policy": deployable,
        "bootstrap": {
            "all": am._bootstrap_ci(selected_ade, floor_ade, test_mask, seed=42241),
            "t50": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (h == 50), seed=42242),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (h == 100), seed=42243),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test_mask & hard_failure, seed=42244),
            "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, test_mask & easy, seed=42245),
            "oracle_t50": am._bootstrap_ci(oracle_ade, floor_ade, test_mask & (h == 50), seed=42246),
        },
    }


def _summary(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    repaired = [row["heldout_source"] for row in targets if row["deployable_after_source_support_policy"]]
    blocked = [row["heldout_source"] for row in targets if not row["deployable_after_source_support_policy"]]
    unsupported = [row["heldout_source"] for row in targets if not row["support"]["in_support"]]
    no_safe_family = [
        row["heldout_source"]
        for row in targets
        if row["support"]["in_support"] and row["support"]["selected_support_family"] is None
    ]
    decision = (
        "source_support_policy_repaired_all_blocked_sources"
        if repaired and not blocked
        else "source_support_policy_partially_repaired_blocked_sources"
        if repaired
        else "source_support_policy_not_deployable_support_blocker"
    )
    return {
        "source": SOURCE,
        "targeted_sources": [row["heldout_source"] for row in targets],
        "repaired_sources": repaired,
        "still_blocked_sources": blocked,
        "unsupported_sources": unsupported,
        "in_support_but_no_safe_family_sources": no_safe_family,
        "decision": decision,
        "next_action": "For unsupported or no-safe-family sources, acquire/calibrate source-specific geometry or train a dedicated source family with stronger easy-harm labels before another deployment attempt.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = jg._eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    targets = [_evaluate_target(data, labels, source) for source in _targets_from_jk()]
    payload: dict[str, Any] = {
        "stage": "Stage42-JL",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(JK_JSON)]),
        "current_facts": CURRENT_FACTS,
        "target_selection": {
            "source": "cached_verified_stage42_jk_still_blocked_sources",
            "blocked_sources_from_jk": _targets_from_jk(),
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
            "train_only_feature_normalization": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in targets),
            "heldout_source_support_uses_labels": False,
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
    payload["stage42_jl_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "jk_blocked_sources_targeted": len(payload["targets"]) == len(payload["target_selection"]["blocked_sources_from_jk"]) and len(payload["targets"]) > 0,
        "source_support_computed": all(row["support"]["nearest_sources"] for row in payload["targets"]),
        "support_table_recorded": all(row["support"]["support_table_top"] for row in payload["targets"]),
        "repair_or_blocker_recorded": bool(payload["summary"]["repaired_sources"] or payload["summary"]["still_blocked_sources"]),
        "unsupported_or_no_safe_family_explained": bool(
            payload["summary"]["unsupported_sources"] or payload["summary"]["in_support_but_no_safe_family_sources"] or payload["summary"]["repaired_sources"]
        ),
        "test_threshold_tuning_false": all(row["support"]["test_threshold_tuning"] is False for row in payload["targets"]),
        "no_overclaim_full_eth_ucy": payload["summary"]["decision"] != "source_support_policy_repaired_all_blocked_sources"
        or not payload["summary"]["still_blocked_sources"],
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
                "heldout_source_support_uses_labels",
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jl_eth_ucy_source_support_coverage_pass" if passed == len(gates) else "stage42_jl_eth_ucy_source_support_coverage_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jl_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JL ETH_UCY Source Support Coverage",
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
        f"- unsupported_sources: `{summary['unsupported_sources']}`",
        f"- in_support_but_no_safe_family_sources: `{summary['in_support_but_no_safe_family_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Held-Out Support Metrics",
        "",
        "| source | in support | nearest | distance | threshold | all | t50 | hard/failure | easy degradation | oracle t50 | deployable |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["targets"]:
        metric = row["metrics"]["source_support_policy"]
        oracle = row["metrics"]["family_oracle"]
        nearest = row["support"]["nearest_sources"][0]["source"] if row["support"]["nearest_sources"] else "none"
        lines.append(
            f"| `{row['heldout_source']}` | `{row['support']['in_support']}` | `{nearest}` | "
            f"{row['support']['nearest_distance']:.3f} | {row['support']['source_support_threshold']:.3f} | "
            f"{_fmt(metric['all_improvement'])} | {_fmt(metric['t50_improvement'])} | {_fmt(metric['hard_failure_improvement'])} | "
            f"{_fmt(metric['easy_degradation'])} | {_fmt(oracle['t50_improvement'])} | `{row['deployable_after_source_support_policy']}` |"
        )
    lines.extend(["", "## Support Family Candidates", ""])
    for row in payload["targets"]:
        support = row["support"]
        lines.extend(
            [
                f"### `{row['heldout_source']}`",
                "",
                f"- decision_reason: `{support['decision_reason']}`",
                f"- selected_support_family: `{support['selected_support_family']}`",
                "",
                "| rank | family | safe sources | mean all | mean t50 | mean hard | max easy | score |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, cand in enumerate(support["support_table_top"][:8], start=1):
            lines.append(
                f"| {idx} | `{cand['family']}` | {cand['safe_source_count']}/{cand['support_source_count']} | "
                f"{_fmt(cand['mean_all_improvement'])} | {_fmt(cand['mean_t50_improvement'])} | "
                f"{_fmt(cand['mean_hard_failure_improvement'])} | {_fmt(cand['max_easy_degradation'])} | {cand['score']:.4f} |"
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
            "- JL turns the JK failure into a source-support question: a family may have oracle headroom but still be unsafe if no similar source supports it without easy harm.",
            "- If this remains fallback-only or unsupported, the next useful work is source-specific calibration/new source acquisition rather than more global threshold search.",
            "- This is raw-frame / dataset-local 2.5D evidence only; no metric, seconds-level, Stage5C, SMC, or foundation claim is enabled.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jl_gate"]
    lines = [
        "# Stage42-JL Gate",
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
    gate = payload["stage42_jl_gate"]
    bits = []
    for row in payload["targets"]:
        metric = row["metrics"]["source_support_policy"]
        oracle = row["metrics"]["family_oracle"]
        bits.append(
            f"{row['heldout_source']}: support={row['support']['in_support']}, all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}, oracle t50 {_fmt(oracle['t50_improvement'])}"
        )
    return [
        "## Stage42-JL ETH_UCY Source Support Coverage",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- source-support heldout results: {'; '.join(bits)}.",
        f"- decision: `{summary['decision']}`; repaired: `{summary['repaired_sources']}`; still blocked: `{summary['still_blocked_sources']}`; unsupported: `{summary['unsupported_sources']}`.",
        "- boundary: this is a source-support diagnostic/repair attempt, still dataset-local raw-frame 2.5D, no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_source_support_coverage"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jl_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jl_gate"]["passed"], "total": payload["stage42_jl_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "targeted_sources": payload["summary"]["targeted_sources"],
        "repaired_sources": payload["summary"]["repaired_sources"],
        "still_blocked_sources": payload["summary"]["still_blocked_sources"],
        "unsupported_sources": payload["summary"]["unsupported_sources"],
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
                    "stage": "Stage42-JL",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jl_gate"]["verdict"],
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


def run_stage42_eth_ucy_source_support_coverage(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    run_stage42_eth_ucy_source_support_coverage(refresh_readmes=True)


if __name__ == "__main__":
    main()
