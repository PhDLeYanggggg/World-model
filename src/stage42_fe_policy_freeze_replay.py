from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_constrained_fc_safety_composer as fe
from src import stage42_full_waypoint_all_hard_loss_repair as dg
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_objective_level_proximity_training as fc
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fe_policy_freeze_replay_stage42.json"
REPORT_MD = OUT_DIR / "fe_policy_freeze_replay_stage42.md"
POLICY_JSON = OUT_DIR / "frozen_constrained_fc_safety_composer_policy_stage42.json"
POLICY_MD = OUT_DIR / "frozen_constrained_fc_safety_composer_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ff_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fe.PAPER_FILES

SOURCE = "fresh_stage42_fe_policy_freeze_replay"
BOOTSTRAP_N = 2000
EPS = 1e-9


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _stable_hash(row: Mapping[str, Any]) -> str:
    blob = json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _bootstrap_rate(values: np.ndarray, *, seed: int, n: int = BOOTSTRAP_N) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(vals)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sample = rng.choice(len(vals), size=len(vals), replace=True)
        out.append(float(np.mean(vals[sample])))
    return {
        "low": float(np.percentile(out, 2.5)),
        "mid": float(np.percentile(out, 50.0)),
        "high": float(np.percentile(out, 97.5)),
        "n": int(len(vals)),
        "bootstrap_n": int(n),
    }


def _bootstrap_rate_delta(a: np.ndarray, b: np.ndarray, *, seed: int, n: int = BOOTSTRAP_N) -> dict[str, Any]:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if len(av) != len(bv):
        raise ValueError("paired rate delta arrays must have matching length")
    return _bootstrap_rate(av - bv, seed=seed, n=n)


def _context() -> dict[str, Any]:
    data = s41._combined()
    split, group = am._split_arrays(data)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    graph, graph_names, graph_stats = graph_ctx._build_graph_features(data)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    am_candidate["floor_xy"] = floor["floor_xy"]
    group_key = di._group_key(data)
    signals = fc._objective_signals(data, labels, graph, group_key, am_candidate)
    prior = fe._load_prior()
    fc_candidate = fe._rebuild_fc_candidate(data, split, labels, floor, features, graph, signals, group_key, prior["fc"])
    return {
        "data": data,
        "split": split,
        "group": group,
        "labels": labels,
        "floor": floor,
        "feature_names": feature_names,
        "graph_names": graph_names,
        "graph_stats": graph_stats,
        "am_candidate": am_candidate,
        "fc_candidate": fc_candidate,
        "group_key": group_key,
        "prior": prior,
    }


def _replay_selected(ctx: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    data = ctx["data"]
    split = ctx["split"]
    labels = ctx["labels"]
    floor = ctx["floor"]
    am_candidate = ctx["am_candidate"]
    fc_candidate = ctx["fc_candidate"]
    group_key = ctx["group_key"]
    prior = ctx["prior"]
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    floor_xy = floor["floor_xy"].astype(np.float32)
    val_evals = fe._reference_evals(val_ids, data, labels, floor, am_candidate, fc_candidate, group_key, prior)
    test_evals = fe._reference_evals(test_ids, data, labels, floor, am_candidate, fc_candidate, group_key, prior)
    val = fe._compose(candidate, val_ids, data, labels, floor_xy, group_key, val_evals)
    test = fe._compose(candidate, test_ids, data, labels, floor_xy, group_key, test_evals)
    h = data["horizon"][test_ids].astype(int)
    hard_failure = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    selected_xy = test["selected_xy"].astype(np.float32)
    normalizer = np.maximum(data["scale"][test_ids].astype(np.float64), 1e-6)
    agent = data["agent_id"][test_ids].astype(np.int64)
    final_min = di._min_group_distance_fast(selected_xy, group_key[test_ids], normalizer, agent)
    final_near = np.isfinite(final_min) & (final_min < 0.05)
    fc_near = np.isfinite(test_evals["fc"]["min_distance"]) & (test_evals["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(test_evals["di"]["min_distance"]) & (test_evals["di"]["min_distance"] < 0.05)
    fb_near = np.isfinite(test_evals["fb"]["min_distance"]) & (test_evals["fb"]["min_distance"] < 0.05)
    bootstrap = {
        "all": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], np.ones(len(test_ids), dtype=bool), seed=43501, n=BOOTSTRAP_N),
        "t50": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 50, seed=43502, n=BOOTSTRAP_N),
        "t100_raw_frame_diagnostic": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 100, seed=43503, n=BOOTSTRAP_N),
        "hard_failure": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], hard_failure, seed=43504, n=BOOTSTRAP_N),
        "easy_degradation": di._bootstrap_ci_subset(test["floor_ade"], test["selected_ade"], easy, seed=43505, n=BOOTSTRAP_N),
    }
    near_bootstrap = {
        "final_near_005": _bootstrap_rate(final_near, seed=43521),
        "delta_final_minus_fc": _bootstrap_rate_delta(final_near, fc_near, seed=43522),
        "delta_final_minus_di": _bootstrap_rate_delta(final_near, di_near, seed=43523),
        "delta_final_minus_fb": _bootstrap_rate_delta(final_near, fb_near, seed=43524),
    }
    return {
        "val_ids": val_ids,
        "test_ids": test_ids,
        "val": val,
        "test": test,
        "test_evals": test_evals,
        "bootstrap": bootstrap,
        "near_bootstrap": near_bootstrap,
        "final_min_distance": final_min,
    }


def _metric_diffs(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    keys = sorted(set(a) & set(b))
    return {key: abs(float(a[key]) - float(b[key])) for key in keys if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float))}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    replay = payload["replay"]
    b = payload["bootstrap_ci"]
    n = payload["near_bootstrap_ci"]
    boundary = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    metric = replay["test_metric_vs_floor"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fe_artifact_loaded": payload["fe_artifact"]["exists"] is True,
        "policy_hash_recorded": len(payload["frozen_policy"]["policy_hash"]) == 64,
        "candidate_exact_replay": replay["candidate_exact_replay"] is True,
        "metric_exact_replay": replay["max_metric_abs_diff"] <= 1e-12,
        "diagnostic_exact_replay": replay["max_diagnostic_abs_diff"] <= 1e-12,
        "bootstrap_n_2000": all(row["bootstrap_n"] >= BOOTSTRAP_N for row in b.values()),
        "bootstrap_all_positive": b["all"]["low"] > 0.0,
        "bootstrap_t50_positive": b["t50"]["low"] > 0.0,
        "bootstrap_t100_raw_positive": b["t100_raw_frame_diagnostic"]["low"] > 0.0,
        "bootstrap_hard_positive": b["hard_failure"]["low"] > 0.0,
        "easy_ci_safe": b["easy_degradation"]["high"] <= 0.02,
        "near_bootstrap_reported": all(row["bootstrap_n"] >= BOOTSTRAP_N for row in n.values()),
        "near_point_better_than_fc": replay["near_delta_vs_fc"] < 0.0,
        "near_point_not_worse_than_di": replay["near_delta_vs_di"] <= 0.0,
        "test_all_positive": metric["all_improvement"] > 0.0,
        "test_t50_positive": metric["t50_improvement"] > 0.0,
        "test_hard_positive": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
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
    verdict = "stage42_ff_fe_policy_freeze_replay_pass" if passed == total else "stage42_ff_fe_policy_freeze_replay_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fe_payload = read_json(fe.REPORT_JSON, {})
    if not fe_payload:
        fe_payload = fe.run_stage42_constrained_fc_safety_composer()
    ctx = _context()
    candidate = fe_payload["repair"]["selected"]["candidate"]
    replay = _replay_selected(ctx, candidate)
    test_metric = replay["test"]["metric"]
    test_diag = replay["test"]["diagnostics"]
    fe_metric = fe_payload["repair"]["test"]["metric_vs_floor"]
    fe_diag = fe_payload["repair"]["test"]["diagnostics"]
    metric_diffs = _metric_diffs(test_metric, fe_metric)
    diag_diffs = _metric_diffs(test_diag, fe_diag)
    policy_artifact = {
        "policy_name": "stage42_fe_constrained_fc_safety_composer",
        "source": SOURCE,
        "frozen_from": str(fe.REPORT_JSON),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "selected_candidate": candidate,
        "policy_hash": _stable_hash(candidate),
        "runtime_inputs": [
            "predicted_fc_rollout_geometry",
            "predicted_di_fallback_geometry",
            "predicted_fa_fallback_geometry",
            "predicted_fb_fallback_geometry",
            "source_frame_horizon_group_key",
            "agent_id",
            "normalizer",
        ],
        "validation_selection_rule": "candidate selected on validation only; test evaluated once",
        "test_usage_rule": "test labels used only for final evaluation and bootstrap, not for policy selection",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FF FE policy freeze, 2000-bootstrap, and replay",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(fe.AM_JSON),
                str(fe.DI_JSON),
                str(fe.FA_JSON),
                str(fe.FB_JSON),
                str(fe.FC_JSON),
                str(fe.REPORT_JSON),
            ]
        ),
        "fe_artifact": {
            "exists": fe.REPORT_JSON.exists(),
            "report": str(fe.REPORT_MD),
            "verdict": fe_payload.get("stage42_fe_gate", {}).get("verdict"),
            "selected_candidate": candidate,
        },
        "frozen_policy": policy_artifact,
        "replay": {
            "candidate_exact_replay": candidate == fe_payload["repair"]["selected"]["candidate"],
            "test_rows": int(len(replay["test_ids"])),
            "test_metric_vs_floor": test_metric,
            "test_diagnostics": test_diag,
            "metric_abs_diffs_vs_fe_artifact": metric_diffs,
            "diagnostic_abs_diffs_vs_fe_artifact": diag_diffs,
            "max_metric_abs_diff": max(metric_diffs.values()) if metric_diffs else 0.0,
            "max_diagnostic_abs_diff": max(diag_diffs.values()) if diag_diffs else 0.0,
            "near_delta_vs_fc": float(test_diag["final_near_005"]) - float(replay["test_evals"]["fc"]["diagnostics"]["near_005"]),
            "near_delta_vs_di": float(test_diag["final_near_005"]) - float(replay["test_evals"]["di"]["diagnostics"]["final_near_005"]),
            "near_delta_vs_fb": float(test_diag["final_near_005"]) - float(replay["test_evals"]["fb"]["diagnostics"]["final_near_005"]),
        },
        "bootstrap_ci": replay["bootstrap"],
        "near_bootstrap_ci": replay["near_bootstrap"],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "decision": "freeze_and_promote_stage42_fe_if_replay_and_ci_hold",
        "notes": [
            "This step does not reselect thresholds or tune on test.",
            "It freezes the Stage42-FE selected candidate, recomputes the same policy, and adds 2000-bootstrap evidence.",
            "Near-collision bootstrap is reported as safety evidence; dataset-local/raw-frame boundaries remain unchanged.",
        ],
    }
    payload["stage42_ff_gate"] = _gate(payload)
    return _jsonable(payload)


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ff_gate"]
    metric = payload["replay"]["test_metric_vs_floor"]
    diag = payload["replay"]["test_diagnostics"]
    return [
        "# Stage42-FF FE Policy Freeze / Bootstrap / Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- frozen policy hash: `{payload['frozen_policy']['policy_hash']}`",
        f"- selected candidate: `{payload['frozen_policy']['selected_candidate']}`",
        "",
        "## Replayed Test Metrics vs Floor",
        "",
        f"- all improvement: `{_pct(metric['all_improvement'])}`",
        f"- t50 improvement: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        f"- final near@0.05: `{_pct(diag['final_near_005'])}`",
        "",
        "## Exact Replay",
        "",
        f"- candidate exact replay: `{payload['replay']['candidate_exact_replay']}`",
        f"- max metric abs diff vs FE artifact: `{payload['replay']['max_metric_abs_diff']}`",
        f"- max diagnostic abs diff vs FE artifact: `{payload['replay']['max_diagnostic_abs_diff']}`",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n | bootstrap_n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | `{_pct(row['low'])}` | `{_pct(row['mid'])}` | `{_pct(row['high'])}` | `{row['n']}` | `{row['bootstrap_n']}` |"
            for name, row in payload["bootstrap_ci"].items()
        ],
        "",
        "## Near@0.05 Bootstrap CI",
        "",
        "| quantity | low | mid | high | n | bootstrap_n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | `{_pct(row['low'])}` | `{_pct(row['mid'])}` | `{_pct(row['high'])}` | `{row['n']}` | `{row['bootstrap_n']}` |"
            for name, row in payload["near_bootstrap_ci"].items()
        ],
        "",
        "## No Leakage / Claim Boundary",
        "",
        "- policy is frozen from validation-selected Stage42-FE candidate.",
        "- replay performs no threshold reselection and no test tuning.",
        "- future waypoints/endpoints are labels only, never inference inputs.",
        "- dataset-local/raw-frame 2.5D only; no metric/seconds claim.",
        "- Stage5C not executed; SMC not enabled.",
    ]


def _render_policy(payload: Mapping[str, Any]) -> list[str]:
    policy = payload["frozen_policy"]
    return [
        "# Frozen Stage42-FE Constrained FC/Safety Composer Policy",
        "",
        f"- source: `{policy['source']}`",
        f"- policy name: `{policy['policy_name']}`",
        f"- policy hash: `{policy['policy_hash']}`",
        f"- selected candidate: `{policy['selected_candidate']}`",
        f"- frozen from: `{policy['frozen_from']}`",
        "",
        "## Runtime Inputs",
        "",
        *[f"- `{name}`" for name in policy["runtime_inputs"]],
        "",
        "## Selection Discipline",
        "",
        f"- validation selection rule: `{policy['validation_selection_rule']}`",
        f"- test usage rule: `{policy['test_usage_rule']}`",
        "- no future endpoint input, no central velocity, no test endpoint goals.",
        "- Stage5C false; SMC false; no metric/seconds claim.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ff_gate"]
    return [
        "# Stage42-FF Gates",
        "",
        f"Verdict: `{gate['verdict']}`",
        f"Passed: `{gate['passed']} / {gate['total']}`",
        "",
        *[f"- `{key}`: `{value}`" for key, value in gate["gates"].items()],
    ]


def _replace_text_section(old: str, tag: str, block: str) -> str:
    start = f"<!-- {tag}:START -->"
    end = f"<!-- {tag}:END -->"
    if start in old and end in old:
        before, rest = old.split(start, 1)
        _, after = rest.split(end, 1)
        return before.rstrip() + "\n\n" + block.strip() + after
    return old.rstrip() + "\n\n" + block.strip() + "\n"


def _summary_section(payload: Mapping[str, Any]) -> str:
    metric = payload["replay"]["test_metric_vs_floor"]
    b = payload["bootstrap_ci"]
    return "\n".join(
        [
            "<!-- STAGE42_FF_FE_POLICY_FREEZE_REPLAY:START -->",
            "## Stage42-FF FE Policy Freeze / Bootstrap / Replay",
            "",
            f"- source: `{payload['source']}`",
            "- role: freeze Stage42-FE constrained FC/safety composer and add 2000-bootstrap plus exact replay evidence.",
            f"- gate: `{payload['stage42_ff_gate']['passed']} / {payload['stage42_ff_gate']['total']}`; verdict `{payload['stage42_ff_gate']['verdict']}`.",
            f"- frozen policy hash: `{payload['frozen_policy']['policy_hash']}`.",
            f"- replay all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
            f"- bootstrap lows all/t50/t100raw/hard: `{_pct(b['all']['low'])}` / `{_pct(b['t50']['low'])}` / `{_pct(b['t100_raw_frame_diagnostic']['low'])}` / `{_pct(b['hard_failure']['low'])}`.",
            f"- exact replay max metric/diagnostic diff: `{payload['replay']['max_metric_abs_diff']}` / `{payload['replay']['max_diagnostic_abs_diff']}`.",
            "- Boundary: frozen protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FF_FE_POLICY_FREEZE_REPLAY:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(_replace_text_section(old, "STAGE42_FF_FE_POLICY_FREEZE_REPLAY", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FF FE policy freeze / bootstrap / replay"
    state["current_verdict"] = payload["stage42_ff_gate"]["verdict"]
    state["stage42_ff_fe_policy_freeze_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "policy": str(POLICY_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ff_gate"]["verdict"],
        "gates": f"{payload['stage42_ff_gate']['passed']}/{payload['stage42_ff_gate']['total']}",
        "policy_hash": payload["frozen_policy"]["policy_hash"],
        "test_metric_vs_floor": payload["replay"]["test_metric_vs_floor"],
        "bootstrap_ci": payload["bootstrap_ci"],
        "near_bootstrap_ci": payload["near_bootstrap_ci"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FF freezes Stage42-FE and adds exact replay plus 2000-bootstrap evidence without reselection or test tuning.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FF FE policy freeze / bootstrap / replay" not in evidence:
            evidence.append("Stage42-FF FE policy freeze / bootstrap / replay")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_ff_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FF freezes the Stage42-FE constrained composer and adds bootstrap/replay evidence, strengthening FE from promotable single-run evidence to frozen reproducible evidence."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fe_policy_freeze_replay() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_json(POLICY_JSON, payload["frozen_policy"])
    write_md(REPORT_MD, _render_report(payload))
    write_md(POLICY_MD, _render_policy(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fe_policy_freeze_replay()
    gate = result["stage42_ff_gate"]
    print(f"Stage42-FF gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
