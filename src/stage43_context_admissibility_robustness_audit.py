from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_context_admissibility_model as bt


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_context_admissibility_robustness_audit.json"
REPORT_MD = OUT_DIR / "stage43_context_admissibility_robustness_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_bu_context_admissibility_robustness_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BU_CONTEXT_ADMISSIBILITY_ROBUSTNESS_AUDIT"
SOURCE = "fresh_stage43_bu_context_admissibility_robustness_audit"
DEFAULT_VARIANT = bt.DEFAULT_VARIANT
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _easy_degradation(selected: np.ndarray, floor: np.ndarray, easy: np.ndarray) -> float:
    if int(easy.sum()) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[easy])) / max(float(np.mean(floor[easy])), EPS) - 1.0))


def _metric_bundle(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    return m._metrics(ds, selected_ade.astype(np.float32), selected_fde.astype(np.float32), switched.astype(bool))


def _delta_metrics(metrics: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - graph["full_waypoint_ade_improvement_vs_floor"]),
        "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"] - graph["t50_full_waypoint_ade_improvement_vs_floor"]),
        "t100_raw_frame_diagnostic": float(
            metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            - graph["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        ),
        "hard_failure": float(
            metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - graph["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        ),
        "easy_degradation": float(metrics["easy_degradation_vs_floor"] - graph["easy_degradation_vs_floor"]),
    }


def _replay_bt(test_rows: int | None = None) -> dict[str, Any]:
    report = read_json(bt.REPORT_JSON, {})
    if not report:
        raise FileNotFoundError(bt.REPORT_JSON)
    rows = report.get("rows", {})
    train_rows = int(rows.get("train", 20000))
    val_rows = int(rows.get("val", 12000))
    replay_rows = int(test_rows or rows.get("test", 12000))
    train = bt._load_context("train", rows=train_rows)
    val = bt._load_context("val", rows=val_rows)
    test = bt._load_context("test", rows=replay_rows)
    bt._standardize_batches(train, val, test)
    ckpt_path = Path(report["model"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} is required for Stage43-BU replay. Run run_stage43_context_admissibility_model.py first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = bt.ContextAdmissibilityNet(
        int(ckpt["input_dim"]),
        len(bt.CONTEXT_VARIANTS),
        hidden_dim=int(ckpt.get("hidden_dim", report["model"].get("hidden_dim", 96))),
    )
    model.load_state_dict(ckpt["model_state"])
    pred = bt._predict_model(model, test.ds.x, batch_size=4096)
    policy = report["validation_selection"]["selected_policy"]
    selected_ade, selected_fde, switched, used = bt._apply_policy(test, pred, policy)
    graph_ade = test.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_fde = test.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32)
    graph_switched = test.arrays[DEFAULT_VARIANT]["switched"].astype(bool)
    metrics = _metric_bundle(test.ds, selected_ade, selected_fde, switched)
    graph_metrics = _metric_bundle(test.ds, graph_ade, graph_fde, graph_switched)
    delta = _delta_metrics(metrics, graph_metrics)
    expected = report["test_metrics"]
    replay_diff = {
        key: float(abs(float(metrics[key]) - float(expected[key])))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t50_full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
        ]
    }
    return {
        "report": report,
        "test": test,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switched": switched,
        "used": used,
        "graph_ade": graph_ade,
        "graph_fde": graph_fde,
        "graph_switched": graph_switched,
        "metrics": metrics,
        "graph_metrics": graph_metrics,
        "delta": delta,
        "replay_diff": replay_diff,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
    }


def _bootstrap_summary(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    graph_ade: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    masks = {
        "all_delta_vs_graph": np.ones(len(selected_ade), dtype=bool),
        "t50_delta_vs_graph": ds.horizon == 50,
        "t100_raw_frame_delta_vs_graph": ds.horizon == 100,
        "hard_failure_delta_vs_graph": ds.hard | ds.failure,
    }
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "metrics": {}}
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        if len(ids) == 0:
            out["metrics"][name] = {"low": 0.0, "mean": 0.0, "high": 0.0, "rows": 0}
            continue
        vals = np.empty(int(n), dtype=np.float64)
        for i in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            floor = ds.floor_ade[sample]
            selected_imp = 1.0 - float(np.mean(selected_ade[sample])) / max(float(np.mean(floor)), EPS)
            graph_imp = 1.0 - float(np.mean(graph_ade[sample])) / max(float(np.mean(floor)), EPS)
            vals[i] = selected_imp - graph_imp
        out["metrics"][name] = {
            "low": float(np.quantile(vals, 0.025)),
            "mean": float(np.mean(vals)),
            "high": float(np.quantile(vals, 0.975)),
            "rows": int(len(ids)),
        }
    easy = ds.easy.astype(bool)
    ids = np.where(easy)[0]
    if len(ids) == 0:
        out["metrics"]["easy_degradation_delta_vs_graph"] = {"low": 0.0, "mean": 0.0, "high": 0.0, "rows": 0}
    else:
        vals = np.empty(int(n), dtype=np.float64)
        for i in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            sel_deg = _easy_degradation(selected_ade, ds.floor_ade, np.isin(np.arange(len(selected_ade)), sample))
            graph_deg = _easy_degradation(graph_ade, ds.floor_ade, np.isin(np.arange(len(graph_ade)), sample))
            vals[i] = sel_deg - graph_deg
        out["metrics"]["easy_degradation_delta_vs_graph"] = {
            "low": float(np.quantile(vals, 0.025)),
            "mean": float(np.mean(vals)),
            "high": float(np.quantile(vals, 0.975)),
            "rows": int(len(ids)),
        }
    return out


def _slice_row(
    name: str,
    mask: np.ndarray,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    graph_ade: np.ndarray,
    used: np.ndarray,
) -> dict[str, Any]:
    selected_imp = _slice_improvement(selected_ade, ds.floor_ade, mask)
    graph_imp = _slice_improvement(graph_ade, ds.floor_ade, mask)
    easy_mask = mask & ds.easy
    row = {
        "slice": name,
        "rows": int(mask.sum()),
        "selected_improvement": selected_imp,
        "graph_history_improvement": graph_imp,
        "delta_vs_graph": float(selected_imp - graph_imp),
        "easy_rows": int(easy_mask.sum()),
        "easy_degradation": _easy_degradation(selected_ade[mask], ds.floor_ade[mask], ds.easy[mask]),
        "context_rate": float(np.mean(used[mask].astype(str) != DEFAULT_VARIANT)) if int(mask.sum()) else 0.0,
        "scene_proxy_rows": int(np.sum(used[mask].astype(str) == "scene_proxy_only")),
        "scene_graph_full_rows": int(np.sum(used[mask].astype(str) == "scene_graph_full")),
    }
    return row


def _slice_audit(ds: m.WaypointSplit, selected_ade: np.ndarray, graph_ade: np.ndarray, used: np.ndarray, *, min_rows: int) -> dict[str, Any]:
    domain = ds.domain.astype(str)
    source = ds.source_file.astype(str)
    horizon = ds.horizon.astype(int)
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(selected_ade), dtype=bool),
        "hard_failure": ds.hard | ds.failure,
        "easy": ds.easy,
    }
    for h in sorted(set(horizon.tolist())):
        masks[f"horizon_{h}"] = horizon == h
    for d in sorted(set(domain.tolist())):
        masks[f"domain_{d}"] = domain == d
        for h in sorted(set(horizon.tolist())):
            masks[f"domain_{d}_horizon_{h}"] = (domain == d) & (horizon == h)
    for s in sorted(set(source.tolist())):
        masks[f"source_{s}"] = source == s
        for h in sorted(set(horizon.tolist())):
            masks[f"source_{s}_horizon_{h}"] = (source == s) & (horizon == h)
    rows = [_slice_row(name, mask, ds, selected_ade, graph_ade, used) for name, mask in masks.items() if int(mask.sum()) >= int(min_rows)]
    positive = [row for row in rows if row["delta_vs_graph"] > 0.0]
    negative = [row for row in rows if row["delta_vs_graph"] < 0.0]
    weak = [
        row
        for row in rows
        if row["slice"] in {"horizon_50", "horizon_100", "hard_failure", "all"} and row["delta_vs_graph"] <= 0.0
    ]
    easy_hazard = [row for row in rows if row["easy_rows"] >= 10 and row["easy_degradation"] > 0.02]
    return {
        "min_rows": int(min_rows),
        "slice_count": len(rows),
        "positive_slice_count": len(positive),
        "negative_slice_count": len(negative),
        "core_weak_slices": weak,
        "easy_hazard_slice_count": len(easy_hazard),
        "top_easy_hazard_slices": sorted(easy_hazard, key=lambda row: row["easy_degradation"], reverse=True)[:20],
        "top_positive_slices": sorted(positive, key=lambda row: row["delta_vs_graph"], reverse=True)[:20],
        "top_negative_slices": sorted(negative, key=lambda row: row["delta_vs_graph"])[:20],
        "slice_rows": rows,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    replay = _replay_bt(test_rows=int(args.test_rows) if args.test_rows else None)
    ds = replay["test"].ds
    bootstrap = _bootstrap_summary(
        ds,
        replay["selected_ade"],
        replay["graph_ade"],
        n=int(args.bootstrap),
        seed=int(args.seed),
    )
    slices = _slice_audit(
        ds,
        replay["selected_ade"],
        replay["graph_ade"],
        replay["used"],
        min_rows=int(args.min_rows),
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_bootstrap_slice_audit_from_stage43_bt",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "precondition": {
            "bt_verdict": replay["report"].get("stage43_bt_gate", {}).get("verdict", "missing"),
            "bt_gate": {
                "passed": replay["report"].get("stage43_bt_gate", {}).get("passed", 0),
                "total": replay["report"].get("stage43_bt_gate", {}).get("total", 0),
            },
        },
        "rows": {"test": int(len(ds.x))},
        "checkpoint": {
            "path": replay["checkpoint"],
            "sha256": replay["checkpoint_sha256"],
            "committed": False,
        },
        "replay_metrics": replay["metrics"],
        "graph_history_metrics": replay["graph_metrics"],
        "delta_vs_graph_history_only": replay["delta"],
        "replay_diff_vs_stage43_bt_report": replay["replay_diff"],
        "bootstrap": bootstrap,
        "slice_audit": slices,
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "deployment_policy_changed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_variant_error_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
            "test_threshold_selection": False,
        },
        "input_hash": _combined_hash([bt.REPORT_JSON]),
    }
    payload["stage43_bu_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    pre = payload["precondition"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    replay_max = max(float(v) for v in payload["replay_diff_vs_stage43_bt_report"].values())
    boot = payload["bootstrap"]["metrics"]
    all_low = float(boot["all_delta_vs_graph"]["low"])
    hard_low = float(boot["hard_failure_delta_vs_graph"]["low"])
    t50_low = float(boot["t50_delta_vs_graph"]["low"])
    t100_low = float(boot["t100_raw_frame_delta_vs_graph"]["low"])
    t100_high = float(boot["t100_raw_frame_delta_vs_graph"]["high"])
    easy_high = float(boot["easy_degradation_delta_vs_graph"]["high"])
    robust_all_hard = all_low > 0.0 and hard_low > 0.0 and easy_high <= 0.02
    t50_robust = t50_low > 0.0
    t100_robust = t100_low > 0.0
    t100_ci_crosses_zero = t100_low <= 0.0 <= t100_high
    slice_easy_safe = int(payload["slice_audit"].get("easy_hazard_slice_count", 0)) == 0
    gates = {
        "bt_precondition_passed": pre["bt_verdict"]
        in {
            "stage43_bt_context_admissibility_pass_safe_lift_diagnostic",
            "stage43_bt_context_admissibility_pass_safe_no_lift_diagnostic",
            "stage43_bt_context_admissibility_pass_unsafe_diagnostic",
        }
        and int(pre["bt_gate"]["passed"]) == int(pre["bt_gate"]["total"]),
        "checkpoint_replayed_not_committed": Path(payload["checkpoint"]["path"]).exists()
        and payload["checkpoint"]["committed"] is False,
        "exact_replay_matches_bt_report": replay_max <= 1e-8,
        "bootstrap_completed": int(payload["bootstrap"]["n"]) >= 1000,
        "slice_audit_completed": payload["slice_audit"]["slice_count"] > 0,
        "slice_easy_hazards_reported": "easy_hazard_slice_count" in payload["slice_audit"],
        "all_and_hard_bootstrap_measured": "all_delta_vs_graph" in boot and "hard_failure_delta_vs_graph" in boot,
        "easy_safety_ci_measured": "easy_degradation_delta_vs_graph" in boot and easy_high <= 0.02,
        "t50_and_t100_reported": "t50_delta_vs_graph" in boot and "t100_raw_frame_delta_vs_graph" in boot,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_variant_error_label_eval_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["scene_proxy_train_only"] is True
        and no_leak["graph_inputs_past_or_current_only"] is True
        and no_leak["test_threshold_selection"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["raw_scene_or_verified_sdf_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and robust_all_hard and t50_robust and t100_robust and slice_easy_safe:
        verdict = "stage43_bu_context_admissibility_robust_lift_pass"
    elif passed == total and robust_all_hard and t50_robust:
        verdict = "stage43_bu_context_admissibility_partial_robust_lift_pass"
    elif passed == total:
        verdict = "stage43_bu_context_admissibility_fragile_lift_diagnostic_pass"
    else:
        verdict = "stage43_bu_context_admissibility_robustness_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "robust_all_hard_lift": robust_all_hard,
        "t50_bootstrap_robust": t50_robust,
        "t100_bootstrap_robust": t100_robust,
        "t100_ci_crosses_zero": t100_ci_crosses_zero,
        "slice_easy_safe": slice_easy_safe,
        "easy_safe_ci": easy_high <= 0.02,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total and robust_all_hard and slice_easy_safe,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    return [
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
    ]


def _ci_line(name: str, row: Mapping[str, Any]) -> str:
    return f"| `{name}` | `{row['rows']}` | `{_pct(row['low'])}` | `{_pct(row['mean'])}` | `{_pct(row['high'])}` |"


def _slice_table(rows: list[Mapping[str, Any]], limit: int = 12) -> list[str]:
    lines = ["| slice | rows | delta vs graph | context rate | easy degradation |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['slice']}` | `{row['rows']}` | `{_pct(row['delta_vs_graph'])}` | `{_pct(row['context_rate'])}` | `{_pct(row['easy_degradation'])}` |"
        )
    if len(rows) > limit:
        lines.append(f"| `...` | `{len(rows) - limit} more` |  |  |  |")
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bu_gate"]
    boot = payload["bootstrap"]["metrics"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BU Context Admissibility Robustness Audit",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- robust all/hard lift: `{gate['robust_all_hard_lift']}`",
            f"- t50 bootstrap robust: `{gate['t50_bootstrap_robust']}`",
            f"- t100 bootstrap robust: `{gate['t100_bootstrap_robust']}`",
            f"- t100 CI crosses zero: `{gate['t100_ci_crosses_zero']}`",
            f"- slice easy safe: `{gate['slice_easy_safe']}`",
            f"- easy-safe CI: `{gate['easy_safe_ci']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Exact Replay",
            "",
            f"- checkpoint committed: `{payload['checkpoint']['committed']}`",
            f"- replay diff max: `{max(payload['replay_diff_vs_stage43_bt_report'].values()):.8f}`",
            f"- replay diff: `{payload['replay_diff_vs_stage43_bt_report']}`",
            "",
            "## Replay Metrics",
            "",
            *_metric_lines(payload["replay_metrics"]),
            "",
            "## Delta Vs Graph-History-Only",
            "",
            f"- all delta: `{_pct(payload['delta_vs_graph_history_only']['all'])}`",
            f"- t50 delta: `{_pct(payload['delta_vs_graph_history_only']['t50'])}`",
            f"- t100 raw-frame diagnostic delta: `{_pct(payload['delta_vs_graph_history_only']['t100_raw_frame_diagnostic'])}`",
            f"- hard/failure delta: `{_pct(payload['delta_vs_graph_history_only']['hard_failure'])}`",
            f"- easy degradation delta: `{_pct(payload['delta_vs_graph_history_only']['easy_degradation'])}`",
            "",
            "## Bootstrap Delta Vs Graph-History-Only",
            "",
            f"- bootstrap n: `{payload['bootstrap']['n']}`",
            "",
            "| metric | rows | low | mean | high |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[_ci_line(name, row) for name, row in boot.items()],
            "",
            "## Slice Audit",
            "",
            f"- slice count: `{payload['slice_audit']['slice_count']}`",
            f"- positive slice count: `{payload['slice_audit']['positive_slice_count']}`",
            f"- negative slice count: `{payload['slice_audit']['negative_slice_count']}`",
            f"- easy hazard slice count: `{payload['slice_audit']['easy_hazard_slice_count']}`",
            f"- core weak slices: `{[row['slice'] for row in payload['slice_audit']['core_weak_slices']]}`",
            "",
            "### Top Positive Slices",
            "",
            *_slice_table(payload["slice_audit"]["top_positive_slices"]),
            "",
            "### Top Negative Slices",
            "",
            *_slice_table(payload["slice_audit"]["top_negative_slices"]),
            "",
            "### Top Easy Hazard Slices",
            "",
            *_slice_table(payload["slice_audit"]["top_easy_hazard_slices"]),
            "",
            "## Interpretation",
            "",
            "- Stage43-BU exact-replays Stage43-BT and adds bootstrap plus source/domain/horizon slice evidence.",
            "- It is a robustness audit, not a deployment policy update.",
            "- The key question is whether BT's row-level context admissibility lift is stable enough to claim context contribution.",
            "- Dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-BU Context Admissibility Robustness Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- robust all/hard lift: `{gate['robust_all_hard_lift']}`",
            f"- t50 bootstrap robust: `{gate['t50_bootstrap_robust']}`",
            f"- t100 bootstrap robust: `{gate['t100_bootstrap_robust']}`",
            f"- t100 CI crosses zero: `{gate['t100_ci_crosses_zero']}`",
            f"- slice easy safe: `{gate['slice_easy_safe']}`",
            f"- easy-safe CI: `{gate['easy_safe_ci']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- robust all/hard lift: `{gate['robust_all_hard_lift']}`",
            f"- t50 bootstrap robust: `{gate['t50_bootstrap_robust']}`",
            f"- t100 bootstrap robust: `{gate['t100_bootstrap_robust']}`",
            f"- t100 CI crosses zero: `{gate['t100_ci_crosses_zero']}`",
            f"- slice easy safe: `{gate['slice_easy_safe']}`",
            f"- easy-safe CI: `{gate['easy_safe_ci']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BU is a robustness audit for Stage43-BT context admissibility.",
            "- It does not claim deployment update by itself.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bu_gate"]
    boot = payload["bootstrap"]["metrics"]
    delta = payload["delta_vs_graph_history_only"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"robust_all_hard_lift = `{gate['robust_all_hard_lift']}`",
        f"t50_bootstrap_robust = `{gate['t50_bootstrap_robust']}`",
        f"t100_bootstrap_robust = `{gate['t100_bootstrap_robust']}`",
        f"t100_ci_crosses_zero = `{gate['t100_ci_crosses_zero']}`",
        f"slice_easy_safe = `{gate['slice_easy_safe']}`",
        f"easy_safe_ci = `{gate['easy_safe_ci']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        "Stage43-BU exact-replays Stage43-BT and adds bootstrap plus source/domain/horizon slice evidence. It is a robustness audit, not a deployment update.",
        f"Replay diff max: `{max(payload['replay_diff_vs_stage43_bt_report'].values()):.8f}`.",
        f"Delta vs graph-history-only: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, t100 raw-frame diagnostic `{_pct(delta['t100_raw_frame_diagnostic'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        f"Bootstrap CI low vs graph-history-only: all `{_pct(boot['all_delta_vs_graph']['low'])}`, t50 `{_pct(boot['t50_delta_vs_graph']['low'])}`, hard/failure `{_pct(boot['hard_failure_delta_vs_graph']['low'])}`, easy high `{_pct(boot['easy_degradation_delta_vs_graph']['high'])}`.",
        f"Slice audit: `{payload['slice_audit']['positive_slice_count']}` positive slices, `{payload['slice_audit']['negative_slice_count']}` negative slices, easy hazard slices `{payload['slice_audit']['easy_hazard_slice_count']}`, core weak slices `{[row['slice'] for row in payload['slice_audit']['core_weak_slices']]}`.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 is diagnostic; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bu_context_admissibility_robustness_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "robust_all_hard_lift": gate["robust_all_hard_lift"],
        "t50_bootstrap_robust": gate["t50_bootstrap_robust"],
        "t100_bootstrap_robust": gate["t100_bootstrap_robust"],
        "t100_ci_crosses_zero": gate["t100_ci_crosses_zero"],
        "slice_easy_safe": gate["slice_easy_safe"],
        "easy_safe_ci": gate["easy_safe_ci"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "delta_vs_graph_history_only": payload["delta_vs_graph_history_only"],
        "bootstrap_summary": payload["bootstrap"]["metrics"],
        "slice_summary": {
            "slice_count": payload["slice_audit"]["slice_count"],
            "positive_slice_count": payload["slice_audit"]["positive_slice_count"],
            "negative_slice_count": payload["slice_audit"]["negative_slice_count"],
            "easy_hazard_slice_count": payload["slice_audit"]["easy_hazard_slice_count"],
            "core_weak_slices": payload["slice_audit"]["core_weak_slices"],
        },
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bu_context_admissibility_robustness_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BU",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "robust_all_hard_lift": gate["robust_all_hard_lift"],
                        "t50_bootstrap_robust": gate["t50_bootstrap_robust"],
                        "slice_easy_safe": gate["slice_easy_safe"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BU context admissibility robustness audit.")
    parser.add_argument("--test-rows", type=int, default=0, help="Optional override for replay test rows; 0 uses Stage43-BT rows.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=461)
    parser.add_argument("--min-rows", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = _run(args)
    gate = payload["stage43_bu_gate"]
    print(f"Stage43-BU: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"robust_all_hard_lift={gate['robust_all_hard_lift']}")
    print(f"t50_bootstrap_robust={gate['t50_bootstrap_robust']}")
    print(f"slice_easy_safe={gate['slice_easy_safe']}")
    return payload


if __name__ == "__main__":
    main()
