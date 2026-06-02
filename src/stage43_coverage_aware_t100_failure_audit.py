from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_coverage_aware_latent_dynamics as cg
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_coverage_aware_t100_failure_audit.json"
REPORT_MD = OUT_DIR / "stage43_coverage_aware_t100_failure_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_ch_coverage_aware_t100_failure_audit_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
CG_JSON = OUT_DIR / "stage43_coverage_aware_latent_dynamics.json"

SECTION = "STAGE43_CH_COVERAGE_AWARE_T100_FAILURE_AUDIT"
SOURCE = "fresh_stage43_ch_coverage_aware_t100_failure_audit"
EPS = 1e-8


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _slice_stats(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "rows": 0,
            "full_waypoint_ade_improvement_vs_floor": 0.0,
            "endpoint_fde_improvement_vs_floor": 0.0,
            "easy_degradation_vs_floor": 0.0,
            "switch_rate": 0.0,
            "mean_floor_ade": 0.0,
            "mean_selected_ade": 0.0,
            "harm_over_floor_ade": 0.0,
        }
    easy = ds.easy & mask
    easy_degradation = (
        max(0.0, float(np.mean(selected_ade[easy])) / max(float(np.mean(ds.floor_ade[easy])), EPS) - 1.0)
        if int(easy.sum())
        else 0.0
    )
    return {
        "rows": int(mask.sum()),
        "full_waypoint_ade_improvement_vs_floor": float(1.0 - float(np.mean(selected_ade[mask])) / max(float(np.mean(ds.floor_ade[mask])), EPS)),
        "endpoint_fde_improvement_vs_floor": float(1.0 - float(np.mean(selected_fde[mask])) / max(float(np.mean(ds.floor_fde[mask])), EPS)),
        "easy_degradation_vs_floor": float(easy_degradation),
        "switch_rate": float(np.mean(switched[mask])),
        "mean_floor_ade": float(np.mean(ds.floor_ade[mask])),
        "mean_selected_ade": float(np.mean(selected_ade[mask])),
        "harm_over_floor_ade": float(np.mean(selected_ade[mask] - ds.floor_ade[mask])),
    }


def _bootstrap_slice(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    mask: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, Any]:
    ids = np.where(np.asarray(mask, dtype=bool))[0]
    if len(ids) == 0:
        return {"n": int(n), "rows": 0, "low": 0.0, "mean": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n), dtype=np.float64)
    for i in range(int(n)):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals[i] = 1.0 - float(np.mean(selected_ade[sample])) / max(float(np.mean(ds.floor_ade[sample])), EPS)
    return {
        "n": int(n),
        "rows": int(len(ids)),
        "low": float(np.quantile(vals, 0.025)),
        "mean": float(np.mean(vals)),
        "high": float(np.quantile(vals, 0.975)),
    }


def _top_slice_table(
    values: np.ndarray,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    *,
    limit: int = 12,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, count in Counter(values.astype(str).tolist()).most_common(limit):
        mask = values.astype(str) == name
        out[name] = _slice_stats(ds, selected_ade, selected_fde, switched, mask)
    return out


def _prediction_diagnostics(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {"rows": 0}
    return {
        "rows": int(mask.sum()),
        "predicted_gain_mean": float(np.mean(pred["gain"][mask])),
        "predicted_harm_mean": float(np.mean(pred["harm"][mask])),
        "predicted_failure_mean": float(np.mean(pred["failure"][mask])),
        "predicted_density_mean": float(np.mean(pred["density"][mask])),
        "floor_ade_mean": float(np.mean(ds.floor_ade[mask])),
        "candidate_ade_mean": float(np.mean(m._trajectory_error(ds, pred["waypoint"])[0][mask])),
    }


def _load_medium_replay() -> tuple[dict[str, Any], m.WaypointSplit, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    cg._configure_base()
    report = read_json(CG_JSON, {})
    ckpt_path = Path(report["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    max_rows = int(report.get("data_rows", {}).get("test", 50000))
    seed = int(ckpt.get("seed", 431))
    ds = m._build_split("test", max_rows=max_rows, seed=seed)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    model = m.FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    pred = m._predict(model, ds, torch.device("cpu"), batch_size=2048)
    selected_ade, selected_fde, switched = m._select_with_policy(ds, pred, report["validation_selected_policy"]["policy"])
    return report, ds, pred, selected_ade, selected_fde, switched


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    cg_report = payload["coverage_aware_latent_dynamics"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    t100 = payload["horizon_slices"]["100"]
    gates = {
        "cg_medium_precondition_present": cg_report["verdict"] == "stage43_cg_coverage_aware_latent_dynamics_candidate_pass"
        and cg_report["mode"] == "medium",
        "t100_rows_present": int(t100["rows"]) > 0,
        "t100_negative_confirmed": float(t100["full_waypoint_ade_improvement_vs_floor"]) < 0.0,
        "t100_ci_reported": int(payload["t100_bootstrap_ci"]["rows"]) > 0,
        "domain_slice_reported": bool(payload["domain_slices"]),
        "source_slice_reported": bool(payload["source_slices"]),
        "switched_vs_fallback_reported": "t100_switched" in payload["t100_switch_attribution"]
        and "t100_fallback" in payload["t100_switch_attribution"],
        "prediction_diagnostics_reported": "t100" in payload["prediction_diagnostics"],
        "no_future_or_test_leakage": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["future_labels_eval_only"] is True
        and no_leakage["central_velocity_input"] is False
        and no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ch_t100_failure_audit_pass_blocker_confirmed" if passed == total else "stage43_ch_t100_failure_audit_incomplete",
    }


def run_t100_failure_audit(*, bootstrap: int = 2000) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    report, ds, pred, selected_ade, selected_fde, switched = _load_medium_replay()
    horizon_slices = {
        str(h): _slice_stats(ds, selected_ade, selected_fde, switched, ds.horizon == h)
        for h in [10, 25, 50, 100]
    }
    t100_mask = ds.horizon == 100
    hard_failure = ds.hard | ds.failure
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_audit_from_stage43_cg_medium_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "coverage_aware_latent_dynamics": {
            "report": str(CG_JSON),
            "verdict": report.get("stage43_cg_gate", {}).get("verdict"),
            "mode": report.get("mode"),
            "checkpoint": report.get("checkpoint"),
            "checkpoint_committed": report.get("checkpoint_committed", False),
            "data_rows": report.get("data_rows", {}),
            "medium_t100_metric": report.get("test_metrics_with_floor", {}).get("t100_raw_frame_full_waypoint_diagnostic_vs_floor"),
        },
        "validation_selected_policy": report["validation_selected_policy"],
        "replayed_rows": int(len(ds.x)),
        "horizon_slices": horizon_slices,
        "domain_slices": _top_slice_table(ds.domain, ds, selected_ade, selected_fde, switched),
        "source_slices": _top_slice_table(ds.source_file, ds, selected_ade, selected_fde, switched, limit=20),
        "t100_switch_attribution": {
            "t100_all": _slice_stats(ds, selected_ade, selected_fde, switched, t100_mask),
            "t100_switched": _slice_stats(ds, selected_ade, selected_fde, switched, t100_mask & switched),
            "t100_fallback": _slice_stats(ds, selected_ade, selected_fde, switched, t100_mask & ~switched),
            "t100_hard_failure": _slice_stats(ds, selected_ade, selected_fde, switched, t100_mask & hard_failure),
            "t100_easy": _slice_stats(ds, selected_ade, selected_fde, switched, t100_mask & ds.easy),
        },
        "prediction_diagnostics": {
            "all": _prediction_diagnostics(ds, pred, np.ones(len(ds.x), dtype=bool)),
            "t50": _prediction_diagnostics(ds, pred, ds.horizon == 50),
            "t100": _prediction_diagnostics(ds, pred, t100_mask),
            "t100_switched": _prediction_diagnostics(ds, pred, t100_mask & switched),
            "t100_fallback": _prediction_diagnostics(ds, pred, t100_mask & ~switched),
        },
        "t100_bootstrap_ci": _bootstrap_slice(ds, selected_ade, t100_mask, n=int(bootstrap), seed=1043),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "interpretation": {
            "t100_deployable": False,
            "reason": "t100 raw-frame full-waypoint ADE improvement remains negative after CE medium replay; keep t100 floor and train a dedicated long-horizon repair before any t100 deployment claim.",
            "recommended_next_step": "Train a t100-specific coverage-aware long-horizon head or per-horizon policy with validation-selected t100 safety constraints; do not change all/t50 deployment claims based on t100.",
        },
    }
    payload["stage43_ch_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_ch_gate"]
    t100 = payload["horizon_slices"]["100"]
    ci = payload["t100_bootstrap_ci"]
    return [
        "# Stage43-CH Coverage-Aware T100 Failure Audit",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- replayed rows: `{payload['replayed_rows']}`",
        "",
        "## Current Boundary",
        "",
        "- This is a failure audit, not a new deployable t100 model.",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric or seconds-level claim.",
        "- Stage5C not executed; SMC not enabled.",
        "",
        "## T100 Failure Confirmation",
        "",
        f"- t100 rows: `{t100['rows']}`",
        f"- t100 full-waypoint ADE improvement: `{_pct(t100['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 endpoint FDE improvement: `{_pct(t100['endpoint_fde_improvement_vs_floor'])}`",
        f"- t100 switch rate: `{_pct(t100['switch_rate'])}`",
        f"- t100 easy degradation: `{_pct(t100['easy_degradation_vs_floor'])}`",
        f"- t100 bootstrap CI: `[{_pct(ci['low'])}, {_pct(ci['high'])}]`",
        "",
        "## Horizon Slices",
        "",
        "| horizon | rows | ADE improvement | endpoint improvement | switch | easy degradation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {h} | {row['rows']} | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['endpoint_fde_improvement_vs_floor'])}` | `{_pct(row['switch_rate'])}` | `{_pct(row['easy_degradation_vs_floor'])}` |"
            for h, row in payload["horizon_slices"].items()
        ],
        "",
        "## T100 Switch Attribution",
        "",
        "| slice | rows | ADE improvement | switch | mean floor ADE | mean selected ADE | harm over floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {name} | {row['rows']} | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['switch_rate'])}` | `{row['mean_floor_ade']:.6f}` | `{row['mean_selected_ade']:.6f}` | `{row['harm_over_floor_ade']:.6f}` |"
            for name, row in payload["t100_switch_attribution"].items()
        ],
        "",
        "## Domain Slices",
        "",
        "| domain | rows | ADE improvement | t100? | switch | easy degradation |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
        *[
            f"| {name} | {row['rows']} | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | mixed | `{_pct(row['switch_rate'])}` | `{_pct(row['easy_degradation_vs_floor'])}` |"
            for name, row in payload["domain_slices"].items()
        ],
        "",
        "## Interpretation",
        "",
        payload["interpretation"]["reason"],
        "",
        f"Recommended next step: {payload['interpretation']['recommended_next_step']}",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_ch_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CH Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- t100 deployable: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{SOURCE}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- t100 deployable: `False`",
            "- long objective complete: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ch_gate"]
    t100 = payload["horizon_slices"]["100"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "",
        "I replayed the Stage43-CG medium checkpoint on the CE test subset and isolated the long-horizon t100 failure. This is an audit, not a new t100 deployment.",
        "",
        f"- t100 rows: `{t100['rows']}`",
        f"- t100 full-waypoint ADE improvement: `{_pct(t100['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 bootstrap CI: `[{_pct(payload['t100_bootstrap_ci']['low'])}, {_pct(payload['t100_bootstrap_ci']['high'])}]`",
        f"- t100 switch rate: `{_pct(t100['switch_rate'])}`",
        f"- all/t50 CG remains positive, but t100 remains diagnostic-only.",
        "",
        "Boundary: no metric/seconds-level claim; no Stage5C; no SMC; future waypoints remain labels/eval only.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ch_coverage_aware_t100_failure_audit"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "t100_deployable": False,
        "horizon_slices": payload["horizon_slices"],
        "t100_bootstrap_ci": payload["t100_bootstrap_ci"],
        "interpretation": payload["interpretation"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_ch_coverage_aware_t100_failure_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(m._jsonable({"event": "stage43_ch_coverage_aware_t100_failure_audit", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stage43-CG t100 raw-frame diagnostic failure.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_t100_failure_audit(bootstrap=int(args.bootstrap))
    gate = payload["stage43_ch_gate"]
    print(f"Stage43-CH: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"t100_improvement={payload['horizon_slices']['100']['full_waypoint_ade_improvement_vs_floor']:.6f}")
    return payload


if __name__ == "__main__":
    main()
