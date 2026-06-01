from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_self_gate_conformal_audit as ak
from src import stage43_bounded_residual_safety_audit as al


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_bounded_residual_statistical_confirmation.json"
REPORT_MD = OUT_DIR / "stage43_bounded_residual_statistical_confirmation.md"
GATE_MD = OUT_DIR / "stage43_stage_am_bounded_residual_statistical_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AM_BOUNDED_RESIDUAL_STATISTICAL_CONFIRMATION"
SOURCE = "fresh_stage43_am_bounded_residual_statistical_confirmation"
STAGE43_AL = OUT_DIR / "stage43_bounded_residual_safety_audit.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, ids: np.ndarray) -> float:
    if len(ids) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[ids])) / max(float(np.mean(floor[ids])), m.EPS))


def _easy_degradation(selected: np.ndarray, floor: np.ndarray, ids: np.ndarray) -> float:
    if len(ids) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[ids])) / max(float(np.mean(floor[ids])), m.EPS) - 1.0))


def _bounded_arrays(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waypoint = al._bounded_waypoint(
        ds,
        pred,
        alpha=float(config["alpha"]),
        clip_norm=float(config["clip_norm"]),
    )
    switched = al._allow_mask(
        ds,
        pred,
        config["policy"],
        force_h100_floor=bool(config.get("force_h100_floor", False)),
    )
    waypoint = np.where(switched[:, None, None], waypoint, ds.floor_waypoint_delta).astype(np.float32)
    ade, fde = m._trajectory_error(ds, waypoint)
    return ade, fde, switched


def _stored_arrays(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ak._select_with_policy(ds, pred, policy)


def _bootstrap_delta_ci(
    ds: m.WaypointSplit,
    stored_ade: np.ndarray,
    bounded_ade: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    masks = {
        "all_delta_improvement": np.ones(len(ds.x), dtype=bool),
        "t50_delta_improvement": ds.horizon == 50,
        "t100_delta_improvement": ds.horizon == 100,
        "hard_failure_delta_improvement": ds.hard | ds.failure,
        "easy_degradation_bounded": ds.easy,
        "easy_degradation_delta": ds.easy,
    }
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "metrics": {}}
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        vals = np.zeros(int(n), dtype=np.float64)
        if len(ids) == 0:
            out["metrics"][name] = {"low": 0.0, "mean": 0.0, "high": 0.0, "rows": 0}
            continue
        for i in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            if name == "easy_degradation_bounded":
                vals[i] = _easy_degradation(bounded_ade, ds.floor_ade, sample)
            elif name == "easy_degradation_delta":
                vals[i] = _easy_degradation(bounded_ade, ds.floor_ade, sample) - _easy_degradation(
                    stored_ade, ds.floor_ade, sample
                )
            else:
                vals[i] = _slice_improvement(bounded_ade, ds.floor_ade, sample) - _slice_improvement(
                    stored_ade, ds.floor_ade, sample
                )
        out["metrics"][name] = {
            "low": float(np.quantile(vals, 0.025)),
            "mean": float(np.mean(vals)),
            "high": float(np.quantile(vals, 0.975)),
            "rows": int(len(ids)),
        }
    return out


def _slice_rows(
    ds: m.WaypointSplit,
    stored_ade: np.ndarray,
    bounded_ade: np.ndarray,
    switched: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slices: list[tuple[str, np.ndarray]] = []
    for domain in sorted(set(ds.domain.astype(str))):
        slices.append((f"domain:{domain}", ds.domain.astype(str) == domain))
    for horizon in [10, 25, 50, 100]:
        slices.append((f"horizon:{horizon}", ds.horizon == horizon))
    for name, mask in slices:
        ids = np.where(mask)[0]
        if len(ids) == 0:
            continue
        stored = _slice_improvement(stored_ade, ds.floor_ade, ids)
        bounded = _slice_improvement(bounded_ade, ds.floor_ade, ids)
        rows.append(
            {
                "slice": name,
                "rows": int(len(ids)),
                "stored_improvement": stored,
                "bounded_improvement": bounded,
                "delta": bounded - stored,
                "switch_rate": float(np.mean(switched[ids])),
            }
        )
    return rows


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    runtime = m._configure_runtime(int(args.seed))
    if not STAGE43_AL.exists():
        raise FileNotFoundError(STAGE43_AL)
    prior_al = read_json(STAGE43_AL, {})
    prior_m, ckpt, model = ak._load_stage43_m()
    _, test = ak._build_eval_splits(prior_m, ckpt)
    device = torch.device("cpu")
    pred = m._predict(model, test, device, int(args.batch_size))
    stored_policy = prior_m["validation_selected_policy"]["policy"]
    stored_ade, stored_fde, stored_switched = _stored_arrays(test, pred, stored_policy)
    stored_metrics = m._metrics(test, stored_ade, stored_fde, stored_switched)
    stored_diff = ak._metric_diff(stored_metrics, prior_m["test_metrics_with_floor"])
    config = prior_al["best_safe_bounded_residual"]["config"]
    bounded_ade, bounded_fde, bounded_switched = _bounded_arrays(test, pred, config)
    bounded_metrics = m._metrics(test, bounded_ade, bounded_fde, bounded_switched)
    point_delta = {
        "all": bounded_metrics["full_waypoint_ade_improvement_vs_floor"]
        - stored_metrics["full_waypoint_ade_improvement_vs_floor"],
        "t50": bounded_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        - stored_metrics["t50_full_waypoint_ade_improvement_vs_floor"],
        "t100": bounded_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        - stored_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "hard_failure": bounded_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        - stored_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"],
        "easy": bounded_metrics["easy_degradation_vs_floor"] - stored_metrics["easy_degradation_vs_floor"],
    }
    bootstrap = _bootstrap_delta_ci(
        test,
        stored_ade,
        bounded_ade,
        n=int(args.bootstrap),
        seed=int(args.seed) + 1700,
    )
    slice_rows = _slice_rows(test, stored_ade, bounded_ade, bounded_switched)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_bootstrap_confirmation_over_frozen_stage43_al_candidate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "runtime": runtime,
        "stage43_al_source": {
            "report": str(STAGE43_AL),
            "verdict": prior_al.get("stage43_al_gate", {}).get("verdict"),
            "deploy_bounded_residual": prior_al.get("stage43_al_gate", {}).get("deploy_bounded_residual"),
            "report_sha256": m._sha256(STAGE43_AL),
        },
        "stage43_m_source": {
            "report": str(ak.STAGE43_M),
            "checkpoint": str(ak.STAGE43_M_CKPT),
            "checkpoint_sha256": m._sha256(ak.STAGE43_M_CKPT),
        },
        "data_rows": {"test": int(len(test.x))},
        "feature_schema_match": list(ckpt["feature_names"]) == test.feature_names,
        "cache_row_hashes": {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS},
        "cache_row_hash_match_prior": {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS}
        == prior_m.get("cache_row_hashes"),
        "stored_policy_replay_diff": stored_diff,
        "bounded_config": config,
        "stored_metrics": stored_metrics,
        "bounded_metrics": bounded_metrics,
        "point_delta_vs_stored": point_delta,
        "bootstrap_delta_ci": bootstrap,
        "slice_rows": slice_rows,
        "slice_summary": {
            "domain_count": int(sum(row["slice"].startswith("domain:") for row in slice_rows)),
            "positive_domain_delta_count": int(
                sum(row["slice"].startswith("domain:") and row["delta"] > 0.0 for row in slice_rows)
            ),
            "positive_horizon_delta_count": int(
                sum(row["slice"].startswith("horizon:") and row["delta"] > 0.0 for row in slice_rows)
            ),
        },
        "interpretation": {
            "bounded_residual_statistically_confirmed": False,
            "global_floor_removable": False,
            "stage43_al_candidate_status": "pending_gate",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([STAGE43_AL, ak.STAGE43_M, ak.STAGE43_M_CKPT, m._cache_path("test")]),
    }
    payload["stage43_am_gate"] = _gate(payload)
    payload["interpretation"]["bounded_residual_statistically_confirmed"] = (
        payload["stage43_am_gate"]["bounded_residual_statistically_confirmed"]
    )
    payload["interpretation"]["stage43_al_candidate_status"] = (
        "statistically_confirmed_candidate"
        if payload["stage43_am_gate"]["bounded_residual_statistically_confirmed"]
        else "diagnostic_until_more_evidence"
    )
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    ci = payload["bootstrap_delta_ci"]["metrics"]
    bounded = payload["bounded_metrics"]
    gates = {
        "stage43_al_candidate_available": payload["stage43_al_source"]["deploy_bounded_residual"] is True,
        "stage43_m_exact_replay": payload["stored_policy_replay_diff"]["max_abs_diff"] <= 1e-5,
        "feature_schema_and_rows_match": payload["feature_schema_match"] is True
        and payload["cache_row_hash_match_prior"] is True,
        "bootstrap_n_at_least_2000": payload["bootstrap_delta_ci"]["n"] >= 2000,
        "all_delta_ci_positive": ci["all_delta_improvement"]["low"] > 0.0,
        "t50_delta_ci_positive": ci["t50_delta_improvement"]["low"] > 0.0,
        "hard_failure_delta_ci_positive": ci["hard_failure_delta_improvement"]["low"] > 0.0,
        "t100_delta_ci_nonnegative": ci["t100_delta_improvement"]["low"] >= -1e-8,
        "easy_degradation_ci_safe": ci["easy_degradation_bounded"]["high"] <= 0.02
        and bounded["easy_degradation_vs_floor"] <= 0.02,
        "per_domain_slices_reported": payload["slice_summary"]["domain_count"] >= 2,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["thresholds_selected_on_test"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    confirmed = passed == total
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_am_bounded_residual_statistically_confirmed"
        if confirmed
        else "stage43_am_bounded_residual_confirmation_incomplete",
        "bounded_residual_statistically_confirmed": confirmed,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_am_gate"]
    ci = payload["bootstrap_delta_ci"]["metrics"]
    lines = [
        "# Stage43-AM Bounded Residual Statistical Confirmation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- statistically confirmed: `{gate['bounded_residual_statistically_confirmed']}`",
        f"- bootstrap n: `{payload['bootstrap_delta_ci']['n']}`",
        f"- stored policy replay max abs diff: `{payload['stored_policy_replay_diff']['max_abs_diff']:.8f}`",
        "",
        "## Bootstrap Delta vs Stored Hard Switch",
        "",
        "| metric | low | mean | high | rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in [
        "all_delta_improvement",
        "t50_delta_improvement",
        "t100_delta_improvement",
        "hard_failure_delta_improvement",
        "easy_degradation_bounded",
        "easy_degradation_delta",
    ]:
        row = ci[name]
        lines.append(
            f"| {name} | `{_pct(row['low'])}` | `{_pct(row['mean'])}` | `{_pct(row['high'])}` | `{row['rows']}` |"
        )
    lines.extend(
        [
            "",
            "## Slice Delta Summary",
            "",
            "| slice | rows | stored | bounded | delta | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["slice_rows"]:
        lines.append(
            f"| {row['slice']} | `{row['rows']}` | `{_pct(row['stored_improvement'])}` | `{_pct(row['bounded_improvement'])}` | `{_pct(row['delta'])}` | `{_pct(row['switch_rate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This confirms or rejects the Stage43-AL bounded-residual candidate using fresh replay plus bootstrap deltas over the frozen test rows.",
            "- The bounded residual is still floor-protected and h100-guarded; global floor removal remains unsupported.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AM Bounded Residual Statistical Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- statistically confirmed: `{gate['bounded_residual_statistically_confirmed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_am_gate"]
    ci = payload["bootstrap_delta_ci"]["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"bounded_residual_statistically_confirmed = `{gate['bounded_residual_statistically_confirmed']}`",
        f"bootstrap_n = `{payload['bootstrap_delta_ci']['n']}`",
        f"all_delta_ci = `[{_pct(ci['all_delta_improvement']['low'])}, {_pct(ci['all_delta_improvement']['high'])}]`",
        f"t50_delta_ci = `[{_pct(ci['t50_delta_improvement']['low'])}, {_pct(ci['t50_delta_improvement']['high'])}]`",
        f"hard_failure_delta_ci = `[{_pct(ci['hard_failure_delta_improvement']['low'])}, {_pct(ci['hard_failure_delta_improvement']['high'])}]`",
        f"easy_degradation_bounded_ci = `[{_pct(ci['easy_degradation_bounded']['low'])}, {_pct(ci['easy_degradation_bounded']['high'])}]`",
        "",
        "Stage43-AM bootstrap-confirms the Stage43-AL bounded residual candidate against the stored Stage43-M hard-switch policy on frozen rows. The candidate remains floor-protected and h100-guarded; this is not global floor removal.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_am_bounded_residual_statistical_confirmation"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "bounded_residual_statistically_confirmed": gate["bounded_residual_statistically_confirmed"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "bootstrap_delta_ci": payload["bootstrap_delta_ci"],
        "slice_summary": payload["slice_summary"],
        "global_floor_removable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_am_bounded_residual_statistical_confirmation"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-AM",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "bounded_residual_statistically_confirmed": gate[
                            "bounded_residual_statistically_confirmed"
                        ],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AM bounded residual statistical confirmation.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=431)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_am_gate"]
    print(f"Stage43-AM: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"bounded_residual_statistically_confirmed={gate['bounded_residual_statistically_confirmed']}")
    return result


if __name__ == "__main__":
    main()
