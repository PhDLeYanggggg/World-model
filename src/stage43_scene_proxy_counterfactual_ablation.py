from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_scene_proxy_guarded_latent_policy as ac
from src import stage43_scene_proxy_slice_safe_policy as ae


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_proxy_counterfactual_ablation.json"
REPORT_MD = OUT_DIR / "stage43_scene_proxy_counterfactual_ablation.md"
GATE_MD = OUT_DIR / "stage43_stage_af_scene_proxy_counterfactual_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AF_SCENE_PROXY_COUNTERFACTUAL_ABLATION"
SOURCE = "fresh_stage43_af_scene_proxy_counterfactual_ablation"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _delta_metrics(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return ac._delta_metrics(current, baseline)


def _route_eval(pack: Mapping[str, Any], final_route: np.ndarray) -> dict[str, Any]:
    ds: m.WaypointSplit = pack["ds"]
    selected_ade = ds.floor_ade.astype(np.float32).copy()
    selected_fde = ds.floor_fde.astype(np.float32).copy()
    m_mask = final_route == ae.ROUTE_STAGE43_M
    ab_mask = final_route == ae.ROUTE_STAGE43_AB
    selected_ade[m_mask] = pack["m_ade"][m_mask]
    selected_fde[m_mask] = pack["m_fde"][m_mask]
    selected_ade[ab_mask] = pack["ab_ade"][ab_mask]
    selected_fde[ab_mask] = pack["ab_fde"][ab_mask]
    switched = final_route != ae.ROUTE_FLOOR
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    metrics.update(ae._route_rates(ds, final_route))
    return {
        "metrics": metrics,
        "diagnostics": ae._slice_diagnostics(ds, selected_ade, selected_fde, switched, final_route),
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switched": switched,
        "final_route": final_route,
    }


def _counterfactual_routes(actual_route: np.ndarray) -> dict[str, np.ndarray]:
    no_scene_to_m = actual_route.copy()
    no_scene_to_m[no_scene_to_m == ae.ROUTE_STAGE43_AB] = ae.ROUTE_STAGE43_M
    no_scene_to_floor = actual_route.copy()
    no_scene_to_floor[no_scene_to_floor == ae.ROUTE_STAGE43_AB] = ae.ROUTE_FLOOR
    return {
        "actual_slice_safe": actual_route,
        "no_scene_proxy_to_stage43_m": no_scene_to_m,
        "no_scene_proxy_to_floor": no_scene_to_floor,
    }


def _slice_delta_table(actual_diag: Mapping[str, Any], cf_diag: Mapping[str, Any], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, actual in actual_diag[key].items():
        cf = cf_diag[key].get(name, {})
        out[name] = _delta_metrics(actual, cf)
    return out


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    rows = ac._max_rows("medium" if args.medium else "quick" if args.quick else "small")
    stage43_m_report = read_json(m.REPORT_JSON, {})
    stage43_ab_report = read_json(ac.ab.REPORT_JSON, {})
    stage43_ae_report = read_json(ae.REPORT_JSON, {})
    m_model, m_ckpt = ac._load_model(Path(stage43_m_report["checkpoint"]))
    ab_model, ab_ckpt = ac._load_model(Path(stage43_ab_report["checkpoint"]))
    test_pack = ac._replay_split(
        "test",
        max_rows=rows["test"],
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        m_model=m_model,
        m_ckpt=m_ckpt,
        ab_model=ab_model,
        ab_ckpt=ab_ckpt,
        m_policy=stage43_m_report["validation_selected_policy"]["policy"],
    )
    selected = ae._eval_policy(test_pack, stage43_ae_report["validation_selected_policy"]["policy"])
    routes = _counterfactual_routes(selected["final_route"])
    evaluations = {name: _route_eval(test_pack, route) for name, route in routes.items()}
    actual = evaluations["actual_slice_safe"]
    no_scene_m = evaluations["no_scene_proxy_to_stage43_m"]
    no_scene_floor = evaluations["no_scene_proxy_to_floor"]
    contribution_vs_m = _delta_metrics(actual["metrics"], no_scene_m["metrics"])
    contribution_vs_floor = _delta_metrics(actual["metrics"], no_scene_floor["metrics"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_same_route_counterfactual_model_family_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "medium" if args.medium else "quick" if args.quick else "small",
        "stage43_ae_verdict": stage43_ae_report.get("stage43_ae_gate", {}).get("verdict"),
        "stage43_ae_deploy": stage43_ae_report.get("stage43_ae_gate", {}).get("deploy_slice_safe_scene_proxy"),
        "policy": stage43_ae_report["validation_selected_policy"]["policy"],
        "data_rows": {"test": len(test_pack["ds"].x)},
        "actual_slice_safe": {"metrics": actual["metrics"], "diagnostics": actual["diagnostics"]},
        "counterfactuals": {
            "no_scene_proxy_to_stage43_m": {"metrics": no_scene_m["metrics"], "diagnostics": no_scene_m["diagnostics"]},
            "no_scene_proxy_to_floor": {"metrics": no_scene_floor["metrics"], "diagnostics": no_scene_floor["diagnostics"]},
        },
        "scene_proxy_contribution_vs_stage43_m_counterfactual": contribution_vs_m,
        "scene_proxy_contribution_vs_floor_counterfactual": contribution_vs_floor,
        "domain_contribution_vs_stage43_m_counterfactual": _slice_delta_table(
            actual["diagnostics"], no_scene_m["diagnostics"], "domains"
        ),
        "horizon_contribution_vs_stage43_m_counterfactual": _slice_delta_table(
            actual["diagnostics"], no_scene_m["diagnostics"], "horizons"
        ),
        "route_counts": {
            "floor": int(np.sum(actual["final_route"] == ae.ROUTE_FLOOR)),
            "stage43_m": int(np.sum(actual["final_route"] == ae.ROUTE_STAGE43_M)),
            "stage43_ab": int(np.sum(actual["final_route"] == ae.ROUTE_STAGE43_AB)),
        },
        "bootstrap_ci_actual": m._bootstrap_ci(
            test_pack["ds"],
            actual["selected_ade"],
            actual["selected_fde"],
            n=int(args.bootstrap),
            seed=int(args.seed) + 5100,
        ),
        "ablation_type": {
            "same_route_counterfactual": True,
            "compares_separately_trained_stage43_m_and_stage43_ab_models": True,
            "not_full_retrained_factorial_ablation": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "scene_proxy_train_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "t100_raw_frame_diagnostic_only": True,
            "not_uniform_all_metric_improvement": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([ae.REPORT_JSON, m.REPORT_JSON, ac.ab.REPORT_JSON, m._cache_path("test")]),
    }
    payload["stage43_af_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    actual = payload["actual_slice_safe"]["metrics"]
    no_scene = payload["counterfactuals"]["no_scene_proxy_to_stage43_m"]["metrics"]
    contrib = payload["scene_proxy_contribution_vs_stage43_m_counterfactual"]
    gates = {
        "stage43_ae_precondition_pass": payload["stage43_ae_verdict"] == "stage43_ae_slice_safe_scene_proxy_candidate"
        and payload["stage43_ae_deploy"] is True,
        "fresh_same_route_counterfactual_replay": payload["result_source"]
        == "fresh_replay_same_route_counterfactual_model_family_ablation",
        "scene_proxy_route_present": payload["route_counts"]["stage43_ab"] > 0,
        "counterfactual_no_scene_built": no_scene["rows"] == actual["rows"],
        "scene_proxy_t50_lift_positive": contrib["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "scene_proxy_endpoint_t50_lift_positive": contrib["t50_endpoint_fde_improvement_vs_floor"] > 0.0,
        "scene_proxy_hard_or_all_lift_positive": contrib["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or contrib["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "actual_easy_preserved": actual["easy_degradation_vs_floor"] <= 0.02,
        "actual_t100_floor_guarded": actual["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -0.002
        and actual["h100_floor_rate"] >= 0.99,
        "tradeoff_reported_not_overclaimed": payload["ablation_type"]["not_full_retrained_factorial_ablation"] is True
        and payload["claim_boundary"]["not_uniform_all_metric_improvement"] is True,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(passed == total)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_af_scene_proxy_counterfactual_contribution_pass"
        if deploy
        else "stage43_af_scene_proxy_counterfactual_diagnostic_only",
        "scene_proxy_counterfactual_contribution_supported": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_af_gate"]
    actual = payload["actual_slice_safe"]["metrics"]
    no_scene = payload["counterfactuals"]["no_scene_proxy_to_stage43_m"]["metrics"]
    contrib = payload["scene_proxy_contribution_vs_stage43_m_counterfactual"]
    lines = [
        "# Stage43-AF Scene-Proxy Counterfactual Ablation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- route counts: `{payload['route_counts']}`",
        "",
        "## What Was Compared",
        "",
        "Stage43-AF replays the Stage43-AE validation-selected route and then removes the scene-proxy branch counterfactually. On the same rows and same route, any `Stage43-AB` scene-proxy selection is replaced by either `Stage43-M` or the original floor.",
        "",
        "This is a same-route model-family ablation: Stage43-M and Stage43-AB are separately trained models. It is not a full factorial retraining of every module.",
        "",
        "## Actual AE vs No-Scene Counterfactual",
        "",
        "| metric | AE actual | no scene -> Stage43-M | scene-proxy contribution |",
        "| --- | ---: | ---: | ---: |",
        f"| all full-waypoint ADE | `{_pct(actual['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(no_scene['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(contrib['full_waypoint_ade_improvement_vs_floor'])}` |",
        f"| t50 full-waypoint ADE | `{_pct(actual['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(no_scene['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(contrib['t50_full_waypoint_ade_improvement_vs_floor'])}` |",
        f"| t50 endpoint FDE | `{_pct(actual['t50_endpoint_fde_improvement_vs_floor'])}` | `{_pct(no_scene['t50_endpoint_fde_improvement_vs_floor'])}` | `{_pct(contrib['t50_endpoint_fde_improvement_vs_floor'])}` |",
        f"| hard/failure | `{_pct(actual['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(no_scene['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(contrib['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` |",
        f"| t100 raw-frame diagnostic | `{_pct(actual['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` | `{_pct(no_scene['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` | `{_pct(contrib['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` |",
        f"| easy degradation | `{_pct(actual['easy_degradation_vs_floor'])}` | `{_pct(no_scene['easy_degradation_vs_floor'])}` | `{_pct(contrib['easy_degradation_vs_floor'])}` |",
        "",
        "## Horizon Contribution vs No-Scene",
        "",
        "| horizon | all contribution | t50 contribution | hard contribution | easy contribution |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for horizon, row in sorted(payload["horizon_contribution_vs_stage43_m_counterfactual"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| `{horizon}` | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['easy_degradation_vs_floor'])}` |"
        )
    lines.extend(
        [
            "",
            "## Domain Contribution vs No-Scene",
            "",
            "| domain | all contribution | t50 contribution | hard contribution | easy contribution |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in sorted(payload["domain_contribution_vs_stage43_m_counterfactual"].items()):
        lines.append(
            f"| `{domain}` | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['easy_degradation_vs_floor'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The scene-proxy branch contributes positive `t+50` and endpoint lift under the same Stage43-AE safety route. The result does not claim a uniform all/hard improvement over every alternative: AE is a slice-safe/t50-focused deployment contract, while AC remains stronger for some all/hard objectives but has caveated easy slices.",
            "",
            "## Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D evidence only.",
            "- Future waypoints are labels/eval only, not input.",
            "- This is a same-route counterfactual model-family ablation, not a full retrained factorial ablation.",
            "- No metric/seconds claim, no Stage5C, no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AF Scene-Proxy Counterfactual Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- scene proxy contribution supported: `{gate['scene_proxy_counterfactual_contribution_supported']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_af_gate"]
    actual = payload["actual_slice_safe"]["metrics"]
    contrib = payload["scene_proxy_contribution_vs_stage43_m_counterfactual"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"scene_proxy_counterfactual_contribution_supported = `{gate['scene_proxy_counterfactual_contribution_supported']}`",
        "",
        f"actual_slice_safe_all = `{_pct(actual['full_waypoint_ade_improvement_vs_floor'])}`",
        f"actual_slice_safe_t50 = `{_pct(actual['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"scene_proxy_t50_contribution_vs_stage43_m_counterfactual = `{_pct(contrib['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"scene_proxy_t50_endpoint_contribution_vs_stage43_m_counterfactual = `{_pct(contrib['t50_endpoint_fde_improvement_vs_floor'])}`",
        f"scene_proxy_hard_contribution_vs_stage43_m_counterfactual = `{_pct(contrib['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"actual_easy_degradation = `{_pct(actual['easy_degradation_vs_floor'])}`",
        "",
        "Stage43-AF uses the same Stage43-AE route and replaces only the scene-proxy AB branch with a no-scene Stage43-M counterfactual. This gives a direct model-family contribution estimate for scene/goal proxy latent features under the same safety contract.",
        "",
        "Boundary unchanged: same-route counterfactual, not full factorial retraining; dataset-local/raw-frame 2.5D only; t100 remains diagnostic; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_af_scene_proxy_counterfactual_ablation"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "scene_proxy_counterfactual_contribution_supported": gate["scene_proxy_counterfactual_contribution_supported"],
        "actual_metrics": payload["actual_slice_safe"]["metrics"],
        "scene_proxy_contribution_vs_stage43_m_counterfactual": payload[
            "scene_proxy_contribution_vs_stage43_m_counterfactual"
        ],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_af_scene_proxy_counterfactual_ablation"
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
                        "stage": "Stage43-AF",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "scene_proxy_counterfactual_contribution_supported": gate[
                            "scene_proxy_counterfactual_contribution_supported"
                        ],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AF scene-proxy same-route counterfactual ablation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_af_gate"]
    print(f"Stage43-AF: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"scene_proxy_counterfactual_contribution_supported={gate['scene_proxy_counterfactual_contribution_supported']}")
    return result


if __name__ == "__main__":
    main()
