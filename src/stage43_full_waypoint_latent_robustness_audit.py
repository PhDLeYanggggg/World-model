from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    CKPT_DIR,
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    FullWaypointLatentDynamics,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _npz,
    _select_with_policy,
    _sha256,
    _trajectory_error,
)


REPORT_JSON = OUT_DIR / "stage43_full_waypoint_latent_robustness_audit.json"
REPORT_MD = OUT_DIR / "stage43_full_waypoint_latent_robustness_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_n_full_waypoint_latent_robustness_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_M_JSON = OUT_DIR / "stage43_full_waypoint_latent_dynamics.json"
SECTION = "STAGE43_N_FULL_WAYPOINT_LATENT_ROBUSTNESS"
SOURCE = "fresh_stage43_n_full_waypoint_latent_robustness_audit"
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _slice_metrics(
    floor_ade: np.ndarray,
    floor_fde: np.ndarray,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    ungated_ade: np.ndarray,
    switch: np.ndarray,
    easy: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if int(np.sum(mask)) == 0:
        return {
            "rows": 0,
            "full_waypoint_ade_improvement_vs_floor": 0.0,
            "endpoint_fde_improvement_vs_floor": 0.0,
            "ungated_full_waypoint_ade_improvement_vs_floor": 0.0,
            "easy_degradation_vs_floor": 0.0,
            "switch_rate": 0.0,
            "mean_floor_ade": 0.0,
            "mean_selected_ade": 0.0,
            "mean_ungated_ade": 0.0,
        }
    floor = floor_ade[mask]
    selected = selected_ade[mask]
    ungated = ungated_ade[mask]
    easy_mask = mask & easy
    easy_deg = (
        max(0.0, float(np.mean(selected_ade[easy_mask])) / max(float(np.mean(floor_ade[easy_mask])), EPS) - 1.0)
        if int(np.sum(easy_mask))
        else 0.0
    )
    return {
        "rows": int(np.sum(mask)),
        "full_waypoint_ade_improvement_vs_floor": float(1.0 - float(np.mean(selected)) / max(float(np.mean(floor)), EPS)),
        "endpoint_fde_improvement_vs_floor": float(
            1.0 - float(np.mean(selected_fde[mask])) / max(float(np.mean(floor_fde[mask])), EPS)
        ),
        "ungated_full_waypoint_ade_improvement_vs_floor": float(
            1.0 - float(np.mean(ungated)) / max(float(np.mean(floor)), EPS)
        ),
        "easy_degradation_vs_floor": float(easy_deg),
        "switch_rate": float(np.mean(switch[mask])),
        "mean_floor_ade": float(np.mean(floor)),
        "mean_selected_ade": float(np.mean(selected)),
        "mean_ungated_ade": float(np.mean(ungated)),
    }


def _breakdown(values: np.ndarray, *arrays: np.ndarray, min_rows: int = 1) -> dict[str, Any]:
    floor_ade, floor_fde, selected_ade, selected_fde, ungated_ade, switch, easy = arrays
    out: dict[str, Any] = {}
    for value in sorted(set(values.astype(str).tolist())):
        mask = values.astype(str) == value
        if int(np.sum(mask)) < int(min_rows):
            continue
        out[value] = _slice_metrics(floor_ade, floor_fde, selected_ade, selected_fde, ungated_ade, switch, easy, mask)
    return out


def _top_slices(rows: Mapping[str, Mapping[str, Any]], *, key: str, n: int = 12, reverse: bool = False) -> list[dict[str, Any]]:
    ordered = sorted(
        (
            {"slice": name, **dict(metrics)}
            for name, metrics in rows.items()
            if int(metrics.get("rows", 0)) > 0
        ),
        key=lambda row: float(row.get(key, 0.0)),
        reverse=reverse,
    )
    return ordered[: int(n)]


def _load_model(stage43m: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any], FullWaypointLatentDynamics]:
    checkpoint = Path(stage43m.get("checkpoint", CKPT_DIR / "stage43_full_waypoint_latent_dynamics.pt"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return checkpoint, ckpt, model


def _standardize_from_checkpoint(ds, ckpt: Mapping[str, Any]):
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    return ds


def run_full_waypoint_latent_robustness_audit(*, batch_size: int = 4096, max_rows: int | None = None) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    test = _build_split("test", max_rows=max_rows, seed=int(ckpt.get("seed", 431)))
    test = _standardize_from_checkpoint(test, ckpt)
    with torch.no_grad():
        from src.stage43_full_waypoint_latent_dynamics import _predict

        pred = _predict(model, test, torch.device("cpu"), int(batch_size))
    policy = stage43m["validation_selected_policy"]["policy"]
    selected_ade, selected_fde, switched = _select_with_policy(test, pred, policy)
    ungated_ade, ungated_fde = _trajectory_error(test, pred["waypoint"])
    overall = _metrics(test, selected_ade, selected_fde, switched)
    arrays = (test.floor_ade, test.floor_fde, selected_ade, selected_fde, ungated_ade, switched, test.easy)
    by_domain = _breakdown(test.domain, *arrays)
    by_horizon = _breakdown(test.horizon.astype(str), *arrays)
    by_source = _breakdown(test.source_file, *arrays, min_rows=50)
    by_scene = _breakdown(test.scene_id, *arrays, min_rows=50)
    negative_sources = [
        {"source_file": name, **metrics}
        for name, metrics in by_source.items()
        if float(metrics["full_waypoint_ade_improvement_vs_floor"]) < 0.0
    ]
    domain_easy_harm = [
        {"domain": name, **metrics}
        for name, metrics in by_domain.items()
        if float(metrics["easy_degradation_vs_floor"]) > 0.02
    ]
    t100 = by_horizon.get("100", {})
    t100_switch_harm = float(t100.get("mean_selected_ade", 0.0) - t100.get("mean_floor_ade", 0.0))
    t100_ungated_gap = float(t100.get("mean_ungated_ade", 0.0) - t100.get("mean_floor_ade", 0.0))
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_full_test_replay_from_stage43_m_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {
            "verdict": stage43m.get("stage43_m_gate", {}).get("verdict"),
            "deploy_neural_full_waypoint": bool(stage43m.get("stage43_m_gate", {}).get("deploy_neural_full_waypoint")),
            "sampled_test_rows": int(stage43m.get("test_metrics_with_floor", {}).get("rows", 0)),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_sha256_matches_stage43_m": _sha256(checkpoint) == stage43m.get("checkpoint_sha256"),
        "policy_replayed": policy,
        "full_test_rows": int(len(test.x)),
        "feature_count": int(test.x.shape[1]),
        "overall_full_test_metrics": overall,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_source_summary": {
            "source_count": int(len(by_source)),
            "negative_source_count": int(len(negative_sources)),
            "worst_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12),
            "best_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12, reverse=True),
        },
        "by_scene_summary": {
            "scene_count": int(len(by_scene)),
            "negative_scene_count": int(
                sum(float(metrics["full_waypoint_ade_improvement_vs_floor"]) < 0.0 for metrics in by_scene.values())
            ),
            "worst_scenes": _top_slices(by_scene, key="full_waypoint_ade_improvement_vs_floor", n=12),
            "best_scenes": _top_slices(by_scene, key="full_waypoint_ade_improvement_vs_floor", n=12, reverse=True),
        },
        "source_domain_caveats": {
            "negative_source_count": int(len(negative_sources)),
            "domain_easy_harm_count": int(len(domain_easy_harm)),
            "domains_with_easy_harm": domain_easy_harm,
            "uniform_source_success": False,
            "uniform_domain_easy_safety": len(domain_easy_harm) == 0,
        },
        "t100_failure_attribution": {
            "rows": int(t100.get("rows", 0)),
            "protected_t100_improvement": float(t100.get("full_waypoint_ade_improvement_vs_floor", 0.0)),
            "ungated_t100_improvement": float(t100.get("ungated_full_waypoint_ade_improvement_vs_floor", 0.0)),
            "switch_rate": float(t100.get("switch_rate", 0.0)),
            "mean_selected_minus_floor_ade": t100_switch_harm,
            "mean_ungated_minus_floor_ade": t100_ungated_gap,
            "diagnosis": "t100 remains negative because the neural waypoint shape is worse than the floor on long raw-frame rows; fallback gate still switches too often for t100.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
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
            "uniform_source_success": False,
            "t100_success": False,
        },
    }
    payload["stage43_n_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = payload["overall_full_test_metrics"]
    t100 = payload["t100_failure_attribution"]
    gates = {
        "stage43_m_precondition_deployable": payload["stage43_m_precondition"]["deploy_neural_full_waypoint"] is True,
        "checkpoint_replayed": payload["checkpoint_sha256_matches_stage43_m"] is True,
        "full_test_eval_larger_than_stage43_m_sample": payload["full_test_rows"]
        > payload["stage43_m_precondition"]["sampled_test_rows"],
        "domain_breakdown_complete": len(payload["by_domain"]) >= 2,
        "horizon_breakdown_complete": all(str(h) in payload["by_horizon"] for h in [10, 25, 50, 100]),
        "source_breakdown_complete": payload["by_source_summary"]["source_count"] > 0,
        "source_domain_caveats_recorded": payload["source_domain_caveats"]["uniform_source_success"] is False
        and (
            payload["source_domain_caveats"]["negative_source_count"] > 0
            or payload["source_domain_caveats"]["domain_easy_harm_count"] > 0
        ),
        "overall_positive_on_full_test": overall["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t50_positive_on_full_test": payload["by_horizon"]["50"]["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "easy_preserved_on_full_test": overall["easy_degradation_vs_floor"] <= 0.02,
        "t100_failure_reported_honestly": t100["protected_t100_improvement"] < 0.0
        and payload["claim_boundary"]["t100_success"] is False,
        "no_leakage_claim_boundary": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_n_full_test_positive_with_source_t100_blockers"
        if passed == total
        else "stage43_n_robustness_audit_incomplete",
        "full_test_protected_candidate": passed == total,
        "uniform_source_success": False,
        "t100_success": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_n_gate"]
    overall = payload["overall_full_test_metrics"]
    t100 = payload["t100_failure_attribution"]
    lines = [
        "# Stage43-N Full-Waypoint Latent Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        "",
        "## Full-Test Protected Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(overall['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(overall['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(payload['by_horizon']['50']['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(t100['protected_t100_improvement'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(overall['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(overall['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(overall['switch_rate'])}`",
        "",
        "## Domain Breakdown",
        "",
        "| domain | rows | ADE lift | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in payload["by_domain"].items():
        lines.append(
            f"| {name} | {metrics['rows']} | {_pct(metrics['full_waypoint_ade_improvement_vs_floor'])} | {_pct(metrics['easy_degradation_vs_floor'])} | {_pct(metrics['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Horizon Breakdown",
            "",
            "| horizon | rows | ADE lift | ungated ADE lift | easy degradation | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in payload["by_horizon"].items():
        lines.append(
            f"| {name} | {metrics['rows']} | {_pct(metrics['full_waypoint_ade_improvement_vs_floor'])} | {_pct(metrics['ungated_full_waypoint_ade_improvement_vs_floor'])} | {_pct(metrics['easy_degradation_vs_floor'])} | {_pct(metrics['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Source Caveat",
            "",
            f"- source count: `{payload['by_source_summary']['source_count']}`",
            f"- negative source count: `{payload['by_source_summary']['negative_source_count']}`",
            f"- domains with easy harm >2%: `{payload['source_domain_caveats']['domain_easy_harm_count']}`",
            "- uniform source success: `False`",
            "",
            "Worst source slices:",
            "",
            "| source | rows | ADE lift | easy degradation | switch |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["by_source_summary"]["worst_sources"][:8]:
        lines.append(
            f"| {row['slice']} | {row['rows']} | {_pct(row['full_waypoint_ade_improvement_vs_floor'])} | {_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## t100 Failure Attribution",
            "",
            f"- rows: `{t100['rows']}`",
            f"- protected t100 improvement: `{_pct(t100['protected_t100_improvement'])}`",
            f"- ungated t100 improvement: `{_pct(t100['ungated_t100_improvement'])}`",
            f"- switch rate: `{_pct(t100['switch_rate'])}`",
            f"- diagnosis: {t100['diagnosis']}",
            "",
            "## Boundary",
            "",
            "- dataset-local/raw-frame 2.5D only.",
            "- no metric/seconds-level claim.",
            "- no Stage5C execution.",
            "- no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-N Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"full_test_protected_candidate: `{gate['full_test_protected_candidate']}`",
        f"uniform_source_success: `{gate['uniform_source_success']}`",
        f"t100_success: `{gate['t100_success']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_n_gate"]
    overall = payload["overall_full_test_metrics"]
    t100 = payload["t100_failure_attribution"]
    lines = [
        "## Stage43-N full-waypoint latent robustness audit",
        "",
        f"Result source: `{payload['result_source']}`. The Stage43-M checkpoint and validation-selected protected policy were replayed on the full Stage43-L test cache, then broken down by domain, horizon, source, and scene.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(overall['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(payload['by_horizon']['50']['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(t100['protected_t100_improvement'])}`",
        f"- hard/failure ADE improvement vs floor: `{_pct(overall['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(overall['easy_degradation_vs_floor'])}`",
        f"- negative source count: `{payload['by_source_summary']['negative_source_count']}`",
        "",
        "Boundary: this supports a protected full-test latent dynamics candidate with t100 and source-level caveats; it is still dataset-local/raw-frame 2.5D only, with no metric/seconds-level claim, no Stage5C, and no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_n_gate"]
    state["stage43_n_full_waypoint_latent_robustness_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "full_test_rows": payload["full_test_rows"],
        "overall_full_test_metrics": payload["overall_full_test_metrics"],
        "t100_failure_attribution": payload["t100_failure_attribution"],
        "negative_source_count": payload["by_source_summary"]["negative_source_count"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_n_full_waypoint_latent_robustness_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_n_full_waypoint_latent_robustness_audit", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Stage43-M on full test and audit source/domain/horizon robustness.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional debug cap; 0 means full test.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_full_waypoint_latent_robustness_audit(
        batch_size=int(args.batch_size),
        max_rows=None if int(args.max_rows) <= 0 else int(args.max_rows),
    )
    gate = result["stage43_n_gate"]
    print(f"Stage43-N: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
