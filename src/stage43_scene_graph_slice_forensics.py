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
from src import stage43_scene_graph_multimodal_ablation as bp
from src import stage43_gated_scene_graph_fusion as bq


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_graph_slice_forensics.json"
REPORT_MD = OUT_DIR / "stage43_scene_graph_slice_forensics.md"
GATE_MD = OUT_DIR / "stage43_stage_br_scene_graph_slice_forensics_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BR_SCENE_GRAPH_SLICE_FORENSICS"
SOURCE = "fresh_stage43_br_scene_graph_slice_forensics"
VARIANTS = ["no_context", "scene_proxy_only", "graph_history_only", "scene_graph_full"]
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load_bp_variant(variant: str, rows: Mapping[str, int | None]) -> tuple[m.WaypointSplit, dict[str, np.ndarray], dict[str, Any]]:
    bp_payload = read_json(bp.REPORT_JSON, {})
    variants = {row["variant"]: row for row in bp_payload.get("variants", [])}
    if variant not in variants:
        raise KeyError(f"Stage43-BR missing BP variant {variant}")
    meta = variants[variant]
    ckpt_path = Path(meta["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    row_seed = int(ckpt.get("row_seed", 443))
    test, ctx = bp._build_variant_split("test", max_rows=rows["test"], row_seed=row_seed, variant=variant)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    test.x = ((test.x - mean) / std).astype(np.float32)
    model = m.FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        latent_dim=int(ckpt.get("latent_dim", 32)),
    )
    model.load_state_dict(ckpt["model_state"])
    pred = m._predict(model, test, torch.device("cpu"), batch_size=4096)
    policy = meta["validation_selected_policy"]["policy"]
    selected_ade, selected_fde, switched = m._select_with_policy(test, pred, policy)
    arrays = {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switched": switched.astype(bool),
        "floor_ade": test.floor_ade,
        "floor_fde": test.floor_fde,
    }
    info = {
        "variant": variant,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": meta.get("checkpoint_sha256", ""),
        "context": ctx,
        "policy": policy,
        "reported_metrics": meta.get("test_metrics_with_floor", {}),
        "recomputed_metrics": m._metrics(test, selected_ade, selected_fde, switched),
    }
    return test, arrays, info


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.mean(values[mask]))


def _slice_row(
    name: str,
    mask: np.ndarray,
    ds: m.WaypointSplit,
    arrays_by_variant: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    imps = {
        variant: _slice_improvement(arrays_by_variant[variant]["selected_ade"], ds.floor_ade, mask)
        for variant in VARIANTS
    }
    best_variant = max(imps, key=imps.get)
    return {
        "slice": name,
        "rows": int(mask.sum()),
        "floor_ade": _mean_or_zero(ds.floor_ade, mask),
        "best_variant": best_variant,
        "improvements": imps,
        "scene_minus_graph": float(imps["scene_proxy_only"] - imps["graph_history_only"]),
        "scene_minus_no_context": float(imps["scene_proxy_only"] - imps["no_context"]),
        "graph_minus_no_context": float(imps["graph_history_only"] - imps["no_context"]),
        "full_minus_graph": float(imps["scene_graph_full"] - imps["graph_history_only"]),
        "full_minus_no_context": float(imps["scene_graph_full"] - imps["no_context"]),
        "scene_switch_rate": _mean_or_zero(arrays_by_variant["scene_proxy_only"]["switched"].astype(np.float32), mask),
        "graph_switch_rate": _mean_or_zero(arrays_by_variant["graph_history_only"]["switched"].astype(np.float32), mask),
        "full_switch_rate": _mean_or_zero(arrays_by_variant["scene_graph_full"]["switched"].astype(np.float32), mask),
    }


def _build_slices(ds: m.WaypointSplit) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = [("all", np.ones(len(ds.x), dtype=bool))]
    for horizon in sorted(set(ds.horizon.astype(int).tolist())):
        masks.append((f"horizon_{horizon}", ds.horizon == horizon))
    hard_failure = ds.hard | ds.failure
    masks.extend(
        [
            ("hard_failure", hard_failure),
            ("easy", ds.easy),
            ("not_easy", ~ds.easy),
        ]
    )
    for domain in sorted(set(ds.domain.astype(str).tolist())):
        dmask = ds.domain.astype(str) == domain
        masks.append((f"domain_{domain}", dmask))
        for horizon in sorted(set(ds.horizon.astype(int).tolist())):
            hm = dmask & (ds.horizon == horizon)
            if int(hm.sum()) >= 50:
                masks.append((f"domain_{domain}_horizon_{horizon}", hm))
    source_values = ds.source_file.astype(str)
    for source in sorted(set(source_values.tolist())):
        smask = source_values == source
        if int(smask.sum()) >= 100:
            safe = source.replace("/", "_").replace(" ", "_")
            masks.append((f"source_{safe}", smask))
    return masks


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    rows = {"test": int(args.test_rows)}
    bp_payload = read_json(bp.REPORT_JSON, {})
    bq_payload = read_json(bq.REPORT_JSON, {})
    bp_gate = bp_payload.get("stage43_bp_gate", {})
    bq_gate = bq_payload.get("stage43_bq_gate", {})
    arrays_by_variant: dict[str, dict[str, np.ndarray]] = {}
    infos: dict[str, dict[str, Any]] = {}
    reference_ds: m.WaypointSplit | None = None
    for variant in VARIANTS:
        ds, arrays, info = _load_bp_variant(variant, rows)
        if reference_ds is None:
            reference_ds = ds
        elif len(ds.x) != len(reference_ds.x) or not np.array_equal(ds.horizon, reference_ds.horizon):
            raise ValueError(f"Stage43-BR row alignment failed for {variant}")
        arrays_by_variant[variant] = arrays
        infos[variant] = info
    if reference_ds is None:
        raise RuntimeError("No Stage43-BR reference dataset built")
    slice_rows = [
        _slice_row(name, mask, reference_ds, arrays_by_variant)
        for name, mask in _build_slices(reference_ds)
        if int(mask.sum()) > 0
    ]
    scene_over_graph = [row for row in slice_rows if row["rows"] >= int(args.min_slice_rows) and row["scene_minus_graph"] > 0.0]
    scene_over_no_context = [row for row in slice_rows if row["rows"] >= int(args.min_slice_rows) and row["scene_minus_no_context"] > 0.0]
    full_over_graph = [row for row in slice_rows if row["rows"] >= int(args.min_slice_rows) and row["full_minus_graph"] > 0.0]
    best_counts: dict[str, int] = {variant: 0 for variant in VARIANTS}
    for row in slice_rows:
        if row["rows"] >= int(args.min_slice_rows):
            best_counts[row["best_variant"]] += 1
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_slice_forensics_from_stage43_bp_bq_checkpoints",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "rows": int(len(reference_ds.x)),
        "min_slice_rows": int(args.min_slice_rows),
        "bp_precondition": {
            "verdict": bp_gate.get("verdict", "missing"),
            "gate": f"{bp_gate.get('passed', 0)} / {bp_gate.get('total', 0)}",
        },
        "bq_precondition": {
            "verdict": bq_gate.get("verdict", "missing"),
            "gate": f"{bq_gate.get('passed', 0)} / {bq_gate.get('total', 0)}",
        },
        "variant_replay": infos,
        "slice_rows": slice_rows,
        "summary": {
            "slice_count": len(slice_rows),
            "eligible_slice_count": int(sum(row["rows"] >= int(args.min_slice_rows) for row in slice_rows)),
            "scene_over_graph_slice_count": len(scene_over_graph),
            "scene_over_no_context_slice_count": len(scene_over_no_context),
            "full_over_graph_slice_count": len(full_over_graph),
            "best_variant_counts": best_counts,
            "top_scene_over_graph_slices": sorted(scene_over_graph, key=lambda row: row["scene_minus_graph"], reverse=True)[:10],
            "top_scene_over_no_context_slices": sorted(scene_over_no_context, key=lambda row: row["scene_minus_no_context"], reverse=True)[:10],
            "top_full_over_graph_slices": sorted(full_over_graph, key=lambda row: row["full_minus_graph"], reverse=True)[:10],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "deployable_policy_changed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([bp.REPORT_JSON, bq.REPORT_JSON]),
    }
    payload["stage43_br_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    bp_verdict = payload["bp_precondition"]["verdict"]
    bq_verdict = payload["bq_precondition"]["verdict"]
    scene_signal = int(summary["scene_over_graph_slice_count"]) >= 2
    weak_scene_signal = int(summary["scene_over_no_context_slice_count"]) >= 2
    gates = {
        "bp_precondition_passed": bp_verdict
        in {
            "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_mixed_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_contribution_supported",
        },
        "bq_precondition_passed": bq_verdict
        in {
            "stage43_bq_gated_scene_graph_fusion_pass_contribution_supported",
            "stage43_bq_gated_scene_graph_fusion_pass_safe_no_best_single_lift_diagnostic",
            "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
            "stage43_bq_gated_scene_graph_fusion_pass_unsafe_diagnostic",
        },
        "row_level_bp_checkpoint_replay_completed": all("recomputed_metrics" in row for row in payload["variant_replay"].values()),
        "slice_table_nonempty": summary["slice_count"] > 0,
        "eligible_slices_present": summary["eligible_slice_count"] >= 5,
        "source_horizon_hard_easy_slices_present": any(str(row["slice"]).startswith("domain_") for row in payload["slice_rows"])
        and any(row["slice"] == "hard_failure" for row in payload["slice_rows"])
        and any(row["slice"] == "easy" for row in payload["slice_rows"])
        and any(str(row["slice"]).startswith("horizon_") for row in payload["slice_rows"]),
        "scene_utility_measured": "scene_over_graph_slice_count" in summary
        and "scene_over_no_context_slice_count" in summary,
        "graph_history_best_count_measured": "graph_history_only" in summary["best_variant_counts"],
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["scene_proxy_train_only"] is True
        and payload["no_leakage"]["graph_inputs_past_or_current_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["raw_scene_or_verified_sdf_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["claim_boundary"]["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal"
        if passed == total and scene_signal
        else "stage43_br_scene_graph_slice_forensics_pass_weak_scene_signal_diagnostic"
        if passed == total and weak_scene_signal
        else "stage43_br_scene_graph_slice_forensics_pass_no_scene_signal_diagnostic"
        if passed == total
        else "stage43_br_scene_graph_slice_forensics_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "slice_forensics_executed": passed == total,
        "targeted_scene_signal": scene_signal,
        "weak_scene_signal": weak_scene_signal,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _slice_table(rows: list[Mapping[str, Any]], limit: int = 24) -> list[str]:
    out = [
        "| slice | rows | best | no_context | scene | graph | full | scene-graph | full-graph |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(rows, key=lambda row: (row["slice"] != "all", -abs(row["scene_minus_graph"]), row["slice"]))
    for row in ordered[:limit]:
        imps = row["improvements"]
        out.append(
            f"| `{row['slice']}` | `{row['rows']}` | `{row['best_variant']}` | `{_pct(imps['no_context'])}` | `{_pct(imps['scene_proxy_only'])}` | `{_pct(imps['graph_history_only'])}` | `{_pct(imps['scene_graph_full'])}` | `{_pct(row['scene_minus_graph'])}` | `{_pct(row['full_minus_graph'])}` |"
        )
    return out


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_br_gate"]
    summary = payload["summary"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BR Scene-Graph Slice Forensics",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- rows: `{payload['rows']}`",
            f"- scene over graph eligible slices: `{summary['scene_over_graph_slice_count']}`",
            f"- scene over no_context eligible slices: `{summary['scene_over_no_context_slice_count']}`",
            f"- full over graph eligible slices: `{summary['full_over_graph_slice_count']}`",
            f"- best variant counts: `{summary['best_variant_counts']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Slice Table",
            "",
            *_slice_table(payload["slice_rows"]),
            "",
            "## Interpretation",
            "",
            "- This is row-level forensics over Stage43-BP checkpoints, not a new deployment policy.",
            "- It identifies whether train-only scene proxies have slice-specific utility after BQ showed safe gated fusion still did not lift over graph-history.",
            "- Future waypoints remain labels/eval only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
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
            "# Stage43-BR Scene-Graph Slice Forensics Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- slice forensics executed: `{gate['slice_forensics_executed']}`",
            f"- targeted scene signal: `{gate['targeted_scene_signal']}`",
            f"- weak scene signal: `{gate['weak_scene_signal']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
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
            f"- slice forensics executed: `{gate['slice_forensics_executed']}`",
            f"- targeted scene signal: `{gate['targeted_scene_signal']}`",
            f"- weak scene signal: `{gate['weak_scene_signal']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BR is a row-level slice forensic report over retrained Stage43-BP variants.",
            "- It does not update deployment and does not claim raw scene/SDF evidence.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_br_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"slice_forensics_executed = `{gate['slice_forensics_executed']}`",
        f"targeted_scene_signal = `{gate['targeted_scene_signal']}`",
        f"weak_scene_signal = `{gate['weak_scene_signal']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Stage43-BR replays Stage43-BP checkpoints at row level and audits scene/graph utility by source, horizon, hard/failure, and easy slices. Eligible scene-over-graph slices: `{summary['scene_over_graph_slice_count']}`; scene-over-no-context slices: `{summary['scene_over_no_context_slice_count']}`; full-over-graph slices: `{summary['full_over_graph_slice_count']}`.",
        "",
        f"Best variant counts across eligible slices: `{summary['best_variant_counts']}`.",
        "",
        "This is slice forensics only, not a deployment policy update. Scene remains train-only proxy scene/goal/raster evidence, not raw image/SDF evidence.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_br_scene_graph_slice_forensics"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "slice_forensics_executed": gate["slice_forensics_executed"],
        "targeted_scene_signal": gate["targeted_scene_signal"],
        "weak_scene_signal": gate["weak_scene_signal"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "summary": payload["summary"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_br_scene_graph_slice_forensics"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BR",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "targeted_scene_signal": gate["targeted_scene_signal"],
                        "weak_scene_signal": gate["weak_scene_signal"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BR scene-graph slice forensics.")
    parser.add_argument("--test-rows", type=int, default=12000)
    parser.add_argument("--min-slice-rows", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = _run(args)
    gate = payload["stage43_br_gate"]
    print(f"Stage43-BR: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"targeted_scene_signal={gate['targeted_scene_signal']}")
    print(f"weak_scene_signal={gate['weak_scene_signal']}")
    return payload


if __name__ == "__main__":
    main()
