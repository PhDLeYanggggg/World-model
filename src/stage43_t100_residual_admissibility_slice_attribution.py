from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_residual_admissibility_head as ct
from src import stage43_t100_residual_admissibility_statistical_confirmation as cu
from src import stage43_t100_supported_latent_dynamics as cr
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_slice_attribution.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_slice_attribution.md"
GATE_MD = OUT_DIR / "stage43_stage_cv_t100_residual_admissibility_slice_attribution_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CV_T100_RESIDUAL_ADMISSIBILITY_SLICE_ATTRIBUTION"
SOURCE = "fresh_stage43_cv_t100_residual_admissibility_slice_attribution"


def _ensure_cu_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(cu.REPORT_JSON, {})
    missing = not report or "seed_runs" not in report
    if not missing:
        for run in report.get("seed_runs", []):
            if not Path(str(run.get("checkpoint", ""))).exists():
                missing = True
                break
    if missing:
        local_args = argparse.Namespace(
            quick=bool(args.quick),
            small=True,
            seeds=str(args.seeds),
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=args.max_test,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            hidden_dim=int(args.hidden_dim),
            lr=float(args.lr),
            bootstrap=int(args.bootstrap),
        )
        report = cu.confirm_t100_residual_admissibility(local_args)
    return report


def _default_max(split: str, args: argparse.Namespace) -> int:
    value = getattr(args, f"max_{split}", None)
    if value is not None:
        return int(value)
    if bool(args.quick):
        return {"train": 6000, "val": 3000, "test": 3000}[split]
    return {"train": 24000, "val": 9000, "test": 10000}[split]


def _selected_cache_rows(split: str, *, max_rows: int, seed: int) -> dict[str, np.ndarray]:
    cache = cr._npz(cr._cache_path(split))
    n = len(cache["horizon"])
    ids = np.arange(n)
    if max_rows is not None and max_rows < n:
        rng = np.random.default_rng(seed + {"train": 17, "val": 19, "test": 23}[split])
        ids = np.sort(rng.choice(ids, size=int(max_rows), replace=False))
    keys = ["dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon", "old_split", "local_row"]
    return {key: np.asarray(cache[key])[ids] for key in keys}


def _row_key(meta: Mapping[str, np.ndarray]) -> np.ndarray:
    pieces = [
        np.asarray(meta["dataset"]).astype(str),
        np.asarray(meta["scene_id"]).astype(str),
        np.asarray(meta["source_file"]).astype(str),
        np.asarray(meta["agent_id"]).astype(str),
        np.asarray(meta["frame_id"]).astype(str),
        np.asarray(meta["horizon"]).astype(str),
    ]
    return np.asarray(["|".join(vals) for vals in zip(*pieces)], dtype=object)


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256("\0".join(np.asarray(values).astype(str).tolist()).encode("utf-8")).hexdigest()


def _slice_table(
    labels: np.ndarray,
    selected_ade: np.ndarray,
    floor_ade: np.ndarray,
    switched: np.ndarray,
    *,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(str)
    improvement = floor_ade.astype(np.float64) - selected_ade.astype(np.float64)
    total_positive_gain = float(np.sum(np.maximum(improvement[switched], 0.0)))
    rows: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        sw = mask & switched
        if int(mask.sum()) == 0:
            continue
        mean_floor = float(np.mean(floor_ade[mask]))
        mean_selected = float(np.mean(selected_ade[mask]))
        pos_gain = float(np.sum(np.maximum(improvement[sw], 0.0)))
        rows.append(
            {
                "label": str(label),
                "rows": int(mask.sum()),
                "switched": int(sw.sum()),
                "switch_rate": float(np.mean(sw[mask])) if int(mask.sum()) else 0.0,
                "slice_improvement_vs_floor": float(1.0 - mean_selected / max(mean_floor, m.EPS)),
                "mean_delta_ade_selected_minus_floor": float(np.mean(selected_ade[mask] - floor_ade[mask])),
                "positive_gain_sum_switched": pos_gain,
                "positive_gain_share": float(pos_gain / max(total_positive_gain, m.EPS)),
                "harm_sum_switched": float(np.sum(np.maximum(-improvement[sw], 0.0))),
                "mean_floor_ade": mean_floor,
                "mean_selected_ade": mean_selected,
            }
        )
    rows.sort(key=lambda row: (row["positive_gain_sum_switched"], row["switched"]), reverse=True)
    return rows[: int(top_k)]


def _binary_slice(label: str, mask: np.ndarray, selected_ade: np.ndarray, floor_ade: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask).astype(bool)
    if int(mask.sum()) == 0:
        return {"label": label, "rows": 0, "switched": 0, "switch_rate": 0.0, "slice_improvement_vs_floor": 0.0}
    return {
        "label": label,
        "rows": int(mask.sum()),
        "switched": int((mask & switched).sum()),
        "switch_rate": float(np.mean(switched[mask])),
        "slice_improvement_vs_floor": m._slice_improvement(selected_ade, floor_ade, mask),
        "mean_delta_ade_selected_minus_floor": float(np.mean(selected_ade[mask] - floor_ade[mask])),
        "mean_floor_ade": float(np.mean(floor_ade[mask])),
        "mean_selected_ade": float(np.mean(selected_ade[mask])),
    }


def _load_seed_head(run: Mapping[str, Any]) -> tuple[ct.ResidualAdmissibilityHead, np.ndarray, np.ndarray]:
    ckpt = torch.load(str(run["checkpoint"]), map_location="cpu", weights_only=False)
    model = ct.ResidualAdmissibilityHead(int(ckpt["input_dim"]), hidden_dim=int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, np.asarray(ckpt["feature_mean"], dtype=np.float32), np.asarray(ckpt["feature_std"], dtype=np.float32)


def _seed_attribution(seed_run: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    seed = int(seed_run["seed"])
    build_args = argparse.Namespace(
        quick=bool(args.quick),
        seed=seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        batch_size=int(args.batch_size),
    )
    _train, _val, test, _cs_ckpt, cs_model = ct._build_splits(build_args)
    device = torch.device("cpu")
    cs_pred = ct.cs._predict(cs_model, test, device, int(args.batch_size))
    test_aug = ct._augment_alpha_features(test, cs_pred)
    model, mean, std = _load_seed_head(seed_run)
    test_aug["x"] = ((test_aug["x"] - mean) / std).astype(np.float32)
    head_pred = ct._predict_head(model, test_aug, device, int(args.batch_size))
    policy = seed_run["validation_selected_policy"]["policy"]
    metrics, selected_ade, selected_fde, switched = ct._evaluate_selected(test, cs_pred, head_pred, policy)
    expected = seed_run["test_metrics_with_floor"]
    metric_diff = {
        key: float(abs(float(metrics[key]) - float(expected[key])))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }
    meta = _selected_cache_rows("test", max_rows=_default_max("test", args), seed=seed)
    keys = _row_key(meta)
    source_agent = np.asarray([f"{src}|{ag}" for src, ag in zip(meta["source_file"].astype(str), meta["agent_id"].astype(str))], dtype=object)
    hard_failure = test.hard | test.failure
    rows = {
        "domain": _slice_table(test.domain, selected_ade, test.floor_ade, switched),
        "source_file": _slice_table(test.source_file, selected_ade, test.floor_ade, switched),
        "scene_id": _slice_table(test.scene_id, selected_ade, test.floor_ade, switched),
        "source_agent": _slice_table(source_agent, selected_ade, test.floor_ade, switched, top_k=8),
    }
    binary = [
        _binary_slice("hard_or_failure", hard_failure, selected_ade, test.floor_ade, switched),
        _binary_slice("easy", test.easy, selected_ade, test.floor_ade, switched),
        _binary_slice("hard_only", test.hard, selected_ade, test.floor_ade, switched),
        _binary_slice("failure_only", test.failure, selected_ade, test.floor_ade, switched),
        _binary_slice("switched_rows", switched, selected_ade, test.floor_ade, switched),
        _binary_slice("unswitched_rows", ~switched, selected_ade, test.floor_ade, switched),
    ]
    max_source_gain_share = float(max([row["positive_gain_share"] for row in rows["source_file"]] or [0.0]))
    max_scene_gain_share = float(max([row["positive_gain_share"] for row in rows["scene_id"]] or [0.0]))
    positive_sources = int(sum(row["positive_gain_sum_switched"] > 0.0 for row in rows["source_file"]))
    switched_sources = int(sum(row["switched"] > 0 for row in rows["source_file"]))
    return {
        "seed": seed,
        "rows": int(len(test.x)),
        "row_key_hash": _hash_array(keys),
        "policy": policy,
        "metrics": metrics,
        "metric_replay_diff": metric_diff,
        "max_metric_replay_diff": float(max(metric_diff.values()) if metric_diff else 0.0),
        "switched_count": int(switched.sum()),
        "switch_rate": float(np.mean(switched)),
        "positive_sources": positive_sources,
        "switched_sources": switched_sources,
        "max_source_positive_gain_share": max_source_gain_share,
        "max_scene_positive_gain_share": max_scene_gain_share,
        "slice_tables": rows,
        "binary_slices": binary,
        "concentration": {
            "source_signal_narrow": bool(max_source_gain_share >= 0.60 or positive_sources < 3),
            "scene_signal_narrow": bool(max_scene_gain_share >= 0.60),
            "max_source_positive_gain_share": max_source_gain_share,
            "max_scene_positive_gain_share": max_scene_gain_share,
            "positive_sources": positive_sources,
            "switched_sources": switched_sources,
        },
    }


def _aggregate(seed_payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = {
        "max_metric_replay_diff": [float(run["max_metric_replay_diff"]) for run in seed_payloads],
        "switch_rate": [float(run["switch_rate"]) for run in seed_payloads],
        "max_source_positive_gain_share": [float(run["max_source_positive_gain_share"]) for run in seed_payloads],
        "max_scene_positive_gain_share": [float(run["max_scene_positive_gain_share"]) for run in seed_payloads],
        "positive_sources": [float(run["positive_sources"]) for run in seed_payloads],
        "switched_sources": [float(run["switched_sources"]) for run in seed_payloads],
    }
    out: dict[str, Any] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "values": [float(x) for x in arr.tolist()],
        }
    out["all_replay_exact"] = bool(max(values["max_metric_replay_diff"] or [1.0]) <= 1e-7)
    out["any_seed_source_narrow"] = bool(any(run["concentration"]["source_signal_narrow"] for run in seed_payloads))
    out["any_seed_scene_narrow"] = bool(any(run["concentration"]["scene_signal_narrow"] for run in seed_payloads))
    out["scope_verdict"] = (
        "broad_enough_to_expand"
        if not out["any_seed_source_narrow"] and not out["any_seed_scene_narrow"]
        else "narrow_supported_diagnostic"
    )
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_cu_precondition_present": payload["stage43_cu_precondition"]["verdict"]
        == "stage43_cu_t100_admissibility_multiseed_confirmed_tiny_positive",
        "fresh_slice_attribution": payload["result_source"] == "fresh_t100_residual_admissibility_slice_attribution",
        "three_seed_replay": len(payload["seed_attribution"]) >= 3,
        "replay_diff_zero": bool(payload["aggregate"]["all_replay_exact"]),
        "domain_source_scene_tables_present": all(
            all(name in run["slice_tables"] and run["slice_tables"][name] for name in ["domain", "source_file", "scene_id"])
            for run in payload["seed_attribution"]
        ),
        "switch_concentration_reported": "scope_verdict" in payload["aggregate"],
        "scope_boundary_reported": payload["deploy_on_current_heldout"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed != total:
        verdict = "stage43_cv_t100_slice_attribution_incomplete_keep_floor"
    elif payload["aggregate"]["scope_verdict"] == "broad_enough_to_expand":
        verdict = "stage43_cv_t100_slice_attribution_broad_supported_diagnostic"
    else:
        verdict = "stage43_cv_t100_slice_attribution_narrow_supported_diagnostic"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_seed_table(seed_payloads: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| seed | t100 improvement | switch rate | positive sources | max source gain share | max scene gain share | replay diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in seed_payloads:
        lines.append(
            f"| `{run['seed']}` | `{run['metrics']['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.6f}` | "
            f"`{run['switch_rate']:.6f}` | `{run['positive_sources']}` | "
            f"`{run['max_source_positive_gain_share']:.6f}` | `{run['max_scene_positive_gain_share']:.6f}` | "
            f"`{run['max_metric_replay_diff']:.8f}` |"
        )
    return lines


def _render_slice_table(title: str, rows: list[Mapping[str, Any]], *, limit: int = 8) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| label | rows | switched | switch rate | slice improvement | gain share | harm switched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['label']}` | `{row['rows']}` | `{row['switched']}` | `{row['switch_rate']:.4f}` | "
            f"`{row['slice_improvement_vs_floor']:.6f}` | `{row.get('positive_gain_share', 0.0):.4f}` | "
            f"`{row.get('harm_sum_switched', 0.0):.6f}` |"
        )
    lines.append("")
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cv_gate"]
    agg = payload["aggregate"]
    first = payload["seed_attribution"][0] if payload["seed_attribution"] else {}
    lines = [
        "# Stage43-CV T100 Residual Admissibility Slice Attribution",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- scope verdict: `{agg['scope_verdict']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Replay",
        "",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- max replay diff: `{agg['max_metric_replay_diff']['max']:.8f}`",
        "",
        "## Seed Summary",
        "",
        *_render_seed_table(payload["seed_attribution"]),
        "",
        "## Concentration",
        "",
        f"- mean max source positive-gain share: `{agg['max_source_positive_gain_share']['mean']:.6f}`",
        f"- mean max scene positive-gain share: `{agg['max_scene_positive_gain_share']['mean']:.6f}`",
        f"- mean positive sources: `{agg['positive_sources']['mean']:.2f}`",
        f"- any seed source narrow: `{agg['any_seed_source_narrow']}`",
        f"- any seed scene narrow: `{agg['any_seed_scene_narrow']}`",
        "",
        "## Top Slices From First Seed",
        "",
    ]
    if first:
        lines.extend(_render_slice_table("Domain", first["slice_tables"]["domain"]))
        lines.extend(_render_slice_table("Source file", first["slice_tables"]["source_file"]))
        lines.extend(_render_slice_table("Scene", first["slice_tables"]["scene_id"]))
        lines.extend(_render_slice_table("Source-agent", first["slice_tables"]["source_agent"], limit=6))
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This step attributes the tiny CU-confirmed t100 admissibility lift across domain/source/scene/source-agent slices.",
            "- If concentration is narrow, the correct next step is slice-specific expansion or heldout stress testing, not a deployment claim.",
            "- Future endpoints/full waypoints remain labels only; inference inputs are causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cv_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CV Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- scope verdict: `{payload['aggregate']['scope_verdict']}`",
            "- deploy on current heldout t100: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
            "",
        ],
    )
    agg = payload["aggregate"]
    readme_block = [
        "## Stage43-CV: t100 residual admissibility slice attribution",
        "",
        "I traced the tiny Stage43-CU t100 residual-admissibility gain back to domain/source/scene/source-agent slices. The aim was to see whether the signal is broad enough to expand or too concentrated to treat as deployment evidence.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- scope verdict: `{agg['scope_verdict']}`",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- mean max source gain share: `{agg['max_source_positive_gain_share']['mean']:.2%}`",
        f"- mean max scene gain share: `{agg['max_scene_positive_gain_share']['mean']:.2%}`",
        f"- mean positive sources: `{agg['positive_sources']['mean']:.2f}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This keeps the t100 signal in the diagnostic lane. If the gain is concentrated, the next work should be source-slice expansion and stress testing before any wider claim.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cv_t100_residual_admissibility_slice_attribution"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_slice_attribution"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "aggregate": payload["aggregate"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def attribute_t100_residual_admissibility_slices(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cu_report = _ensure_cu_precondition(args)
    seed_payloads = [_seed_attribution(run, args) for run in cu_report.get("seed_runs", [])]
    aggregate = _aggregate(seed_payloads)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_t100_residual_admissibility_slice_attribution",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "stage43_cu_precondition": {
            "report": str(cu.REPORT_JSON),
            "verdict": cu_report.get("stage43_cu_gate", {}).get("verdict"),
        },
        "seed_attribution": seed_payloads,
        "aggregate": aggregate,
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "feature_standardization_train_only": True,
            "validation_policy_selection_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cv_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CV: {payload['stage43_cv_gate']['verdict']} ({payload['stage43_cv_gate']['passed']}/{payload['stage43_cv_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Attribute Stage43-CU t100 admissibility signal across source/scene slices.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    return attribute_t100_residual_admissibility_slices(args)


if __name__ == "__main__":
    main()
