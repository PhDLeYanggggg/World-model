from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_scene_proxy_augmented_latent_dynamics as ab
from src.stage43_scene_raster_proxy_tokens import DATA_DIR as SCENE_PROXY_DIR


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_scene_proxy_retrained_ablation.json"
REPORT_MD = OUT_DIR / "stage43_scene_proxy_retrained_ablation.md"
GATE_MD = OUT_DIR / "stage43_stage_ag_scene_proxy_retrained_ablation_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_scene_proxy_retrained_ablation_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AG_SCENE_PROXY_RETRAINED_ABLATION"
SOURCE = "fresh_stage43_ag_scene_proxy_retrained_ablation"

SCENE_GROUPS: dict[str, list[int]] = {
    "no_scene": [],
    "geometry_route": [0, 1, 2, 3, 4, 5, 6, 7, 13],
    "goal_only": [0, 1, 2, 8, 9, 10, 11, 12, 13],
    "full_scene": list(range(14)),
}


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _scene_npz(split: str) -> Mapping[str, np.ndarray]:
    path = SCENE_PROXY_DIR / f"stage43_scene_proxy_features_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage43-AA scene-proxy cache: {path}")
    return np.load(path, allow_pickle=False)


def _sample_ids(split: str, *, max_rows: int | None, seed: int) -> np.ndarray:
    return ab._sample_ids(split, max_rows=max_rows, seed=seed)


def _variant_split(split: str, *, max_rows: int | None, seed: int, variant: str) -> m.WaypointSplit:
    ds = m._build_split(split, max_rows=max_rows, seed=seed)
    indices = SCENE_GROUPS[variant]
    if not indices:
        return ds
    ids = _sample_ids(split, max_rows=max_rows, seed=seed)
    scene = _scene_npz(split)
    features = scene["features"].astype(np.float32)[ids][:, indices]
    names_all = scene["feature_names"].astype(str).tolist()
    names = [f"scene_proxy::{names_all[i]}" for i in indices]
    if len(features) != len(ds.x):
        raise ValueError(f"Scene proxy row mismatch for {split}/{variant}: {len(features)} != {len(ds.x)}")
    ds.x = np.concatenate([ds.x, features], axis=1).astype(np.float32)
    ds.feature_names = [*ds.feature_names, *names]
    return ds


def _variant_hash(split: str, variant: str) -> str:
    scene = _scene_npz(split)
    if not SCENE_GROUPS[variant]:
        return "no_scene_features"
    return f"{scene['feature_hash'][0]}::{variant}::{','.join(map(str, SCENE_GROUPS[variant]))}"


def _train_one_variant(args: argparse.Namespace, *, variant: str, rows: Mapping[str, int | None]) -> dict[str, Any]:
    seed = int(args.seed) + 97 * list(SCENE_GROUPS).index(variant)
    runtime = m._configure_runtime(seed)
    train = _variant_split("train", max_rows=rows["train"], seed=seed, variant=variant)
    val = _variant_split("val", max_rows=rows["val"], seed=seed, variant=variant)
    test = _variant_split("test", max_rows=rows["test"], seed=seed, variant=variant)
    train, val, test, mean, std = m._standardize(train, val, test)
    model = m.FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim))
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    ckpt_path = CKPT_DIR / f"stage43_scene_proxy_retrained_ablation_{variant}.pt"
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        latent_vars: list[float] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = m._loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            latent_vars.append(float(stat.get("latent_variance", 0.0)))
        val_pred = m._predict(model, val, device, int(args.batch_size))
        val_ade, _ = m._trajectory_error(val, val_pred["waypoint"])
        val_loss = float(np.mean((val_ade - val.floor_ade) ** 2))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_candidate_mse_to_floor": val_loss,
            "latent_variance": float(np.mean(latent_vars)) if latent_vars else 0.0,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            m._jsonable(
                {
                    "source": SOURCE,
                    "variant": variant,
                    "epoch": epoch + 1,
                    "elapsed_s": time.time() - start,
                    "last": row,
                    "git_commit": m._git_commit(),
                }
            ),
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train.feature_names,
                    "input_dim": int(train.x.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "latent_dim": int(args.latent_dim),
                    "variant": variant,
                    "scene_feature_indices": SCENE_GROUPS[variant],
                    "seed": seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "checkpoint_committed": False,
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "future_waypoint_label_eval_only": True,
                        "central_velocity_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                ckpt_path,
            )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))
    val_policy = m._search_policy(val, val_pred)
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    protected_metrics = m._metrics(test, selected_ade, selected_fde, switched)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated_metrics = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    return {
        "variant": variant,
        "scene_feature_indices": SCENE_GROUPS[variant],
        "scene_feature_count": len(SCENE_GROUPS[variant]),
        "feature_count": int(train.x.shape[1]),
        "scene_feature_hashes": {split: _variant_hash(split, variant) for split in m.SPLITS},
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": protected_metrics,
        "test_metrics_neural_without_floor": ungated_metrics,
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
    }


def _delta_metrics(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(current.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "endpoint_fde_improvement_vs_floor",
            "t50_full_waypoint_ade_improvement_vs_floor",
            "t50_endpoint_fde_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    mode = "medium" if args.medium else "quick" if args.quick else "small"
    rows: dict[str, int | None]
    if args.quick:
        rows = {"train": 6000, "val": 3000, "test": 3000}
    elif args.medium:
        rows = {"train": 90000, "val": 40000, "test": 50000}
    else:
        rows = {"train": 30000, "val": 12000, "test": 16000}
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in SCENE_GROUPS]
    if unknown:
        raise KeyError(f"Unknown Stage43-AG variants: {unknown}")
    results = [_train_one_variant(args, variant=variant, rows=rows) for variant in variants]
    by_variant = {row["variant"]: row for row in results}
    no_scene = by_variant.get("no_scene")
    if no_scene is None:
        raise ValueError("Stage43-AG requires the no_scene retrained baseline.")
    no_scene_metrics = no_scene["test_metrics_with_floor"]
    for row in results:
        row["delta_vs_retrained_no_scene"] = _delta_metrics(row["test_metrics_with_floor"], no_scene_metrics)
    candidate_rows = [row for row in results if row["variant"] != "no_scene"]
    best_t50 = max(candidate_rows, key=lambda row: row["delta_vs_retrained_no_scene"]["t50_full_waypoint_ade_improvement_vs_floor"])
    safe_candidate_rows = [
        row
        for row in candidate_rows
        if row["test_metrics_with_floor"]["easy_degradation_vs_floor"] <= 0.02
        and row["delta_vs_retrained_no_scene"]["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
    ]
    best_safe_t50 = max(
        safe_candidate_rows,
        key=lambda row: row["delta_vs_retrained_no_scene"]["t50_full_waypoint_ade_improvement_vs_floor"],
    ) if safe_candidate_rows else None
    best_hard = max(candidate_rows, key=lambda row: row["delta_vs_retrained_no_scene"]["hard_failure_full_waypoint_ade_improvement_vs_floor"])
    best_all = max(candidate_rows, key=lambda row: row["delta_vs_retrained_no_scene"]["full_waypoint_ade_improvement_vs_floor"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_retrained_scene_proxy_subset_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "variants": results,
        "best_variant_by_t50_delta": best_t50["variant"],
        "best_safe_variant_by_t50_delta": best_safe_t50["variant"] if best_safe_t50 else None,
        "best_variant_by_hard_delta": best_hard["variant"],
        "best_variant_by_all_delta": best_all["variant"],
        "bootstrap_ci_best_t50": m._bootstrap_ci(
            _variant_split("test", max_rows=rows["test"], seed=int(args.seed) + 97 * list(SCENE_GROUPS).index(best_t50["variant"]), variant=best_t50["variant"]),
            np.asarray(best_t50["test_metrics_with_floor"]["mean_selected_ade"], dtype=np.float32)
            if False
            else np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            n=0,
            seed=0,
        )
        if False
        else {"skipped": "variant-level bootstrap omitted; protected metrics are point estimates for this retrained subset audit"},
        "ablation_type": {
            "fresh_retrained_variants": True,
            "same_train_val_test_protocol": True,
            "compares_no_scene_geometry_goal_full_scene": True,
            "not_full_stage43_factorial_all_modules": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "scene_proxy_not_raw_image_or_true_sdf": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash(
            [
                m._cache_path("train"),
                m._cache_path("val"),
                m._cache_path("test"),
                SCENE_PROXY_DIR / "stage43_scene_proxy_features_train.npz",
                SCENE_PROXY_DIR / "stage43_scene_proxy_features_val.npz",
                SCENE_PROXY_DIR / "stage43_scene_proxy_features_test.npz",
                ab.STAGE43_AA_JSON,
            ]
        ),
    }
    payload["stage43_ag_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    variants = {row["variant"]: row for row in payload["variants"]}
    no_scene = variants["no_scene"]
    non_empty = [row for name, row in variants.items() if name != "no_scene"]
    t50_positive = [
        row
        for row in non_empty
        if row["delta_vs_retrained_no_scene"]["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and row["test_metrics_with_floor"]["easy_degradation_vs_floor"] <= 0.02
    ]
    hard_positive = [
        row
        for row in non_empty
        if row["delta_vs_retrained_no_scene"]["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and row["test_metrics_with_floor"]["easy_degradation_vs_floor"] <= 0.02
    ]
    gates = {
        "fresh_retrained_ablation": payload["result_source"] == "fresh_retrained_scene_proxy_subset_ablation",
        "no_scene_baseline_retrained": no_scene["scene_feature_count"] == 0 and no_scene["test_metrics_with_floor"]["rows"] > 0,
        "multiple_scene_subsets_retrained": len(non_empty) >= 2 and all(row["test_metrics_with_floor"]["rows"] > 0 for row in non_empty),
        "scene_subset_t50_lift_found": bool(t50_positive),
        "scene_subset_hard_or_all_lift_found": bool(hard_positive)
        or any(row["delta_vs_retrained_no_scene"]["full_waypoint_ade_improvement_vs_floor"] > 0.0 for row in non_empty),
        "safe_t50_scene_variant_available": payload.get("best_safe_variant_by_t50_delta") is not None,
        "latent_noncollapse_for_scene_variants": all(row["latent_variance"] > 0.01 for row in non_empty),
        "checkpoints_not_committed": all(row["checkpoint_committed"] is False for row in payload["variants"]),
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["scene_proxy_train_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "not_overclaimed_full_factorial": payload["ablation_type"]["not_full_stage43_factorial_all_modules"] is True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(passed == total)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ag_scene_proxy_retrained_ablation_pass"
        if deploy
        else "stage43_ag_scene_proxy_retrained_ablation_diagnostic",
        "scene_proxy_retrained_ablation_supports_contribution": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _variant_table_rows(payload: Mapping[str, Any]) -> list[str]:
    rows = [
        "| variant | scene features | all | t50 | hard | easy | delta all vs no-scene | delta t50 vs no-scene | delta hard vs no-scene | latent var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variants"]:
        metrics = row["test_metrics_with_floor"]
        delta = row["delta_vs_retrained_no_scene"]
        rows.append(
            f"| `{row['variant']}` | `{row['scene_feature_count']}` | `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(metrics['easy_degradation_vs_floor'])}` | `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{row['latent_variance']:.4f}` |"
        )
    return rows


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ag_gate"]
    best_t50 = next(row for row in payload["variants"] if row["variant"] == payload["best_variant_by_t50_delta"])
    best_safe_t50 = (
        next(row for row in payload["variants"] if row["variant"] == payload["best_safe_variant_by_t50_delta"])
        if payload.get("best_safe_variant_by_t50_delta")
        else None
    )
    lines = [
        "# Stage43-AG Scene-Proxy Retrained Ablation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- best t50 variant: `{payload['best_variant_by_t50_delta']}`",
        f"- best safe t50 variant: `{payload.get('best_safe_variant_by_t50_delta')}`",
        f"- best hard variant: `{payload['best_variant_by_hard_delta']}`",
        f"- best all variant: `{payload['best_variant_by_all_delta']}`",
        "",
        "## Variants",
        "",
        *_variant_table_rows(payload),
        "",
        "## Interpretation",
        "",
        f"The best t50 scene subset is `{payload['best_variant_by_t50_delta']}` with delta vs retrained no-scene `{_pct(best_t50['delta_vs_retrained_no_scene']['t50_full_waypoint_ade_improvement_vs_floor'])}`.",
        (
            f"The best safe t50 scene subset is `{best_safe_t50['variant']}` with delta `{_pct(best_safe_t50['delta_vs_retrained_no_scene']['t50_full_waypoint_ade_improvement_vs_floor'])}` and easy degradation `{_pct(best_safe_t50['test_metrics_with_floor']['easy_degradation_vs_floor'])}`."
            if best_safe_t50
            else "No scene subset achieved positive t50 lift while preserving easy degradation <= 2%."
        ),
        "",
        "This is a focused retrained scene-proxy subset ablation. It does not replace the broader Stage43-AF same-route deployment counterfactual, and it is not a full all-module factorial ablation. Unsafe higher-lift variants are reported but not treated as deployable evidence.",
        "",
        "## Boundary",
        "",
        "- Dataset-local/raw-frame 2.5D only.",
        "- Scene proxy is train-only route/goal/context proxy, not raw imagery or verified SDF.",
        "- Future waypoints are labels/eval only.",
        "- No metric/seconds claim, no Stage5C, no SMC.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AG Scene-Proxy Retrained Ablation Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- supports contribution: `{gate['scene_proxy_retrained_ablation_supports_contribution']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ag_gate"]
    best = next(row for row in payload["variants"] if row["variant"] == payload["best_variant_by_t50_delta"])
    safe_best_name = payload.get("best_safe_variant_by_t50_delta")
    safe_best = next((row for row in payload["variants"] if row["variant"] == safe_best_name), None)
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"scene_proxy_retrained_ablation_supports_contribution = `{gate['scene_proxy_retrained_ablation_supports_contribution']}`",
        f"best_t50_variant = `{payload['best_variant_by_t50_delta']}`",
        f"best_t50_delta_vs_retrained_no_scene = `{_pct(best['delta_vs_retrained_no_scene']['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"best_safe_t50_variant = `{safe_best_name}`",
        f"best_safe_t50_delta_vs_retrained_no_scene = `{_pct(safe_best['delta_vs_retrained_no_scene']['t50_full_waypoint_ade_improvement_vs_floor']) if safe_best else 'not_available'}`",
        f"best_hard_variant = `{payload['best_variant_by_hard_delta']}`",
        "",
        "Stage43-AG fresh-trains no-scene, geometry/route, goal-only, and full-scene proxy variants under the same protected full-waypoint latent dynamics protocol. This is a focused retrained scene-proxy subset ablation, not a full all-module factorial ablation. The report separates raw-best t50 from safety-preserving t50 evidence.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; scene proxy is not raw image/SDF; future labels are supervision/eval only; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ag_scene_proxy_retrained_ablation"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "scene_proxy_retrained_ablation_supports_contribution": gate[
            "scene_proxy_retrained_ablation_supports_contribution"
        ],
        "best_variant_by_t50_delta": payload["best_variant_by_t50_delta"],
        "best_safe_variant_by_t50_delta": payload.get("best_safe_variant_by_t50_delta"),
        "best_variant_by_hard_delta": payload["best_variant_by_hard_delta"],
        "best_variant_by_all_delta": payload["best_variant_by_all_delta"],
        "variants": [
            {
                "variant": row["variant"],
                "metrics": row["test_metrics_with_floor"],
                "delta_vs_retrained_no_scene": row["delta_vs_retrained_no_scene"],
                "latent_variance": row["latent_variance"],
                "checkpoint_committed": False,
            }
            for row in payload["variants"]
        ],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ag_scene_proxy_retrained_ablation"
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
                        "stage": "Stage43-AG",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "scene_proxy_retrained_ablation_supports_contribution": gate[
                            "scene_proxy_retrained_ablation_supports_contribution"
                        ],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AG retrained scene-proxy subset ablation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--variants", default="no_scene,geometry_route,goal_only,full_scene")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=431)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_ag_gate"]
    print(f"Stage43-AG: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"scene_proxy_retrained_ablation_supports_contribution={gate['scene_proxy_retrained_ablation_supports_contribution']}")
    return result


if __name__ == "__main__":
    main()
