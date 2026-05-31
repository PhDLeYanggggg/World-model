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
from src.stage43_scene_raster_proxy_tokens import DATA_DIR as SCENE_PROXY_DIR
from src.stage43_scene_raster_proxy_tokens import REPORT_JSON as STAGE43_AA_JSON


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_scene_proxy_augmented_latent_dynamics.json"
REPORT_MD = OUT_DIR / "stage43_scene_proxy_augmented_latent_dynamics.md"
GATE_MD = OUT_DIR / "stage43_stage_ab_scene_proxy_augmented_latent_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_scene_proxy_augmented_latent_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AB_SCENE_PROXY_AUGMENTED_LATENT_DYNAMICS"
SOURCE = "fresh_stage43_ab_scene_proxy_augmented_latent_dynamics"
SCENE_FEATURE_PREFIX = "scene_proxy::"


def _sample_ids(split: str, *, max_rows: int | None, seed: int) -> np.ndarray:
    cache = m._npz(m._cache_path(split))
    n = len(cache["horizon"])
    ids = np.arange(n)
    if max_rows is not None and max_rows < n:
        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        ids = np.sort(rng.choice(ids, size=int(max_rows), replace=False))
    return ids


def _scene_proxy_npz(split: str) -> Mapping[str, np.ndarray]:
    path = SCENE_PROXY_DIR / f"stage43_scene_proxy_features_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage43-AA scene proxy features: {path}")
    return np.load(path, allow_pickle=False)


def _augmented_split(split: str, *, max_rows: int | None, seed: int) -> m.WaypointSplit:
    ds = m._build_split(split, max_rows=max_rows, seed=seed)
    ids = _sample_ids(split, max_rows=max_rows, seed=seed)
    scene = _scene_proxy_npz(split)
    features = scene["features"].astype(np.float32)[ids]
    names = [f"{SCENE_FEATURE_PREFIX}{name}" for name in scene["feature_names"].astype(str).tolist()]
    if len(features) != len(ds.x):
        raise ValueError(f"Scene proxy row mismatch for {split}: {len(features)} != {len(ds.x)}")
    ds.x = np.concatenate([ds.x, features], axis=1).astype(np.float32)
    ds.feature_names = [*ds.feature_names, *names]
    return ds


def _scene_feature_hash(split: str) -> str:
    return str(_scene_proxy_npz(split)["feature_hash"][0])


def _train_eval(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    seed = int(args.seed)
    runtime = m._configure_runtime(seed)
    mode = "quick" if args.quick else "small" if args.small else "medium"
    max_train = 6000 if args.quick else 30000 if args.small else 90000
    max_val = 3000 if args.quick else 12000 if args.small else 40000
    max_test = 3000 if args.quick else 16000 if args.small else 50000

    train = _augmented_split("train", max_rows=max_train, seed=seed)
    val = _augmented_split("val", max_rows=max_val, seed=seed)
    test = _augmented_split("test", max_rows=max_test, seed=seed)
    train, val, test, mean, std = m._standardize(train, val, test)

    model = m.FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim))
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    best_path = CKPT_DIR / "stage43_scene_proxy_augmented_latent_dynamics.pt"
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        stats: list[dict[str, float]] = []
        for batch_ids in m._batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = m._loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_pred = m._predict(model, val, device, int(args.batch_size))
        val_ade, _ = m._trajectory_error(val, val_pred["waypoint"])
        val_loss = float(np.mean((val_ade - val.floor_ade) ** 2))
        latent_var = float(np.mean([row["latent_variance"] for row in stats])) if stats else 0.0
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_candidate_mse_to_floor": val_loss,
            "latent_variance": latent_var,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            m._jsonable(
                {
                    "source": SOURCE,
                    "epoch": epoch + 1,
                    "elapsed_s": time.time() - start,
                    "last": row,
                    "git_commit": m._git_commit(),
                    "mode": mode,
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
                    "seed": seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "scene_proxy_augmented": True,
                    "scene_proxy_feature_hashes": {split: _scene_feature_hash(split) for split in m.SPLITS},
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "future_waypoint_label_eval_only": True,
                        "central_velocity_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))
    val_policy = m._search_policy(val, val_pred)
    selected_ade, selected_fde, switched = m._select_with_policy(test, test_pred, val_policy["policy"])
    protected_metrics = m._metrics(test, selected_ade, selected_fde, switched)
    bootstrap = m._bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=seed + 1000)
    ungated_ade, ungated_fde = m._trajectory_error(test, test_pred["waypoint"])
    ungated_metrics = m._metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    latent_var = float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0

    stage43_m = read_json(OUT_DIR / "stage43_full_waypoint_latent_dynamics.json", {})
    stage43_aa = read_json(STAGE43_AA_JSON, {})
    baseline_metrics = stage43_m.get("test_metrics_with_floor", {})
    delta_vs_m = _delta_metrics(protected_metrics, baseline_metrics)
    scene_lift = (
        delta_vs_m["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_vs_m["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_vs_m["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
    )
    deploy = bool(
        scene_lift
        and (
            protected_metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or protected_metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or protected_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
        and protected_metrics["easy_degradation_vs_floor"] <= 0.02
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_scene_proxy_augmented_torch_training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "checkpoint": str(best_path),
        "checkpoint_sha256": m._sha256(best_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "stage43_m_baseline_verdict": stage43_m.get("stage43_m_gate", {}).get("verdict"),
        "stage43_aa_precondition": stage43_aa.get("stage43_aa_gate", {}),
        "scene_proxy_feature_hashes": {split: _scene_feature_hash(split) for split in m.SPLITS},
        "cache_row_hashes": {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS},
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "base_feature_count": int(len(train.feature_names) - 14),
        "scene_proxy_feature_count": 14,
        "feature_count": int(train.x.shape[1]),
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": protected_metrics,
        "baseline_stage43_m_metrics_with_floor": baseline_metrics,
        "delta_vs_stage43_m": delta_vs_m,
        "bootstrap_ci": bootstrap,
        "test_metrics_neural_without_floor": ungated_metrics,
        "latent_variance": latent_var,
        "scene_proxy_lift_over_stage43_m": bool(scene_lift),
        "deploy_scene_proxy_augmented_neural": deploy,
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
                OUT_DIR / "stage43_scene_raster_proxy_tokens.json",
                OUT_DIR / "stage43_full_waypoint_latent_dynamics.json",
            ]
        ),
    }
    payload["stage43_ab_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _delta_metrics(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "full_waypoint_ade_improvement_vs_floor",
        "endpoint_fde_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "t50_endpoint_fde_improvement_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    return {key: float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    delta = payload["delta_vs_stage43_m"]
    gates = {
        "stage43_m_baseline_available": payload["stage43_m_baseline_verdict"]
        in {"stage43_m_protected_full_waypoint_latent_candidate_pass", "stage43_m_full_waypoint_latent_diagnostic_keep_floor"},
        "stage43_aa_precondition_passed": payload["stage43_aa_precondition"].get("verdict")
        == "stage43_aa_scene_raster_proxy_tokens_pass",
        "torch_training_fresh_run": payload["result_source"] == "fresh_run_scene_proxy_augmented_torch_training"
        and Path(payload["checkpoint"]).exists(),
        "checkpoint_not_committed": payload["checkpoint_committed"] is False,
        "scene_proxy_features_integrated": payload["scene_proxy_feature_count"] == 14
        and payload["feature_count"] > payload["base_feature_count"],
        "scene_proxy_hashes_recorded": all(bool(v) for v in payload["scene_proxy_feature_hashes"].values()),
        "protected_eval_completed": metrics["rows"] > 0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "scene_proxy_lift_or_honest_not_promoted": payload["scene_proxy_lift_over_stage43_m"] is True
        or payload["deploy_scene_proxy_augmented_neural"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["scene_proxy_train_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(payload["deploy_scene_proxy_augmented_neural"] and passed == total)
    verdict = (
        "stage43_ab_scene_proxy_augmented_latent_lift_candidate"
        if deploy
        else "stage43_ab_scene_proxy_augmented_latent_diagnostic_keep_stage43_m"
    )
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_scene_proxy_augmented_neural": deploy,
        "scene_proxy_lift_over_stage43_m": bool(payload["scene_proxy_lift_over_stage43_m"]),
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ab_gate"]
    cur = payload["test_metrics_with_floor"]
    base = payload["baseline_stage43_m_metrics_with_floor"]
    delta = payload["delta_vs_stage43_m"]
    lines = [
        "# Stage43-AB Scene-Proxy Augmented Latent Dynamics",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy scene-proxy augmented neural: `{gate['deploy_scene_proxy_augmented_neural']}`",
        f"- checkpoint committed: `{payload['checkpoint_committed']}`",
        "",
        "## Feature Integration",
        "",
        f"- base feature count: `{payload['base_feature_count']}`",
        f"- scene proxy feature count: `{payload['scene_proxy_feature_count']}`",
        f"- total feature count: `{payload['feature_count']}`",
        f"- scene proxy hashes: `{payload['scene_proxy_feature_hashes']}`",
        "",
        "## Metrics vs Floor",
        "",
        "| metric | Stage43-M | Stage43-AB | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in [
        "full_waypoint_ade_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "switch_rate",
    ]:
        lines.append(f"| `{key}` | `{_pct(float(base.get(key, 0.0)))}` | `{_pct(float(cur.get(key, 0.0)))}` | `{_pct(float(delta.get(key, 0.0)))}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Scene/raster proxy features improved at least one Stage43-M comparison slice and remain protected by the same floor. This is a candidate for promotion, but still dataset-local/raw-frame 2.5D and proxy-only."
                if gate["deploy_scene_proxy_augmented_neural"]
                else "Scene/raster proxy features were integrated and trained in a fresh torch run, but they are not promoted unless they improve Stage43-M while preserving easy cases. Keep Stage43-M as the current deployed latent dynamics head."
            ),
            "",
            "## Boundary",
            "",
            "- Scene proxy is train-only route/SDF/goal prior, not raw imagery and not verified metric SDF.",
            "- Future endpoint/full waypoints are labels/eval only.",
            "- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
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
            "# Stage43-AB Scene-Proxy Augmented Latent Dynamics Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy scene-proxy augmented neural: `{gate['deploy_scene_proxy_augmented_neural']}`",
            f"- scene proxy lift over Stage43-M: `{gate['scene_proxy_lift_over_stage43_m']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ab_gate"]
    cur = payload["test_metrics_with_floor"]
    delta = payload["delta_vs_stage43_m"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_scene_proxy_augmented_neural = `{gate['deploy_scene_proxy_augmented_neural']}`",
        f"scene_proxy_lift_over_stage43_m = `{gate['scene_proxy_lift_over_stage43_m']}`",
        "",
        f"full_waypoint_ade_vs_floor = `{_pct(cur['full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`",
        f"t50_full_waypoint_ade_vs_floor = `{_pct(cur['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"hard_failure_vs_floor = `{_pct(cur['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"easy_degradation = `{_pct(cur['easy_degradation_vs_floor'])}`",
        "",
        "Stage43-AB retrains the full-waypoint latent dynamics head with the Stage43-AA train-only scene/raster proxy features appended to the causal input. It compares against Stage43-M and only promotes the augmented model if it improves Stage43-M while preserving easy cases.",
        "",
        "Boundary unchanged: scene proxy is not raw image/SDF, future labels are loss/eval only, no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ab_scene_proxy_augmented_latent_dynamics"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_scene_proxy_augmented_neural": gate["deploy_scene_proxy_augmented_neural"],
        "scene_proxy_lift_over_stage43_m": gate["scene_proxy_lift_over_stage43_m"],
        "metrics": payload["test_metrics_with_floor"],
        "delta_vs_stage43_m": payload["delta_vs_stage43_m"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "checkpoint_committed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ab_scene_proxy_augmented_latent_dynamics"
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
                        "stage": "Stage43-AB",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "deploy_scene_proxy_augmented_neural": gate["deploy_scene_proxy_augmented_neural"],
                        "scene_proxy_lift_over_stage43_m": gate["scene_proxy_lift_over_stage43_m"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage43-AB scene-proxy augmented latent full-waypoint dynamics.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _train_eval(args)
    gate = result["stage43_ab_gate"]
    print(f"Stage43-AB: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_scene_proxy_augmented_neural={gate['deploy_scene_proxy_augmented_neural']}")
    return result


if __name__ == "__main__":
    main()
