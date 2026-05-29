from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import (
    CKPT_DIR,
    DATA35,
    OUT_DIR,
    ProtectedLatentStateModel,
    SplitData,
    _batch_indices,
    _build_split,
    _configure_runtime,
    _git_commit,
    _jsonable,
    _loss,
    _metrics,
    _predict,
    _search_policy,
    _select_with_policy,
    _sha256,
    _standardize,
)
from src.stage43_source_level_heldout_split import REPORT_JSON as SPLIT_REPORT_JSON
from src.stage43_source_level_heldout_split import build_source_level_split


REPORT_JSON = OUT_DIR / "stage43_source_level_latent_eval.json"
REPORT_MD = OUT_DIR / "stage43_source_level_latent_eval.md"
TRAINING_JSON = OUT_DIR / "stage43_source_level_latent_training.json"
TRAINING_MD = OUT_DIR / "stage43_source_level_latent_training.md"
GATE_MD = OUT_DIR / "stage43_stage_g_source_level_latent_gate.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_source_level_latent_heartbeat.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_G_SOURCE_LEVEL_PROTECTED_LATENT"
SOURCE = "fresh_stage43_g_source_level_protected_latent"
OLD_SPLITS = ["train", "val", "test"]


def _load_manifest() -> Mapping[str, Any]:
    if not SPLIT_REPORT_JSON.exists():
        return build_source_level_split()
    return read_json(SPLIT_REPORT_JSON, {})


def _source_files_for_old_split(old_split: str) -> np.ndarray:
    geo = np.load(DATA35 / f"expanded_external_{old_split}.npz", allow_pickle=False)
    return geo["source_file"].astype(str)


def _subset_split_data(ds: SplitData, ids: np.ndarray, split_name: str) -> SplitData:
    return SplitData(
        split=split_name,
        x=ds.x[ids].copy(),
        y_delta=ds.y_delta[ids].copy(),
        y_failure=ds.y_failure[ids].copy(),
        y_gain=ds.y_gain[ids].copy(),
        y_harm=ds.y_harm[ids].copy(),
        y_occupancy=ds.y_occupancy[ids].copy(),
        horizon=ds.horizon[ids].copy(),
        domain=ds.domain[ids].copy(),
        floor_err=ds.floor_err[ids].copy(),
        strongest_err=ds.strongest_err[ids].copy(),
        candidate_err_ref=ds.candidate_err_ref[ids].copy(),
        hard=ds.hard[ids].copy(),
        failure=ds.failure[ids].copy(),
        easy=ds.easy[ids].copy(),
        scale=ds.scale[ids].copy(),
        feature_names=list(ds.feature_names),
    )


def _concat_split_data(rows: list[SplitData], split_name: str) -> SplitData:
    first = rows[0]
    return SplitData(
        split=split_name,
        x=np.concatenate([row.x for row in rows], axis=0),
        y_delta=np.concatenate([row.y_delta for row in rows], axis=0),
        y_failure=np.concatenate([row.y_failure for row in rows], axis=0),
        y_gain=np.concatenate([row.y_gain for row in rows], axis=0),
        y_harm=np.concatenate([row.y_harm for row in rows], axis=0),
        y_occupancy=np.concatenate([row.y_occupancy for row in rows], axis=0),
        horizon=np.concatenate([row.horizon for row in rows], axis=0),
        domain=np.concatenate([row.domain for row in rows], axis=0),
        floor_err=np.concatenate([row.floor_err for row in rows], axis=0),
        strongest_err=np.concatenate([row.strongest_err for row in rows], axis=0),
        candidate_err_ref=np.concatenate([row.candidate_err_ref for row in rows], axis=0),
        hard=np.concatenate([row.hard for row in rows], axis=0),
        failure=np.concatenate([row.failure for row in rows], axis=0),
        easy=np.concatenate([row.easy for row in rows], axis=0),
        scale=np.concatenate([row.scale for row in rows], axis=0),
        feature_names=list(first.feature_names),
    )


def _maybe_sample(ds: SplitData, max_rows: int | None, seed: int) -> SplitData:
    if max_rows is None or len(ds.x) <= max_rows:
        return ds
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(np.arange(len(ds.x)), size=int(max_rows), replace=False))
    return _subset_split_data(ds, ids, ds.split)


def build_source_level_datasets(
    *,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    seed: int = 443,
) -> tuple[SplitData, SplitData, SplitData, Mapping[str, Any]]:
    manifest = _load_manifest()
    assignments = {str(k): str(v) for k, v in manifest["source_assignments"].items()}
    by_new_split: dict[str, list[SplitData]] = {"train": [], "val": [], "test": []}
    for old_split in OLD_SPLITS:
        old_ds = _build_split(old_split, max_rows=None, seed=seed)
        sources = _source_files_for_old_split(old_split)
        for new_split in ["train", "val", "test"]:
            ids = np.where(np.asarray([assignments[str(src)] == new_split for src in sources], dtype=bool))[0]
            if len(ids):
                by_new_split[new_split].append(_subset_split_data(old_ds, ids, new_split))
    train = _concat_split_data(by_new_split["train"], "train")
    val = _concat_split_data(by_new_split["val"], "val")
    test = _concat_split_data(by_new_split["test"], "test")
    train = _maybe_sample(train, max_train, seed + 10)
    val = _maybe_sample(val, max_val, seed + 20)
    test = _maybe_sample(test, max_test, seed + 30)
    return train, val, test, manifest


def _domain_metrics(ds: SplitData, selected: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in sorted(set(ds.domain.astype(str).tolist())):
        mask = ds.domain.astype(str) == domain
        out[domain] = _metrics(_subset_split_data(ds, np.where(mask)[0], ds.split), selected[mask], switched[mask])
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    domains = set(payload["test_domains"])
    gates = {
        "source_level_split_precondition_passed": payload["source_level_split"]["verdict"] == "stage43_f_source_level_split_ready",
        "torch_training_fresh_run": payload["result_source"] == "fresh_run" and Path(payload["checkpoint"]).exists(),
        "test_contains_all_domains": domains >= {"ETH_UCY", "TrajNet", "UCY"},
        "latent_noncollapse": payload["latent_variance"] > 0.01,
        "protected_eval_completed": metrics["rows"] > 0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "neural_has_any_protected_lift": (
            metrics["all_improvement_vs_floor"] > 0.0
            or metrics["t50_improvement_vs_floor"] > 0.0
            or metrics["hard_failure_improvement_vs_floor"] > 0.0
        ),
        "domain_metrics_reported": set(payload["domain_metrics"].keys()) >= {"ETH_UCY", "TrajNet", "UCY"},
        "ungated_neural_reported": "all_improvement_vs_floor" in payload["test_metrics_neural_without_floor"],
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(
        passed == total
        and (
            metrics["all_improvement_vs_floor"] > 0.0
            or metrics["t50_improvement_vs_floor"] > 0.0
            or metrics["hard_failure_improvement_vs_floor"] > 0.0
        )
    )
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_g_source_level_latent_candidate_pass" if passed == total else "stage43_g_source_level_latent_diagnostic_only",
        "deploy_neural": deploy,
    }


def run_training(
    *,
    mode: str = "small",
    epochs: int = 5,
    batch_size: int = 512,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    lr: float = 1e-3,
    seed: int = 443,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    runtime = _configure_runtime(seed)
    if mode == "quick":
        max_train, max_val, max_test = 4000, 2000, 2000
        epochs = min(epochs, 2)
    elif mode == "small":
        max_train, max_val, max_test = 60000, 30000, 30000
    elif mode == "full":
        max_train = max_val = max_test = None
    else:
        raise ValueError(f"Unknown Stage43-G mode: {mode}")
    train, val, test, manifest = build_source_level_datasets(max_train=max_train, max_val=max_val, max_test=max_test, seed=seed)
    raw_rows = {"train": len(train.x), "val": len(val.x), "test": len(test.x)}
    train, val, test, mean, std = _standardize(train, val, test)
    device = torch.device("cpu")
    model = ProtectedLatentStateModel(train.x.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    ckpt_path = CKPT_DIR / f"stage43_source_level_latent_{mode}.pt"
    started = time.time()
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        latent_vars: list[float] = []
        for ids in _batch_indices(len(train.x), int(batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stats = _loss(model, train, ids, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            latent_vars.append(float(stats["latent_variance"]))
        val_pred = _predict(model, val, device, int(batch_size))
        val_policy = _search_policy(val, val_pred)
        val_metric = float(val_policy["metrics"]["all_improvement_vs_floor"] + val_policy["metrics"]["t50_improvement_vs_floor"])
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "latent_variance": float(np.mean(latent_vars)) if latent_vars else 0.0,
            "val_objective_proxy": val_metric,
            "val_policy": val_policy,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            {
                "source": SOURCE,
                "mode": mode,
                "epoch": epoch + 1,
                "elapsed_s": time.time() - started,
                "last": row,
                "git_commit": _git_commit(),
            },
        )
        if val_metric > best_score:
            best_score = val_metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train.feature_names,
                    "input_dim": int(train.x.shape[1]),
                    "hidden_dim": int(hidden_dim),
                    "latent_dim": int(latent_dim),
                    "seed": int(seed),
                    "epoch": epoch + 1,
                    "mode": mode,
                    "runtime": runtime,
                    "source_level_split_report": str(SPLIT_REPORT_JSON),
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "central_velocity_official_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                ckpt_path,
            )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, device, int(batch_size))
    test_pred = _predict(model, test, device, int(batch_size))
    val_policy = _search_policy(val, val_pred)
    selected, switched = _select_with_policy(test, test_pred, val_policy["policy"])
    candidate, candidate_switch = _select_with_policy(
        test,
        {"delta": test_pred["delta"], "gain": np.ones(len(test.x)), "harm": np.zeros(len(test.x)), "failure": np.ones(len(test.x))},
        {"gain_threshold": 0.0, "harm_threshold": 1.0, "failure_threshold": 0.0},
    )
    metrics = _metrics(test, selected, switched)
    candidate_metrics = _metrics(test, candidate, candidate_switch)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "mode": mode,
        "checkpoint": str(ckpt_path),
        "checkpoint_committed": False,
        "checkpoint_sha256": _sha256(ckpt_path),
        "runtime": runtime,
        "data_rows": raw_rows,
        "source_level_split": {
            "report": str(SPLIT_REPORT_JSON),
            "verdict": manifest["stage43_f_gate"]["verdict"],
            "gate": f"{manifest['stage43_f_gate']['passed']} / {manifest['stage43_f_gate']['total']}",
            "row_hash": manifest["pool"]["row_hash"],
            "split_summary": manifest["split_summary"],
        },
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_domains": sorted(set(test.domain.astype(str).tolist())),
        "test_metrics_with_floor": metrics,
        "test_metrics_neural_without_floor": candidate_metrics,
        "domain_metrics": _domain_metrics(test, selected, switched),
        "safety_floor_intervention": {
            "switch_rate": float(np.mean(switched)),
            "fallback_rate": float(1.0 - np.mean(switched)),
            "interpretation": "floor_not_active_on_test" if float(np.mean(switched)) >= 0.999 else "floor_active_on_some_rows",
            "next_required_audit": "bootstrap and safety stress before replacing the frozen Stage37/Stage42 floor",
        },
        "latent_variance": float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0,
        "claim_boundary": {
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
        },
    }
    payload["stage43_g_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    training = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "generated_at_utc": payload["generated_at_utc"],
        "mode": payload["mode"],
        "checkpoint": payload["checkpoint"],
        "checkpoint_committed": False,
        "runtime": payload["runtime"],
        "data_rows": payload["data_rows"],
        "training_history": payload["training_history"],
        "source_level_split": payload["source_level_split"],
    }
    write_json(TRAINING_JSON, _jsonable(training))
    gate = payload["stage43_g_gate"]
    metrics = payload["test_metrics_with_floor"]
    cand = payload["test_metrics_neural_without_floor"]
    lines = [
        "# Stage43-G Source-Level Protected Latent Model",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- deploy neural: `{gate['deploy_neural']}`",
        f"- mode: `{payload['mode']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        "- checkpoint committed: `False`",
        f"- latent variance: `{payload['latent_variance']:.6f}`",
        f"- source split row hash: `{payload['source_level_split']['row_hash']}`",
        f"- safety floor fallback rate: `{payload['safety_floor_intervention']['fallback_rate']:.6f}`",
        "",
        "## Protected Test Metrics vs Safety Floor",
        "",
        f"- rows: `{metrics['rows']}`",
        f"- all improvement: `{metrics['all_improvement_vs_floor']:.6f}`",
        f"- t50 improvement: `{metrics['t50_improvement_vs_floor']:.6f}`",
        f"- t100 raw-frame diagnostic: `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`",
        f"- hard/failure improvement: `{metrics['hard_failure_improvement_vs_floor']:.6f}`",
        f"- easy degradation: `{metrics['easy_degradation_vs_floor']:.6f}`",
        f"- switch rate: `{metrics['switch_rate']:.6f}`",
        "",
        "## Safety Floor Interpretation",
        "",
        f"- intervention status: `{payload['safety_floor_intervention']['interpretation']}`",
        f"- next required audit: {payload['safety_floor_intervention']['next_required_audit']}",
        "- Because switch rate is part of the evidence, a full-switch result must not be described as final floor replacement until bootstrap and safety stress pass.",
        "",
        "## Domain Metrics",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {domain} | {row['rows']} | {row['all_improvement_vs_floor']:.6f} | {row['t50_improvement_vs_floor']:.6f} | {row['t100_raw_frame_diagnostic_vs_floor']:.6f} | {row['hard_failure_improvement_vs_floor']:.6f} | {row['easy_degradation_vs_floor']:.6f} | {row['switch_rate']:.6f} |"
            for domain, row in payload["domain_metrics"].items()
        ],
        "",
        "## Ungated Neural Diagnostic",
        "",
        f"- all improvement: `{cand['all_improvement_vs_floor']:.6f}`",
        f"- t50 improvement: `{cand['t50_improvement_vs_floor']:.6f}`",
        f"- hard/failure improvement: `{cand['hard_failure_improvement_vs_floor']:.6f}`",
        f"- easy degradation: `{cand['easy_degradation_vs_floor']:.6f}`",
        "",
        "No Stage5C, no SMC, no metric/seconds-level claim, no true-3D/foundation claim.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        TRAINING_MD,
        [
            "# Stage43-G Source-Level Protected Latent Training",
            "",
            f"- source: `{training['source']}`",
            f"- mode: `{training['mode']}`",
            f"- checkpoint: `{training['checkpoint']}`",
            "- checkpoint committed: `False`",
            f"- data rows: `{training['data_rows']}`",
            f"- runtime: `{training['runtime']}`",
            "",
            "This training uses the Stage43-F source-file-level split. Future endpoint/waypoint labels remain loss/eval only.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-G Source-Level Latent Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy neural: `{gate['deploy_neural']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_g_gate"]
    metrics = payload["test_metrics_with_floor"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_neural = `{gate['deploy_neural']}`",
        "",
        "Stage43-G trains a fresh protected latent-state model on the Stage43-F source-file-level split, where ETH_UCY, TrajNet, and UCY all appear in held-out test through disjoint source files. This replaces the earlier UCY-only checkpoint for multi-domain evaluation.",
        "",
        f"Protected test metrics vs floor: all `{metrics['all_improvement_vs_floor']:.6f}`, t50 `{metrics['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{metrics['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{metrics['easy_degradation_vs_floor']:.6f}`.",
        "",
        f"Safety note: test switch rate is `{metrics['switch_rate']:.6f}` and fallback rate is `{payload['safety_floor_intervention']['fallback_rate']:.6f}`. This means the full split result needs bootstrap and safety-stress confirmation before it can replace the frozen floor as a deployment policy.",
        "",
        "This remains dataset-local/raw-frame 2.5D evidence. Stage5C and SMC are disabled; no metric/seconds/true-3D/foundation claim is made.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_g_source_level_protected_latent"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_neural": gate["deploy_neural"],
        "mode": payload["mode"],
        "data_rows": payload["data_rows"],
        "test_domains": payload["test_domains"],
        "metrics": payload["test_metrics_with_floor"],
        "domain_metrics": payload["domain_metrics"],
        "safety_floor_intervention": payload["safety_floor_intervention"],
        "source_level_split": payload["source_level_split"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "small", "full"], default="small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=443)
    args = parser.parse_args(argv)
    return run_training(
        mode=args.mode,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        lr=float(args.lr),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    result = main()
    gate = result["stage43_g_gate"]
    print(f"Stage43-G source-level latent model: {gate['verdict']} ({gate['passed']}/{gate['total']})")
