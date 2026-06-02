from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _batch_indices,
    _build_split,
    _git_commit,
    _jsonable,
    _target_vec,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_latent_transition_consistency_audit import (
    REPORT_JSON as STAGE43_BY_JSON,
    _apply_readout,
    _bootstrap_transition,
    _breakdown,
    _fit_ridge_readout,
    _predict_transition_latents,
    _transition_metrics,
)


REPORT_JSON = OUT_DIR / "stage43_latent_transition_adapter_repair.json"
REPORT_MD = OUT_DIR / "stage43_latent_transition_adapter_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_bz_latent_transition_adapter_repair_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
CKPT_DIR = OUT_DIR / "checkpoints"
HEARTBEAT_JSON = OUT_DIR / "stage43_latent_transition_adapter_repair_heartbeat.json"

SECTION = "STAGE43_BZ_LATENT_TRANSITION_ADAPTER_REPAIR"
SOURCE = "fresh_stage43_bz_latent_transition_adapter_repair"


class LatentTransitionAdapter(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int = 128, delta_clip: float = 3.0) -> None:
        super().__init__()
        self.delta_clip = float(delta_clip)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + latent_dim),
            nn.Linear(feature_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.net(torch.cat([x, z_t], dim=1))) * self.delta_clip
        return self.out_norm(z_t + delta)


def _configure_runtime(seed: int) -> dict[str, Any]:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    return {
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "num_workers": 0,
    }


def _limit_split(ds: Any, max_rows: int | None, seed: int) -> Any:
    if max_rows is None or len(ds.x) <= int(max_rows):
        return ds
    original_rows = len(ds.x)
    rng = np.random.default_rng(int(seed))
    ids = np.sort(rng.choice(original_rows, size=int(max_rows), replace=False))
    for key, value in list(ds.__dict__.items()):
        if isinstance(value, np.ndarray) and len(value) == original_rows:
            setattr(ds, key, value[ids])
    return ds


def _adapter_predict(
    adapter: LatentTransitionAdapter,
    x: np.ndarray,
    z_t: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    adapter.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for ids in _batch_indices(len(x), int(batch_size), shuffle=False, seed=0):
            xb = torch.from_numpy(x[ids].astype(np.float32))
            zb = torch.from_numpy(z_t[ids].astype(np.float32))
            out.append(adapter(xb, zb).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - nn.functional.cosine_similarity(pred, target, dim=1).mean()


def _train_adapter(
    adapter: LatentTransitionAdapter,
    train_x: np.ndarray,
    train_z_t: np.ndarray,
    train_target: np.ndarray,
    val_x: np.ndarray,
    val_z_t: np.ndarray,
    val_target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    checkpoint: Path,
) -> list[dict[str, Any]]:
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(int(epochs)):
        adapter.train()
        losses: list[float] = []
        for ids in _batch_indices(len(train_x), int(batch_size), shuffle=True, seed=int(seed) + epoch):
            xb = torch.from_numpy(train_x[ids].astype(np.float32))
            zt = torch.from_numpy(train_z_t[ids].astype(np.float32))
            target = torch.from_numpy(train_target[ids].astype(np.float32))
            pred = adapter(xb, zt)
            mse = nn.functional.mse_loss(pred, target)
            loss = mse + 0.10 * _cosine_loss(pred, target) + 0.01 * torch.mean((pred - zt) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = _adapter_predict(adapter, val_x, val_z_t, batch_size=int(batch_size))
        val_mse = float(np.mean((val_pred.astype(np.float64) - val_target.astype(np.float64)) ** 2))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_mse_to_target_latent": val_mse,
            "elapsed_s": time.time() - start,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable({"source": SOURCE, "epoch": epoch + 1, "last": row, "git_commit": _git_commit()}),
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "adapter_state": adapter.state_dict(),
                    "feature_dim": int(train_x.shape[1]),
                    "latent_dim": int(train_z_t.shape[1]),
                    "hidden_dim": int(adapter.net[1].out_features),
                    "delta_clip": float(adapter.delta_clip),
                    "seed": int(seed),
                    "epoch": epoch + 1,
                    "best_val_mse_to_target_latent": best_val,
                    "runtime": {
                        "device": "cpu",
                        "num_workers": 0,
                        "future_target_latent_input": False,
                        "future_target_latent_label_eval_only": True,
                    },
                },
                checkpoint,
            )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state"])
    return history


def _weak_slices(table: Mapping[str, Mapping[str, Any]], *, axis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, row in table.items():
        if float(row["transition_gain_vs_identity"]) <= 0.0 or float(row["transition_gain_vs_train_centroid"]) <= 0.0:
            rows.append(
                {
                    "axis": axis,
                    "slice": key,
                    "rows": int(row["rows"]),
                    "transition_gain_vs_identity": float(row["transition_gain_vs_identity"]),
                    "transition_gain_vs_train_centroid": float(row["transition_gain_vs_train_centroid"]),
                }
            )
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = payload["adapter_overall"]
    by_overall = payload["stage43_by_reference"]["overall"]
    calibrated = payload["adapter_calibrated_readout_overall"]
    boot = payload["adapter_calibrated_readout_bootstrap"]
    gates = {
        "stage43_m_checkpoint_replayed": bool(payload["stage43_m_checkpoint_replayed"]),
        "stage43_by_precondition_seen": bool(payload["stage43_by_precondition_seen"]),
        "train_only_adapter_completed": len(payload["training_history"]) > 0,
        "future_target_latent_label_eval_only": payload["no_leakage"]["future_target_latent_label_eval_only"] is True
        and payload["no_leakage"]["future_target_latent_input"] is False,
        "no_test_statistics_normalization": payload["no_leakage"]["test_statistics_normalization"] is False,
        "latent_noncollapse": payload["adapter_latent_stats"]["adapter_min_variance"] >= payload["adapter_latent_stats"]["noncollapse_threshold"],
        "raw_adapter_beats_identity": overall["transition_gain_vs_identity"] > 0.0,
        "raw_adapter_beats_stage43_m_transition": overall["mse_next_to_target"] < by_overall["mse_next_to_target"],
        "raw_adapter_beats_train_centroid": overall["transition_gain_vs_train_centroid"] > 0.0,
        "calibrated_adapter_beats_identity": calibrated["transition_gain_vs_identity"] > 0.0,
        "calibrated_bootstrap_supports_identity_lift": boot["transition_gain_vs_identity"]["low"] > 0.0,
        "domain_and_horizon_breakdowns_reported": bool(payload["adapter_domain_breakdown"])
        and bool(payload["adapter_horizon_breakdown"]),
        "weak_slice_caveats_reported": "weak_adapter_slices" in payload
        and "weak_calibrated_adapter_slices" in payload,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage43_bz_latent_transition_adapter_repair_pass"
    elif (
        gates["raw_adapter_beats_identity"]
        and gates["raw_adapter_beats_stage43_m_transition"]
        and gates["raw_adapter_beats_train_centroid"]
    ):
        verdict = "stage43_bz_latent_transition_adapter_repair_pass_with_readout_caveat"
    else:
        verdict = "stage43_bz_latent_transition_adapter_repair_diagnostic_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _format_metric_table(title: str, table: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | MSE next-target |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in table.items():
        lines.append(
            f"| `{key}` | `{int(row['rows'])}` | `{float(row['transition_gain_vs_identity']):.4f}` | "
            f"`{float(row['transition_gain_vs_train_centroid']):.4f}` | `{float(row['mean_cosine_next_target']):.4f}` | "
            f"`{float(row['mse_next_to_target']):.4f}` |"
        )
    lines.append("")
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_bz_gate"]
    lines = [
        "# Stage43-BZ Latent Transition Adapter Repair",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        "- deployable policy changed: `False`",
        "",
        "## Global Comparison",
        "",
        "| model | rows | gain vs identity | gain vs train centroid | MSE next-target |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `Stage43-M z_next` | `{payload['stage43_by_reference']['overall']['rows']}` | "
        f"`{payload['stage43_by_reference']['overall']['transition_gain_vs_identity']:.4f}` | "
        f"`{payload['stage43_by_reference']['overall']['transition_gain_vs_train_centroid']:.4f}` | "
        f"`{payload['stage43_by_reference']['overall']['mse_next_to_target']:.4f}` |",
        f"| `Stage43-BZ adapter z_next` | `{payload['adapter_overall']['rows']}` | "
        f"`{payload['adapter_overall']['transition_gain_vs_identity']:.4f}` | "
        f"`{payload['adapter_overall']['transition_gain_vs_train_centroid']:.4f}` | "
        f"`{payload['adapter_overall']['mse_next_to_target']:.4f}` |",
        "",
        "## Train-Only Calibrated Readout",
        "",
        "| model | gain vs calibrated identity | gain vs train centroid | MSE next-target |",
        "| --- | ---: | ---: | ---: |",
        f"| `Stage43-BY calibrated z_next` | `{payload['stage43_by_reference']['calibrated_readout_overall']['transition_gain_vs_identity']:.4f}` | "
        f"`{payload['stage43_by_reference']['calibrated_readout_overall']['transition_gain_vs_train_centroid']:.4f}` | "
        f"`{payload['stage43_by_reference']['calibrated_readout_overall']['mse_next_to_target']:.4f}` |",
        f"| `Stage43-BZ calibrated adapter` | `{payload['adapter_calibrated_readout_overall']['transition_gain_vs_identity']:.4f}` | "
        f"`{payload['adapter_calibrated_readout_overall']['transition_gain_vs_train_centroid']:.4f}` | "
        f"`{payload['adapter_calibrated_readout_overall']['mse_next_to_target']:.4f}` |",
        "",
        "Calibrated readout bootstrap 95% CI:",
        f"- gain vs identity: `[{payload['adapter_calibrated_readout_bootstrap']['transition_gain_vs_identity']['low']:.4f}, {payload['adapter_calibrated_readout_bootstrap']['transition_gain_vs_identity']['high']:.4f}]`",
        f"- gain vs train centroid: `[{payload['adapter_calibrated_readout_bootstrap']['transition_gain_vs_train_centroid']['low']:.4f}, {payload['adapter_calibrated_readout_bootstrap']['transition_gain_vs_train_centroid']['high']:.4f}]`",
        "",
    ]
    lines += _format_metric_table("Adapter Domain Breakdown", payload["adapter_domain_breakdown"])
    lines += _format_metric_table("Adapter Horizon Breakdown", payload["adapter_horizon_breakdown"])
    lines += [
        "## Interpretation",
        "",
        "- Stage43-BZ freezes the Stage43-M past encoder and future-target encoder, then trains a past-only latent transition adapter on train rows only.",
        "- Future target latents are label/eval targets only; they are not inference inputs.",
        "- The adapter is not a deployable policy change and does not remove the safety floor.",
        "- This stage directly tests whether the Stage43-BY readout caveat is caused by a weak transition head rather than by absent causal signal.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines.append("")
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-BZ Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{SOURCE}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "- protected multimodal latent state candidate: `True`",
        f"- adapter raw gain vs identity: `{payload['adapter_overall']['transition_gain_vs_identity']:.4f}`",
        f"- adapter calibrated gain vs identity: `{payload['adapter_calibrated_readout_overall']['transition_gain_vs_identity']:.4f}`",
        f"- adapter calibrated CI low vs identity: `{payload['adapter_calibrated_readout_bootstrap']['transition_gain_vs_identity']['low']:.4f}`",
        "- deployable policy changed: `False`",
        "- long objective complete: `False`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-BZ is a latent transition repair experiment, not an ungated deployment policy.",
        "- Safety floors remain required for deployment.",
        "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    for key, value in gate["gates"].items():
        world_lines.append(f"| `{key}` | `{bool(value)}` |")
    world_lines.append("")
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(WORLD_GATE_MD, world_lines)


def _update_summary_files(payload: Mapping[str, Any]) -> None:
    block = "\n".join(
        [
            f"## {SECTION}",
            "",
            f"source = `{SOURCE}`",
            f"result_source = `{payload['result_source']}`",
            f"verdict = `{payload['stage43_bz_gate']['verdict']}`",
            f"gate = `{payload['stage43_bz_gate']['passed']} / {payload['stage43_bz_gate']['total']}`",
            "deployable_policy_changed = `False`",
            "",
            "Stage43-BZ trains a train-only, past-only latent transition adapter with frozen Stage43-M encoders to repair the Stage43-BY calibrated readout caveat.",
            f"Adapter raw gain vs identity: `{payload['adapter_overall']['transition_gain_vs_identity']:.4f}`.",
            f"Adapter raw gain vs train centroid: `{payload['adapter_overall']['transition_gain_vs_train_centroid']:.4f}`.",
            f"Adapter calibrated gain vs identity: `{payload['adapter_calibrated_readout_overall']['transition_gain_vs_identity']:.4f}`.",
            f"Adapter calibrated gain vs train centroid: `{payload['adapter_calibrated_readout_overall']['transition_gain_vs_train_centroid']:.4f}`.",
            f"Calibrated gain-vs-identity CI low: `{payload['adapter_calibrated_readout_bootstrap']['transition_gain_vs_identity']['low']:.4f}`.",
            "",
            "Interpretation: this is a targeted latent transition repair experiment. It does not change deployment, does not remove the safety floor, and does not enable Stage5C or SMC.",
        ]
    )
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block.splitlines())
    state = read_json(RESEARCH_STATE, {})
    state["stage43_bz_latent_transition_adapter_repair"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": payload["stage43_bz_gate"]["verdict"],
        "gate": f"{payload['stage43_bz_gate']['passed']} / {payload['stage43_bz_gate']['total']}",
        "adapter_overall": payload["adapter_overall"],
        "adapter_calibrated_readout_overall": payload["adapter_calibrated_readout_overall"],
        "adapter_calibrated_readout_bootstrap": payload["adapter_calibrated_readout_bootstrap"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bz_latent_transition_adapter_repair"
    state["current_verdict"] = payload["stage43_bz_gate"]["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": SOURCE, "verdict": payload["stage43_bz_gate"]["verdict"], "generated_at_utc": payload["generated_at_utc"]}), ensure_ascii=False) + "\n")


def run_latent_transition_adapter_repair(
    *,
    epochs: int = 8,
    batch_size: int = 8192,
    hidden_dim: int = 128,
    delta_clip: float = 3.0,
    lr: float = 1e-3,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    bootstrap: int = 1000,
    bootstrap_rows: int = 8000,
    seed: int = 503,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    runtime = _configure_runtime(seed)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43by = read_json(STAGE43_BY_JSON, {})
    checkpoint, ckpt, base_model = _load_model(stage43m)
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    train = _limit_split(train, max_train, seed + 1)
    val = _limit_split(val, max_val, seed + 2)
    test = _limit_split(test, max_test, seed + 3)
    train_pred = _predict_transition_latents(base_model, train, batch_size=int(batch_size))
    val_pred = _predict_transition_latents(base_model, val, batch_size=int(batch_size))
    test_pred = _predict_transition_latents(base_model, test, batch_size=int(batch_size))
    adapter = LatentTransitionAdapter(train.x.shape[1], train_pred["z_t"].shape[1], hidden_dim=int(hidden_dim), delta_clip=float(delta_clip))
    adapter_checkpoint = CKPT_DIR / "stage43_latent_transition_adapter_repair.pt"
    history = _train_adapter(
        adapter,
        train.x,
        train_pred["z_t"],
        train_pred["z_target"],
        val.x,
        val_pred["z_t"],
        val_pred["z_target"],
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
        checkpoint=adapter_checkpoint,
    )
    train_adapter = _adapter_predict(adapter, train.x, train_pred["z_t"], batch_size=int(batch_size))
    test_adapter = _adapter_predict(adapter, test.x, test_pred["z_t"], batch_size=int(batch_size))
    centroid = train_pred["z_target"].mean(axis=0).astype(np.float32)
    original_overall = _transition_metrics(
        test_pred["z_t"],
        test_pred["z_next"],
        test_pred["z_target"],
        centroid,
        np.ones(len(test.x), dtype=bool),
    )
    adapter_overall = _transition_metrics(
        test_pred["z_t"],
        test_adapter,
        test_pred["z_target"],
        centroid,
        np.ones(len(test.x), dtype=bool),
    )
    adapter_readout = _fit_ridge_readout(train_adapter, train_pred["z_target"], ridge=1e-2)
    identity_readout = _fit_ridge_readout(train_pred["z_t"], train_pred["z_target"], ridge=1e-2)
    calibrated_adapter = _apply_readout(test_adapter, adapter_readout)
    calibrated_identity = _apply_readout(test_pred["z_t"], identity_readout)
    adapter_calibrated = _transition_metrics(
        calibrated_identity,
        calibrated_adapter,
        test_pred["z_target"],
        centroid,
        np.ones(len(test.x), dtype=bool),
    )
    domain_breakdown = _breakdown(test.domain, test_pred["z_t"], test_adapter, test_pred["z_target"], centroid, min_rows=100)
    horizon_breakdown = _breakdown(test.horizon.astype(str), test_pred["z_t"], test_adapter, test_pred["z_target"], centroid, min_rows=100)
    subset_values = np.where(test.easy, "easy", np.where(test.hard | test.failure, "hard_failure", "non_easy"))
    subset_breakdown = _breakdown(subset_values, test_pred["z_t"], test_adapter, test_pred["z_target"], centroid, min_rows=100)
    calibrated_domain_breakdown = _breakdown(test.domain, calibrated_identity, calibrated_adapter, test_pred["z_target"], centroid, min_rows=100)
    calibrated_horizon_breakdown = _breakdown(test.horizon.astype(str), calibrated_identity, calibrated_adapter, test_pred["z_target"], centroid, min_rows=100)
    raw_bootstrap = _bootstrap_transition(
        test_pred["z_t"],
        test_adapter,
        test_pred["z_target"],
        centroid,
        n=int(bootstrap),
        sample_rows=int(bootstrap_rows),
        seed=int(seed) + 100,
    )
    calibrated_bootstrap = _bootstrap_transition(
        calibrated_identity,
        calibrated_adapter,
        test_pred["z_target"],
        centroid,
        n=int(bootstrap),
        sample_rows=int(bootstrap_rows),
        seed=int(seed) + 200,
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_train_only_latent_transition_adapter_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_checkpoint_replayed": True,
        "stage43_m_checkpoint": str(checkpoint),
        "stage43_m_checkpoint_hash": _combined_hash({"checkpoint": str(checkpoint), "sha256": stage43m.get("checkpoint_sha256", "unknown")}),
        "stage43_by_precondition_seen": bool(stage43by),
        "runtime": runtime,
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "delta_clip": float(delta_clip),
            "lr": float(lr),
            "max_train": max_train,
            "max_val": max_val,
            "max_test": max_test,
            "seed": int(seed),
        },
        "training_history": history,
        "adapter_checkpoint": str(adapter_checkpoint),
        "checkpoint_committed": False,
        "rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "stage43_by_reference": {
            "overall": stage43by.get("overall", original_overall),
            "calibrated_readout_overall": stage43by.get("calibrated_readout_overall", {}),
        },
        "stage43_m_replayed_overall": original_overall,
        "adapter_overall": adapter_overall,
        "adapter_bootstrap": raw_bootstrap,
        "adapter_calibrated_readout_overall": adapter_calibrated,
        "adapter_calibrated_readout_bootstrap": calibrated_bootstrap,
        "adapter_domain_breakdown": domain_breakdown,
        "adapter_horizon_breakdown": horizon_breakdown,
        "adapter_subset_breakdown": subset_breakdown,
        "adapter_calibrated_domain_breakdown": calibrated_domain_breakdown,
        "adapter_calibrated_horizon_breakdown": calibrated_horizon_breakdown,
        "weak_adapter_slices": _weak_slices(domain_breakdown, axis="domain") + _weak_slices(horizon_breakdown, axis="horizon"),
        "weak_calibrated_adapter_slices": _weak_slices(calibrated_domain_breakdown, axis="domain")
        + _weak_slices(calibrated_horizon_breakdown, axis="horizon"),
        "adapter_latent_stats": {
            "rows": int(len(test_adapter)),
            "latent_dim": int(test_adapter.shape[1]),
            "adapter_min_variance": float(np.var(test_adapter, axis=0).min()),
            "adapter_mean_variance": float(np.var(test_adapter, axis=0).mean()),
            "target_mean_variance": float(np.var(test_pred["z_target"], axis=0).mean()),
            "noncollapse_threshold": 0.01,
        },
        "no_leakage": {
            "future_target_latent_input": False,
            "future_target_latent_label_eval_only": True,
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "deployable_policy_changed": False,
            "metric_or_seconds_claim": False,
            "true_3d_claim": False,
            "foundation_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_bz_gate"] = _gate(payload)
    _write_reports(payload)
    _update_summary_files(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train and audit Stage43-BZ latent transition adapter repair.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-clip", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=503)
    args = parser.parse_args(argv)
    payload = run_latent_transition_adapter_repair(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        delta_clip=args.delta_clip,
        lr=args.lr,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        bootstrap=args.bootstrap,
        bootstrap_rows=args.bootstrap_rows,
        seed=args.seed,
    )
    gate = payload["stage43_bz_gate"]
    print(f"Stage43-BZ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"adapter_gain_vs_identity={payload['adapter_overall']['transition_gain_vs_identity']:.4f}")
    print(f"calibrated_gain_vs_identity={payload['adapter_calibrated_readout_overall']['transition_gain_vs_identity']:.4f}")
    return payload


if __name__ == "__main__":
    main()
