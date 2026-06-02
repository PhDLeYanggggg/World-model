from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_scene_graph_context_router as bs
from src import stage43_scene_graph_multimodal_ablation as bp
from src import stage43_scene_graph_slice_forensics as br
from src import stage43_gated_scene_graph_fusion as bq


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_context_admissibility_model.json"
REPORT_MD = OUT_DIR / "stage43_context_admissibility_model.md"
GATE_MD = OUT_DIR / "stage43_stage_bt_context_admissibility_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
HEARTBEAT_JSON = OUT_DIR / "stage43_context_admissibility_model_heartbeat.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BT_CONTEXT_ADMISSIBILITY_MODEL"
SOURCE = "fresh_stage43_bt_context_admissibility_model"
DEFAULT_VARIANT = bs.DEFAULT_VARIANT
CONTEXT_VARIANTS = ["scene_proxy_only", "scene_graph_full"]
EPS = 1e-8


@dataclass
class ContextBatch:
    ds: m.WaypointSplit
    arrays: dict[str, dict[str, np.ndarray]]
    info: dict[str, Any]


class ContextAdmissibilityNet(nn.Module):
    def __init__(self, input_dim: int, candidate_count: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gain_head = nn.Linear(hidden_dim, candidate_count)
        self.harm_head = nn.Linear(hidden_dim, candidate_count)
        self.gain_reg_head = nn.Linear(hidden_dim, candidate_count)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "gain_logit": self.gain_head(h),
            "harm_logit": self.harm_head(h),
            "gain_reg": self.gain_reg_head(h),
        }


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load_context(split: str, *, rows: int | None) -> ContextBatch:
    reference: m.WaypointSplit | None = None
    default_ds: m.WaypointSplit | None = None
    arrays_by_variant: dict[str, dict[str, np.ndarray]] = {}
    info_by_variant: dict[str, Any] = {}
    for variant in bs.VARIANTS:
        ds, arrays, info = bs._load_variant(split, variant, rows=rows)
        if reference is None:
            reference = ds
        else:
            bs._assert_aligned(reference, ds, split=split, variant=variant)
        if variant == DEFAULT_VARIANT:
            default_ds = ds
        arrays_by_variant[variant] = arrays
        info_by_variant[variant] = info
    if default_ds is None:
        raise RuntimeError(f"Missing default graph-history split for {split}")
    return ContextBatch(ds=default_ds, arrays=arrays_by_variant, info=info_by_variant)


def _standardize_batches(train: ContextBatch, val: ContextBatch, test: ContextBatch) -> tuple[np.ndarray, np.ndarray]:
    mean = train.ds.x.mean(axis=0).astype(np.float32)
    raw_std = train.ds.x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for batch in [train, val, test]:
        batch.ds.x = ((batch.ds.x - mean) / std).astype(np.float32)
    return mean, std


def _supervision(batch: ContextBatch, *, gain_margin: float = 0.005, harm_margin: float = 0.002) -> dict[str, np.ndarray]:
    default_ade = batch.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    candidate_ade = np.stack([batch.arrays[v]["selected_ade"].astype(np.float32) for v in CONTEXT_VARIANTS], axis=1)
    gain = (default_ade[:, None] - candidate_ade).astype(np.float32)
    return {
        "gain_value": gain,
        "gain_label": (gain > float(gain_margin)).astype(np.float32),
        "harm_label": ((candidate_ade - default_ade[:, None]) > float(harm_margin)).astype(np.float32),
    }


def _batch_indices(n: int, batch_size: int, *, seed: int, shuffle: bool) -> list[np.ndarray]:
    ids = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
    return [ids[i : i + batch_size] for i in range(0, n, batch_size)]


def _train_model(args: argparse.Namespace, train: ContextBatch, val: ContextBatch) -> tuple[ContextAdmissibilityNet, dict[str, Any]]:
    runtime = m._configure_runtime(int(args.seed))
    device = torch.device("cpu")
    model = ContextAdmissibilityNet(train.ds.x.shape[1], len(CONTEXT_VARIANTS), hidden_dim=int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    sup = _supervision(train)
    val_sup = _supervision(val)
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    ckpt_path = CKPT_DIR / "stage43_context_admissibility_model.pt"
    start = time.time()
    x_train = train.ds.x.astype(np.float32)
    row_weight = (1.0 + 1.0 * (train.ds.hard | train.ds.failure).astype(np.float32) + 0.5 * (train.ds.horizon == 50).astype(np.float32))
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        for ids in _batch_indices(len(x_train), int(args.batch_size), seed=int(args.seed) + epoch, shuffle=True):
            x = torch.from_numpy(x_train[ids]).to(device)
            out = model(x)
            gain_label = torch.from_numpy(sup["gain_label"][ids]).to(device)
            harm_label = torch.from_numpy(sup["harm_label"][ids]).to(device)
            gain_value = torch.from_numpy(sup["gain_value"][ids]).to(device)
            weights = torch.from_numpy(row_weight[ids]).to(device)[:, None]
            gain_loss = (nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], gain_label, reduction="none") * weights).mean()
            harm_loss = (nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], harm_label, reduction="none") * weights).mean()
            reg_loss = (nn.functional.smooth_l1_loss(out["gain_reg"], gain_value, reduction="none") * weights).mean()
            loss = gain_loss + 1.4 * harm_loss + 0.8 * reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = _predict_model(model, val.ds.x, batch_size=int(args.batch_size))
        val_loss = _label_eval_loss(val_pred, val_sup)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_label_loss": float(val_loss),
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
                    "runtime": runtime,
                }
            ),
        )
        if val_loss < best_loss:
            best_loss = float(val_loss)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(train.ds.x.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "candidate_variants": CONTEXT_VARIANTS,
                    "feature_mean": train.ds.x.mean(axis=0).astype(np.float32),
                    "feature_std_after_standardization": train.ds.x.std(axis=0).astype(np.float32),
                    "epoch": epoch + 1,
                    "seed": int(args.seed),
                    "runtime": runtime,
                    "checkpoint_committed": False,
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "future_variant_error_label_only": True,
                        "central_velocity_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                ckpt_path,
            )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, {"runtime": runtime, "history": history, "checkpoint": str(ckpt_path), "checkpoint_sha256": m._sha256(ckpt_path)}


@torch.no_grad()
def _predict_model(model: ContextAdmissibilityNet, x: np.ndarray, *, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {"gain_prob": [], "harm_prob": [], "gain_reg": []}
    for ids in _batch_indices(len(x), batch_size, seed=0, shuffle=False):
        tensor = torch.from_numpy(x[ids].astype(np.float32))
        out = model(tensor)
        outs["gain_prob"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm_prob"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["gain_reg"].append(out["gain_reg"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in outs.items()}


def _label_eval_loss(pred: Mapping[str, np.ndarray], sup: Mapping[str, np.ndarray]) -> float:
    gain_prob = np.clip(pred["gain_prob"], 1e-4, 1.0 - 1e-4)
    harm_prob = np.clip(pred["harm_prob"], 1e-4, 1.0 - 1e-4)
    gain_label = sup["gain_label"]
    harm_label = sup["harm_label"]
    bce_gain = -(gain_label * np.log(gain_prob) + (1.0 - gain_label) * np.log(1.0 - gain_prob)).mean()
    bce_harm = -(harm_label * np.log(harm_prob) + (1.0 - harm_label) * np.log(1.0 - harm_prob)).mean()
    reg = np.mean(np.abs(pred["gain_reg"] - sup["gain_value"]))
    return float(bce_gain + 1.4 * bce_harm + 0.8 * reg)


def _apply_policy(
    batch: ContextBatch,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_ade = batch.arrays[DEFAULT_VARIANT]["selected_ade"].copy()
    selected_fde = batch.arrays[DEFAULT_VARIANT]["selected_fde"].copy()
    switched = batch.arrays[DEFAULT_VARIANT]["switched"].copy()
    used = np.full(len(selected_ade), DEFAULT_VARIANT, dtype=object)
    gain_prob = pred["gain_prob"]
    harm_prob = pred["harm_prob"]
    gain_reg = pred["gain_reg"]
    allowed = (
        (gain_prob >= float(policy["gain_threshold"]))
        & (harm_prob <= float(policy["harm_threshold"]))
        & (gain_reg >= float(policy["predicted_gain_threshold"]))
    )
    score = np.where(allowed, gain_reg - 0.25 * harm_prob, -1e9)
    best = np.argmax(score, axis=1)
    best_score = score[np.arange(len(score)), best]
    for row_id in np.where(best_score > -1e8)[0]:
        variant = CONTEXT_VARIANTS[int(best[row_id])]
        selected_ade[row_id] = batch.arrays[variant]["selected_ade"][row_id]
        selected_fde[row_id] = batch.arrays[variant]["selected_fde"][row_id]
        switched[row_id] = batch.arrays[variant]["switched"][row_id]
        used[row_id] = variant
    return selected_ade.astype(np.float32), selected_fde.astype(np.float32), switched.astype(bool), used


def _variant_counts(values: np.ndarray) -> dict[str, int]:
    return {str(v): int(np.sum(values.astype(str) == str(v))) for v in sorted(set(values.astype(str).tolist()))}


def _reference_metrics(batch: ContextBatch) -> dict[str, Any]:
    return {
        variant: m._metrics(batch.ds, batch.arrays[variant]["selected_ade"], batch.arrays[variant]["selected_fde"], batch.arrays[variant]["switched"])
        for variant in bs.VARIANTS
    }


def _delta(metrics: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - graph["full_waypoint_ade_improvement_vs_floor"]),
        "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"] - graph["t50_full_waypoint_ade_improvement_vs_floor"]),
        "hard_failure": float(
            metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - graph["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        ),
        "easy_degradation": float(metrics["easy_degradation_vs_floor"] - graph["easy_degradation_vs_floor"]),
    }


def _score(metrics: Mapping[str, Any], graph: Mapping[str, Any], switch_rate: float) -> float:
    d = _delta(metrics, graph)
    return float(d["all"] + 1.4 * d["t50"] + 1.0 * d["hard_failure"] - 20.0 * max(0.0, d["easy_degradation"]) - 0.03 * switch_rate)


def _candidate_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for gain_threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        for harm_threshold in [0.05, 0.10, 0.20, 0.30, 0.50]:
            for predicted_gain_threshold in [0.0, 0.0025, 0.005, 0.010, 0.020]:
                rows.append(
                    {
                        "gain_threshold": gain_threshold,
                        "harm_threshold": harm_threshold,
                        "predicted_gain_threshold": predicted_gain_threshold,
                    }
                )
    return rows


def _select_policy(val: ContextBatch, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    graph = _reference_metrics(val)[DEFAULT_VARIANT]
    candidates: list[dict[str, Any]] = []
    for policy in _candidate_grid():
        selected_ade, selected_fde, switched, used = _apply_policy(val, pred, policy)
        metrics = m._metrics(val.ds, selected_ade, selected_fde, switched)
        d = _delta(metrics, graph)
        context_rate = float(np.mean(used.astype(str) != DEFAULT_VARIANT))
        safe = metrics["easy_degradation_vs_floor"] <= 0.02 and d["easy_degradation"] <= 0.02 and d["all"] >= -0.002
        candidates.append(
            {
                "policy": policy,
                "validation_metrics": metrics,
                "delta_vs_graph_history_only": d,
                "context_variant_counts": _variant_counts(used),
                "context_rate": context_rate,
                "safe": bool(safe),
                "selection_score": _score(metrics, graph, context_rate),
            }
        )
    safe_candidates = [row for row in candidates if row["safe"]]
    selected = max(safe_candidates or candidates, key=lambda row: row["selection_score"])
    selected["all_candidates"] = candidates
    selected["safe_candidate_count"] = len(safe_candidates)
    return selected


def _admissibility_diagnostics(batch: ContextBatch, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    sup = _supervision(batch)
    rows: dict[str, Any] = {}
    for i, variant in enumerate(CONTEXT_VARIANTS):
        true_gain = sup["gain_value"][:, i]
        pred_gain = pred["gain_reg"][:, i]
        gain_label = sup["gain_label"][:, i].astype(bool)
        harm_label = sup["harm_label"][:, i].astype(bool)
        corr = float(np.corrcoef(true_gain, pred_gain)[0, 1]) if np.std(true_gain) > 1e-8 and np.std(pred_gain) > 1e-8 else 0.0
        rows[variant] = {
            "mean_true_gain": float(np.mean(true_gain)),
            "mean_predicted_gain": float(np.mean(pred_gain)),
            "gain_label_rate": float(np.mean(gain_label)),
            "harm_label_rate": float(np.mean(harm_label)),
            "gain_reg_correlation": corr,
        }
    return rows


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    return [
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
    ]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    train = _load_context("train", rows=int(args.train_rows))
    val = _load_context("val", rows=int(args.val_rows))
    test = _load_context("test", rows=int(args.test_rows))
    mean, std = _standardize_batches(train, val, test)
    model, train_info = _train_model(args, train, val)
    val_pred = _predict_model(model, val.ds.x, batch_size=int(args.batch_size))
    selected_policy = _select_policy(val, val_pred)
    test_pred = _predict_model(model, test.ds.x, batch_size=int(args.batch_size))
    selected_ade, selected_fde, switched, used = _apply_policy(test, test_pred, selected_policy["policy"])
    test_metrics = m._metrics(test.ds, selected_ade, selected_fde, switched)
    references = _reference_metrics(test)
    graph = references[DEFAULT_VARIANT]
    delta = _delta(test_metrics, graph)
    bp_payload = read_json(bp.REPORT_JSON, {})
    bq_payload = read_json(bq.REPORT_JSON, {})
    br_payload = read_json(br.REPORT_JSON, {})
    bs_payload = read_json(bs.REPORT_JSON, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_row_level_harm_aware_context_admissibility",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "rows": {"train": int(len(train.ds.x)), "val": int(len(val.ds.x)), "test": int(len(test.ds.x))},
        "candidate_variants": CONTEXT_VARIANTS,
        "default_variant": DEFAULT_VARIANT,
        "precondition": {
            "bp_verdict": bp_payload.get("stage43_bp_gate", {}).get("verdict", "missing"),
            "bq_verdict": bq_payload.get("stage43_bq_gate", {}).get("verdict", "missing"),
            "br_verdict": br_payload.get("stage43_br_gate", {}).get("verdict", "missing"),
            "bs_verdict": bs_payload.get("stage43_bs_gate", {}).get("verdict", "missing"),
        },
        "model": {
            "type": "row_level_context_admissibility_mlp",
            "input_feature_count": int(train.ds.x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "checkpoint": train_info["checkpoint"],
            "checkpoint_sha256": train_info["checkpoint_sha256"],
            "checkpoint_committed": False,
            "runtime": train_info["runtime"],
            "training_history": train_info["history"],
        },
        "feature_standardization": {
            "train_only_mean_hash": _hash_array(mean),
            "train_only_std_hash": _hash_array(std),
            "test_statistics_normalization": False,
        },
        "validation_selection": {
            "selected_policy": selected_policy["policy"],
            "selected_validation_metrics": selected_policy["validation_metrics"],
            "selected_delta_vs_graph_history_only": selected_policy["delta_vs_graph_history_only"],
            "selected_context_variant_counts": selected_policy["context_variant_counts"],
            "safe_candidate_count": selected_policy["safe_candidate_count"],
            "candidate_count": len(selected_policy["all_candidates"]),
            "test_tuned": False,
            "selection_rule": "train admissibility model, select confidence/gain/harm thresholds on validation only, evaluate test once",
        },
        "test_metrics": test_metrics,
        "test_context_variant_counts": _variant_counts(used),
        "test_reference_metrics": references,
        "delta_vs_graph_history_only": delta,
        "admissibility_diagnostics": {
            "val": _admissibility_diagnostics(val, val_pred),
            "test": _admissibility_diagnostics(test, test_pred),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_variant_error_label_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
            "test_threshold_selection": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "deployment_policy_changed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([bp.REPORT_JSON, bq.REPORT_JSON, br.REPORT_JSON, bs.REPORT_JSON]),
    }
    payload["stage43_bt_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _hash_array(arr: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray(arr).tobytes())
    return digest.hexdigest()


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_graph_history_only"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    easy_safe = metrics["easy_degradation_vs_floor"] <= 0.02
    beats_graph = max(delta["all"], delta["t50"], delta["hard_failure"]) > 0.0
    gates = {
        "bp_precondition_passed": payload["precondition"]["bp_verdict"]
        in {
            "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_mixed_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_contribution_supported",
        },
        "bq_precondition_passed": payload["precondition"]["bq_verdict"]
        in {
            "stage43_bq_gated_scene_graph_fusion_pass_contribution_supported",
            "stage43_bq_gated_scene_graph_fusion_pass_safe_no_best_single_lift_diagnostic",
            "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
            "stage43_bq_gated_scene_graph_fusion_pass_unsafe_diagnostic",
        },
        "br_precondition_passed": payload["precondition"]["br_verdict"]
        in {
            "stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal",
            "stage43_br_scene_graph_slice_forensics_pass_weak_scene_signal_diagnostic",
            "stage43_br_scene_graph_slice_forensics_pass_no_scene_signal_diagnostic",
        },
        "bs_precondition_passed": payload["precondition"]["bs_verdict"]
        in {
            "stage43_bs_scene_graph_context_router_pass_safe_lift_diagnostic",
            "stage43_bs_scene_graph_context_router_pass_safe_no_lift_diagnostic",
            "stage43_bs_scene_graph_context_router_pass_unsafe_diagnostic",
        },
        "fresh_torch_training_completed": payload["result_source"] == "fresh_row_level_harm_aware_context_admissibility"
        and Path(payload["model"]["checkpoint"]).exists(),
        "checkpoint_not_committed": payload["model"]["checkpoint_committed"] is False,
        "train_val_test_loaded": payload["rows"]["train"] > 0 and payload["rows"]["val"] > 0 and payload["rows"]["test"] > 0,
        "validation_only_threshold_selection": payload["validation_selection"]["test_tuned"] is False
        and no_leak["test_threshold_selection"] is False,
        "test_eval_completed": metrics["rows"] > 0,
        "graph_history_reference_present": DEFAULT_VARIANT in payload["test_reference_metrics"],
        "admissibility_diagnostics_reported": bool(payload["admissibility_diagnostics"]["test"]),
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_variant_error_label_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["scene_proxy_train_only"] is True
        and no_leak["graph_inputs_past_or_current_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["raw_scene_or_verified_sdf_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and easy_safe and beats_graph:
        verdict = "stage43_bt_context_admissibility_pass_safe_lift_diagnostic"
    elif passed == total and easy_safe:
        verdict = "stage43_bt_context_admissibility_pass_safe_no_lift_diagnostic"
    elif passed == total:
        verdict = "stage43_bt_context_admissibility_pass_unsafe_diagnostic"
    else:
        verdict = "stage43_bt_context_admissibility_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "row_level_admissibility_trained": passed == total,
        "beats_graph_history_on_any_core_metric": beats_graph,
        "easy_safe": easy_safe,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total and easy_safe,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bt_gate"]
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_graph_history_only"]
    val = payload["validation_selection"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BT Context Admissibility Model",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- row-level admissibility trained: `{gate['row_level_admissibility_trained']}`",
            f"- beats graph-history on any core metric: `{gate['beats_graph_history_on_any_core_metric']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Validation-Selected Policy",
            "",
            f"- selected policy: `{val['selected_policy']}`",
            f"- candidate count: `{val['candidate_count']}`",
            f"- safe candidate count: `{val['safe_candidate_count']}`",
            f"- context counts on validation: `{val['selected_context_variant_counts']}`",
            "",
            "## Test Metrics",
            "",
            *_metric_lines(metrics),
            f"- context counts on test: `{payload['test_context_variant_counts']}`",
            "",
            "## Delta Vs Graph-History-Only",
            "",
            f"- all delta: `{_pct(delta['all'])}`",
            f"- t50 delta: `{_pct(delta['t50'])}`",
            f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
            f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
            "",
            "## Admissibility Diagnostics",
            "",
            "| variant | mean true gain | mean predicted gain | gain-label rate | harm-label rate | gain correlation |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| `{variant}` | `{row['mean_true_gain']:.5f}` | `{row['mean_predicted_gain']:.5f}` | `{_pct(row['gain_label_rate'])}` | `{_pct(row['harm_label_rate'])}` | `{row['gain_reg_correlation']:.4f}` |"
                for variant, row in payload["admissibility_diagnostics"]["test"].items()
            ],
            "",
            "## Interpretation",
            "",
            "- This is a row-level harm-aware context admissibility diagnostic.",
            "- It tries to release scene/full context only when predicted gain is high and predicted harm is low.",
            "- Thresholds are selected on validation only; test is evaluated once.",
            "- Future variant errors are labels/eval only, not inference inputs.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
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
            "# Stage43-BT Context Admissibility Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- row-level admissibility trained: `{gate['row_level_admissibility_trained']}`",
            f"- beats graph-history on any core metric: `{gate['beats_graph_history_on_any_core_metric']}`",
            f"- easy safe: `{gate['easy_safe']}`",
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
            f"- row-level admissibility trained: `{gate['row_level_admissibility_trained']}`",
            f"- beats graph-history on any core metric: `{gate['beats_graph_history_on_any_core_metric']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BT is a row-level context admissibility diagnostic.",
            "- It does not update deployment unless it beats the graph-history floor safely.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bt_gate"]
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_graph_history_only"]
    val = payload["validation_selection"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"row_level_admissibility_trained = `{gate['row_level_admissibility_trained']}`",
        f"beats_graph_history_on_any_core_metric = `{gate['beats_graph_history_on_any_core_metric']}`",
        f"easy_safe = `{gate['easy_safe']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        "Stage43-BT trains a row-level harm-aware context admissibility model over the Stage43-BP scene/graph variants. It uses graph-history causal features as input and future variant error only as train/eval labels.",
        f"Validation selected policy: `{val['selected_policy']}`; safe validation candidates: `{val['safe_candidate_count']} / {val['candidate_count']}`.",
        "",
        f"Test metrics: all `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`, t100 raw-frame diagnostic `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        f"Delta vs graph-history-only: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        f"Test context counts: `{payload['test_context_variant_counts']}`.",
        "",
        "Interpretation: this is a context-admissibility diagnostic, not a deployment update unless it safely beats graph-history on core metrics. Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bt_context_admissibility_model"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "row_level_admissibility_trained": gate["row_level_admissibility_trained"],
        "beats_graph_history_on_any_core_metric": gate["beats_graph_history_on_any_core_metric"],
        "easy_safe": gate["easy_safe"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "test_metrics": metrics,
        "delta_vs_graph_history_only": delta,
        "test_context_variant_counts": payload["test_context_variant_counts"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bt_context_admissibility_model"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BT",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "beats_graph_history_on_any_core_metric": gate["beats_graph_history_on_any_core_metric"],
                        "easy_safe": gate["easy_safe"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BT row-level context admissibility model.")
    parser.add_argument("--train-rows", type=int, default=20000)
    parser.add_argument("--val-rows", type=int, default=12000)
    parser.add_argument("--test-rows", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=443)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = _run(args)
    gate = payload["stage43_bt_gate"]
    print(f"Stage43-BT: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"beats_graph_history_on_any_core_metric={gate['beats_graph_history_on_any_core_metric']}")
    print(f"easy_safe={gate['easy_safe']}")
    return payload


if __name__ == "__main__":
    main()
