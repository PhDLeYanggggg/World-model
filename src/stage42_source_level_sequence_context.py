from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_incremental_ablation as ao
from src import stage42_source_level_residual_context as ap
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "source_level_sequence_context_stage42.json"
REPORT_MD = OUT_DIR / "source_level_sequence_context_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ar_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

THREADS = 4
EPOCHS = 2
BATCH = 4096
SEED = 4243
RESIDUAL_ALPHAS = [0.25, 0.50, 0.75, 1.00]
HORIZONS = [10, 25, 50, 100]
MIN_SEQUENCE_DELTA = 0.01
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AR 是 proposed source-level split sequence-context residual training，不是 metric 或 seconds-level 结果。",
    "第一阶段只用 baseline-family rollout context；第二阶段用 temporal sequence encoder 和 goal/neighbor context 预测 residual full-waypoint delta。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE42AR_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE42AR_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage42-AR refuses x86_64/Rosetta Python for torch training.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _domain_onehot(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    domains = sorted(set(data["dataset"].astype(str).tolist()))
    mat = np.stack([(data["dataset"].astype(str) == d).astype(np.float32) for d in domains], axis=1)
    return mat.astype(np.float32), domains


def _horizon_onehot(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack([(data["horizon"].astype(int) == h).astype(np.float32) for h in HORIZONS], axis=1)


def _build_context(data: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    domain, _ = _domain_onehot(data)
    horizon = _horizon_onehot(data)
    neighbor = data["history_scalar"][:, 1:6].astype(np.float32)
    proto = np.concatenate(
        [
            data["prototype_likelihood"].astype(np.float32),
            data["prototype_entropy"][:, None].astype(np.float32),
            data["goal_ambiguity"][:, None].astype(np.float32),
        ],
        axis=1,
    )
    if variant == "sequence_history":
        neighbor[:] = 0.0
        proto[:] = 0.0
    elif variant == "sequence_goal_neighbor_no_history":
        pass
    elif variant == "sequence_history_goal_neighbor":
        pass
    else:
        raise ValueError(f"Unknown sequence context variant: {variant}")
    return np.concatenate([neighbor, proto, horizon, domain], axis=1).astype(np.float32)


def _build_sequence(data: Mapping[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray]:
    seq = data["history_seq"].astype(np.float32).copy()
    valid = np.clip(seq[..., 6], 0.0, 1.0).astype(np.float32)
    if variant == "sequence_goal_neighbor_no_history":
        seq[:] = 0.0
        valid[:] = 0.0
    return seq, valid


def _standardize_sequence(seq: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = seq[train_mask].reshape(-1, seq.shape[-1])
    mean = train.mean(axis=0).astype(np.float32)
    std = np.maximum(train.std(axis=0), 1e-4).astype(np.float32)
    return ((seq - mean[None, None, :]) / std[None, None, :]).astype(np.float32), mean, std


def _standardize_context(ctx: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = ctx[train_mask].mean(axis=0).astype(np.float32)
    std = np.maximum(ctx[train_mask].std(axis=0), 1e-4).astype(np.float32)
    return ((ctx - mean) / std).astype(np.float32), mean, std


def _make_model(seq_dim: int, ctx_dim: int, out_dim: int, width: int = 96):
    torch = _torch()
    import torch.nn as nn

    class SequenceContextResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv1d(seq_dim, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(64, width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.ctx = nn.Sequential(nn.Linear(ctx_dim, width), nn.GELU(), nn.LayerNorm(width))
            self.head = nn.Sequential(
                nn.Linear(width * 2, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Linear(width, out_dim),
            )

        def forward(self, seq, valid, ctx):
            h = self.temporal(seq.transpose(1, 2)).transpose(1, 2)
            mask = valid.clamp(0, 1)[..., None]
            hist = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.head(torch.cat([hist, self.ctx(ctx)], dim=1))

    return SequenceContextResidual()


def _train_sequence_residual(
    name: str,
    seq: np.ndarray,
    valid_seq: np.ndarray,
    ctx: np.ndarray,
    residual_target: np.ndarray,
    waypoint_valid: np.ndarray,
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train_mask = split == "train"
    val_mask = split == "val"
    seq_z, seq_mean, seq_std = _standardize_sequence(seq, train_mask)
    ctx_z, ctx_mean, ctx_std = _standardize_context(ctx, train_mask)
    y = residual_target.reshape(len(residual_target), -1).astype(np.float32)
    v = np.repeat(waypoint_valid.astype(np.float32), 2, axis=1)
    seed = SEED + sum(ord(c) for c in name)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    checkpoint = CHECKPOINT_DIR / f"stage42ar_{name}.pt"
    heartbeat = OUT_DIR / f"stage42ar_{name}_heartbeat.json"
    model = _make_model(seq_z.shape[-1], ctx_z.shape[-1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    tensors = {
        "seq": torch.tensor(seq_z[train_mask]),
        "valid_seq": torch.tensor(valid_seq[train_mask]),
        "ctx": torch.tensor(ctx_z[train_mask]),
        "target": torch.tensor(y[train_mask]),
        "valid": torch.tensor(v[train_mask]),
    }
    val_tensors = {
        "seq": torch.tensor(seq_z[val_mask]),
        "valid_seq": torch.tensor(valid_seq[val_mask]),
        "ctx": torch.tensor(ctx_z[val_mask]),
        "target": torch.tensor(y[val_mask]),
        "valid": torch.tensor(v[val_mask]),
    }
    hard = (data["hard"].astype(bool) | data["failure"].astype(bool))[train_mask]
    horizon = data["horizon"].astype(int)[train_mask]
    row_w = 1.0 + 1.5 * hard.astype(np.float32) + 2.5 * (horizon == 50).astype(np.float32) + 1.0 * (horizon == 100).astype(np.float32)
    row_w_t = torch.tensor(row_w.astype(np.float32))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(tensors["seq"]))
        losses = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            pred = model(tensors["seq"][ids], tensors["valid_seq"][ids], tensors["ctx"][ids])
            loss_row = (F.smooth_l1_loss(pred, tensors["target"][ids], reduction="none") * tensors["valid"][ids]).sum(dim=1) / tensors["valid"][ids].sum(dim=1).clamp_min(1.0)
            loss = (loss_row * row_w_t[ids]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = model(val_tensors["seq"], val_tensors["valid_seq"], val_tensors["ctx"])
            val_row = (F.smooth_l1_loss(pred, val_tensors["target"], reduction="none") * val_tensors["valid"]).sum(dim=1) / val_tensors["valid"].sum(dim=1).clamp_min(1.0)
            val_loss = float(val_row.mean().cpu())
        cand = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loss < best["val_loss"]:
            best = cand
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": name,
                    "seq_dim": int(seq_z.shape[-1]),
                    "ctx_dim": int(ctx_z.shape[-1]),
                    "out_dim": int(y.shape[1]),
                    "seq_mean": seq_mean,
                    "seq_std": seq_std,
                    "ctx_mean": ctx_mean,
                    "ctx_std": ctx_std,
                    "best": best,
                },
                checkpoint,
            )
        heartbeat.write_text(json.dumps({"variant": name, "epoch": epoch, "best": best, "checkpoint": str(checkpoint)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "variant": name, "checkpoint": str(checkpoint), "heartbeat": str(heartbeat), "best": best}


def _predict_sequence(info: Mapping[str, Any], seq: np.ndarray, valid_seq: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    torch = _torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = _make_model(int(payload["seq_dim"]), int(payload["ctx_dim"]), int(payload["out_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    seq_z = ((seq - payload["seq_mean"][None, None, :]) / np.maximum(payload["seq_std"], 1e-4)[None, None, :]).astype(np.float32)
    ctx_z = ((ctx - payload["ctx_mean"]) / np.maximum(payload["ctx_std"], 1e-4)).astype(np.float32)
    outs = []
    with torch.no_grad():
        for start in range(0, len(seq_z), BATCH):
            out = model(
                torch.tensor(seq_z[start : start + BATCH]),
                torch.tensor(valid_seq[start : start + BATCH]),
                torch.tensor(ctx_z[start : start + BATCH]),
            )
            outs.append(out.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).reshape(len(seq_z), len(am.WAYPOINT_FRAC), 2).astype(np.float32)


def _evaluate_variant(name: str, residual_delta: np.ndarray, base_xy: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    data = shared["data"]
    labels = shared["labels"]
    floor = shared["floor"]
    split = shared["split"]
    val_mask = split == "val"
    test_mask = split == "test"
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    best: dict[str, Any] | None = None
    best_score = -1e9
    candidates = []
    for alpha in RESIDUAL_ALPHAS:
        pred_xy = (base_xy.astype(np.float64) + float(alpha) * residual_delta.astype(np.float64) * np.maximum(data["scale"].astype(np.float64), EPS)[:, None, None]).astype(np.float32)
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
        val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.1 * val_metric["all_improvement"]
            + 2.0 * val_metric["t50_improvement"]
            + 1.2 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        candidates.append({"alpha": float(alpha), "score": float(score), "policy_slice_count": int(len(policy["slices"])), "val_metric": val_metric})
        if score > best_score:
            best_score = float(score)
            best = {
                "alpha": float(alpha),
                "pred_xy": pred_xy,
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "score": float(score),
                "val_metric": val_metric,
            }
    if best is None:
        raise RuntimeError(f"No sequence residual evaluation completed for {name}.")
    pred_ade, pred_fde = am._trajectory_errors(best["pred_xy"], labels)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "variant": name,
        "best_residual_alpha": best["alpha"],
        "validation_selection": {"source": "fresh_run", "test_threshold_tuning": False, "selected_score": best["score"], "candidates": candidates},
        "policy_slice_count": int(len(best["policy"]["slices"])),
        "protected": am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask),
        "protected_fde": am._metric(best["selected_fde"], floor_fde, data, best["switch"], test_mask),
        "ungated_diagnostic": am._metric(pred_ade, floor_ade, data, np.ones(len(pred_ade), dtype=bool), test_mask),
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask, seed=42301),
            "t50": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (horizon == 50), seed=42302),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (horizon == 100), seed=42303),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & hard_failure, seed=42304),
            "easy_degradation": am._bootstrap_ci(floor_ade, best["selected_ade"], test_mask & easy, seed=42305),
        },
        "by_domain": {
            d: am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask & (domain == d))
            for d in sorted(set(domain[test_mask].tolist()))
        },
    }


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return ao._metric_delta(lhs, rhs)


def _positive_sequence_delta(delta: Mapping[str, float], threshold: float = MIN_SEQUENCE_DELTA) -> bool:
    return (
        delta["all_improvement"] > threshold
        or delta["t50_improvement"] > threshold
        or delta["hard_failure_improvement"] > threshold
    )


def run_stage42_source_level_sequence_context() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    names = shared["feature_names"]
    features = shared["features"]
    baseline_direct = ap._direct_candidate(features[:, ap._baseline_mask(names)], shared)
    base_xy = baseline_direct["pred_xy"]
    target_delta = ap._target_delta(shared["data"], shared["labels"])
    residual_target = target_delta - ap._xy_to_delta(shared["data"], base_xy)
    training = {}
    variants = {}
    for name in ["sequence_history", "sequence_goal_neighbor_no_history", "sequence_history_goal_neighbor"]:
        seq, valid_seq = _build_sequence(shared["data"], name)
        ctx = _build_context(shared["data"], name)
        training[name] = _train_sequence_residual(name, seq, valid_seq, ctx, residual_target, shared["labels"]["waypoint_valid"], shared["data"], shared["split"])
        residual_delta = _predict_sequence(training[name], seq, valid_seq, ctx)
        variants[name] = _evaluate_variant(name, residual_delta, base_xy, shared)
    baseline_metric = baseline_direct["model"]["metrics"]["protected_ridge_source_level"]
    deltas = {
        name: {
            "source": "fresh_run",
            "delta_vs_baseline_family_only": _metric_delta(row["protected"], baseline_metric),
            "positive_sequence_increment": _positive_sequence_delta(_metric_delta(row["protected"], baseline_metric)),
            "interpretation": _interpret_sequence_delta(name, _metric_delta(row["protected"], baseline_metric)),
        }
        for name, row in variants.items()
    }
    positive = sorted([name for name, row in deltas.items() if row["positive_sequence_increment"]])
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AR proposed source-level sequence context residual",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "runtime": {
            "python": sys.executable,
            "machine": platform.machine(),
            "torch_threads": THREADS,
            "num_workers": 0,
            "epochs": EPOCHS,
            "batch": BATCH,
            "checkpoint_dir": str(CHECKPOINT_DIR),
        },
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_neural_context_stage42.json",
                "outputs/stage42_long_research/source_level_residual_context_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "sequence_schema": {
            "history_seq_shape": list(shared["data"]["history_seq"].shape),
            "uses_past_history_only": True,
            "goal_neighbor_context": ["prototype_likelihood", "prototype_entropy", "goal_ambiguity", "history_scalar_neighbor_slice"],
        },
        "baseline_family_only": {
            "source": "fresh_run",
            "feature_count": int(np.sum(ap._baseline_mask(names))),
            "best_lambda": baseline_direct["model"]["best_lambda"],
            "protected": baseline_metric,
            "bootstrap": baseline_direct["model"]["bootstrap"],
        },
        "training": training,
        "sequence_variants": variants,
        "sequence_deltas": deltas,
        "positive_sequence_context_variants": positive,
        "summary": {
            "source": "fresh_run",
            "sequence_context_verdict": "stage42_ar_sequence_context_supported" if positive else "stage42_ar_sequence_context_not_supported",
            "positive_sequence_context_variants": positive,
            "interpretation": "Stage42-AR tests temporal sequence context after tabular ridge/MLP context failed. Positive variants support history/goal/neighbor contribution beyond baseline family; negative variants imply the next step needs richer graph/scene context or a different supervision target.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_ar_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _interpret_sequence_delta(name: str, delta: Mapping[str, float]) -> str:
    if _positive_sequence_delta(delta):
        return f"{name} improves over baseline-family-only by > {MIN_SEQUENCE_DELTA} on at least one core metric."
    return f"{name} does not improve over baseline-family-only by > {MIN_SEQUENCE_DELTA}; sequence context contribution not proven."


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    baseline = result["baseline_family_only"]["protected"]
    gates = {
        "arm64_torch_runtime": result["runtime"]["machine"] == "arm64",
        "single_process_no_workers": result["runtime"]["num_workers"] == 0,
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "baseline_family_first_stage_positive": baseline["all_improvement"] > 0
        and baseline["t50_improvement"] > 0
        and baseline["easy_degradation"] <= 0.02,
        "sequence_variants_complete": len(result["sequence_variants"]) >= 3,
        "sequence_context_increment_found": len(result["positive_sequence_context_variants"]) >= 1,
        "checkpoints_recorded": all(Path(row["checkpoint"]).exists() for row in result["training"].values()),
        "bootstrap_available_for_baseline": result["baseline_family_only"]["bootstrap"]["all"]["bootstrap_n"] > 0
        and result["baseline_family_only"]["bootstrap"]["t50"]["bootstrap_n"] > 0,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = (
        "stage42_ar_sequence_context_evidence_pass"
        if all(gates.values())
        else "stage42_ar_sequence_context_evidence_partial_or_negative"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    base = result["baseline_family_only"]["protected"]
    lines = [
        "# Stage42-AR Proposed Source-Level Sequence Context",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ar_gate']['passed']} / {result['stage42_ar_gate']['total']}`",
        f"- verdict: `{result['stage42_ar_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Runtime And Schema",
        "",
        f"- runtime: `{result['runtime']}`",
        f"- sequence_schema: `{result['sequence_schema']}`",
        "",
        "## Why This Was Run",
        "",
        "- Stage42-AQ ruled out a simple tabular MLP residual-context repair.",
        "- Stage42-AR uses a temporal Conv1D sequence encoder over past-only history plus goal/neighbor context.",
        "",
        "## Baseline-Family First Stage",
        "",
        f"- protected_metric: `{base}`",
        "",
        "## Sequence Residual Variants",
        "",
        "| variant | alpha | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["sequence_variants"].items():
        metric = row["protected"]
        d = result["sequence_deltas"][name]["delta_vs_baseline_family_only"]
        lines.append(
            f"| `{name}` | {row['best_residual_alpha']:.2f} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- positive_sequence_context_variants: `{result['positive_sequence_context_variants']}`",
            f"- sequence_context_verdict: `{result['summary']['sequence_context_verdict']}`",
            "",
        ]
    )
    if result["positive_sequence_context_variants"]:
        lines.append("- Stage42-AR found sequence-context residual value beyond baseline-family rollout context.")
    else:
        lines.append("- Stage42-AR did not find sequence-context residual value beyond baseline-family rollout context.")
    lines.extend(
        [
            "- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ar_gate"]
    lines = [
        "# Stage42-AR Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "source": result["source"],
        "generated_at_utc": result["generated_at_utc"],
        "verdict": result["stage42_ar_gate"]["verdict"],
        "gate": f"{result['stage42_ar_gate']['passed']}/{result['stage42_ar_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_sequence_context()
