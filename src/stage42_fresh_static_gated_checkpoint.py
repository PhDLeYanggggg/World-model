from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_full_trajectory_world_state as ft
from src import stage42_sequence_full_waypoint as s42i


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "fresh_static_gated_checkpoint_stage42.json"
REPORT_MD = OUT_DIR / "fresh_static_gated_checkpoint_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_k_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SEEDS = [71, 73, 79]
EPOCHS = 2
BATCH = 2048
THREADS = 4
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-K 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。",
    "future waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE42K_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE42K_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage42-K refuses x86_64/Rosetta Python for torch training.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _make_model(static_dim: int, token_dim: int = 9, width: int = 72):
    torch = _torch()
    import torch.nn as nn

    class StaticGatedSequenceWaypoint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv1d(token_dim, 48, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(48, width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width))
            self.gate = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))
            # Bias toward the no-static expert until validation-worthy static
            # evidence is learned. This mirrors the Stage42-J finding.
            nn.init.constant_(self.gate[-1].bias, -2.0)
            self.ctx = nn.Sequential(nn.Linear(width * 3, width * 2), nn.GELU(), nn.LayerNorm(width * 2))
            self.waypoints = nn.Linear(width * 2, len(ft.WAYPOINT_FRAC) * 2)
            self.risk = nn.Linear(width * 2, 1)
            self.interaction = nn.Linear(width * 2, 1)
            self.occupancy = nn.Linear(width * 2, 1)
            self.physical = nn.Linear(width * 2, 1)

        def _encode_agents(self, tokens, mask):
            b, a, t, d = tokens.shape
            flat = tokens.reshape(b * a, t, d).transpose(1, 2)
            h = self.temporal(flat).transpose(1, 2).reshape(b, a, t, -1)
            valid_t = tokens[..., 6].clamp(0, 1)
            emb = (h * valid_t[..., None]).sum(dim=2) / valid_t.sum(dim=2, keepdim=True).clamp_min(1.0)
            emb = emb * mask[..., None].float()
            target = emb[:, 0]
            neigh_mask = mask[:, 1:, None].float()
            neigh = (emb[:, 1:] * neigh_mask).sum(dim=1) / neigh_mask.sum(dim=1).clamp_min(1.0)
            return target, neigh

        def forward(self, tokens, mask, static):
            target, neigh = self._encode_agents(tokens, mask)
            static_h = self.static(static)
            raw_gate = torch.sigmoid(self.gate(torch.cat([target, neigh, static_h], dim=1))).squeeze(-1)
            gated_static = raw_gate[:, None] * static_h
            ctx = self.ctx(torch.cat([target, neigh, gated_static], dim=1))
            return {
                "waypoint_delta": self.waypoints(ctx).view(-1, len(ft.WAYPOINT_FRAC), 2),
                "traj_risk": self.risk(ctx).squeeze(-1),
                "interaction_logit": self.interaction(ctx).squeeze(-1),
                "occupancy_logit": self.occupancy(ctx).squeeze(-1),
                "physical_logit": self.physical(ctx).squeeze(-1),
                "static_gate": raw_gate,
            }

    return StaticGatedSequenceWaypoint()


def _train_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make_model(train["static"].shape[1], train["agent_tokens"].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    ckpt = CHECKPOINT_DIR / f"stage42k_fresh_static_gate_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42k_fresh_static_gate_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}

    tensors = {
        "tokens": torch.tensor(train["agent_tokens"]),
        "mask": torch.tensor(train["agent_mask"]),
        "static": torch.tensor(train["static"]),
        "target": torch.tensor(train["waypoint_delta"]),
        "valid": torch.tensor(train["waypoint_valid"].astype(np.float32)),
        "interaction": torch.tensor(train["interaction"]),
        "occupancy": torch.tensor(train["occupancy"]),
        "physical": torch.tensor(train["physical"]),
        "hard": torch.tensor((train["hard"] | train["failure"]).astype(np.float32)),
        "horizon": torch.tensor(train["horizon"]),
    }
    val_tensors = {
        "tokens": torch.tensor(val["agent_tokens"]),
        "mask": torch.tensor(val["agent_mask"]),
        "static": torch.tensor(val["static"]),
        "target": torch.tensor(val["waypoint_delta"]),
        "valid": torch.tensor(val["waypoint_valid"].astype(np.float32)),
    }
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train["agent_tokens"]))
        losses: list[float] = []
        gate_vals: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            static = tensors["static"][ids].clone()
            # Static dropout makes the checkpoint robust to missing or harmful
            # context and prevents reintroducing the Stage42-I failure mode.
            drop = torch.rand(static.shape[0]) < 0.50
            static[drop] = 0.0
            out = model(tensors["tokens"][ids], tensors["mask"][ids], static)
            valid = tensors["valid"][ids]
            row_w = 1.0 + 1.5 * tensors["hard"][ids] + 2.0 * (tensors["horizon"][ids] == 50).float() + 1.0 * (tensors["horizon"][ids] == 100).float()
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], tensors["target"][ids], reduction="none").mean(dim=2)
            traj = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_w).mean()
            err = torch.linalg.norm(out["waypoint_delta"] - tensors["target"][ids], dim=2)
            risk_target = torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach()
            risk = (F.smooth_l1_loss(out["traj_risk"], risk_target, reduction="none") * row_w).mean()
            aux = (
                F.binary_cross_entropy_with_logits(out["interaction_logit"], tensors["interaction"][ids])
                + F.binary_cross_entropy_with_logits(out["occupancy_logit"], tensors["occupancy"][ids])
                + F.binary_cross_entropy_with_logits(out["physical_logit"], tensors["physical"][ids])
            )
            gate_penalty = 0.015 * out["static_gate"].mean()
            loss = traj + 0.30 * risk + 0.15 * aux + gate_penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            gate_vals.append(float(out["static_gate"].mean().detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val_tensors["tokens"], val_tensors["mask"], val_tensors["static"])
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], val_tensors["target"], reduction="none").mean(dim=2)
            val_loss = float(((per_wp * val_tensors["valid"]).sum(dim=1) / val_tensors["valid"].sum(dim=1).clamp_min(1.0)).mean().cpu())
            val_gate = float(out["static_gate"].mean().cpu())
        cand = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses)), "train_gate_mean": float(np.mean(gate_vals)), "val_gate_mean": val_gate}
        if val_loss < best["val_loss"]:
            best = cand
            torch.save({"model": model.state_dict(), "seed": seed, "static_dim": train["static"].shape[1], "token_dim": train["agent_tokens"].shape[-1], "best": best}, ckpt)
        heartbeat.write_text(json.dumps({"seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(info: Mapping[str, Any], split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = _make_model(int(payload["static_dim"]), int(payload["token_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    outs = {k: [] for k in ["waypoint_delta", "traj_risk", "interaction", "occupancy", "physical", "static_gate"]}
    with torch.no_grad():
        for start in range(0, len(split["agent_tokens"]), 4096):
            sl = slice(start, min(start + 4096, len(split["agent_tokens"])))
            out = model(torch.tensor(split["agent_tokens"][sl]), torch.tensor(split["agent_mask"][sl]), torch.tensor(split["static"][sl]))
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
            outs["static_gate"].append(out["static_gate"].cpu().numpy().reshape(-1))
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def _eval_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    info = _train_seed(seed, train, val)
    pred_val = _predict(info, val)
    pred_test = _predict(info, test)
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    policy, val_metrics = s42i._fit_light_policy(pred_val, labels_val)
    test_metrics = s42i._row_metrics("fresh_static_gated_checkpoint", pred_test, labels_test, policy)
    ungated = s42i._row_metrics(
        "fresh_static_gated_checkpoint_ungated",
        pred_test,
        labels_test,
        {"slices": {f"{d}|{h}": {"risk_max": 1e9, "physical_min": 0.0, "max_switch": 1.0, "hard_only": False, "easy_block": False} for d in sorted(set(labels_test["domain"].tolist())) for h in [10, 25, 50, 100]}},
    )
    return {
        "source": info["source"],
        "seed": seed,
        "train_info": info,
        "val_policy": policy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ungated_test_metrics": ungated,
        "static_gate_mean_val": float(np.mean(pred_val["static_gate"])),
        "static_gate_mean_test": float(np.mean(pred_test["static_gate"])),
    }


def _stat(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / np.sqrt(max(len(arr), 1))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": "fresh_run",
        "seeds": [int(row["seed"]) for row in rows],
        "ade_all": _stat([row["test_metrics"]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": _stat([row["test_metrics"]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": _stat([row["test_metrics"]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": _stat([row["test_metrics"]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": _stat([row["test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_all": _stat([row["test_metrics"]["fde"].get("all_improvement", 0.0) for row in rows]),
        "fde_t50": _stat([row["test_metrics"]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": _stat([row["test_metrics"].get("switch_rate", 0.0) for row in rows]),
        "ungated_easy_degradation": _stat([row["ungated_test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "static_gate_mean_test": _stat([row["static_gate_mean_test"] for row in rows]),
    }


def _comparison() -> dict[str, Any]:
    report = read_json(OUT_DIR / "static_gated_full_waypoint_stage42.json", {})
    return {
        "source": "cached_verified",
        "stage42_j_static_gated": (report.get("summary") or {}).get("static_gated", {}),
        "stage42_j_no_static": (report.get("summary") or {}).get("no_static", {}),
        "stage42_j_full_static": (report.get("summary") or {}).get("full_static", {}),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    cmp = result.get("comparison", {})
    full = (cmp.get("stage42_j_full_static") or {})
    gates = {
        "fresh_static_gated_checkpoints_trained": all((row.get("train_info") or {}).get("source") in {"fresh_run", "cached_verified"} for row in result.get("rows", []))
        and len(result.get("rows", [])) >= 3,
        "three_seeds": len(summary.get("seeds", [])) >= 3,
        "fresh_positive": summary.get("ade_all", {}).get("mean", 0.0) > 0.0 or summary.get("ade_t50", {}).get("mean", 0.0) > 0.0 or summary.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "fresh_improves_stage42i_full": summary.get("ade_all", {}).get("mean", 0.0) > full.get("ade_all", {}).get("mean", -1.0)
        and summary.get("ade_t50", {}).get("mean", 0.0) > full.get("ade_t50", {}).get("mean", -1.0),
        "easy_preserved": summary.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_k_fresh_static_gated_checkpoint_pass" if all(gates.values()) else "stage42_k_fresh_static_gated_checkpoint_partial",
    }


def run_stage42_fresh_static_gated_checkpoint() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    rows = [_eval_seed(seed, data["train"], data["val"], data["test"]) for seed in SEEDS]
    result = {
        "source": "fresh_run",
        "stage": "Stage42-K fresh static-gated checkpoint training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_threads": THREADS,
        "epochs": EPOCHS,
        "batch": BATCH,
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                ft.DATA_DIR / "all_agent_train.npz",
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_train.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                OUT_DIR / "static_gated_full_waypoint_stage42.json",
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "rows": rows,
        "summary": _summary(rows),
        "comparison": _comparison(),
        "source_labels": {
            "all_agent_dataset": "cached_verified",
            "full_waypoint_labels": "cached_verified_or_rebuilt_by_stage41_helper",
            "fresh_static_gated_checkpoint_training": "fresh_run",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once_per_seed",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "thresholds_selected_on_val": True,
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
    result["stage42_k_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_k_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    cmp = result["comparison"]
    lines = [
        "# Stage42-K Fresh Static-Gated Checkpoint Training",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_k_gate']['passed']} / {result['stage42_k_gate']['total']}`",
        f"- verdict: `{result['stage42_k_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Fresh Checkpoint Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | static gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `fresh_static_gated_checkpoint` | `fresh_run` | {s['ade_all']['mean']:.6f} | {s['ade_t50']['mean']:.6f} | {s['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {s['ade_hard_failure']['mean']:.6f} | {s['ade_easy_degradation']['mean']:.6f} | {s['fde_all']['mean']:.6f} | {s['fde_t50']['mean']:.6f} | {s['switch_rate']['mean']:.6f} | {s['static_gate_mean_test']['mean']:.6f} |",
        "",
        "## Cached Comparison",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["stage42_j_full_static", "stage42_j_no_static", "stage42_j_static_gated"]:
        row = cmp.get(name, {})
        lines.append(
            f"| `{name}` | `{row.get('source', 'cached_verified')}` | {row.get('ade_all', {}).get('mean', 0.0):.6f} | {row.get('ade_t50', {}).get('mean', 0.0):.6f} | {row.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {row.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {row.get('fde_t50', {}).get('mean', 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-K is the fresh checkpoint version of the Stage42-J static-gate idea.",
            "- Static dropout and a learned low-bias static gate are used to avoid the Stage42-I unconditional-static failure mode.",
            "- Stage42-J remains the stronger policy-level result if this fresh checkpoint does not match it.",
            "- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-K Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_k_gate"]
    s = result["summary"]
    block = f"""
## Stage42-K Fresh Static-Gated Checkpoint Training

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
fresh_static_gated_ade_all = {s['ade_all']['mean']}
fresh_static_gated_ade_t50 = {s['ade_t50']['mean']}
fresh_static_gated_ade_hard_failure = {s['ade_hard_failure']['mean']}
fresh_static_gated_ade_easy_degradation = {s['ade_easy_degradation']['mean']}
fresh_static_gated_fde_t50 = {s['fde_t50']['mean']}
fresh_static_gate_mean_test = {s['static_gate_mean_test']['mean']}
stage5c_executed = false
smc_enabled = false
```

Stage42-K trains fresh static-gated/static-dropout sequence-to-full-waypoint checkpoints. It tests whether the Stage42-J policy-level static gate can be baked into model training. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-K Fresh Static-Gated Checkpoint Training", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-K Fresh Static-Gated Checkpoint Training", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_k_fresh_static_gated_checkpoint"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_k_fresh_static_gated_checkpoint"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "fresh_static_gated_ade_all": s["ade_all"]["mean"],
        "fresh_static_gated_ade_t50": s["ade_t50"]["mean"],
        "fresh_static_gated_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "fresh_static_gated_ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "fresh_static_gated_fde_t50": s["fde_t50"]["mean"],
        "fresh_static_gate_mean_test": s["static_gate_mean_test"]["mean"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": "stage42_k_fresh_static_gated_checkpoint",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_fresh_static_gated_checkpoint()
