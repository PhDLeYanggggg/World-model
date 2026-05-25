from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage42_retrained_ablation as s42g


OUT_DIR = Path("outputs/stage42_long_research")
DATA_NPZ = Path("data/stage41_world_model/combined_external.npz")
META_JSON = Path("data/stage41_world_model/combined_meta.json")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "sequence_ablation_stage42.json"
REPORT_MD = OUT_DIR / "sequence_ablation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_h_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SEEDS = [31, 37, 43]
EPOCHS = 2
BATCH = 4096
THREADS = 4
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "External 数据仍是 dataset-local / unverified weak metric diagnostic。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "future endpoints / family_fde 只作为 supervised label/eval，不作为 inference input。",
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE42H_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE42H_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage42-H refuses x86_64/Rosetta Python for torch training.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _base_feature_names() -> list[str]:
    return list(read_json(META_JSON, {}).get("feature_names", []))


def _domain_one_hot(dataset: np.ndarray) -> tuple[np.ndarray, list[str]]:
    domains = sorted(set(map(str, dataset.tolist())))
    index = {d: i for i, d in enumerate(domains)}
    out = np.zeros((len(dataset), len(domains)), dtype=np.float32)
    for i, d in enumerate(map(str, dataset.tolist())):
        out[i, index[d]] = 1.0
    return out, [f"domain_{d}" for d in domains]


def _horizon_one_hot(horizon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    hs = [10, 25, 50, 100]
    return np.stack([(horizon.astype(int) == h).astype(np.float32) for h in hs], axis=1), [f"horizon_{h}" for h in hs]


def _assemble_static(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    base = data["stage37_features"].astype(np.float32)
    base_names = _base_feature_names()
    hist_scalar = data["history_scalar"].astype(np.float32)
    hist_names = [
        "history_path_length",
        "history_neighbor_count",
        "history_min_neighbor_dist",
        "history_density",
        "history_ttc",
        "history_closing_speed",
        "history_curvature",
        "history_turn_angle",
        "history_valid_len",
    ]
    proto = data["prototype_likelihood"].astype(np.float32)
    proto_names = [f"prototype_likelihood_{i}" for i in range(proto.shape[1])]
    misc = np.stack(
        [
            data["prototype_entropy"].astype(np.float32),
            data["goal_ambiguity"].astype(np.float32),
            data["track_length"].astype(np.float32) / 100.0,
        ],
        axis=1,
    )
    misc_names = ["prototype_entropy", "goal_ambiguity", "track_length_scaled"]
    horizon_x, horizon_names = _horizon_one_hot(data["horizon"])
    domain_x, domain_names = _domain_one_hot(data["dataset"].astype(str))
    x = np.concatenate([base, hist_scalar, proto, misc, horizon_x, domain_x], axis=1).astype(np.float32)
    names = base_names + hist_names + proto_names + misc_names + horizon_names + domain_names
    groups = {"history": [], "neighbor_interaction": [], "goal_scene": [], "domain": []}
    for i, name in enumerate(names):
        low = name.lower()
        if low.startswith("history_"):
            groups["history"].append(i)
        if any(key in low for key in ["neighbor", "density", "ttc", "closing", "interaction"]):
            groups["neighbor_interaction"].append(i)
        if any(key in low for key in ["prototype", "goal", "ambiguity", "exit_like"]):
            groups["goal_scene"].append(i)
        if low.startswith("domain_"):
            groups["domain"].append(i)
    return np.nan_to_num(x, posinf=1e6, neginf=-1e6), names, groups


def _apply_static_mask(x: np.ndarray, drop: list[int]) -> np.ndarray:
    out = x.copy()
    if drop:
        out[:, sorted(set(drop))] = 0.0
    return out


def _split_mask(data: Mapping[str, np.ndarray], split: str) -> np.ndarray:
    return data["old_split"].astype(str) == split


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    return tuple(((x.astype(np.float32) - mean) / std).astype(np.float32) for x in (train, *others))


def _standardize_seq(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0).astype(np.float32)
    std = np.maximum(train.reshape(-1, train.shape[-1]).std(axis=0), 1e-4).astype(np.float32)
    return tuple(((x.astype(np.float32) - mean[None, None, :]) / std[None, None, :]).astype(np.float32) for x in (train, *others))


def _labels_for_split(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    y = data["family_fde"][mask].astype(np.float32)
    floor = np.clip(data["safe_strongest_idx_old"][mask].astype(int), 0, y.shape[1] - 1)
    return {
        "y": y,
        "horizon": data["horizon"][mask],
        "hard": data["hard"][mask],
        "failure": data["failure"][mask],
        "easy": data["easy"][mask],
        "floor_idx": floor,
        "oracle_err": np.min(y, axis=1),
    }


def _make_model(seq_dim: int, static_dim: int, out_dim: int):
    torch = _torch()
    import torch.nn as nn

    class CausalSequenceFDE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv1d(seq_dim, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(32, 48, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.static = nn.Sequential(nn.Linear(static_dim, 96), nn.GELU(), nn.LayerNorm(96))
            self.head = nn.Sequential(nn.Linear(48 + 48 + 96, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, out_dim))

        def forward(self, seq, static):
            h = self.seq(seq.transpose(1, 2))
            pooled = h.mean(dim=2)
            last = h[:, :, -1]
            s = self.static(static)
            return self.head(torch.cat([pooled, last, s], dim=1))

    return CausalSequenceFDE()


def _train_predict_variant(
    *,
    variant: str,
    seed: int,
    seq_train: np.ndarray,
    seq_val: np.ndarray,
    seq_test: np.ndarray,
    static_train: np.ndarray,
    static_val: np.ndarray,
    static_test: np.ndarray,
    labels_train: Mapping[str, np.ndarray],
    labels_val: Mapping[str, np.ndarray],
    labels_test: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    model = _make_model(seq_train.shape[2], static_train.shape[1], labels_train["y"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    train_seq_t = torch.tensor(seq_train)
    train_static_t = torch.tensor(static_train)
    train_y_t = torch.tensor(np.log1p(labels_train["y"]).astype(np.float32))
    hard_weight = torch.tensor((1.0 + 1.5 * (labels_train["hard"].astype(bool) | labels_train["failure"].astype(bool))).astype(np.float32))
    val_seq_t = torch.tensor(seq_val)
    val_static_t = torch.tensor(static_val)
    val_y_t = torch.tensor(np.log1p(labels_val["y"]).astype(np.float32))
    heartbeat = OUT_DIR / f"stage42h_{variant}_seed{seed}_heartbeat.json"
    ckpt = CHECKPOINT_DIR / f"stage42h_{variant}_seed{seed}.pt"
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_seq_t))
        losses = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            pred = model(train_seq_t[ids], train_static_t[ids])
            loss_row = F.smooth_l1_loss(pred, train_y_t[ids], reduction="none").mean(dim=1)
            loss = (loss_row * hard_weight[ids]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_seq_t, val_static_t)
            val_loss = float(F.smooth_l1_loss(val_pred, val_y_t).detach().cpu())
        best = min(best, {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}, key=lambda row: row["val_loss"])
        if best["epoch"] == epoch:
            torch.save({"model": model.state_dict(), "variant": variant, "seed": seed, "best": best}, ckpt)
        heartbeat.write_text(json.dumps({"variant": variant, "seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    def predict(seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for start in range(0, len(seq), BATCH):
                raw = model(torch.tensor(seq[start : start + BATCH]), torch.tensor(static[start : start + BATCH]))
                outs.append(np.maximum(0.0, np.expm1(raw.detach().cpu().numpy())))
        return np.concatenate(outs, axis=0).astype(np.float32)

    val_pred_np = predict(seq_val, static_val)
    test_pred_np = predict(seq_test, static_test)
    return val_pred_np, test_pred_np, {"checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _evaluate_prediction(variant: str, seed: int, val_pred: np.ndarray, test_pred: np.ndarray, labels_val: Mapping[str, np.ndarray], labels_test: Mapping[str, np.ndarray], *, no_safe_switch: bool = False, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = s42g._choose_policy_on_val(val_pred, labels_val["y"], labels_val, no_safe_switch=no_safe_switch)
    floor_idx = labels_test["floor_idx"].astype(int)
    if policy.get("no_safe_switch"):
        selected_idx = np.argmin(test_pred, axis=1).astype(np.int64)
    else:
        selected_idx = s42g._select_from_pred(test_pred, floor_idx, policy["confidence_min"], policy["gain_min"], policy["max_switch_rate"])
    selected_err = labels_test["y"][np.arange(len(labels_test["y"])), selected_idx]
    floor_err = labels_test["y"][np.arange(len(labels_test["y"])), floor_idx]
    metric = s42g._metrics(selected_err, floor_err, labels_test, selected_idx, floor_idx)
    return {"source": "fresh_run", "variant": variant, "seed": seed, "policy": policy, **metric, **(dict(extra or {}))}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stat(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        half = 1.96 * std / np.sqrt(max(len(arr), 1))
        return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}

    out = {}
    for name in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == name]
        out[name] = {
            "source": "fresh_run",
            "seeds": [r["seed"] for r in sub],
            "all": stat([r["all_improvement"] for r in sub]),
            "t50": stat([r["t50_improvement"] for r in sub]),
            "t100_raw_frame_diagnostic": stat([r["t100_raw_frame_diagnostic_improvement"] for r in sub]),
            "hard_failure": stat([r["hard_failure_improvement"] for r in sub]),
            "easy_degradation": stat([r["easy_degradation"] for r in sub]),
            "switch_rate": stat([r["switch_rate"] for r in sub]),
        }
    return out


def _contribution(summary: Mapping[str, Any], full_name: str = "sequence_full_safe_switch") -> dict[str, Any]:
    full = summary.get(full_name, {})
    out = {}
    for name, item in summary.items():
        if name == full_name:
            continue
        out[name] = {
            "all_delta_full_minus_ablation": full.get("all", {}).get("mean", 0.0) - item.get("all", {}).get("mean", 0.0),
            "t50_delta_full_minus_ablation": full.get("t50", {}).get("mean", 0.0) - item.get("t50", {}).get("mean", 0.0),
            "hard_delta_full_minus_ablation": full.get("hard_failure", {}).get("mean", 0.0) - item.get("hard_failure", {}).get("mean", 0.0),
            "easy_delta_ablation_minus_full": item.get("easy_degradation", {}).get("mean", 0.0) - full.get("easy_degradation", {}).get("mean", 0.0),
        }
    return out


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary") or {}
    contrib = result.get("contribution_vs_sequence_full") or {}
    full = summary.get("sequence_full_safe_switch") or {}
    history = contrib.get("sequence_no_history_tokens") or {}
    gates = {
        "sequence_models_trained": all(k in summary for k in ["sequence_full_safe_switch", "sequence_no_history_tokens", "sequence_no_goal_scene_tokens", "sequence_no_neighbor_interaction_tokens"]),
        "three_seeds": all(len((summary.get(k) or {}).get("seeds", [])) >= 3 for k in ["sequence_full_safe_switch", "sequence_no_history_tokens", "sequence_no_goal_scene_tokens", "sequence_no_neighbor_interaction_tokens"]),
        "full_sequence_safe": full.get("easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "history_contribution_answered": "t50_delta_full_minus_ablation" in history,
        "at_least_one_positive_sequence_component": any(v.get("t50_delta_full_minus_ablation", 0.0) > 0 or v.get("hard_delta_full_minus_ablation", 0.0) > 0 for v in contrib.values()),
        "no_safe_switch_diagnosed": "sequence_full_no_safe_switch" in summary,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
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
        "history_t50_delta": history.get("t50_delta_full_minus_ablation"),
        "verdict": "stage42_h_sequence_ablation_pass" if all(gates.values()) else "stage42_h_sequence_ablation_partial",
    }


def run_stage42_sequence_ablation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    data = dict(np.load(DATA_NPZ, allow_pickle=True))
    static_all, static_names, groups = _assemble_static(data)
    seq_all = data["history_seq"].astype(np.float32)
    masks = {split: _split_mask(data, split) for split in ["train", "val", "test"]}
    labels = {split: _labels_for_split(data, masks[split]) for split in ["train", "val", "test"]}
    variant_config = {
        "sequence_full_safe_switch": {"seq_zero": False, "drop_static": []},
        "sequence_no_history_tokens": {"seq_zero": True, "drop_static": groups["history"]},
        "sequence_no_goal_scene_tokens": {"seq_zero": False, "drop_static": groups["goal_scene"]},
        "sequence_no_neighbor_interaction_tokens": {"seq_zero": False, "drop_static": groups["neighbor_interaction"]},
        "sequence_no_domain_expert": {"seq_zero": False, "drop_static": groups["domain"]},
    }
    rows = []
    for variant, cfg in variant_config.items():
        static_variant = _apply_static_mask(static_all, cfg["drop_static"])
        seq_variant = np.zeros_like(seq_all) if cfg["seq_zero"] else seq_all
        seq_train, seq_val, seq_test = _standardize_seq(seq_variant[masks["train"]], seq_variant[masks["val"]], seq_variant[masks["test"]])
        static_train, static_val, static_test = _standardize(static_variant[masks["train"]], static_variant[masks["val"]], static_variant[masks["test"]])
        for seed in SEEDS:
            val_pred, test_pred, train_info = _train_predict_variant(
                variant=variant,
                seed=seed,
                seq_train=seq_train,
                seq_val=seq_val,
                seq_test=seq_test,
                static_train=static_train,
                static_val=static_val,
                static_test=static_test,
                labels_train=labels["train"],
                labels_val=labels["val"],
                labels_test=labels["test"],
            )
            rows.append(_evaluate_prediction(variant, seed, val_pred, test_pred, labels["val"], labels["test"], extra={"train_info": train_info}))
            if variant == "sequence_full_safe_switch":
                rows.append(_evaluate_prediction("sequence_full_no_safe_switch", seed, val_pred, test_pred, labels["val"], labels["test"], no_safe_switch=True, extra={"train_info": train_info}))
    summary = _summarize(rows)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-H causal sequence ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_threads": THREADS,
        "epochs": EPOCHS,
        "batch": BATCH,
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([DATA_NPZ, META_JSON]),
        "dataset_rows": {split: int(np.sum(mask)) for split, mask in masks.items()},
        "feature_schema": {
            "sequence_shape": list(seq_all.shape[1:]),
            "static_dim": int(static_all.shape[1]),
            "static_group_sizes": {k: len(v) for k, v in groups.items()},
            "static_feature_names": static_names,
        },
        "source_labels": {
            "external_combined_dataset": "cached_verified",
            "sequence_model_training": "fresh_run",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once_per_variant_seed",
        },
        "rows": rows,
        "summary": summary,
        "contribution_vs_sequence_full": _contribution(summary),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "family_fde_used_as_label_only": True,
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
            "sequence_ablation_not_full_transformer_or_jepa": True,
        },
    }
    result["stage42_h_gate"] = _gate(result)
    _write_json(REPORT_JSON, result)
    _write_report(result)
    _write_gate(result["stage42_h_gate"])
    _append_readme_and_state(result)
    _append_ledger("stage42_h_sequence_ablation", "success", result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-H Causal Sequence Ablation",
        "",
        "- source: `fresh_run`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_h_gate']['passed']} / {result['stage42_h_gate']['total']}`",
        f"- verdict: `{result['stage42_h_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Metrics",
        "",
        "| variant | all mean | t50 mean | t100 diag mean | hard mean | easy mean | switch mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(result["summary"].items()):
        lines.append(
            f"| `{name}` | {item['all']['mean']:.6f} | {item['t50']['mean']:.6f} | {item['t100_raw_frame_diagnostic']['mean']:.6f} | {item['hard_failure']['mean']:.6f} | {item['easy_degradation']['mean']:.6f} | {item['switch_rate']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Contribution Deltas",
            "",
            "`full_minus_ablation > 0` means the removed component helped the full sequence model on that slice.",
            "",
            "| ablation | all delta | t50 delta | hard delta | easy delta ablation-minus-full |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(result["contribution_vs_sequence_full"].items()):
        lines.append(
            f"| `{name}` | {row['all_delta_full_minus_ablation']:.6f} | {row['t50_delta_full_minus_ablation']:.6f} | {row['hard_delta_full_minus_ablation']:.6f} | {row['easy_delta_ablation_minus_full']:.6f} |"
        )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-H Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- history_t50_delta: `{gate.get('history_t50_delta')}`",
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
    gate = result["stage42_h_gate"]
    full = result["summary"].get("sequence_full_safe_switch", {})
    hist_delta = (result.get("contribution_vs_sequence_full") or {}).get("sequence_no_history_tokens", {})
    block = f"""
## Stage42-H Causal Sequence Ablation

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
sequence_full_all = {full.get('all', {}).get('mean')}
sequence_full_t50 = {full.get('t50', {}).get('mean')}
sequence_full_hard_failure = {full.get('hard_failure', {}).get('mean')}
sequence_full_easy_degradation = {full.get('easy_degradation', {}).get('mean')}
history_t50_delta_full_minus_no_history = {hist_delta.get('t50_delta_full_minus_ablation')}
stage5c_executed = false
smc_enabled = false
```

Stage42-H trains a causal temporal sequence encoder, not a flattened-history ridge selector. It answers whether history tokens help under a sequence model while keeping val-only safety selection and test-once evaluation. This is still dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-H Causal Sequence Ablation", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-H Causal Sequence Ablation", block)

    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_h_causal_sequence_ablation"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_h_causal_sequence_ablation"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "sequence_full_all": full.get("all", {}).get("mean"),
        "sequence_full_t50": full.get("t50", {}).get("mean"),
        "sequence_full_hard_failure": full.get("hard_failure", {}).get("mean"),
        "sequence_full_easy_degradation": full.get("easy_degradation", {}).get("mean"),
        "history_t50_delta_full_minus_no_history": hist_delta.get("t50_delta_full_minus_ablation"),
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(step: str, status: str, result: Mapping[str, Any]) -> None:
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_sequence_ablation()
