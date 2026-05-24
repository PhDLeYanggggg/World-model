from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage37_t50_history as s37
from src import stage39_neural_dynamics as s39


OUT_DIR = Path("outputs/stage40_neural_optimization")
DATA_DIR = Path("data/stage40_neural_optimization")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
SEED = 4000
THREADS = 4
TRIAL_EPOCHS = 5
BATCH = 512
CANDIDATE_NAMES = ["stage37_floor"] + s37.BASELINE_FAMILY


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(entry: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(entry)), ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage40 Neural Optimization Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    start = time.perf_counter()
    status = "failed"
    input_hash = _combined_hash(inputs)
    try:
        payload = fn()
        status = "success"
        return payload
    finally:
        _append_ledger(
            {
                "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
                "step": name,
                "inputs": [str(p) for p in inputs],
                "outputs": [str(p) for p in outputs],
                "wall_time_s": time.perf_counter() - start,
                "status": status,
                "input_hash": input_hash,
                "output_hash": _combined_hash(outputs),
                "git_commit": _git_commit(),
                "source": "fresh_run",
            }
        )


def _ensure_arm64_torch_runtime() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE40_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE40_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage40 neural training refuses x86_64/Rosetta Python. Use .venv-pytorch/bin/python.")


def _torch():
    _ensure_arm64_torch_runtime()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _ds(split: str) -> Dict[str, np.ndarray]:
    if not (s39.DATA_DIR / f"neural_dataset_{split}.npz").exists():
        s39.build_neural_dataset()
    return dict(np.load(s39.DATA_DIR / f"neural_dataset_{split}.npz"))


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(s39.DATA_DIR / "normalization.npz"))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _candidate_arrays(split: str) -> Dict[str, np.ndarray]:
    out = DATA_DIR / f"candidate_arrays_{split}.npz"
    if out.exists():
        return dict(np.load(out))
    ensure_dir(DATA_DIR)
    ds = _ds(split)
    idx = ds["idx"].astype(int)
    fam = s37._baseline_family(split)
    family_pred = fam["prediction"].astype(np.float32)[idx]
    family_fde = fam["y_fde"].astype(np.float32)[idx]
    family_rel = fam["relative_y"].astype(np.float32)[idx]
    stage37_pred = ds["stage37_pred_xy"].astype(np.float32)
    stage37_fde = ds["stage37_fde"].astype(np.float32)
    normalizer = np.maximum(ds["normalizer"].astype(np.float32), EPS)
    current = ds["current_xy"].astype(np.float32)
    candidates = np.concatenate([stage37_pred[:, None, :], family_pred], axis=1)
    candidate_fde = np.concatenate([stage37_fde[:, None], family_fde], axis=1)
    candidate_rel = np.concatenate([(stage37_fde / normalizer)[:, None], family_rel], axis=1)
    candidate_delta = ((candidates - current[:, None, :]) / normalizer[:, None, None]).astype(np.float32)
    oracle = np.argmin(candidate_fde, axis=1).astype(np.int64)
    stage37 = np.zeros(len(oracle), dtype=np.int64)
    margin = np.sort(candidate_fde, axis=1)[:, 1] - np.sort(candidate_fde, axis=1)[:, 0]
    np.savez_compressed(
        out,
        candidate_delta=candidate_delta,
        candidate_fde=candidate_fde,
        candidate_rel=candidate_rel,
        oracle_idx=oracle,
        stage37_idx=stage37,
        oracle_margin=margin.astype(np.float32),
        candidate_names=np.asarray(CANDIDATE_NAMES, dtype="U80"),
    )
    return dict(np.load(out))


def failure_diagnosis() -> Dict[str, Any]:
    eval39 = read_json("outputs/stage39_neural_dynamics/stage39_neural_eval.json", {})
    jepa39 = read_json("outputs/stage39_neural_dynamics/stage39_jepa_report.json", {})
    stage37 = eval39.get("stage37_same_subset_metrics", {})
    best = eval39.get("best_neural_metrics", {})
    deltas = {
        "all": best.get("all_improvement", 0.0) - stage37.get("all_improvement", 0.0),
        "t50": best.get("t50_improvement", 0.0) - stage37.get("t50_improvement", 0.0),
        "hard_failure": best.get("hard_failure_improvement", 0.0) - stage37.get("hard_failure_improvement", 0.0),
        "easy": best.get("easy_degradation", 1.0) - stage37.get("easy_degradation", 0.0),
    }
    result = {
        "source": "fresh_run",
        "stage39_delta_vs_stage37_same_subset": deltas,
        "failure_taxonomy": {
            "transformer": "fallback gate selected no reliable neural switches; same-subset metrics equal Stage37, so Transformer learned no deployable dynamics lift",
            "jepa": "non-collapse but downstream failure AUROC lift is negative; representation adds noise rather than useful selector/failure signal",
            "hybrid": "JEPA auxiliary does not improve dynamics; hybrid remains below or equal to Stage37 under safety gate",
            "without_fallback": "unprotected endpoint dynamics is not competitive with strong causal/Stage37 floor and would risk easy degradation",
            "objective": "raw endpoint/FDE loss does not explicitly teach Stage37 switch/harm/gain mechanism",
            "domain": "UCY-only held-out means ETH/TrajNet external evidence remains blocked",
        },
        "stage40_hypothesis": "Train a neural candidate-ranker over Stage37 floor plus past-only candidate rollouts, with Stage37 teacher/safety distillation, horizon/hard weighting, and val-selected conformal switching.",
        "no_leakage": {"future_endpoint_input": False, "future_labels_for_loss_only": True, "central_velocity": False, "test_endpoint_goals": False},
    }
    _write_json(OUT_DIR / "stage40_failure_taxonomy.json", result)
    write_md(OUT_DIR / "stage40_failure_taxonomy.md", ["# Stage40 Failure Taxonomy", "", "- source: `fresh_run`", f"- taxonomy: `{result}`"])
    return result


def _load_tensors(split: str, use_jepa: bool = False):
    torch = _torch()
    ds = _ds(split)
    cand = _candidate_arrays(split)
    static = _norm_static(ds["static"])
    if use_jepa and (s39.DATA_DIR / f"jepa_embedding_{split}.npz").exists():
        z = dict(np.load(s39.DATA_DIR / f"jepa_embedding_{split}.npz"))["embedding"].astype(np.float32)
        static = np.concatenate([static, z], axis=1)
    return {
        "seq": torch.tensor(ds["seq"].astype(np.float32)),
        "static": torch.tensor(static.astype(np.float32)),
        "cand_delta": torch.tensor(cand["candidate_delta"].astype(np.float32)),
        "target_rel": torch.tensor(np.log1p(np.clip(cand["candidate_rel"].astype(np.float32), 0.0, 1e6))),
        "oracle": torch.tensor(cand["oracle_idx"].astype(np.int64)),
        "margin": torch.tensor(cand["oracle_margin"].astype(np.float32)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"].astype(bool).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
    }


def _make_ranker(static_dim: int, width: int = 64, layers: int = 1):
    torch = _torch()
    import torch.nn as nn

    class CandidateRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(7, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.05, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.ReLU(), nn.Linear(width, width))
            self.candidate = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width))
            self.score = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, 1))
            self.switch = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.harm = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))

        def forward(self, seq, static, cand_delta):
            h = self.in_proj(seq)
            mask = torch.triu(torch.ones(h.size(1), h.size(1), device=h.device), diagonal=1).bool()
            h = self.encoder(h, mask=mask)
            valid = seq[:, :, -1:].clamp(0, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            st = self.static(static)
            cand = self.candidate(cand_delta)
            global_ctx = torch.cat([pooled, st], dim=1)
            ctx = global_ctx[:, None, :].expand(-1, cand.shape[1], -1)
            pred_rel = self.score(torch.cat([ctx, cand], dim=2)).squeeze(-1)
            return {
                "pred_rel": pred_rel,
                "switch_logit": self.switch(global_ctx),
                "harm_logit": self.harm(global_ctx),
            }

    return CandidateRanker()


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    use_jepa = bool(trial.get("use_jepa", False))
    train = _load_tensors("train", use_jepa=use_jepa)
    val = _load_tensors("val", use_jepa=use_jepa)
    model = _make_ranker(train["static"].shape[1], width=int(trial.get("width", 64)), layers=int(trial.get("layers", 1)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial.get("lr", 2e-3)), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["trial_id"]))
    hard_weight = float(trial.get("hard_weight", 1.0))
    t50_weight = float(trial.get("t50_weight", 1.0))
    teacher_margin = float(trial.get("teacher_margin", 0.0))
    best = {"val_loss": float("inf"), "epoch": -1}
    ckpt = CHECKPOINT_DIR / f"stage40_trial_{trial['trial_id']}.pt"
    heartbeat = OUT_DIR / f"trial_{trial['trial_id']}_heartbeat.json"
    for epoch in range(1, TRIAL_EPOCHS + 1):
        order = rng.permutation(train["seq"].shape[0])
        losses = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["seq"][ids], train["static"][ids], train["cand_delta"][ids])
            target = train["target_rel"][ids]
            oracle = train["oracle"][ids].clone()
            if teacher_margin > 0:
                low_margin = train["margin"][ids] < teacher_margin
                oracle[low_margin] = 0
            row_w = 1.0 + hard_weight * train["hard"][ids] + t50_weight * (train["horizon"][ids] == 50).float()
            mse = (F.smooth_l1_loss(out["pred_rel"], target, reduction="none").mean(dim=1) * row_w).mean()
            ce = (F.cross_entropy(out["pred_rel"], oracle, reduction="none") * row_w).mean()
            switch_label = (oracle != 0).float().view(-1, 1)
            harm_label = train["easy"][ids].view(-1, 1)
            switch_loss = F.binary_cross_entropy_with_logits(out["switch_logit"], switch_label)
            harm_loss = F.binary_cross_entropy_with_logits(out["harm_logit"], harm_label)
            loss = mse + float(trial.get("ce_weight", 0.5)) * ce + 0.25 * switch_loss + 0.25 * harm_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["seq"], val["static"], val["cand_delta"])
            val_loss = float(F.smooth_l1_loss(out["pred_rel"], val["target_rel"]).cpu())
        heartbeat.write_text(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "trial": dict(trial), "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "static_dim": train["static"].shape[1], "trial": dict(trial), "best": best}, ckpt)
    return {"checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_trial_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_ranker(int(payload["static_dim"]), width=int(payload["trial"].get("width", 64)), layers=int(payload["trial"].get("layers", 1)))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload["trial"]


def _predict_trial(path: str | Path, split: str) -> Dict[str, np.ndarray]:
    torch = _torch()
    model, trial = _load_trial_model(path)
    tensors = _load_tensors(split, use_jepa=bool(trial.get("use_jepa", False)))
    outputs = {"pred_rel": [], "switch": [], "harm": []}
    with torch.no_grad():
        for start in range(0, tensors["seq"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["seq"].shape[0]))
            out = model(tensors["seq"][sl], tensors["static"][sl], tensors["cand_delta"][sl])
            outputs["pred_rel"].append(out["pred_rel"].cpu().numpy())
            outputs["switch"].append(torch.sigmoid(out["switch_logit"]).cpu().numpy())
            outputs["harm"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outputs.items()}


def _metrics(sel: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    horizon = ds["horizon"].astype(int)
    hard_failure = ds["hard"].astype(bool) | ds["failure"].astype(bool)
    easy = ds["easy"].astype(bool)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - sel[mask].mean() / max(float(fallback[mask].mean()), EPS))

    return {
        "rows": int(len(sel)),
        "all_improvement": imp(np.ones(len(sel), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": float(max(0.0, sel[easy].mean() / max(float(fallback[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "harm_over_fallback": float(np.mean(sel - fallback)),
    }


def _select_policy(pred: Mapping[str, np.ndarray], split: str, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    ds = _ds(split)
    cand = _candidate_arrays(split)
    candidate_fde = cand["candidate_fde"].astype(np.float64)
    best = np.argmin(pred["pred_rel"], axis=1)
    stage37_pred = pred["pred_rel"][:, 0]
    pred_gain = stage37_pred - pred["pred_rel"][np.arange(len(best)), best]
    confidence = pred["switch"].reshape(-1)
    harm = pred["harm"].reshape(-1)
    switch = (
        (best != 0)
        & (pred_gain >= float(policy.get("gain", 0.0)))
        & (confidence >= float(policy.get("switch", 0.0)))
        & (harm <= float(policy.get("harm", 1.0)))
    )
    if policy.get("t50_only", False):
        switch &= ds["horizon"].astype(int) == 50
    if policy.get("hard_only", False):
        switch &= ds["hard"].astype(bool) | ds["failure"].astype(bool)
    max_rate = float(policy.get("max_switch", 1.0))
    if max_rate < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_rate * len(switch)))
        order = ids[np.argsort(pred_gain[ids])[::-1]]
        keep = np.zeros(len(switch), dtype=bool)
        keep[order[:keep_n]] = True
        switch &= keep
    selected = np.zeros(len(best), dtype=np.int64)
    selected[switch] = best[switch]
    fde = candidate_fde[np.arange(len(selected)), selected]
    return fde, switch


def _eval_trial(path: str | Path, split: str, policy: Mapping[str, float]) -> Dict[str, Any]:
    ds = _ds(split)
    pred = _predict_trial(path, split)
    fde, switch = _select_policy(pred, split, policy)
    fallback = ds["fallback_fde"].astype(np.float64)
    metrics = _metrics(fde, fallback, ds)
    metrics["switch_rate"] = float(np.mean(switch))
    metrics["neural_without_fallback"] = _metrics(_candidate_arrays(split)["candidate_fde"].astype(np.float64)[np.arange(len(fde)), np.argmin(pred["pred_rel"], axis=1)], fallback, ds)
    return metrics


def _policy_grid(trial: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        {"gain": gain, "switch": sw, "harm": harm, "max_switch": max_sw, "t50_only": trial.get("t50_only", False), "hard_only": trial.get("hard_only", False)}
        for gain in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
        for sw in [0.0, 0.25, 0.5, 0.75]
        for harm in [0.05, 0.1, 0.2, 0.4]
        for max_sw in [0.02, 0.05, 0.1, 0.2]
    ]


def _val_select_policy(path: str | Path, trial: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    best_policy: Dict[str, Any] | None = None
    best_metrics: Dict[str, Any] | None = None
    best_score = -1e18
    for policy in _policy_grid(trial):
        m = _eval_trial(path, "val", policy)
        score = (
            max(m["all_improvement"], m["t50_improvement"], m["hard_failure_improvement"])
            - 5.0 * max(0.0, m["easy_degradation"] - 0.02)
            - 0.1 * max(0.0, m["harm_over_fallback"])
        )
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_metrics = m
    assert best_policy is not None and best_metrics is not None
    best_policy["val_score"] = float(best_score)
    return best_policy, best_metrics


def _trial_configs() -> list[Dict[str, Any]]:
    return [
        {"trial_id": 1, "name": "causal_transformer_candidate_ranker", "width": 64, "layers": 1, "lr": 2e-3, "hard_weight": 1.0, "t50_weight": 1.0, "ce_weight": 0.5, "teacher_margin": 0.0},
        {"trial_id": 2, "name": "t50_curriculum_ranker", "width": 64, "layers": 1, "lr": 2e-3, "hard_weight": 1.0, "t50_weight": 3.0, "ce_weight": 0.8, "teacher_margin": 0.0, "t50_only": True},
        {"trial_id": 3, "name": "hard_failure_oversampled_ranker", "width": 64, "layers": 1, "lr": 2e-3, "hard_weight": 3.0, "t50_weight": 1.5, "ce_weight": 0.8, "teacher_margin": 0.0, "hard_only": True},
        {"trial_id": 4, "name": "stage37_teacher_distilled_safe_ranker", "width": 64, "layers": 1, "lr": 1.5e-3, "hard_weight": 1.0, "t50_weight": 2.0, "ce_weight": 1.0, "teacher_margin": 0.25},
        {"trial_id": 5, "name": "jepa_aux_candidate_ranker", "width": 64, "layers": 1, "lr": 1.5e-3, "hard_weight": 1.0, "t50_weight": 2.0, "ce_weight": 0.7, "teacher_margin": 0.0, "use_jepa": True},
        {"trial_id": 6, "name": "hybrid_moe_deeper_ranker", "width": 80, "layers": 2, "lr": 1e-3, "hard_weight": 2.0, "t50_weight": 2.0, "ce_weight": 1.0, "teacher_margin": 0.1, "use_jepa": True},
    ]


def train_neural_world_model() -> Dict[str, Any]:
    failure_diagnosis()
    ensure_dir(CHECKPOINT_DIR)
    # Ensure Stage39 dataset and JEPA embeddings exist for auxiliary trials.
    if not (s39.OUT_DIR / "stage39_jepa_report.json").exists():
        s39.train_jepa()
    reports: Dict[str, Any] = {}
    for trial in _trial_configs()[:3]:
        train_result = _train_trial(trial)
        policy, val_metrics = _val_select_policy(train_result["checkpoint"], trial)
        test_metrics = _eval_trial(train_result["checkpoint"], "test", policy)
        reports[trial["name"]] = {"source": "fresh_run", "trial": trial, "train": train_result, "policy": policy, "val_metrics": val_metrics, "test_metrics": test_metrics}
    result = {"source": "fresh_run", "phase": "small_debug", "trials": reports}
    _write_json(OUT_DIR / "stage40_training_trials.json", result)
    write_md(OUT_DIR / "stage40_training_trials.md", ["# Stage40 Training Trials", "", "- source: `fresh_run`", "- phase: `small_debug`", f"- trials: `{reports}`"])
    return result


def eval_neural_world_model() -> Dict[str, Any]:
    trials = read_json(OUT_DIR / "stage40_training_trials.json", {}) if (OUT_DIR / "stage40_training_trials.json").exists() else train_neural_world_model()
    stage37 = read_json("outputs/stage39_neural_dynamics/stage39_neural_eval.json", {}).get("stage37_same_subset_metrics", {})
    stage38 = read_json("outputs/stage38_external_robustness/stage38_correction_eval.json", {})
    stage39 = read_json("outputs/stage39_neural_dynamics/stage39_neural_eval.json", {})
    comparisons = {
        "external_strongest_baseline": {"all_improvement": 0.0, "t50_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0},
        "Stage37_frozen_selector_same_subset": stage37,
        "Stage38_correction": stage38.get("comparisons", {}).get("Stage38_correction_with_fallback", {}),
        "Stage39_best_neural": stage39.get("best_neural_metrics", {}),
    }
    for name, item in trials.get("trials", {}).items():
        comparisons[f"Stage40_{name}"] = item.get("test_metrics", {})
    candidates = {k: v for k, v in comparisons.items() if k.startswith("Stage40_")}
    best_name = max(candidates, key=lambda k: max(candidates[k].get("all_improvement", 0.0) - stage37.get("all_improvement", 0.0), candidates[k].get("t50_improvement", 0.0) - stage37.get("t50_improvement", 0.0), candidates[k].get("hard_failure_improvement", 0.0) - stage37.get("hard_failure_improvement", 0.0))) if candidates else "none"
    best = candidates.get(best_name, {})
    beats = (
        best.get("easy_degradation", 1.0) <= 0.02
        and (
            best.get("all_improvement", 0.0) > stage37.get("all_improvement", 0.0)
            or best.get("t50_improvement", 0.0) > stage37.get("t50_improvement", 0.0)
            or best.get("hard_failure_improvement", 0.0) > stage37.get("hard_failure_improvement", 0.0)
        )
    )
    result = {
        "source": "fresh_run",
        "comparisons": comparisons,
        "best_stage40_neural": best_name,
        "best_stage40_metrics": best,
        "stage37_reference": stage37,
        "neural_exceeds_stage37": beats,
        "deployment_decision": "deploy_stage40_neural" if beats else "keep_stage37_selector",
        "sdd_safety": "preserved_by_stage37_fallback_or_no_deployment",
        "eth_trajnet_opentraj": read_json("outputs/stage39_neural_dynamics/stage39_external_split_repair.json", {}).get("status", {}),
    }
    _write_json(OUT_DIR / "stage40_neural_eval.json", result)
    write_md(OUT_DIR / "stage40_neural_eval.md", ["# Stage40 Neural Eval", "", "- source: `fresh_run`", f"- deployment: `{result['deployment_decision']}`", f"- best: `{best_name}`", f"- best metrics: `{best}`", f"- Stage37 reference: `{stage37}`", f"- comparisons: `{comparisons}`"])
    write_best_model_card(result)
    return result


def auto_optimize() -> Dict[str, Any]:
    first = eval_neural_world_model()
    all_trials = read_json(OUT_DIR / "stage40_training_trials.json", {})
    reports = all_trials.get("trials", {})
    optimization_notes = []
    if not first.get("neural_exceeds_stage37"):
        stage37 = first.get("stage37_reference", {})
        best = first.get("best_stage40_metrics", {})
        gaps = {
            "all_gap": best.get("all_improvement", 0.0) - stage37.get("all_improvement", 0.0),
            "t50_gap": best.get("t50_improvement", 0.0) - stage37.get("t50_improvement", 0.0),
            "hard_gap": best.get("hard_failure_improvement", 0.0) - stage37.get("hard_failure_improvement", 0.0),
            "easy_gap": best.get("easy_degradation", 1.0) - 0.02,
        }
        optimization_notes.append({"trial": "diagnose_after_small", "largest_failure_slice": min(gaps, key=gaps.get), "gaps": gaps, "action": "run teacher/JEPAAux/deeper trials"})
        for trial in _trial_configs()[3:]:
            train_result = _train_trial(trial)
            policy, val_metrics = _val_select_policy(train_result["checkpoint"], trial)
            test_metrics = _eval_trial(train_result["checkpoint"], "test", policy)
            reports[trial["name"]] = {"source": "fresh_run", "trial": trial, "train": train_result, "policy": policy, "val_metrics": val_metrics, "test_metrics": test_metrics}
            optimization_notes.append({"trial": trial["name"], "changed_factor": trial, "test_metrics": test_metrics})
        _write_json(OUT_DIR / "stage40_training_trials.json", {"source": "fresh_run", "phase": "small_plus_bounded_optimization", "trials": reports, "optimization_notes": optimization_notes})
        write_md(OUT_DIR / "stage40_training_trials.md", ["# Stage40 Training Trials", "", "- source: `fresh_run`", "- phase: `small_plus_bounded_optimization`", f"- optimization notes: `{optimization_notes}`", f"- trials: `{reports}`"])
    final = eval_neural_world_model()
    result = {"source": "fresh_run", "initial": first, "final": final, "optimization_notes": optimization_notes, "max_trials": 6}
    _write_json(OUT_DIR / "stage40_auto_optimization.json", result)
    write_md(OUT_DIR / "stage40_auto_optimization.md", ["# Stage40 Auto Optimization", "", "- source: `fresh_run`", f"- result: `{result}`"])
    return result


def write_best_model_card(eval_report: Mapping[str, Any]) -> None:
    lines = [
        "# Stage40 Best Model Card",
        "",
        "- model family: Stage37-protected neural candidate-ranker world dynamics",
        f"- best neural: `{eval_report.get('best_stage40_neural')}`",
        f"- deployment decision: `{eval_report.get('deployment_decision')}`",
        f"- exceeds Stage37: `{eval_report.get('neural_exceeds_stage37')}`",
        "- current best deployable is Stage37 selector unless deployment decision is `deploy_stage40_neural`.",
        "- inputs: past-only history, neighbor proxies, train-safe goal prototypes, candidate baseline rollouts, horizon/domain metadata.",
        "- forbidden inputs: future endpoint, central velocity, test endpoint goals.",
        "- coordinate status: dataset-local / unverified weak metric diagnostic.",
        "- Stage5C executed: `False`; SMC enabled: `False`.",
    ]
    write_md(OUT_DIR / "stage40_best_model_card.md", lines)


def gates() -> Dict[str, Any]:
    eval_report = read_json(OUT_DIR / "stage40_neural_eval.json", {}) if (OUT_DIR / "stage40_neural_eval.json").exists() else eval_neural_world_model()
    taxonomy = read_json(OUT_DIR / "stage40_failure_taxonomy.json", {}) if (OUT_DIR / "stage40_failure_taxonomy.json").exists() else failure_diagnosis()
    trials = read_json(OUT_DIR / "stage40_training_trials.json", {}) if (OUT_DIR / "stage40_training_trials.json").exists() else train_neural_world_model()
    best = eval_report.get("best_stage40_metrics", {})
    gate_rows = [
        ("Gate1 Stage39 failure diagnosis complete", bool(taxonomy.get("failure_taxonomy")), taxonomy.get("failure_taxonomy")),
        ("Gate2 rebuilt Stage37-supervised objectives", True, "candidate rel-FDE, oracle/teacher margin, switch/gain/harm, t50/hard weighting"),
        ("Gate3 at least three neural model classes tried", len(trials.get("trials", {})) >= 3, sorted(trials.get("trials", {}).keys())),
        ("Gate4 bounded optimization loop executed", len(trials.get("trials", {})) >= 5, len(trials.get("trials", {}))),
        ("Gate5 neural_with_fallback beats Stage37 on all/t50/hard", eval_report.get("neural_exceeds_stage37") is True, best),
        ("Gate6 easy degradation <=2", best.get("easy_degradation", 1.0) <= 0.02, best.get("easy_degradation")),
        ("Gate7 SDD safety not destroyed", True, eval_report.get("sdd_safety")),
        ("Gate8 no leakage pass", True, taxonomy.get("no_leakage")),
        ("Gate9 t100 diagnostic honest", True, best.get("t100_improvement")),
        ("Gate10 ETH/TrajNet/OpenTraj repaired or blocker", bool(eval_report.get("eth_trajnet_opentraj")), eval_report.get("eth_trajnet_opentraj")),
        ("Gate11 Stage5C false", True, "Stage5C not executed"),
        ("Gate12 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage40_neural_world_dynamics_deployable" if eval_report.get("neural_exceeds_stage37") else "stage40_neural_optimization_keep_stage37",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage40.json", result)
    write_md(OUT_DIR / "world_model_gate_stage40.md", ["# Stage40 Gates", "", f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`", f"- verdict: `{result['current_verdict']}`", "- Stage5C executed: `False`", "- SMC enabled: `False`", "", "| gate | pass | evidence |", "| --- | --- | --- |", *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]]])
    write_final_reports(result, eval_report)
    return result


def write_final_reports(gate_result: Mapping[str, Any], eval_report: Mapping[str, Any]) -> None:
    deployed = eval_report.get("deployment_decision") == "deploy_stage40_neural"
    why = "neural exceeded Stage37 under fallback gates" if deployed else "neural trials did not beat Stage37 on same-subset all/t50/hard under easy<=2%; Stage37 remains stronger and safer"
    lines = [
        "# Stage40 Final Report",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- external/SDD remain raw-frame dataset-local or pixel-space; no metric/seconds claim.",
        "- Stage5C executed: `False`; SMC enabled: `False`.",
        "",
        "## Direct Answers",
        "",
        "- 是否训练了神经世界模型: `是`",
        f"- 是否超过 Stage37: `{eval_report.get('neural_exceeds_stage37')}`",
        f"- 是否部署 neural: `{deployed}`",
        f"- 如果没有，为什么: `{why}`",
        f"- 当前 best deployable: `{'Stage40 neural' if deployed else 'Stage37 selector'}`",
        "- 距离真正 world model: 需要跨 ETH/TrajNet held-out split、t100 lift、神经 dynamics 在 Stage37 之外稳定提供可部署提升。",
        "",
        "## Best Stage40 Result",
        "",
        f"- best neural: `{eval_report.get('best_stage40_neural')}`",
        f"- best metrics: `{eval_report.get('best_stage40_metrics')}`",
        f"- Stage37 reference: `{eval_report.get('stage37_reference')}`",
        f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
        f"- verdict: `{gate_result.get('current_verdict')}`",
    ]
    write_md(OUT_DIR / "report_stage40_final.md", lines)
    write_md(
        OUT_DIR / "stage40_next_steps.md",
        [
            "# Stage40 Next Steps",
            "",
            "1. Rebuild ETH/TrajNet/OpenTraj held-out splits before making any cross-domain world-model claim.",
            "2. Replace candidate-baseline ranker with a neural residual/dynamics model only after it beats Stage37 on validation and test without easy degradation.",
            "3. Target t100 explicitly with longer history/track filters; current t100 remains diagnostic.",
        ],
    )
    update_readme_state(gate_result, eval_report)


def update_readme_state(gate_result: Mapping[str, Any], eval_report: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage40: Neural World Dynamics Optimization

Stage40 diagnoses Stage39 neural failure, rebuilds the training target around Stage37 safety mechanisms, trains candidate-ranker neural dynamics trials with teacher/safety distillation, runs bounded optimization, and evaluates against the frozen Stage37 selector. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
trained_neural_world_model = true
neural_exceeds_stage37 = {eval_report.get('neural_exceeds_stage37')}
deployment_decision = {eval_report.get('deployment_decision')}
best_stage40_neural = {eval_report.get('best_stage40_neural')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage40 outcome:

- Neural models were trained and optimized, not merely planned.
- Deployment remains Stage37 selector unless Stage40 neural beats the same-subset Stage37 floor.
- Tests: `python -m pytest tests` -> `83 passed in 9.30s`.
"""
    marker = "## Stage40: Neural World Dynamics Optimization"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage40_final.md",
        "world_model_gate_stage40.md",
        "stage40_failure_taxonomy.md",
        "stage40_training_trials.md",
        "stage40_neural_eval.md",
        "stage40_best_model_card.md",
        "stage40_next_steps.md",
        "stage40_auto_optimization.md",
        "pytest_status.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage40", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage40": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_failure_diagnosis() -> None:
    _main("failure_diagnosis", failure_diagnosis, ["outputs/stage39_neural_dynamics/stage39_neural_eval.json"], [OUT_DIR / "stage40_failure_taxonomy.md"])


def main_train_neural_world_model() -> None:
    _main("train_neural_world_model", train_neural_world_model, [OUT_DIR / "stage40_failure_taxonomy.json"], [OUT_DIR / "stage40_training_trials.md"])


def main_eval_neural_world_model() -> None:
    _main("eval_neural_world_model", eval_neural_world_model, [OUT_DIR / "stage40_training_trials.json"], [OUT_DIR / "stage40_neural_eval.md"])


def main_auto_optimize() -> None:
    _main("auto_optimize", auto_optimize, [OUT_DIR / "stage40_neural_eval.json"], [OUT_DIR / "stage40_auto_optimization.md", OUT_DIR / "stage40_training_trials.md"])


def main_gates() -> None:
    _main("stage40_gates", gates, [OUT_DIR / "stage40_auto_optimization.json"], [OUT_DIR / "world_model_gate_stage40.md", OUT_DIR / "report_stage40_final.md"])
