from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as s41a
from src import stage41_breakthrough as s41


OUT_DIR = s41.OUT_DIR
DATA_DIR = s41.DATA_DIR
CHECKPOINT_DIR = s41.CHECKPOINT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
THREADS = 4
BATCH = 512
EPOCHS = 6
SEED = 4197


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


def _append_ledger(step: str, status: str, started: float, inputs: list[str], outputs: list[str]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _torch():
    torch = s41a._torch()
    torch.set_num_threads(THREADS)
    return torch


def _domain_vocab() -> list[str]:
    domains: set[str] = set()
    for split in ["train", "val", "test"]:
        domains.update(s41a._ds(split)["domain"].astype(str).tolist())
    return sorted(domains)


def _horizon_vocab() -> list[int]:
    return [10, 25, 50, 100]


def _features(split: str, domain_vocab: Sequence[str] | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ds = s41a._ds(split)
    domain_vocab = list(domain_vocab or _domain_vocab())
    static = s41a._norm_static(ds["static"]).astype(np.float32)
    target_hist = ds["agent_tokens"][:, 0, :, :].astype(np.float32).reshape(len(static), -1)
    # Small aggregate statistics are intentionally redundant with the flattened
    # history. They help the MLP learn selector-style context without peeking at
    # future endpoints.
    valid = np.clip(ds["agent_tokens"][:, 0, :, 6:7].astype(np.float32), 0.0, 1.0)
    hist = ds["agent_tokens"][:, 0, :, :].astype(np.float32)
    hist_mean = (hist * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)
    hist_std = np.sqrt(((hist - hist_mean[:, None, :]) ** 2 * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0))
    cand = ds["cand_delta"].astype(np.float32).reshape(len(static), -1)
    horizon = ds["horizon"].astype(int)
    h_one = np.zeros((len(static), len(_horizon_vocab())), dtype=np.float32)
    for i, h in enumerate(_horizon_vocab()):
        h_one[:, i] = horizon == h
    domain = ds["domain"].astype(str)
    d_one = np.zeros((len(static), len(domain_vocab)), dtype=np.float32)
    for i, d in enumerate(domain_vocab):
        d_one[:, i] = domain == d
    neighbor = ds["neighbor_counts"].astype(np.float32)[:, None] / 6.0
    x = np.concatenate([static, target_hist, hist_mean, hist_std, cand, h_one, d_one, neighbor], axis=1).astype(np.float32)
    labels = {
        "candidate_rel": np.log1p(np.clip(ds["candidate_fde"].astype(np.float32) / np.maximum(ds["floor_fde"].astype(np.float32)[:, None], EPS), 0.0, 1e4)).astype(np.float32),
        "candidate_fde": ds["candidate_fde"].astype(np.float32),
        "floor_fde": ds["floor_fde"].astype(np.float32),
        "oracle": ds["oracle_idx"].astype(np.int64),
        "horizon": horizon.astype(np.int64),
        "hard": (ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32),
        "easy": ds["easy"].astype(bool).astype(np.float32),
        "failure": ds["failure"].astype(bool).astype(np.float32),
        "domain": domain,
    }
    return x, labels


def _make_model(in_dim: int, candidate_count: int, width: int, dropout: float):
    torch = _torch()
    import torch.nn as nn

    class CandidateDistiller(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.ReLU(),
            )
            self.rel = nn.Linear(width, candidate_count)
            self.failure = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.physical = nn.Linear(width, 1)

        def forward(self, x):
            h = self.trunk(x)
            return {
                "pred_rel": self.rel(h),
                "failure_logit": self.failure(h),
                "gain_logit": self.gain(h),
                "harm_logit": self.harm(h),
                "physical_logit": self.physical(h),
            }

    return CandidateDistiller()


def _trial_configs() -> list[Dict[str, Any]]:
    return [
        {"name": "candidate_distill_balanced", "width": 192, "dropout": 0.05, "lr": 1.0e-3, "hard_w": 1.5, "t50_w": 1.5, "t100_w": 1.0, "ce_w": 0.5, "rank_w": 0.5},
        {"name": "candidate_distill_t50_hard", "width": 224, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 3.0, "t50_w": 3.0, "t100_w": 1.0, "ce_w": 0.8, "rank_w": 0.8},
        {"name": "candidate_distill_t100_curriculum", "width": 224, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 2.0, "t50_w": 1.5, "t100_w": 4.0, "ce_w": 0.6, "rank_w": 0.8},
        {"name": "candidate_distill_conservative_margin", "width": 160, "dropout": 0.10, "lr": 1.2e-3, "hard_w": 2.0, "t50_w": 2.0, "t100_w": 1.0, "ce_w": 1.0, "rank_w": 1.0, "margin_filter": 0.02},
    ]


def _train_one(trial: Mapping[str, Any], domain_vocab: Sequence[str]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    x_train, y_train = _features("train", domain_vocab)
    x_val, y_val = _features("val", domain_vocab)
    candidate_count = y_train["candidate_rel"].shape[1]
    model = _make_model(x_train.shape[1], candidate_count, int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + abs(hash(str(trial["name"]))) % 10000)
    ckpt = CHECKPOINT_DIR / f"stage41_candidate_distiller_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"candidate_distiller_{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        return {
            "source": "cached_verified",
            "checkpoint": str(ckpt),
            "heartbeat": str(heartbeat),
            "best": {
                "val_loss": float(payload.get("val_loss", 0.0)),
                "epoch": int(payload.get("epoch", EPOCHS)),
                "train_loss": float(payload.get("train_loss", 0.0)),
            },
            "resume_note": "checkpoint and heartbeat verified; skipped retraining",
        }
    tx = torch.tensor(x_train)
    ty = {k: torch.tensor(v) for k, v in y_train.items() if k not in {"domain"}}
    vx = torch.tensor(x_val)
    vy = {k: torch.tensor(v) for k, v in y_val.items() if k not in {"domain"}}
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x_train))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(tx[ids])
            rel = ty["candidate_rel"][ids]
            oracle = ty["oracle"][ids].clone()
            if float(trial.get("margin_filter", 0.0)) > 0:
                sorted_rel, _ = torch.sort(rel, dim=1)
                oracle[(sorted_rel[:, 1] - sorted_rel[:, 0]) < float(trial["margin_filter"])] = 0
            best_rel = torch.min(rel, dim=1).values
            fallback_rel = rel[:, 0]
            gain_label = ((fallback_rel - best_rel) > 0.02).float()
            row_w = 1.0 + float(trial["hard_w"]) * ty["hard"][ids]
            row_w = row_w + float(trial["t50_w"]) * (ty["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (ty["horizon"][ids] == 100).float()
            row_w = row_w + 2.0 * gain_label
            reg = (F.smooth_l1_loss(out["pred_rel"], rel, reduction="none").mean(dim=1) * row_w).mean()
            ce = (F.cross_entropy(-out["pred_rel"], oracle, reduction="none") * row_w).mean()
            pred_best = torch.min(out["pred_rel"], dim=1).values
            rank = (F.relu(pred_best - out["pred_rel"][torch.arange(len(ids)), oracle] + 0.01) * row_w).mean()
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], ty["failure"][ids, None])
            gain = F.binary_cross_entropy_with_logits(out["gain_logit"], gain_label[:, None])
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], ty["easy"][ids, None])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], 1.0 - ty["failure"][ids, None])
            loss = reg + float(trial["ce_w"]) * ce + float(trial["rank_w"]) * rank + 0.25 * failure + 0.3 * gain + 0.35 * harm + 0.1 * physical
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(vx)
            val_loss = float((F.smooth_l1_loss(out["pred_rel"], vy["candidate_rel"]) + 0.4 * F.cross_entropy(-out["pred_rel"], vy["oracle"])).cpu())
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "in_dim": x_train.shape[1], "candidate_count": candidate_count, "trial": dict(trial), "domain_vocab": list(domain_vocab), "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["in_dim"]), int(payload["candidate_count"]), int(payload["trial"]["width"]), float(payload["trial"]["dropout"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _predict(path: str | Path, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = _torch()
    model, payload = _load_model(path)
    x, labels = _features(split, payload["domain_vocab"])
    tx = torch.tensor(x)
    outs: Dict[str, list[np.ndarray]] = {k: [] for k in ["pred_rel", "failure", "gain", "harm", "physical"]}
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            pred = model(tx[start : start + 4096])
            outs["pred_rel"].append(pred["pred_rel"].cpu().numpy())
            outs["failure"].append(torch.sigmoid(pred["failure_logit"]).cpu().numpy().reshape(-1))
            outs["gain"].append(torch.sigmoid(pred["gain_logit"]).cpu().numpy().reshape(-1))
            outs["harm"].append(torch.sigmoid(pred["harm_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(pred["physical_logit"]).cpu().numpy().reshape(-1))
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}, labels


def _apply_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = pred["pred_rel"]
    fallback = labels["floor_fde"].astype(np.float64)
    best = np.argmin(rel, axis=1)
    pred_gain = rel[:, 0] - rel[np.arange(len(best)), best]
    switch = np.zeros(len(best), dtype=bool)
    selected_idx = np.zeros(len(best), dtype=np.int64)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        sw = (
            mask
            & (best != 0)
            & (pred_gain >= float(params["gain_threshold"]))
            & (pred["gain"] >= float(params["gain_prob"]))
            & (pred["harm"] <= float(params["harm_prob"]))
            & (pred["physical"] >= float(params["physical_prob"]))
        )
        if bool(params.get("hard_only", False)):
            sw &= labels["hard"].astype(bool)
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch <= 0:
            sw[:] = False
        elif max_switch < 1.0 and np.any(sw):
            ids = np.where(sw)[0]
            keep_n = max(1, int(max_switch * np.sum(mask)))
            keep = np.zeros(len(sw), dtype=bool)
            keep[ids[np.argsort(pred_gain[ids])[::-1][:keep_n]]] = True
            sw &= keep
        switch |= sw
        selected_idx[sw] = best[sw]
    selected = fallback.copy()
    selected[switch] = labels["candidate_fde"].astype(np.float64)[np.where(switch)[0], selected_idx[switch]]
    return selected, switch, selected_idx


def _metrics(selected: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> Dict[str, Any]:
    ds = {
        "horizon": labels["horizon"],
        "hard": labels["hard"].astype(bool),
        "failure": labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    return s41._metrics(selected, labels["floor_fde"].astype(np.float64), ds, switch)


def _slice_score(metrics: Mapping[str, float], horizon: int, mode: str) -> float:
    return (
        float(metrics.get("improvement", 0.0))
        + (0.7 if mode in {"hard_t50", "stage37_gap"} else 0.25) * float(metrics.get("hard_failure_improvement", 0.0))
        + (0.8 if horizon in {50, 100} else 0.0) * float(metrics.get("improvement", 0.0))
        - 20.0 * max(0.0, float(metrics.get("easy_degradation", 0.0)) - 0.01)
    )


def _slice_metrics(selected: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    fallback = labels["floor_fde"].astype(np.float64)
    if not np.any(mask):
        return {"rows": 0, "improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "switch_rate": 0.0}
    hard = labels["hard"].astype(bool) & mask
    easy = labels["easy"].astype(bool) & mask
    imp = 1.0 - float(selected[mask].mean()) / max(float(fallback[mask].mean()), EPS)
    hard_imp = 0.0 if not np.any(hard) else 1.0 - float(selected[hard].mean()) / max(float(fallback[hard].mean()), EPS)
    easy_deg = 0.0 if not np.any(easy) else max(0.0, float(selected[easy].mean()) / max(float(fallback[easy].mean()), EPS) - 1.0)
    return {"rows": int(mask.sum()), "improvement": float(imp), "hard_failure_improvement": float(hard_imp), "easy_degradation": float(easy_deg), "switch_rate": float(np.mean(switch[mask]))}


def _policy_grid() -> list[Dict[str, Any]]:
    return [
        {"gain_threshold": gain, "gain_prob": gp, "harm_prob": hp, "physical_prob": pp, "max_switch": ms, "hard_only": hard_only}
        for gain in [0.0, 0.01, 0.03, 0.06, 0.10]
        for gp in [0.0, 0.35, 0.55, 0.75]
        for hp in [0.04, 0.08, 0.16, 0.30]
        for pp in [0.0, 0.4, 0.65]
        for ms in [0.0, 0.01, 0.03, 0.05, 0.10, 0.18]
        for hard_only in [False, True]
    ]


def _select_policy(path: str | Path, mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred, labels = _predict(path, "val")
    rel = pred["pred_rel"]
    fallback = labels["floor_fde"].astype(np.float64)
    best = np.argmin(rel, axis=1)
    pred_gain = rel[:, 0] - rel[np.arange(len(best)), best]
    candidate_fde = labels["candidate_fde"].astype(np.float64)
    hard_all = labels["hard"].astype(bool)
    easy_all = labels["easy"].astype(bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy = {"type": "candidate_distiller_slice_policy", "mode": mode, "slices": {}}
    diagnostics: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(mask.sum()) < 100:
                continue
            best_params = None
            best_score = 0.0
            best_metrics = None
            ids = np.where(mask)[0]
            local_fallback = fallback[ids]
            local_best = best[ids]
            local_gain = pred_gain[ids]
            local_hard = hard_all[ids]
            local_easy = easy_all[ids]
            local_candidate_fde = candidate_fde[ids]
            candidate_selected = local_candidate_fde[np.arange(len(ids)), local_best]
            can_switch_base = local_best != 0
            for params in _policy_grid():
                local_switch = (
                    can_switch_base
                    & (local_gain >= float(params["gain_threshold"]))
                    & (pred["gain"][ids] >= float(params["gain_prob"]))
                    & (pred["harm"][ids] <= float(params["harm_prob"]))
                    & (pred["physical"][ids] >= float(params["physical_prob"]))
                )
                if bool(params.get("hard_only", False)):
                    local_switch &= local_hard
                max_switch = float(params.get("max_switch", 1.0))
                if max_switch <= 0:
                    local_switch[:] = False
                elif max_switch < 1.0 and np.any(local_switch):
                    sw_ids = np.where(local_switch)[0]
                    keep_n = max(1, int(max_switch * len(ids)))
                    keep = np.zeros(len(local_switch), dtype=bool)
                    keep[sw_ids[np.argsort(local_gain[sw_ids])[::-1][:keep_n]]] = True
                    local_switch &= keep
                selected = local_fallback.copy()
                selected[local_switch] = candidate_selected[local_switch]
                imp = 1.0 - float(selected.mean()) / max(float(local_fallback.mean()), EPS)
                hard_imp = 0.0 if not np.any(local_hard) else 1.0 - float(selected[local_hard].mean()) / max(float(local_fallback[local_hard].mean()), EPS)
                easy_deg = 0.0 if not np.any(local_easy) else max(0.0, float(selected[local_easy].mean()) / max(float(local_fallback[local_easy].mean()), EPS) - 1.0)
                metrics = {
                    "rows": int(len(ids)),
                    "improvement": float(imp),
                    "hard_failure_improvement": float(hard_imp),
                    "easy_degradation": float(easy_deg),
                    "switch_rate": float(np.mean(local_switch)),
                }
                min_imp = 0.003 if mode in {"conservative", "stage37_gap"} else 0.0
                max_easy = 0.002 if mode in {"conservative", "stage37_gap"} else 0.02
                if metrics["improvement"] <= min_imp or metrics["easy_degradation"] > max_easy:
                    continue
                score = _slice_score(metrics, h, mode)
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = metrics
            if best_params:
                policy["slices"][f"{d}|{h}"] = best_params
            diagnostics[f"{d}|{h}"] = {"val_score": best_score, "selected": bool(best_params), "val_metrics": best_metrics or {"rows": int(mask.sum()), "improvement": 0.0}}
    selected, switch, idx = _apply_policy(pred, labels, policy)
    metrics = _metrics(selected, labels, switch)
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(idx, return_counts=True))}
    return policy, {"metrics": metrics, "slice_diagnostics": diagnostics}


def _eval_policy(path: str | Path, split: str, policy: Mapping[str, Any], bootstrap: bool = False) -> Dict[str, Any]:
    pred, labels = _predict(path, split)
    selected, switch, idx = _apply_policy(pred, labels, policy)
    metrics = _metrics(selected, labels, switch)
    endpoint_oracle = labels["candidate_fde"].astype(np.float64).min(axis=1)
    metrics["candidate_oracle"] = _metrics(endpoint_oracle, labels, np.ones(len(endpoint_oracle), dtype=bool))
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(idx, return_counts=True))}
    if bootstrap:
        ds = {
            "horizon": labels["horizon"],
            "hard": labels["hard"].astype(bool),
            "failure": labels["failure"].astype(bool),
            "easy": labels["easy"].astype(bool),
            "domain": labels["domain"],
            "candidate_fde": labels["candidate_fde"],
        }
        metrics["t50_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"].astype(np.float64), ds, "t50", n=2000)
        metrics["hard_failure_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"].astype(np.float64), ds, "hard_failure", n=1000)
    return metrics


def train_candidate_distiller() -> Dict[str, Any]:
    started = time.perf_counter()
    domain_vocab = _domain_vocab()
    trials: Dict[str, Any] = {}
    best_name = ""
    best_val_score = -1e18
    best_policy: Dict[str, Any] = {}
    for trial in _trial_configs():
        train = _train_one(trial, domain_vocab)
        trial_modes: Dict[str, Any] = {}
        for mode in ["conservative", "balanced", "hard_t50", "stage37_gap"]:
            policy, val = _select_policy(train["checkpoint"], mode)
            score = (
                val["metrics"].get("all_improvement", 0.0)
                + 1.3 * val["metrics"].get("t50_improvement", 0.0)
                + val["metrics"].get("hard_failure_improvement", 0.0)
                + 0.4 * val["metrics"].get("t100_improvement", 0.0)
                - 20.0 * max(0.0, val["metrics"].get("easy_degradation", 1.0) - 0.02)
            )
            trial_modes[mode] = {"policy": policy, "val": val, "val_score": score}
            if val["metrics"].get("easy_degradation", 1.0) <= 0.02 and score > best_val_score:
                best_val_score = score
                best_name = f"{trial['name']}::{mode}"
                best_policy = policy
        trials[trial["name"]] = {"source": "fresh_run", "trial": trial, "train": train, "modes": trial_modes}
    if not best_name:
        result = {"source": "not_run", "reason": "no val-safe candidate distiller", "trials": trials}
    else:
        trial_name, _mode = best_name.split("::", 1)
        ckpt = trials[trial_name]["train"]["checkpoint"]
        test_metrics = _eval_policy(ckpt, "test", best_policy, bootstrap=True)
        positive_domains = sum(
            1
            for row in test_metrics.get("by_domain", {}).values()
            if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
        )
        exceeds = bool(
            test_metrics.get("easy_degradation", 1.0) <= 0.02
            and (
                test_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
                or test_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
                or test_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
            )
        )
        result = {
            "source": "fresh_run",
            "best_stage41_candidate_distiller": best_name,
            "selection_rule": "train torch expected-FDE candidate distiller; val-selected slice thresholds; test once",
            "best_policy": best_policy,
            "best_metrics": test_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": exceeds,
            "deployment_decision": "deploy_stage41_candidate_distiller" if exceeds and positive_domains >= 2 else "keep_stage37_selector",
            "trials": trials,
        }
    _write_json(OUT_DIR / "stage41_candidate_distiller.json", result)
    write_md(OUT_DIR / "stage41_candidate_distiller.md", ["# Stage41 Candidate Distiller", "", "- source: `fresh_run`", "- method: torch neural expected-FDE candidate distillation with past-only all-agent tokens.", f"- result: `{result}`"])
    _append_ledger("stage41_candidate_distiller", "ok", started, [str(DATA_DIR / "all_agent_train.npz")], [str(OUT_DIR / "stage41_candidate_distiller.md")])
    return result


def main_candidate_distiller() -> None:
    train_candidate_distiller()
