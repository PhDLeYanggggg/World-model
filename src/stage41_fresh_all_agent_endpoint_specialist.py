from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as aa
from src import stage41_breakthrough as s41
from src import stage41_fresh_confirmation as fresh


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
CHECKPOINT_DIR = fresh.CHECKPOINT_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
EPS = 1e-6
THREADS = 4
BATCH = 512
EPOCHS = 5
SEED = 4193
DOMAINS = ["ETH_UCY", "TrajNet", "UCY"]


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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
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
    torch = aa._torch()
    torch.set_num_threads(THREADS)
    return torch


def _domain_onehot(domain: np.ndarray) -> np.ndarray:
    out = np.zeros((len(domain), len(DOMAINS)), dtype=np.float32)
    d = domain.astype(str)
    for i, name in enumerate(DOMAINS):
        out[:, i] = d == name
    return out


def _ds(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"all_agent_{split}.npz"
    if not path.exists() or not (DATA_DIR / "normalization.npz").exists():
        fresh.build_source_rotation_split()
        with fresh._ProtoPatch() as patched:
            patched.build_stratified_all_agent_dataset()
    return fresh._fresh_ds(split)


def _norm_static(ds: Mapping[str, np.ndarray]) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    static = ((ds["static"].astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)
    return np.concatenate([static, _domain_onehot(ds["domain"].astype(str))], axis=1).astype(np.float32)


def _load_tensors(split: str):
    torch = _torch()
    ds = _ds(split)
    candidate_rel = np.log1p(np.clip(ds["candidate_fde"].astype(np.float32) / np.maximum(ds["floor_fde"].astype(np.float32)[:, None], EPS), 0.0, 1e6))
    floor = ds["floor_fde"].astype(np.float32)
    oracle_gain = np.maximum(0.0, floor - ds["candidate_fde"].astype(np.float32).min(axis=1))
    return {
        "agent_tokens": torch.tensor(ds["agent_tokens"].astype(np.float32)),
        "agent_mask": torch.tensor(ds["agent_mask"].astype(bool)),
        "static": torch.tensor(_norm_static(ds)),
        "cand_delta": torch.tensor(ds["cand_delta"].astype(np.float32)),
        "target_delta": torch.tensor(ds["target_delta"].astype(np.float32)),
        "candidate_rel": torch.tensor(candidate_rel.astype(np.float32)),
        "oracle": torch.tensor(ds["oracle_idx"].astype(np.int64)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"].astype(bool).astype(np.float32)),
        "failure": torch.tensor(ds["failure"].astype(bool).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
        "oracle_gain": torch.tensor(oracle_gain.astype(np.float32)),
        "domain": ds["domain"].astype(str),
        "raw": ds,
    }


def _make_model(static_dim: int, width: int = 88, dropout: float = 0.08):
    torch = _torch()
    import torch.nn as nn

    class FreshAllAgentEndpoint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, width))
            self.role = nn.Embedding(aa.MAX_AGENTS, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.candidate = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            self.endpoint = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
            self.endpoint_risk = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.candidate_score = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
            self.gain = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.harm = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.failure = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.physical = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, agent_tokens, agent_mask, static, cand_delta):
            b, a, t, _ = agent_tokens.shape
            x = self.temporal(agent_tokens) + self.role.weight[:a][None, :, None, :]
            valid_time = agent_tokens[..., 6].clamp(0, 1)
            agent_emb = (x * valid_time[..., None]).sum(dim=2) / valid_time.sum(dim=2, keepdim=True).clamp_min(1.0)
            agent_h = self.agent_encoder(agent_emb)
            valid_agent = agent_mask[:, :, None].float()
            pooled = (agent_h * valid_agent).sum(dim=1) / valid_agent.sum(dim=1).clamp_min(1.0)
            ctx = self.ctx(torch.cat([pooled, self.static(static)], dim=1))
            cand = self.candidate(cand_delta)
            ctx_rep = ctx[:, None, :].expand(-1, cand.shape[1], -1)
            return {
                "endpoint_delta": self.endpoint(ctx),
                "endpoint_risk": self.endpoint_risk(ctx).squeeze(-1),
                "candidate_score": self.candidate_score(torch.cat([ctx_rep, cand], dim=2)).squeeze(-1),
                "gain_logit": self.gain(ctx).squeeze(-1),
                "harm_logit": self.harm(ctx).squeeze(-1),
                "failure_logit": self.failure(ctx).squeeze(-1),
                "physical_logit": self.physical(ctx).squeeze(-1),
            }

    return FreshAllAgentEndpoint()


TRIALS = [
    {"name": "fresh_all_agent_balanced_endpoint", "width": 80, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 2.0, "t50_w": 3.0, "t100_w": 2.0, "endpoint_w": 1.4, "score_w": 1.0, "gain_w": 1.0, "easy_w": 1.6, "seed": 1},
    {"name": "fresh_all_agent_t50_endpoint", "width": 88, "dropout": 0.08, "lr": 7.0e-4, "hard_w": 2.5, "t50_w": 5.0, "t100_w": 1.0, "endpoint_w": 1.8, "score_w": 0.8, "gain_w": 1.4, "easy_w": 1.8, "seed": 2},
    {"name": "fresh_all_agent_long_hard_endpoint", "width": 96, "dropout": 0.10, "lr": 6.0e-4, "hard_w": 4.0, "t50_w": 2.5, "t100_w": 4.0, "endpoint_w": 1.6, "score_w": 1.0, "gain_w": 1.2, "easy_w": 2.0, "seed": 3},
]


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train")
    val = _load_tensors("val")
    model = _make_model(train["static"].shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    rng = np.random.default_rng(SEED + int(trial["seed"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["agent_tokens"].shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["agent_tokens"][ids], train["agent_mask"][ids], train["static"][ids], train["cand_delta"][ids])
            row_w = 1.0 + float(trial["hard_w"]) * train["hard"][ids]
            row_w = row_w + float(trial["t50_w"]) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (train["horizon"][ids] == 100).float()
            endpoint_err = torch.linalg.norm(out["endpoint_delta"] - train["target_delta"][ids], dim=1)
            endpoint = (F.smooth_l1_loss(out["endpoint_delta"], train["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            endpoint_risk = (F.smooth_l1_loss(out["endpoint_risk"], torch.log1p(endpoint_err.detach()), reduction="none") * row_w).mean()
            score = (F.smooth_l1_loss(out["candidate_score"], train["candidate_rel"][ids], reduction="none").mean(dim=1) * row_w).mean()
            ce = (F.cross_entropy(-out["candidate_score"], train["oracle"][ids], reduction="none") * row_w).mean()
            gain_target = (train["oracle_gain"][ids] > 0.02).float()
            gain = F.binary_cross_entropy_with_logits(out["gain_logit"], gain_target)
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], train["easy"][ids])
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], train["failure"][ids])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], 1.0 - train["failure"][ids])
            loss = (
                float(trial["endpoint_w"]) * endpoint
                + 0.5 * endpoint_risk
                + float(trial["score_w"]) * score
                + 0.4 * ce
                + float(trial["gain_w"]) * gain
                + float(trial["easy_w"]) * harm
                + 0.25 * failure
                + 0.1 * physical
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["agent_tokens"], val["agent_mask"], val["static"], val["cand_delta"])
            val_loss = float(
                (
                    F.smooth_l1_loss(out["endpoint_delta"], val["target_delta"])
                    + F.smooth_l1_loss(out["candidate_score"], val["candidate_rel"])
                    + F.binary_cross_entropy_with_logits(out["harm_logit"], val["easy"])
                ).cpu()
            )
        best_candidate = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "best": best_candidate}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = best_candidate
            torch.save({"model": model.state_dict(), "trial": dict(trial), "static_dim": train["static"].shape[1], "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_model(int(payload["static_dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, trial


def _predict(path: str | Path, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = _torch()
    model, _trial = _load_model(path)
    tensors = _load_tensors(split)
    outs: Dict[str, list[np.ndarray]] = {k: [] for k in ["endpoint_delta", "endpoint_risk", "candidate_score", "gain", "harm", "failure", "physical"]}
    with torch.no_grad():
        for start in range(0, tensors["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["agent_tokens"].shape[0]))
            out = model(tensors["agent_tokens"][sl], tensors["agent_mask"][sl], tensors["static"][sl], tensors["cand_delta"][sl])
            outs["endpoint_delta"].append(out["endpoint_delta"].cpu().numpy())
            outs["endpoint_risk"].append(out["endpoint_risk"].cpu().numpy())
            outs["candidate_score"].append(out["candidate_score"].cpu().numpy())
            outs["gain"].append(torch.sigmoid(out["gain_logit"]).cpu().numpy())
            outs["harm"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
            outs["failure"].append(torch.sigmoid(out["failure_logit"]).cpu().numpy())
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy())
    pred = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}
    ds = tensors["raw"]
    labels = {
        "floor_fde": ds["floor_fde"].astype(np.float64),
        "candidate_fde": ds["candidate_fde"].astype(np.float64),
        "current_xy": ds["current_xy"].astype(np.float64),
        "future_xy": ds["future_xy"].astype(np.float64),
        "normalizer": ds["normalizer"].astype(np.float64),
        "horizon": ds["horizon"].astype(np.int64),
        "hard": (ds["hard"].astype(bool) | ds["failure"].astype(bool)),
        "easy": ds["easy"].astype(bool),
        "failure": ds["failure"].astype(bool),
        "domain": ds["domain"].astype(str),
        "scene_id": ds["scene_id"].astype(str),
        "source_file": ds["source_file"].astype(str),
    }
    return pred, labels


def _predict_ensemble(paths: Sequence[str | Path], split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    preds: list[Dict[str, np.ndarray]] = []
    labels_ref: Dict[str, np.ndarray] | None = None
    for path in paths:
        pred, labels = _predict(path, split)
        preds.append(pred)
        labels_ref = labels if labels_ref is None else labels_ref
    if not preds or labels_ref is None:
        raise ValueError("fresh all-agent endpoint ensemble requires at least one checkpoint")
    return {k: np.mean([p[k] for p in preds], axis=0).astype(np.float32) for k in preds[0].keys()}, labels_ref


def _endpoint_fde(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    xy = labels["current_xy"] + pred["endpoint_delta"].astype(np.float64) * labels["normalizer"][:, None]
    return np.linalg.norm(xy - labels["future_xy"], axis=1)


def _metric(selected: np.ndarray, fallback: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> Dict[str, Any]:
    return fresh._metric_from_labels(selected, fallback, labels, switch)


def _apply_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fallback = labels["floor_fde"].astype(np.float64)
    endpoint_fde = _endpoint_fde(pred, labels)
    score = pred["candidate_score"].astype(np.float64)
    best = np.argmin(score, axis=1)
    cand_gain = score[:, 0] - score[np.arange(len(best)), best]
    endpoint_gain = score[:, 0] - pred["endpoint_risk"].astype(np.float64)
    selected = fallback.copy()
    switch = np.zeros(len(fallback), dtype=bool)
    source = np.zeros(len(fallback), dtype=np.int16)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        if not np.any(mask):
            continue
        cand_switch = (
            mask
            & (best != 0)
            & (cand_gain >= float(params.get("candidate_gain_min", 0.0)))
            & (pred["gain"] >= float(params.get("gain_prob_min", 0.0)))
            & (pred["harm"] <= float(params.get("harm_prob_max", 1.0)))
            & (pred["physical"] >= float(params.get("physical_prob_min", 0.0)))
        )
        endpoint_switch = (
            mask
            & (endpoint_gain >= float(params.get("endpoint_gain_min", 0.0)))
            & (pred["endpoint_risk"] <= float(params.get("endpoint_risk_max", 1e9)))
            & (pred["gain"] >= float(params.get("gain_prob_min", 0.0)))
            & (pred["harm"] <= float(params.get("harm_prob_max", 1.0)))
            & (pred["physical"] >= float(params.get("physical_prob_min", 0.0)))
        )
        if params.get("hard_only", False):
            cand_switch &= hard
            endpoint_switch &= hard
        if params.get("easy_block", True):
            cand_switch &= ~easy
            endpoint_switch &= ~easy
        local_switch = cand_switch | endpoint_switch
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch <= 0:
            local_switch[:] = False
        elif max_switch < 1.0 and np.any(local_switch):
            ids = np.where(local_switch)[0]
            local_gain = np.maximum(cand_gain, endpoint_gain)
            keep_n = max(1, int(max_switch * int(np.sum(mask))))
            keep = np.zeros(len(local_switch), dtype=bool)
            keep[ids[np.argsort(local_gain[ids])[::-1][:keep_n]]] = True
            local_switch &= keep
        cand_sel = local_switch & cand_switch
        ep_sel = local_switch & endpoint_switch & ((~cand_switch) | (pred["endpoint_risk"] <= score[np.arange(len(best)), best]))
        selected[cand_sel] = labels["candidate_fde"][np.where(cand_sel)[0], best[cand_sel]]
        source[cand_sel] = best[cand_sel].astype(np.int16)
        selected[ep_sel] = endpoint_fde[ep_sel]
        source[ep_sel] = -1
        switch |= cand_sel | ep_sel
    return selected, switch, source


def _policy_grid() -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    # Bounded hypothesis grid: broad enough to test candidate-vs-endpoint,
    # hard-only, and easy-safe switching, but small enough that validation
    # selection is not the runtime bottleneck.
    for candidate_gain_min in [0.0, 0.03]:
        for endpoint_gain_min in [-0.01, 0.01]:
            for endpoint_risk_max in [0.08, 0.16, 0.35]:
                for harm_prob_max in [0.10, 0.25]:
                    for gain_prob_min in [0.0, 0.35]:
                        for max_switch in [0.0, 0.08, 0.20, 0.50]:
                            base = {
                                "candidate_gain_min": candidate_gain_min,
                                "endpoint_gain_min": endpoint_gain_min,
                                "endpoint_risk_max": endpoint_risk_max,
                                "harm_prob_max": harm_prob_max,
                                "gain_prob_min": gain_prob_min,
                                "physical_prob_min": 0.0,
                                "max_switch": max_switch,
                                "easy_block": True,
                            }
                            out.append({**base, "hard_only": False})
                            out.append({**base, "hard_only": True})
    return out


GRID = _policy_grid()


def _score(metrics: Mapping[str, Any]) -> float:
    max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 1.8 * float(metrics.get("t50_improvement", 0.0))
        + 0.8 * float(metrics.get("t100_improvement", 0.0))
        + 1.4 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 45.0 * max(0.0, max_domain_easy - 0.02)
        - 0.2 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
    )


def _fit_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy = {"type": "fresh_all_agent_endpoint_domain_horizon_policy", "slices": {}}
    diagnostics: Dict[str, Any] = {}
    selected = labels["floor_fde"].copy()
    switch = np.zeros(len(selected), dtype=bool)
    source = np.zeros(len(selected), dtype=np.int16)
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            best_params: Dict[str, Any] | None = None
            best_metrics: Dict[str, Any] | None = None
            best_score = 0.0
            masked_labels = {k: v[mask] for k, v in labels.items()}
            masked_pred = {k: v[mask] for k, v in pred.items()}
            for params in GRID:
                local_policy = {"slices": {f"{d}|{h}": params}}
                sel, sw, _src = _apply_policy(masked_pred, masked_labels, {"slices": {f"{d}|{h}": params}})
                metrics = _metric(sel, masked_labels["floor_fde"], masked_labels, sw)
                max_easy = float(metrics.get("easy_degradation", 0.0))
                if metrics.get("all_improvement", 0.0) <= 0.0 or max_easy > 0.02:
                    continue
                score = _score(metrics)
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = metrics
            if best_params is not None:
                policy["slices"][f"{d}|{h}"] = best_params
                local_policy = {"slices": {f"{d}|{h}": best_params}}
                sel, sw, src = _apply_policy({k: v[mask] for k, v in pred.items()}, {k: v[mask] for k, v in labels.items()}, local_policy)
                selected[mask] = sel
                switch[mask] = sw
                source[mask] = src
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "val_score": float(best_score), "val_metrics": best_metrics or {"rows": int(np.sum(mask)), "all_improvement": 0.0}}
    metrics = _metric(selected, labels["floor_fde"], labels, switch)
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))}
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _eval_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any], bootstrap: bool = False) -> Dict[str, Any]:
    selected, switch, source = _apply_policy(pred, labels, policy)
    metrics = _metric(selected, labels["floor_fde"], labels, switch)
    endpoint = _endpoint_fde(pred, labels)
    metrics["endpoint_without_fallback"] = _metric(endpoint, labels["floor_fde"], labels, np.ones(len(endpoint), dtype=bool))
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))}
    if bootstrap:
        ds = {"horizon": labels["horizon"], "hard": labels["hard"], "failure": labels["failure"], "easy": labels["easy"], "domain": labels["domain"], "candidate_fde": labels["candidate_fde"]}
        metrics["all_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"], ds, "all", n=2000)
        metrics["t50_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"], ds, "t50", n=2000)
        metrics["hard_failure_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"], ds, "hard_failure", n=1000)
    return metrics


def _positive_domains(metrics: Mapping[str, Any]) -> int:
    return int(sum(1 for row in (metrics.get("by_domain") or {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0))


def run_fresh_all_agent_endpoint_specialist() -> Dict[str, Any]:
    started = time.perf_counter()
    fresh.build_source_rotation_split()
    if all((DATA_DIR / f"all_agent_{split}.npz").exists() for split in ["train", "val", "test"]):
        dataset_report = read_json(OUT_DIR / "stage41_stratified_dataset.json", {})
    else:
        with fresh._ProtoPatch() as patched:
            dataset_report = patched.build_stratified_all_agent_dataset()
    trials: Dict[str, Any] = {}
    best_name = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    best_paths: list[str] = []
    for trial in TRIALS:
        train = _train_trial(trial)
        pred_val, labels_val = _predict(train["checkpoint"], "val")
        policy, val_metrics = _fit_policy(pred_val, labels_val)
        score = _score(val_metrics)
        trials[trial["name"]] = {"source": train.get("source"), "trial": trial, "train": train, "policy": policy, "val_metrics": val_metrics, "val_score": score}
        if val_metrics.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
            best_name = trial["name"]
            best_score = float(score)
            best_policy = policy
            best_paths = [str(train["checkpoint"])]
    paths = [row["train"]["checkpoint"] for row in trials.values() if Path(str(row["train"]["checkpoint"])).exists()]
    if len(paths) >= 2:
        pred_val, labels_val = _predict_ensemble(paths, "val")
        policy, val_metrics = _fit_policy(pred_val, labels_val)
        score = _score(val_metrics)
        trials["fresh_all_agent_endpoint_ensemble"] = {"source": "fresh_run", "paths": paths, "policy": policy, "val_metrics": val_metrics, "val_score": score}
        if val_metrics.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
            best_name = "fresh_all_agent_endpoint_ensemble"
            best_score = float(score)
            best_policy = policy
            best_paths = paths
    if not best_policy:
        result: Dict[str, Any] = {"source": "not_run", "reason": "no validation-safe fresh all-agent endpoint policy", "trials": trials}
    else:
        if len(best_paths) == 1:
            pred_test, labels_test = _predict(best_paths[0], "test")
        else:
            pred_test, labels_test = _predict_ensemble(best_paths, "test")
        test_metrics = _eval_policy(pred_test, labels_test, best_policy, bootstrap=True)
        positive = _positive_domains(test_metrics)
        max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (test_metrics.get("by_domain") or {}).values()] or [0.0])
        full_pass = bool(
            test_metrics.get("easy_degradation", 1.0) <= 0.02
            and max_domain_easy <= 0.02
            and positive >= 2
            and test_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            and test_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            and test_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
        result = {
            "source": "fresh_run",
            "protocol_status": "fresh_source_rotation_all_agent_endpoint_specialist",
            "best_name": best_name,
            "best_score": best_score,
            "best_policy": best_policy,
            "best_metrics": test_metrics,
            "positive_external_domains": positive,
            "max_domain_easy_degradation": max_domain_easy,
            "neural_exceeds_stage37_by_gate_margin": full_pass,
            "deployment_decision": "fresh_all_agent_endpoint_candidate_needs_independent_acceptance" if full_pass else "diagnostic_keep_m3w_neural_v1_endpoint_candidate",
            "dataset_report": dataset_report,
            "trials": trials,
            "no_leakage": {
                "future_endpoint_input": False,
                "future_endpoint_label_eval_only": True,
                "central_velocity": False,
                "test_endpoint_goals": False,
                "selection_split": "val",
                "test_used_once_for_final_eval": True,
            },
            "caveat": "This uses source-rotation fresh external split and full all-agent neighbor tokens, but it is still dataset-local raw-frame 2.5D and not Stage5C/SMC.",
        }
    _write_json(OUT_DIR / "stage41_fresh_all_agent_endpoint_specialist.json", result)
    write_md(
        OUT_DIR / "stage41_fresh_all_agent_endpoint_specialist.md",
        [
            "# Stage41 Fresh All-Agent Endpoint Specialist",
            "",
            "- source: `fresh_run`",
            f"- protocol status: `{result.get('protocol_status')}`",
            f"- best name: `{result.get('best_name')}`",
            f"- deployment decision: `{result.get('deployment_decision')}`",
            f"- positive external domains: `{result.get('positive_external_domains')}`",
            f"- max domain easy degradation: `{result.get('max_domain_easy_degradation')}`",
            f"- metrics: `{result.get('best_metrics')}`",
            "",
            "Strict claims:",
            "",
            "- true 3D world model: `False`",
            "- foundation world model: `False`",
            "- metric/seconds claim: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
        ],
    )
    _append_ledger("stage41_fresh_all_agent_endpoint_specialist", "ok", started, [DATA_DIR / "all_agent_train.npz"], [OUT_DIR / "stage41_fresh_all_agent_endpoint_specialist.md"])
    return result


def main_fresh_all_agent_endpoint_specialist() -> None:
    run_fresh_all_agent_endpoint_specialist()


if __name__ == "__main__":
    print(json.dumps(_jsonable(run_fresh_all_agent_endpoint_specialist()), indent=2, ensure_ascii=False))
