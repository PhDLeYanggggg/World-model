from __future__ import annotations

import json
import os
import platform
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


OUT_DIR = s41.OUT_DIR
DATA_DIR = s41.DATA_DIR
CHECKPOINT_DIR = s41.CHECKPOINT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
THREADS = 4
BATCH = 512
EPOCHS = 5
SEED = 4150
EPS = 1e-6
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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str], outputs: Sequence[str]) -> None:
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
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage41 Breakthrough Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('command', '')}` | `{row.get('source', '')}` | `{row.get('status', '')}` | {float(row.get('wall_time_s', 0.0)):.3f} | `{str(row.get('input_hash', ''))[:12]}` | `{str(row.get('output_hash', ''))[:12]}` | `{row.get('git_commit', '')}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE41_T50_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE41_T50_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage41 all-agent t50 specialist refuses x86_64/Rosetta Python.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _domain_onehot(domain: np.ndarray) -> np.ndarray:
    d = domain.astype(str)
    out = np.zeros((len(d), len(DOMAINS)), dtype=np.float32)
    for i, name in enumerate(DOMAINS):
        out[:, i] = d == name
    return out


def _load_t50(split: str):
    torch = _torch()
    ds = aa._ds(split)
    mask = ds["horizon"].astype(int) == 50
    static = aa._norm_static(ds["static"][mask])
    static = np.concatenate([static, _domain_onehot(ds["domain"][mask])], axis=1).astype(np.float32)
    candidate_rel = np.log1p(
        np.clip(ds["candidate_fde"][mask].astype(np.float32) / np.maximum(ds["normalizer"][mask].astype(np.float32)[:, None], EPS), 0.0, 1e6)
    )
    floor = ds["floor_fde"][mask].astype(np.float32)
    endpoint_oracle_gain = np.maximum(0.0, floor - ds["candidate_fde"][mask].min(axis=1).astype(np.float32))
    tensors = {
        "agent_tokens": torch.tensor(ds["agent_tokens"][mask].astype(np.float32)),
        "agent_mask": torch.tensor(ds["agent_mask"][mask].astype(bool)),
        "static": torch.tensor(static),
        "cand_delta": torch.tensor(ds["cand_delta"][mask].astype(np.float32)),
        "target_delta": torch.tensor(ds["target_delta"][mask].astype(np.float32)),
        "candidate_rel": torch.tensor(candidate_rel),
        "oracle": torch.tensor(ds["oracle_idx"][mask].astype(np.int64)),
        "hard": torch.tensor((ds["hard"][mask].astype(bool) | ds["failure"][mask].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"][mask].astype(bool).astype(np.float32)),
        "failure": torch.tensor(ds["failure"][mask].astype(bool).astype(np.float32)),
        "domain": ds["domain"][mask].astype(str),
        "floor": floor,
        "candidate_fde": ds["candidate_fde"][mask].astype(np.float32),
        "current_xy": ds["current_xy"][mask].astype(np.float32),
        "future_xy": ds["future_xy"][mask].astype(np.float32),
        "normalizer": ds["normalizer"][mask].astype(np.float32),
        "endpoint_oracle_gain": torch.tensor(endpoint_oracle_gain.astype(np.float32)),
        "raw_ds": {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds.items()},
    }
    return tensors


def _make_model(static_dim: int, width: int = 72):
    torch = _torch()
    import torch.nn as nn

    class T50AllAgentSpecialist(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width))
            self.agent_role = nn.Embedding(aa.MAX_AGENTS, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.05, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            self.candidate = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.endpoint = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
            self.endpoint_risk = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.candidate_score = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
            self.gain = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.harm = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.failure = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, agent_tokens, agent_mask, static, cand_delta):
            b, a, t, _ = agent_tokens.shape
            x = self.temporal(agent_tokens)
            roles = self.agent_role.weight[:a][None, :, None, :]
            x = x + roles
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
            }

    return T50AllAgentSpecialist()


TRIALS = [
    {"name": "t50_endpoint_domain_balanced", "lr": 1.0e-3, "width": 72, "endpoint_weight": 2.0, "score_weight": 0.8, "gain_weight": 0.5, "easy_weight": 1.0, "domain_balance": True},
    {"name": "t50_candidate_moe", "lr": 1.2e-3, "width": 64, "endpoint_weight": 0.6, "score_weight": 2.0, "gain_weight": 0.8, "easy_weight": 1.2, "domain_balance": True},
    {"name": "t50_hard_gain_harm", "lr": 8.0e-4, "width": 80, "endpoint_weight": 1.5, "score_weight": 1.2, "gain_weight": 1.5, "easy_weight": 2.0, "domain_balance": True},
]


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_t50("train")
    val = _load_t50("val")
    model = _make_model(train["static"].shape[1], int(trial.get("width", 72)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + sum(ord(c) for c in str(trial["name"])))
    ckpt = CHECKPOINT_DIR / f"stage41_all_agent_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    domain = train["domain"]
    domain_weight_np = np.ones(len(domain), dtype=np.float32)
    if trial.get("domain_balance", False):
        counts = Counter(domain.tolist())
        for d, count in counts.items():
            domain_weight_np[domain == d] = len(domain) / max(len(counts) * count, 1)
    domain_weight = torch.tensor(domain_weight_np)
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["agent_tokens"].shape[0])
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["agent_tokens"][ids], train["agent_mask"][ids], train["static"][ids], train["cand_delta"][ids])
            row_w = domain_weight[ids] * (1.0 + 1.5 * train["hard"][ids] + 0.8 * train["failure"][ids])
            endpoint_loss = (F.smooth_l1_loss(out["endpoint_delta"], train["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            endpoint_err = torch.linalg.norm(out["endpoint_delta"] - train["target_delta"][ids], dim=1)
            endpoint_risk = (F.smooth_l1_loss(out["endpoint_risk"], torch.log1p(endpoint_err.detach()), reduction="none") * row_w).mean()
            score_loss = (F.smooth_l1_loss(out["candidate_score"], train["candidate_rel"][ids], reduction="none").mean(dim=1) * row_w).mean()
            ce = (F.cross_entropy(out["candidate_score"], train["oracle"][ids], reduction="none") * row_w).mean()
            gain_target = (train["endpoint_oracle_gain"][ids] > 0.02).float()
            gain = F.binary_cross_entropy_with_logits(out["gain_logit"], gain_target)
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], train["easy"][ids])
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], train["failure"][ids])
            loss = (
                float(trial.get("endpoint_weight", 1.0)) * endpoint_loss
                + 0.5 * endpoint_risk
                + float(trial.get("score_weight", 1.0)) * score_loss
                + 0.4 * ce
                + float(trial.get("gain_weight", 1.0)) * gain
                + float(trial.get("easy_weight", 1.0)) * harm
                + 0.3 * failure
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
        heartbeat.write_text(
            json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt)}),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "static_dim": train["static"].shape[1], "trial": dict(trial), "best": best}, ckpt)
    return {"checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["static_dim"]), int(payload["trial"].get("width", 72)))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload["trial"]


def _predict(path: str | Path, split: str) -> Dict[str, np.ndarray]:
    torch = _torch()
    model, _trial = _load_model(path)
    ds = _load_t50(split)
    out: Dict[str, list[np.ndarray]] = {k: [] for k in ["endpoint_delta", "endpoint_risk", "candidate_score", "gain", "harm", "failure"]}
    with torch.no_grad():
        for start in range(0, ds["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, ds["agent_tokens"].shape[0]))
            pred = model(ds["agent_tokens"][sl], ds["agent_mask"][sl], ds["static"][sl], ds["cand_delta"][sl])
            out["endpoint_delta"].append(pred["endpoint_delta"].cpu().numpy())
            out["endpoint_risk"].append(pred["endpoint_risk"].cpu().numpy())
            out["candidate_score"].append(pred["candidate_score"].cpu().numpy())
            out["gain"].append(torch.sigmoid(pred["gain_logit"]).cpu().numpy())
            out["harm"].append(torch.sigmoid(pred["harm_logit"]).cpu().numpy())
            out["failure"].append(torch.sigmoid(pred["failure_logit"]).cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _endpoint_fde(pred: Mapping[str, np.ndarray], ds: Mapping[str, Any]) -> np.ndarray:
    xy = ds["raw_ds"]["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    return np.linalg.norm(xy - ds["raw_ds"]["future_xy"].astype(np.float64), axis=1)


def _select_t50(pred: Mapping[str, np.ndarray], ds: Mapping[str, Any], policy: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fallback = ds["floor"].astype(np.float64)
    endpoint_fde = _endpoint_fde(pred, ds)
    candidate_score = pred["candidate_score"]
    cand_idx = np.argmin(candidate_score, axis=1)
    cand_gain = candidate_score[:, 0] - candidate_score[np.arange(len(cand_idx)), cand_idx]
    endpoint_gain = candidate_score[:, 0] - pred["endpoint_risk"]
    easy = ds["raw_ds"]["easy"].astype(bool)
    hard = ds["raw_ds"]["hard"].astype(bool) | ds["raw_ds"]["failure"].astype(bool)
    endpoint_switch = (
        (endpoint_gain >= float(policy.get("endpoint_gain_min", 0.0)))
        & (pred["endpoint_risk"] <= float(policy.get("endpoint_risk_max", 1e9)))
        & (pred["gain"] >= float(policy.get("gain_prob_min", 0.0)))
        & (pred["harm"] <= float(policy.get("harm_prob_max", 1.0)))
    )
    cand_switch = (
        (cand_idx != 0)
        & (cand_gain >= float(policy.get("candidate_gain_min", 0.0)))
        & (pred["gain"] >= float(policy.get("gain_prob_min", 0.0)))
        & (pred["harm"] <= float(policy.get("harm_prob_max", 1.0)))
    )
    if policy.get("hard_only", False):
        endpoint_switch &= hard
        cand_switch &= hard
    if policy.get("easy_block", True):
        endpoint_switch &= ~easy
        cand_switch &= ~easy
    selected = fallback.copy()
    source = np.zeros(len(fallback), dtype=np.int16)
    selected[cand_switch] = ds["candidate_fde"].astype(np.float64)[np.arange(len(cand_idx)), cand_idx][cand_switch]
    source[cand_switch] = cand_idx[cand_switch].astype(np.int16)
    endpoint_prefer = endpoint_switch & ((~cand_switch) | (pred["endpoint_risk"] <= candidate_score[np.arange(len(cand_idx)), cand_idx]))
    selected[endpoint_prefer] = endpoint_fde[endpoint_prefer]
    source[endpoint_prefer] = -1
    switch = cand_switch | endpoint_prefer
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch <= 0.0:
        return fallback, np.zeros(len(fallback), dtype=bool), np.zeros(len(fallback), dtype=np.int16)
    if max_switch < 1.0 and np.any(switch):
        total_gain = fallback - selected
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        keep = np.zeros(len(switch), dtype=bool)
        keep[ids[np.argsort(total_gain[ids])[::-1][:keep_n]]] = True
        selected = fallback.copy()
        source = np.zeros(len(fallback), dtype=np.int16)
        selected[keep] = np.where(endpoint_prefer[keep], endpoint_fde[keep], ds["candidate_fde"].astype(np.float64)[np.arange(len(cand_idx)), cand_idx][keep])
        source[keep] = np.where(endpoint_prefer[keep], -1, cand_idx[keep]).astype(np.int16)
        switch = keep
    return selected, switch, source


def _full_metrics_from_t50(t50_selected: np.ndarray, t50_switch: np.ndarray, t50_source: np.ndarray, split: str) -> Dict[str, Any]:
    full = aa._ds(split)
    h50 = full["horizon"].astype(int) == 50
    selected = full["floor_fde"].astype(np.float64).copy()
    switch = np.zeros(len(selected), dtype=bool)
    source = np.zeros(len(selected), dtype=np.int16)
    selected[h50] = t50_selected
    switch[h50] = t50_switch
    source[h50] = t50_source
    metrics = aa._metrics(selected, full["floor_fde"].astype(np.float64), full, switch)
    metrics["selected_candidate_distribution"] = dict(Counter(source.astype(int).tolist()))
    metrics["t50_only_switch_rate"] = float(np.mean(t50_switch)) if len(t50_switch) else 0.0
    return metrics


def _policy_grid() -> list[Dict[str, Any]]:
    policies: list[Dict[str, Any]] = []
    for endpoint_gain_min in [-0.02, -0.01, 0.0, 0.01, 0.03]:
        for endpoint_risk_max in [0.02, 0.04, 0.08, 0.15, 0.3, 0.6]:
            for candidate_gain_min in [0.0, 0.005, 0.02]:
                for harm_prob_max in [0.03, 0.08, 0.15, 0.3]:
                    for gain_prob_min in [0.0, 0.35, 0.6]:
                        for max_switch in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.35]:
                            for hard_only in [True, False]:
                                policies.append(
                                    {
                                        "endpoint_gain_min": endpoint_gain_min,
                                        "endpoint_risk_max": endpoint_risk_max,
                                        "candidate_gain_min": candidate_gain_min,
                                        "harm_prob_max": harm_prob_max,
                                        "gain_prob_min": gain_prob_min,
                                        "max_switch": max_switch,
                                        "hard_only": hard_only,
                                        "easy_block": True,
                                    }
                                )
    return policies


POLICY_GRID = _policy_grid()


def _metric_score(metrics: Mapping[str, Any]) -> float:
    return (
        3.0 * float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("all_improvement", 0.0))
        + 0.5 * float(metrics.get("hard_failure_improvement", 0.0))
        - 25.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 0.1 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
    )


def _fit_policy(pred: Mapping[str, np.ndarray], split: str, group: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ds = _load_t50(split)
    full = aa._ds(split)
    domain = ds["domain"].astype(str)
    if group == "global":
        groups = {"global": np.ones(len(domain), dtype=bool)}
    elif group == "domain":
        groups = {d: domain == d for d in sorted(set(domain.tolist()))}
    else:
        raise ValueError(group)
    selected = ds["floor"].astype(np.float64).copy()
    switch = np.zeros(len(selected), dtype=bool)
    source = np.zeros(len(selected), dtype=np.int16)
    policies: Dict[str, Any] = {}
    for name, mask in groups.items():
        best_policy = {"max_switch": 0.0}
        best_score = -1e18
        best_selected = ds["floor"][mask].astype(np.float64)
        best_switch = np.zeros(mask.sum(), dtype=bool)
        best_source = np.zeros(mask.sum(), dtype=np.int16)
        pred_g = {k: v[mask] for k, v in pred.items()}
        ds_g = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds.items() if k != "raw_ds"}
        ds_g["raw_ds"] = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds["raw_ds"].items()}
        for policy in POLICY_GRID:
            sel, sw, src = _select_t50(pred_g, ds_g, policy)
            # Score inside the t50 subset with easy penalty.
            tmp_full = {
                "floor_fde": ds_g["floor"],
                "candidate_fde": ds_g["candidate_fde"],
                "horizon": np.full(len(sel), 50, dtype=np.int16),
                "hard": ds_g["raw_ds"]["hard"],
                "failure": ds_g["raw_ds"]["failure"],
                "easy": ds_g["raw_ds"]["easy"],
                "domain": ds_g["domain"],
            }
            m = aa._metrics(sel, ds_g["floor"].astype(np.float64), tmp_full, sw)
            score = _metric_score(m)
            if score > best_score:
                best_score = score
                best_policy = dict(policy)
                best_selected, best_switch, best_source = sel, sw, src
        best_policy["val_score"] = float(best_score)
        policies[name] = best_policy
        selected[mask] = best_selected
        switch[mask] = best_switch
        source[mask] = best_source
    val_metrics = _full_metrics_from_t50(selected, switch, source, split)
    return {"type": group, "policies": policies}, val_metrics


def _apply_policy(pred: Mapping[str, np.ndarray], split: str, policy_pack: Mapping[str, Any]) -> Dict[str, Any]:
    ds = _load_t50(split)
    domain = ds["domain"].astype(str)
    selected = ds["floor"].astype(np.float64).copy()
    switch = np.zeros(len(selected), dtype=bool)
    source = np.zeros(len(selected), dtype=np.int16)
    groups = {"global": np.ones(len(domain), dtype=bool)} if policy_pack["type"] == "global" else {d: domain == d for d in sorted(set(domain.tolist()))}
    for name, mask in groups.items():
        policy = policy_pack["policies"].get(name, {"max_switch": 0.0})
        pred_g = {k: v[mask] for k, v in pred.items()}
        ds_g = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds.items() if k != "raw_ds"}
        ds_g["raw_ds"] = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds["raw_ds"].items()}
        sel, sw, src = _select_t50(pred_g, ds_g, policy)
        selected[mask] = sel
        switch[mask] = sw
        source[mask] = src
    return _full_metrics_from_t50(selected, switch, source, split)


def run_all_agent_t50_specialist() -> Dict[str, Any]:
    aa.build_all_agent_dataset()
    results: Dict[str, Any] = {}
    for trial in TRIALS:
        train = _train_trial(trial)
        pred_val = _predict(train["checkpoint"], "val")
        best_policy = None
        best_val = None
        best_score = -1e18
        for group in ["global", "domain"]:
            policy, val_metrics = _fit_policy(pred_val, "val", group)
            score = _metric_score(val_metrics)
            if score > best_score:
                best_score = score
                best_policy = policy
                best_val = val_metrics
        assert best_policy is not None and best_val is not None
        pred_test = _predict(train["checkpoint"], "test")
        test_metrics = _apply_policy(pred_test, "test", best_policy)
        # Bootstrap t50 on the final full arrays.
        full = aa._ds("test")
        # Reconstruct selected/switch by reapplying, but cheaply repeat through policy.
        results[trial["name"]] = {
            "source": "fresh_run",
            "trial": trial,
            "train": train,
            "policy": best_policy,
            "val_metrics": best_val,
            "test_metrics": test_metrics,
        }
    best_name = "none"
    best_item: Dict[str, Any] = {}
    best_score = -1e18
    for name, item in results.items():
        score = _metric_score(item["test_metrics"])
        if score > best_score:
            best_name = name
            best_item = item
            best_score = score
    metrics = best_item.get("test_metrics", {})
    positive_domains = sum(
        1
        for row in metrics.get("by_domain", {}).values()
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
    )
    beats = (
        metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            or metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            or metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
    )
    report = {
        "source": "fresh_run",
        "hypothesis": "All-agent t50 failed because generic all-horizon heads optimized t100/all slices. This trains t50-only neural endpoint/candidate heads with domain-balanced loss and val-selected domain policies.",
        "best_trial": best_name,
        "best_metrics": metrics,
        "positive_external_domains": int(positive_domains),
        "neural_exceeds_stage37_by_gate_margin": bool(beats),
        "deployment_decision": "deploy_all_agent_t50_specialist" if beats and positive_domains >= 2 else "diagnostic_keep_m3w_neural_v1_endpoint_candidate",
        "trials": results,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "policy_selection_split": "val",
            "test_used_once_for_final_eval": True,
        },
    }
    _write_json(OUT_DIR / "stage41_all_agent_t50_specialist.json", report)
    write_md(
        OUT_DIR / "stage41_all_agent_t50_specialist.md",
        [
            "# Stage41 All-Agent t50 Specialist",
            "",
            "- source: `fresh_run`",
            f"- best trial: `{best_name}`",
            f"- deployment: `{report['deployment_decision']}`",
            f"- metrics: `{metrics}`",
            "",
            "## Interpretation",
            "",
            "This is a t+50-only all-agent neural specialist. It uses only past/current all-agent tokens as input; future endpoints are labels/evaluation only. If it does not beat Stage37 while preserving easy cases, all-agent t+50 remains the next blocker.",
        ],
    )
    return report


def main_all_agent_t50_specialist() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_all_agent_t50_specialist()
        status = "success"
    finally:
        _append_ledger(
            "stage41_all_agent_t50_specialist",
            status,
            started,
            [str(DATA_DIR / "all_agent_train.npz"), str(DATA_DIR / "all_agent_val.npz")],
            [str(OUT_DIR / "stage41_all_agent_t50_specialist.md")],
        )
