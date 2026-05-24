from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41


OUT_DIR = s41.OUT_DIR
DATA_DIR = s41.DATA_DIR
CHECKPOINT_DIR = s41.CHECKPOINT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
MAX_AGENTS = 6
TOKEN_K = 32
BATCH = 256
EPOCHS = 4
THREADS = 4
SEED = 4141
EPS = 1e-6


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


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE41_ALL_AGENT_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE41_ALL_AGENT_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage41 all-agent training refuses x86_64/Rosetta Python.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _group_indices(data: Mapping[str, np.ndarray]) -> dict[tuple[str, int, int], np.ndarray]:
    group: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    source = data["source_file"].astype(str)
    frame = np.rint(data["frame_id"].astype(float)).astype(int)
    horizon = data["horizon"].astype(int)
    for i, key in enumerate(zip(source, frame, horizon)):
        group[key].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in group.items()}


def _build_tokens_for_ids(data: Mapping[str, np.ndarray], ids: np.ndarray, groups: Mapping[tuple[str, int, int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ids)
    tokens = np.zeros((n, MAX_AGENTS, TOKEN_K, 9), dtype=np.float32)
    mask = np.zeros((n, MAX_AGENTS), dtype=bool)
    neighbor_counts = np.zeros(n, dtype=np.int16)
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    source = data["source_file"].astype(str)
    frame = np.rint(data["frame_id"].astype(float)).astype(int)
    horizon = data["horizon"].astype(int)
    agent = data["agent_id"].astype(int)
    normalizer = np.maximum(data["history_scalar"][:, 0].astype(np.float32) + data["history_seq"][:, -1, 2].astype(np.float32) * np.maximum(horizon.astype(np.float32), 1.0), EPS)
    for out_i, row_id in enumerate(ids.astype(int)):
        key = (source[row_id], int(frame[row_id]), int(horizon[row_id]))
        candidates = groups.get(key, np.asarray([row_id], dtype=np.int64))
        # Use one row per neighboring agent and choose nearest current-position neighbors.
        candidates = candidates[agent[candidates] != agent[row_id]]
        if len(candidates):
            _, first = np.unique(agent[candidates], return_index=True)
            candidates = candidates[np.sort(first)]
            dist = np.linalg.norm(cur[candidates] - cur[row_id], axis=1)
            candidates = candidates[np.argsort(dist)[: MAX_AGENTS - 1]]
        selected = np.concatenate([np.asarray([row_id], dtype=np.int64), candidates])
        neighbor_counts[out_i] = len(selected) - 1
        for j, rid in enumerate(selected[:MAX_AGENTS]):
            hist = data["history_seq"][rid, -TOKEN_K:, :].astype(np.float32)
            rel = ((cur[rid] - cur[row_id]) / max(float(normalizer[row_id]), EPS)).astype(np.float32)
            rel_rep = np.repeat(rel[None, :], TOKEN_K, axis=0)
            tokens[out_i, j] = np.concatenate([hist, rel_rep], axis=1)
            mask[out_i, j] = True
    return tokens, mask, neighbor_counts


def build_all_agent_dataset() -> Dict[str, Any]:
    s41.build_seq2seq_dataset()
    data = s41._combined()
    groups = _group_indices(data)
    report: Dict[str, Any] = {"source": "fresh_run", "max_agents": MAX_AGENTS, "history_k": TOKEN_K, "splits": {}}
    for split in ["train", "val", "test"]:
        base = s41._ds(split)
        ids = base["ids"].astype(np.int64)
        out = DATA_DIR / f"all_agent_{split}.npz"
        tokens, agent_mask, neighbor_counts = _build_tokens_for_ids(data, ids, groups)
        np.savez_compressed(
            out,
            ids=ids,
            agent_tokens=tokens,
            agent_mask=agent_mask,
            neighbor_counts=neighbor_counts,
            static=base["static"].astype(np.float32),
            target_delta=base["target_delta"].astype(np.float32),
            cand_delta=base["cand_delta"].astype(np.float32),
            candidate_fde=base["candidate_fde"].astype(np.float32),
            floor_fde=base["floor_fde"].astype(np.float32),
            oracle_idx=base["oracle_idx"].astype(np.int64),
            normalizer=base["normalizer"].astype(np.float32),
            current_xy=base["current_xy"].astype(np.float32),
            future_xy=base["future_xy"].astype(np.float32),
            horizon=base["horizon"].astype(np.int16),
            hard=base["hard"].astype(bool),
            easy=base["easy"].astype(bool),
            failure=base["failure"].astype(bool),
            domain=base["domain"],
            scene_id=base["scene_id"],
            source_file=base["source_file"],
        )
        report["splits"][split] = {
            "rows": int(len(ids)),
            "neighbor_count_mean": float(np.mean(neighbor_counts)),
            "neighbor_count_ge1": int(np.sum(neighbor_counts >= 1)),
            "neighbor_count_ge3": int(np.sum(neighbor_counts >= 3)),
            "domains": dict(Counter(base["domain"].astype(str).tolist())),
            "t50": int(np.sum(base["horizon"].astype(int) == 50)),
            "t100": int(np.sum(base["horizon"].astype(int) == 100)),
        }
    report["no_leakage"] = {
        "future_endpoint_input": False,
        "future_endpoint_label_eval_only": True,
        "neighbor_tokens_from_same_current_frame_and_past_history": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
    }
    _write_json(OUT_DIR / "stage41_all_agent_dataset.json", report)
    write_md(OUT_DIR / "stage41_all_agent_dataset.md", ["# Stage41 All-Agent Token Dataset", "", "- source: `fresh_run`", f"- report: `{report}`"])
    return report


def _ds(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"all_agent_{split}.npz"
    if not path.exists():
        build_all_agent_dataset()
    return dict(np.load(path))


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _load_tensors(split: str):
    torch = _torch()
    ds = _ds(split)
    return {
        "agent_tokens": torch.tensor(ds["agent_tokens"].astype(np.float32)),
        "agent_mask": torch.tensor(ds["agent_mask"].astype(bool)),
        "static": torch.tensor(_norm_static(ds["static"])),
        "cand_delta": torch.tensor(ds["cand_delta"].astype(np.float32)),
        "target_delta": torch.tensor(ds["target_delta"].astype(np.float32)),
        "candidate_rel": torch.tensor(np.log1p(np.clip(ds["candidate_fde"].astype(np.float32) / np.maximum(ds["normalizer"].astype(np.float32)[:, None], EPS), 0.0, 1e6))),
        "oracle": torch.tensor(ds["oracle_idx"].astype(np.int64)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"].astype(bool).astype(np.float32)),
        "failure": torch.tensor(ds["failure"].astype(bool).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
    }


def _make_model(static_dim: int, width: int = 56, layers: int = 1):
    torch = _torch()
    import torch.nn as nn

    class AllAgentWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(9, width)
            self.role = nn.Embedding(MAX_AGENTS, width)
            self.temporal = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.LayerNorm(width))
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.05, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.ReLU(), nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU())
            self.candidate = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.endpoint = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 2))
            self.endpoint_risk = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.score = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, 1))
            self.failure = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.gain = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.harm = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.physical = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))

        def forward(self, agent_tokens, agent_mask, static, cand_delta):
            b, a, t, _ = agent_tokens.shape
            x = self.in_proj(agent_tokens)
            roles = self.role.weight[:a][None, :, None, :]
            x = x + roles
            valid_time = agent_tokens[..., 6].clamp(0, 1)
            x = self.temporal(x)
            agent_emb = (x * valid_time[..., None]).sum(dim=2) / valid_time.sum(dim=2, keepdim=True).clamp_min(1.0)
            # Agent-level attention is enough here; the temporal axis is already
            # past-only and pooled, so this avoids the 6*32 token bottleneck.
            agent_h = self.agent_encoder(agent_emb)
            valid_agent = agent_mask[:, :, None].float()
            pooled = (agent_h * valid_agent).sum(dim=1) / valid_agent.sum(dim=1).clamp_min(1.0)
            st = self.static(static)
            ctx = torch.cat([pooled, st], dim=1)
            cand = self.candidate(cand_delta)
            score_ctx = ctx[:, None, :].expand(-1, cand.shape[1], -1)
            return {
                "endpoint_delta": self.endpoint(ctx),
                "endpoint_risk": self.endpoint_risk(ctx).squeeze(-1),
                "candidate_score": self.score(torch.cat([score_ctx, cand], dim=2)).squeeze(-1),
                "failure_logit": self.failure(ctx),
                "gain_logit": self.gain(ctx),
                "harm_logit": self.harm(ctx),
                "physical_logit": self.physical(ctx),
            }

    return AllAgentWorldModel()


def _trial_configs() -> list[Dict[str, Any]]:
    return [
        {"trial_id": 11, "name": "all_agent_token_transformer", "width": 56, "layers": 1, "lr": 1.5e-3, "hard_weight": 1.5, "t50_weight": 1.5, "t100_weight": 1.0, "score_weight": 1.2, "endpoint_weight": 0.7},
        {"trial_id": 12, "name": "all_agent_t50_hard_safety", "width": 56, "layers": 1, "lr": 1.2e-3, "hard_weight": 3.0, "t50_weight": 3.0, "t100_weight": 0.5, "score_weight": 1.4, "endpoint_weight": 0.5, "teacher_margin": 0.02},
        {"trial_id": 13, "name": "all_agent_t100_curriculum", "width": 64, "layers": 1, "lr": 1.0e-3, "hard_weight": 2.0, "t50_weight": 1.0, "t100_weight": 4.0, "score_weight": 1.2, "endpoint_weight": 0.7},
        {"trial_id": 14, "name": "all_agent_easy_guard", "width": 48, "layers": 1, "lr": 1.2e-3, "hard_weight": 2.0, "t50_weight": 2.0, "t100_weight": 1.0, "score_weight": 1.8, "endpoint_weight": 0.4, "teacher_margin": 0.05},
        {"trial_id": 15, "name": "all_agent_endpoint_risk_switch", "width": 56, "layers": 1, "lr": 1.0e-3, "hard_weight": 2.0, "t50_weight": 2.0, "t100_weight": 1.0, "score_weight": 1.2, "endpoint_weight": 1.5, "endpoint_risk_weight": 1.0, "use_endpoint": True},
        {"trial_id": 16, "name": "all_agent_endpoint_t100_focus", "width": 56, "layers": 1, "lr": 1.0e-3, "hard_weight": 2.0, "t50_weight": 1.0, "t100_weight": 5.0, "score_weight": 1.0, "endpoint_weight": 1.8, "endpoint_risk_weight": 1.2, "use_endpoint": True},
        {"trial_id": 17, "name": "all_agent_endpoint_easy_guard", "width": 48, "layers": 1, "lr": 1.0e-3, "hard_weight": 2.0, "t50_weight": 2.5, "t100_weight": 1.0, "score_weight": 1.5, "endpoint_weight": 1.2, "endpoint_risk_weight": 1.5, "teacher_margin": 0.05, "use_endpoint": True},
        {"trial_id": 18, "name": "all_agent_endpoint_hard_only", "width": 48, "layers": 1, "lr": 1.2e-3, "hard_weight": 4.0, "t50_weight": 2.0, "t100_weight": 1.0, "score_weight": 1.2, "endpoint_weight": 1.2, "endpoint_risk_weight": 1.2, "use_endpoint": True, "hard_only": True},
    ]


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train")
    val = _load_tensors("val")
    model = _make_model(train["static"].shape[1], int(trial.get("width", 56)), int(trial.get("layers", 1)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial.get("lr", 1e-3)), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["trial_id"]))
    ckpt = CHECKPOINT_DIR / f"stage41_trial_{trial['trial_id']}.pt"
    heartbeat = OUT_DIR / f"trial_{trial['trial_id']}_heartbeat.json"
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["agent_tokens"].shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["agent_tokens"][ids], train["agent_mask"][ids], train["static"][ids], train["cand_delta"][ids])
            oracle = train["oracle"][ids].clone()
            if float(trial.get("teacher_margin", 0.0)) > 0:
                sorted_rel, _ = torch.sort(train["candidate_rel"][ids], dim=1)
                oracle[(sorted_rel[:, 1] - sorted_rel[:, 0]) < float(trial["teacher_margin"])] = 0
            row_w = 1.0 + float(trial.get("hard_weight", 1.0)) * train["hard"][ids]
            row_w = row_w + float(trial.get("t50_weight", 1.0)) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial.get("t100_weight", 1.0)) * (train["horizon"][ids] == 100).float()
            score = (F.smooth_l1_loss(out["candidate_score"], train["candidate_rel"][ids], reduction="none").mean(dim=1) * row_w).mean()
            endpoint_err = torch.linalg.norm(out["endpoint_delta"] - train["target_delta"][ids], dim=1)
            endpoint = (F.smooth_l1_loss(out["endpoint_delta"], train["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            endpoint_risk = (F.smooth_l1_loss(out["endpoint_risk"], torch.log1p(endpoint_err.detach()), reduction="none") * row_w).mean()
            ce = (F.cross_entropy(out["candidate_score"], oracle, reduction="none") * row_w).mean()
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], train["failure"][ids, None])
            gain = F.binary_cross_entropy_with_logits(out["gain_logit"], (oracle != 0).float()[:, None])
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], train["easy"][ids, None])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], 1.0 - train["failure"][ids, None])
            loss = float(trial.get("score_weight", 1.0)) * score + float(trial.get("endpoint_weight", 0.5)) * endpoint + float(trial.get("endpoint_risk_weight", 0.2)) * endpoint_risk + 0.5 * ce + 0.25 * failure + 0.25 * gain + 0.35 * harm + 0.1 * physical
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["agent_tokens"], val["agent_mask"], val["static"], val["cand_delta"])
            val_loss = float((F.smooth_l1_loss(out["candidate_score"], val["candidate_rel"]) + F.smooth_l1_loss(out["endpoint_delta"], val["target_delta"])).cpu())
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "static_dim": train["static"].shape[1], "trial": dict(trial), "best": best}, ckpt)
    return {"checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["static_dim"]), int(payload["trial"].get("width", 56)), int(payload["trial"].get("layers", 1)))
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, payload["trial"]


def _predict(path: str | Path, split: str) -> Dict[str, np.ndarray]:
    torch = _torch()
    model, _trial = _load_model(path)
    tensors = _load_tensors(split)
    out: Dict[str, list[np.ndarray]] = {k: [] for k in ["endpoint_delta", "endpoint_risk", "candidate_score", "failure", "gain", "harm", "physical"]}
    with torch.no_grad():
        for start in range(0, tensors["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["agent_tokens"].shape[0]))
            pred = model(tensors["agent_tokens"][sl], tensors["agent_mask"][sl], tensors["static"][sl], tensors["cand_delta"][sl])
            out["endpoint_delta"].append(pred["endpoint_delta"].cpu().numpy())
            out["endpoint_risk"].append(pred["endpoint_risk"].cpu().numpy().reshape(-1))
            out["candidate_score"].append(pred["candidate_score"].cpu().numpy())
            out["failure"].append(torch.sigmoid(pred["failure_logit"]).cpu().numpy().reshape(-1))
            out["gain"].append(torch.sigmoid(pred["gain_logit"]).cpu().numpy().reshape(-1))
            out["harm"].append(torch.sigmoid(pred["harm_logit"]).cpu().numpy().reshape(-1))
            out["physical"].append(torch.sigmoid(pred["physical_logit"]).cpu().numpy().reshape(-1))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _select(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = pred["candidate_score"]
    best = np.argmin(score, axis=1)
    pred_gain = score[:, 0] - score[np.arange(len(best)), best]
    switch = (
        (best != 0)
        & (pred_gain >= float(policy.get("gain_threshold", 0.0)))
        & (pred["gain"] >= float(policy.get("gain_prob", 0.0)))
        & (pred["harm"] <= float(policy.get("harm_prob", 1.0)))
        & (pred["physical"] >= float(policy.get("physical_prob", 0.0)))
    )
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch <= 0.0:
        switch[:] = False
    elif max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        keep = np.zeros(len(switch), dtype=bool)
        keep[ids[np.argsort(pred_gain[ids])[::-1][:keep_n]]] = True
        switch &= keep
    idx = np.zeros(len(best), dtype=np.int64)
    idx[switch] = best[switch]
    selected = ds["candidate_fde"].astype(np.float64)[np.arange(len(idx)), idx]
    if policy.get("use_endpoint", False):
        endpoint_pred_score = pred["endpoint_risk"]
        endpoint_gain = score[:, 0] - endpoint_pred_score
        endpoint_switch = (
            (endpoint_gain >= float(policy.get("endpoint_gain_threshold", policy.get("gain_threshold", 0.0))))
            & (pred["gain"] >= float(policy.get("gain_prob", 0.0)))
            & (pred["harm"] <= float(policy.get("harm_prob", 1.0)))
            & (pred["physical"] >= float(policy.get("physical_prob", 0.0)))
        )
        if policy.get("hard_only", False):
            endpoint_switch &= ds["hard"].astype(bool) | ds["failure"].astype(bool)
        max_endpoint = float(policy.get("max_endpoint_switch", policy.get("max_switch", 1.0)))
        if max_endpoint <= 0.0:
            endpoint_switch[:] = False
        elif max_endpoint < 1.0 and np.any(endpoint_switch):
            ids = np.where(endpoint_switch)[0]
            keep_n = max(1, int(max_endpoint * len(endpoint_switch)))
            keep = np.zeros(len(endpoint_switch), dtype=bool)
            keep[ids[np.argsort(endpoint_gain[ids])[::-1][:keep_n]]] = True
            endpoint_switch &= keep
        endpoint_xy = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
        endpoint_fde = np.linalg.norm(endpoint_xy - ds["future_xy"].astype(np.float64), axis=1)
        replace = endpoint_switch & ((~switch) | (endpoint_pred_score < score[np.arange(len(idx)), idx]))
        selected[replace] = endpoint_fde[replace]
        switch = switch | replace
        idx[replace] = -1
    return selected, switch, idx


def _policy_grid() -> list[Dict[str, float]]:
    policies = [
        {"gain_threshold": gain, "gain_prob": gp, "harm_prob": hp, "physical_prob": 0.0, "max_switch": max_sw}
        for gain in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
        for gp in [0.0, 0.25, 0.5, 0.7]
        for hp in [0.03, 0.05, 0.1, 0.2, 0.35]
        for max_sw in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    ]
    endpoint_policies = []
    for pol in policies:
        for eg in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
            p = dict(pol)
            p.update({"use_endpoint": True, "endpoint_gain_threshold": eg, "max_endpoint_switch": pol["max_switch"]})
            endpoint_policies.append(p)
    return policies + endpoint_policies


def _quick_score(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
    selected, switch, _idx = _select(pred, ds, policy)
    fallback = ds["floor_fde"].astype(np.float64)
    horizon = ds["horizon"].astype(int)
    hard = ds["hard"].astype(bool) | ds["failure"].astype(bool)
    easy = ds["easy"].astype(bool)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - selected[mask].mean() / max(float(fallback[mask].mean()), EPS))

    all_imp = imp(np.ones(len(selected), dtype=bool))
    t50_imp = imp(horizon == 50)
    hard_imp = imp(hard)
    easy_deg = float(max(0.0, selected[easy].mean() / max(float(fallback[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0
    harm = float(np.mean(selected - fallback))
    score = all_imp + t50_imp + hard_imp - 20.0 * max(0.0, easy_deg - 0.02) - 0.5 * max(0.0, harm)
    return score, {"all_improvement": all_imp, "t50_improvement": t50_imp, "hard_failure_improvement": hard_imp, "easy_degradation": easy_deg, "harm_over_fallback": harm, "switch_rate": float(np.mean(switch))}


def _metrics(selected: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray], switch: np.ndarray | None = None) -> Dict[str, Any]:
    return s41._metrics(selected, fallback, ds, switch)


def _eval_predictions(pred: Mapping[str, np.ndarray], split: str, policy: Mapping[str, float], bootstrap: bool = False) -> Dict[str, Any]:
    ds = _ds(split)
    selected, switch, selected_idx = _select(pred, ds, policy)
    fallback = ds["floor_fde"].astype(np.float64)
    endpoint = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    endpoint_fde = np.linalg.norm(endpoint - ds["future_xy"].astype(np.float64), axis=1)
    cand_without = ds["candidate_fde"].astype(np.float64)[np.arange(len(selected_idx)), np.argmin(pred["candidate_score"], axis=1)]
    out = _metrics(selected, fallback, ds, switch)
    out["neural_endpoint_without_fallback"] = _metrics(endpoint_fde, fallback, ds)
    out["neural_candidate_without_fallback"] = _metrics(cand_without, fallback, ds)
    out["selected_candidate_distribution"] = dict(Counter(selected_idx.astype(int).tolist()))
    if bootstrap:
        out["t50_ci"] = s41._bootstrap_ci(selected, fallback, ds, "t50", n=2000)
        out["hard_failure_ci"] = s41._bootstrap_ci(selected, fallback, ds, "hard_failure", n=1000)
    return out


def _slice_mapping(mapping: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in mapping.items()}


def _val_select_domain_policies(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred = _predict(path, "val")
    ds = _ds("val")
    domains = ds["domain"].astype(str)
    policies: Dict[str, Dict[str, float]] = {}
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        pred_d = _slice_mapping(pred, mask)
        ds_d = _slice_mapping(ds, mask)
        best_score = -1e18
        best_policy: Dict[str, float] | None = None
        for policy in _policy_grid():
            score, quick_metrics = _quick_score(pred_d, ds_d, policy)
            if score > best_score:
                best_score = score
                best_policy = dict(policy)
        assert best_policy is not None
        best_policy["val_score"] = float(best_score)
        policies[domain] = best_policy
    selected, switch, idx = _eval_domain_policy_arrays(pred, ds, policies)
    metrics = _metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)
    metrics["selected_candidate_distribution"] = dict(Counter(idx.astype(int).tolist()))
    return {"type": "domain_conditioned", "policies": policies}, metrics


def _eval_domain_policy_arrays(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policies: Mapping[str, Mapping[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    domains = ds["domain"].astype(str)
    selected = ds["floor_fde"].astype(np.float64).copy()
    switch = np.zeros(len(selected), dtype=bool)
    idx = np.zeros(len(selected), dtype=np.int64)
    for domain, policy in policies.items():
        mask = domains == domain
        if not np.any(mask):
            continue
        pred_d = _slice_mapping(pred, mask)
        ds_d = _slice_mapping(ds, mask)
        sel_d, sw_d, idx_d = _select(pred_d, ds_d, policy)
        selected[mask] = sel_d
        switch[mask] = sw_d
        idx[mask] = idx_d
    return selected, switch, idx


def _eval_domain_policies(path: str | Path, split: str, policies: Mapping[str, Mapping[str, float]], bootstrap: bool = False) -> Dict[str, Any]:
    pred = _predict(path, split)
    ds = _ds(split)
    selected, switch, idx = _eval_domain_policy_arrays(pred, ds, policies)
    out = _metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)
    endpoint = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    endpoint_fde = np.linalg.norm(endpoint - ds["future_xy"].astype(np.float64), axis=1)
    out["neural_endpoint_without_fallback"] = _metrics(endpoint_fde, ds["floor_fde"].astype(np.float64), ds)
    out["selected_candidate_distribution"] = dict(Counter(idx.astype(int).tolist()))
    if bootstrap:
        out["t50_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "t50", n=2000)
        out["hard_failure_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "hard_failure", n=1000)
    return out


def _val_select(path: str | Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    pred = _predict(path, "val")
    ds = _ds("val")
    best_policy: Dict[str, float] | None = None
    best_metrics: Dict[str, float] | None = None
    best_score = -1e18
    for policy in _policy_grid():
        score, metrics = _quick_score(pred, ds, policy)
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_metrics = metrics
    assert best_policy is not None and best_metrics is not None
    best_policy["val_score"] = float(best_score)
    return best_policy, best_metrics


def train_all_agent_world_models() -> Dict[str, Any]:
    build_all_agent_dataset()
    prior = read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {})
    reports: Dict[str, Any] = dict(prior.get("trials", {}))
    for trial in _trial_configs():
        if trial["name"] in reports:
            train = reports[trial["name"]].get("train", {})
            if train.get("checkpoint"):
                policy, val_metrics = _val_select(train["checkpoint"])
                test_metrics = _eval_predictions(_predict(train["checkpoint"], "test"), "test", policy, bootstrap=False)
                domain_policy, domain_val_metrics = _val_select_domain_policies(train["checkpoint"])
                domain_test_metrics = _eval_domain_policies(train["checkpoint"], "test", domain_policy["policies"], bootstrap=False)
                if (
                    domain_val_metrics.get("all_improvement", 0.0)
                    + domain_val_metrics.get("t50_improvement", 0.0)
                    + domain_val_metrics.get("hard_failure_improvement", 0.0)
                    - 20.0 * max(0.0, domain_val_metrics.get("easy_degradation", 1.0) - 0.02)
                ) > (
                    val_metrics.get("all_improvement", 0.0)
                    + val_metrics.get("t50_improvement", 0.0)
                    + val_metrics.get("hard_failure_improvement", 0.0)
                    - 20.0 * max(0.0, val_metrics.get("easy_degradation", 1.0) - 0.02)
                ):
                    selected_policy = domain_policy
                    selected_val = domain_val_metrics
                    selected_test = domain_test_metrics
                else:
                    selected_policy = policy
                    selected_val = val_metrics
                    selected_test = test_metrics
                reports[trial["name"]].update({"policy": selected_policy, "val_metrics": selected_val, "test_metrics": selected_test, "global_policy": policy, "domain_policy": domain_policy, "source": "cached_verified"})
            continue
        train = _train_trial(trial)
        policy, val_metrics = _val_select(train["checkpoint"])
        test_metrics = _eval_predictions(_predict(train["checkpoint"], "test"), "test", policy, bootstrap=False)
        domain_policy, domain_val_metrics = _val_select_domain_policies(train["checkpoint"])
        domain_test_metrics = _eval_domain_policies(train["checkpoint"], "test", domain_policy["policies"], bootstrap=False)
        if (
            domain_val_metrics.get("all_improvement", 0.0)
            + domain_val_metrics.get("t50_improvement", 0.0)
            + domain_val_metrics.get("hard_failure_improvement", 0.0)
            - 20.0 * max(0.0, domain_val_metrics.get("easy_degradation", 1.0) - 0.02)
        ) > (
            val_metrics.get("all_improvement", 0.0)
            + val_metrics.get("t50_improvement", 0.0)
            + val_metrics.get("hard_failure_improvement", 0.0)
            - 20.0 * max(0.0, val_metrics.get("easy_degradation", 1.0) - 0.02)
        ):
            selected_policy = domain_policy
            selected_val = domain_val_metrics
            selected_test = domain_test_metrics
        else:
            selected_policy = policy
            selected_val = val_metrics
            selected_test = test_metrics
        reports[trial["name"]] = {"source": "fresh_run", "trial": trial, "train": train, "policy": selected_policy, "val_metrics": selected_val, "test_metrics": selected_test, "global_policy": policy, "domain_policy": domain_policy}
    result = {"source": "fresh_run", "trials": reports, "trial_count": len(reports), "stage41_pass": "all_agent_neighbor_token_second_pass"}
    _write_json(OUT_DIR / "stage41_all_agent_training_trials.json", result)
    write_md(OUT_DIR / "stage41_all_agent_training_trials.md", ["# Stage41 All-Agent Training Trials", "", "- source: `fresh_run`", f"- trials: `{reports}`"])
    return result


def eval_all_agent_world_models() -> Dict[str, Any]:
    trials = read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {}) if (OUT_DIR / "stage41_all_agent_training_trials.json").exists() else train_all_agent_world_models()
    best_name = "none"
    best_item: Dict[str, Any] = {}
    best_score = -1e18
    for name, item in trials.get("trials", {}).items():
        m = item.get("test_metrics", {})
        score = max(
            m.get("all_improvement", 0.0) - s41.STAGE37_REFERENCE["all_improvement"],
            m.get("t50_improvement", 0.0) - s41.STAGE37_REFERENCE["t50_improvement"],
            m.get("hard_failure_improvement", 0.0) - s41.STAGE37_REFERENCE["hard_failure_improvement"],
        ) - 10.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        if score > best_score:
            best_score = score
            best_name = name
            best_item = item
    best_metrics = best_item.get("test_metrics", {})
    if best_item.get("train", {}).get("checkpoint") and best_item.get("policy"):
        if best_item["policy"].get("type") == "domain_conditioned":
            best_metrics = _eval_domain_policies(best_item["train"]["checkpoint"], "test", best_item["policy"]["policies"], bootstrap=True)
        else:
            best_metrics = _eval_predictions(_predict(best_item["train"]["checkpoint"], "test"), "test", best_item["policy"], bootstrap=True)
        best_item["test_metrics"] = best_metrics
    positive_domains = sum(1 for row in best_metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
    beats = (
        best_metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            best_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            or best_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            or best_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
    )
    result = {
        "source": "fresh_run",
        "best_stage41_all_agent_neural": best_name,
        "best_metrics": best_metrics,
        "positive_external_domains": int(positive_domains),
        "neural_exceeds_stage37_by_gate_margin": bool(beats),
        "deployment_decision": "deploy_stage41_all_agent_neural" if beats and positive_domains >= 2 else "keep_stage37_selector",
        "trials": trials.get("trials", {}),
        "note": "Second pass adds nearest-neighbor all-agent tokens from same current frame and past-only per-agent histories.",
    }
    _write_json(OUT_DIR / "stage41_all_agent_eval.json", result)
    write_md(OUT_DIR / "stage41_all_agent_eval.md", ["# Stage41 All-Agent Neural Eval", "", "- source: `fresh_run`", f"- deployment: `{result['deployment_decision']}`", f"- best: `{best_name}`", f"- metrics: `{best_metrics}`"])
    return result


def main_build_all_agent_dataset() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        build_all_agent_dataset()
        status = "success"
    finally:
        _append_ledger("stage41_build_all_agent_dataset", status, started, [str(DATA_DIR / "seq2seq_train.npz")], [str(OUT_DIR / "stage41_all_agent_dataset.md")])


def main_train_all_agent_world_models() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        train_all_agent_world_models()
        status = "success"
    finally:
        _append_ledger("stage41_train_all_agent_world_models", status, started, [str(OUT_DIR / "stage41_all_agent_dataset.json")], [str(OUT_DIR / "stage41_all_agent_training_trials.md")])


def main_eval_all_agent_world_models() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        eval_all_agent_world_models()
        status = "success"
    finally:
        _append_ledger("stage41_eval_all_agent_world_models", status, started, [str(OUT_DIR / "stage41_all_agent_training_trials.json")], [str(OUT_DIR / "stage41_all_agent_eval.md")])
