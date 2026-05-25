from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage35_selective_transfer as s35
from src import stage41_all_agent as aa
from src import stage41_breakthrough as s41
from src import stage41_fresh_confirmation as fresh


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
CHECKPOINT_DIR = fresh.CHECKPOINT_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
WAYPOINT_FRAC = np.asarray([0.25, 0.50, 0.75, 1.0], dtype=np.float32)
THREADS = 4
BATCH = 512
EPOCHS = 4
SEED = 4197
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


def _fresh_ds(split: str) -> Dict[str, np.ndarray]:
    if not fresh.SPLIT_PATH.exists():
        fresh.build_source_rotation_split()
    path = DATA_DIR / f"all_agent_{split}.npz"
    if not path.exists() or not (DATA_DIR / "normalization.npz").exists():
        with fresh._ProtoPatch() as patched:
            patched.build_stratified_all_agent_dataset()
    return fresh._fresh_ds(split)


def _track_reader(path: str | Path) -> np.ndarray:
    p = Path(str(path))
    lower = str(p).lower()
    if p.name == "obsmat.txt" and ("/eth/" in lower or "/ucy/" in lower):
        return s35._read_obsmat(p)
    return s35._read_four_col(p)


def _track_map(source_files: Sequence[str]) -> Dict[tuple[str, int], np.ndarray]:
    out: Dict[tuple[str, int], np.ndarray] = {}
    for source in sorted(set(map(str, source_files))):
        path = Path(source)
        if not path.exists():
            continue
        arr = _track_reader(path)
        if len(arr) == 0:
            continue
        for agent in np.unique(arr[:, 1]).astype(int):
            tr = arr[arr[:, 1] == agent]
            tr = tr[np.argsort(tr[:, 0])]
            out[(source, int(agent))] = tr.astype(np.float64)
    return out


def _lookup_waypoints(track: np.ndarray, frame: float, horizon: int, endpoint_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.zeros((len(WAYPOINT_FRAC), 2), dtype=np.float32)
    valid = np.zeros(len(WAYPOINT_FRAC), dtype=bool)
    frames = track[:, 0].astype(np.float64)
    # Raw external trajectories are irregular. This tolerance allows the next
    # observed future point while keeping labels tied to the requested horizon.
    tolerance = max(2.0, 0.30 * float(max(horizon, 1)))
    for i, frac in enumerate(WAYPOINT_FRAC):
        target = float(frame) + float(horizon) * float(frac)
        j = int(np.searchsorted(frames, target, side="left"))
        if j < len(frames) and frames[j] <= target + tolerance:
            pts[i] = track[j, 2:4].astype(np.float32)
            valid[i] = True
    # Endpoint was already audited as label/eval only in earlier stages. Use it
    # only as the final waypoint label if raw interpolation misses the exact row.
    if not valid[-1]:
        pts[-1] = endpoint_xy.astype(np.float32)
        valid[-1] = True
    return pts, valid


def _future_interaction_labels(waypoints: np.ndarray, valid: np.ndarray, ds: Mapping[str, np.ndarray], ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ids)
    interaction = np.zeros(n, dtype=bool)
    occupancy = np.zeros(n, dtype=bool)
    physical = np.ones(n, dtype=bool)
    data = s41._combined()
    source = data["source_file"].astype(str)[ids]
    frame = np.rint(data["frame_id"].astype(float)[ids]).astype(int)
    horizon = data["horizon"].astype(int)[ids]
    agent = data["agent_id"].astype(int)[ids]
    groups: Dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(source, frame, horizon)):
        groups[key].append(i)
    norm = np.maximum(ds["normalizer"].astype(np.float32), EPS)
    for members in groups.values():
        if len(members) < 2:
            continue
        mem = np.asarray(members, dtype=np.int64)
        for local_i, row in enumerate(mem):
            other = mem[agent[mem] != agent[row]]
            if len(other) == 0:
                continue
            close_count = 0
            min_dist = np.inf
            for w in range(len(WAYPOINT_FRAC)):
                if not valid[row, w]:
                    continue
                other_valid = valid[other, w]
                if not np.any(other_valid):
                    continue
                d = np.linalg.norm(waypoints[other[other_valid], w] - waypoints[row, w], axis=1)
                if len(d):
                    min_dist = min(min_dist, float(np.min(d)))
                    close_count += int(np.sum(d <= max(0.05 * float(norm[row]), 1e-3)))
            interaction[row] = min_dist <= max(0.08 * float(norm[row]), 1e-3)
            occupancy[row] = close_count > 0
    # A simple physical-validity target: labels with huge jagged jumps are kept
    # for trajectory loss but marked physically suspicious for the auxiliary head.
    cur = ds["current_xy"].astype(np.float32)
    for i in range(n):
        pts = np.concatenate([cur[i][None, :], waypoints[i, valid[i]]], axis=0)
        if len(pts) < 3:
            physical[i] = False
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        physical[i] = bool(np.all(np.isfinite(seg)) and np.max(seg) <= max(4.0 * float(np.median(seg) + EPS), 0.5 * float(norm[i])))
    return interaction, occupancy, physical


def build_full_trajectory_labels() -> Dict[str, Any]:
    data = s41._combined()
    source_files = data["source_file"].astype(str).tolist()
    tracks = _track_map(source_files)
    report: Dict[str, Any] = {
        "source": "fresh_run",
        "waypoint_fractions": WAYPOINT_FRAC.tolist(),
        "splits": {},
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
    }
    for split in ["train", "val", "test"]:
        ds = _fresh_ds(split)
        ids = ds["ids"].astype(np.int64)
        waypoints = np.zeros((len(ids), len(WAYPOINT_FRAC), 2), dtype=np.float32)
        valid = np.zeros((len(ids), len(WAYPOINT_FRAC)), dtype=bool)
        missing_track = 0
        for out_i, rid in enumerate(ids):
            source = str(data["source_file"][rid])
            agent = int(data["agent_id"][rid])
            track = tracks.get((source, agent))
            if track is None:
                missing_track += 1
                waypoints[out_i, -1] = ds["future_xy"][out_i]
                valid[out_i, -1] = True
                continue
            pts, mask = _lookup_waypoints(
                track,
                float(data["frame_id"][rid]),
                int(data["horizon"][rid]),
                ds["future_xy"][out_i].astype(np.float32),
            )
            waypoints[out_i] = pts
            valid[out_i] = mask
        interaction, occupancy, physical = _future_interaction_labels(waypoints, valid, ds, ids)
        target_delta = ((waypoints - ds["current_xy"][:, None, :]) / np.maximum(ds["normalizer"][:, None, None], EPS)).astype(np.float32)
        np.savez_compressed(
            DATA_DIR / f"full_trajectory_{split}.npz",
            ids=ids,
            waypoint_xy=waypoints,
            waypoint_valid=valid,
            waypoint_delta=target_delta,
            interaction_future_close=interaction,
            occupancy_future_dense=occupancy,
            physical_valid=physical,
        )
        full_mask = np.all(valid, axis=1)
        report["splits"][split] = {
            "rows": int(len(ids)),
            "full_waypoint_rows": int(np.sum(full_mask)),
            "endpoint_only_rows": int(np.sum(valid[:, -1] & ~full_mask)),
            "missing_track_rows": int(missing_track),
            "interaction_positive": int(np.sum(interaction)),
            "occupancy_positive": int(np.sum(occupancy)),
            "physical_valid": int(np.sum(physical)),
            "domains": dict(Counter(ds["domain"].astype(str).tolist())),
            "t50": int(np.sum(ds["horizon"].astype(int) == 50)),
            "t100": int(np.sum(ds["horizon"].astype(int) == 100)),
        }
    _write_json(OUT_DIR / "stage41_full_trajectory_labels.json", report)
    write_md(
        OUT_DIR / "stage41_full_trajectory_labels.md",
        [
            "# Stage41 Full-Trajectory Label Reconstruction",
            "",
            "- source: `fresh_run`",
            "- future waypoints are label/evaluation only; inputs remain past-only all-agent tokens.",
            f"- waypoint fractions: `{WAYPOINT_FRAC.tolist()}`",
            f"- splits: `{report['splits']}`",
            f"- no leakage: `{report['no_leakage']}`",
        ],
    )
    return report


def _traj(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"full_trajectory_{split}.npz"
    if not path.exists():
        build_full_trajectory_labels()
    return dict(np.load(path))


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _load_tensors(split: str):
    torch = _torch()
    ds = _fresh_ds(split)
    tr = _traj(split)
    return {
        "agent_tokens": torch.tensor(ds["agent_tokens"].astype(np.float32)),
        "agent_mask": torch.tensor(ds["agent_mask"].astype(bool)),
        "static": torch.tensor(_norm_static(ds["static"])),
        "cand_delta": torch.tensor(ds["cand_delta"].astype(np.float32)),
        "waypoint_delta": torch.tensor(tr["waypoint_delta"].astype(np.float32)),
        "waypoint_valid": torch.tensor(tr["waypoint_valid"].astype(bool)),
        "interaction": torch.tensor(tr["interaction_future_close"].astype(np.float32)),
        "occupancy": torch.tensor(tr["occupancy_future_dense"].astype(np.float32)),
        "physical": torch.tensor(tr["physical_valid"].astype(np.float32)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"].astype(bool).astype(np.float32)),
        "failure": torch.tensor(ds["failure"].astype(bool).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
        "raw": ds,
        "traj": tr,
    }


def _make_model(static_dim: int, width: int = 80, dropout: float = 0.08):
    torch = _torch()
    import torch.nn as nn

    class FullTrajectoryWorldState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, width))
            self.role = nn.Embedding(aa.MAX_AGENTS, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.waypoints = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(WAYPOINT_FRAC) * 2))
            self.traj_risk = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.interaction = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.occupancy = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.physical = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, agent_tokens, agent_mask, static):
            b, a, _t, _ = agent_tokens.shape
            x = self.temporal(agent_tokens) + self.role.weight[:a][None, :, None, :]
            valid_time = agent_tokens[..., 6].clamp(0, 1)
            agent_emb = (x * valid_time[..., None]).sum(dim=2) / valid_time.sum(dim=2, keepdim=True).clamp_min(1.0)
            agent_h = self.agent_encoder(agent_emb)
            valid_agent = agent_mask[:, :, None].float()
            pooled = (agent_h * valid_agent).sum(dim=1) / valid_agent.sum(dim=1).clamp_min(1.0)
            ctx = self.ctx(torch.cat([pooled, self.static(static)], dim=1))
            return {
                "waypoint_delta": self.waypoints(ctx).view(-1, len(WAYPOINT_FRAC), 2),
                "traj_risk": self.traj_risk(ctx).squeeze(-1),
                "interaction_logit": self.interaction(ctx).squeeze(-1),
                "occupancy_logit": self.occupancy(ctx).squeeze(-1),
                "physical_logit": self.physical(ctx).squeeze(-1),
            }

    return FullTrajectoryWorldState()


TRIALS = [
    {"name": "full_traj_balanced", "width": 80, "dropout": 0.08, "lr": 8e-4, "hard_w": 2.0, "t50_w": 2.5, "t100_w": 2.0, "aux_w": 0.35, "seed": 1},
    {"name": "full_traj_t50_hard", "width": 88, "dropout": 0.08, "lr": 7e-4, "hard_w": 3.5, "t50_w": 5.0, "t100_w": 1.0, "aux_w": 0.30, "seed": 2},
    {"name": "full_traj_long_horizon", "width": 96, "dropout": 0.10, "lr": 6e-4, "hard_w": 2.5, "t50_w": 2.0, "t100_w": 5.0, "aux_w": 0.40, "seed": 3},
]


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train")
    val = _load_tensors("val")
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    model = _make_model(train["static"].shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["seed"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["agent_tokens"].shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["agent_tokens"][ids], train["agent_mask"][ids], train["static"][ids])
            valid = train["waypoint_valid"][ids].float()
            row_w = 1.0 + float(trial["hard_w"]) * train["hard"][ids]
            row_w = row_w + float(trial["t50_w"]) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (train["horizon"][ids] == 100).float()
            err = torch.linalg.norm(out["waypoint_delta"] - train["waypoint_delta"][ids], dim=2)
            traj = ((F.smooth_l1_loss(out["waypoint_delta"], train["waypoint_delta"][ids], reduction="none").mean(dim=2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_w).mean()
            risk = (F.smooth_l1_loss(out["traj_risk"], torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach(), reduction="none") * row_w).mean()
            interaction = F.binary_cross_entropy_with_logits(out["interaction_logit"], train["interaction"][ids])
            occupancy = F.binary_cross_entropy_with_logits(out["occupancy_logit"], train["occupancy"][ids])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], train["physical"][ids])
            loss = traj + 0.35 * risk + float(trial["aux_w"]) * (interaction + occupancy + physical)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["agent_tokens"], val["agent_mask"], val["static"])
            valid = val["waypoint_valid"].float()
            val_loss = float(((F.smooth_l1_loss(out["waypoint_delta"], val["waypoint_delta"], reduction="none").mean(dim=2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).mean().cpu())
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


def _predict(path: str | Path, split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = _torch()
    model, _trial = _load_model(path)
    tensors = _load_tensors(split)
    outs: Dict[str, list[np.ndarray]] = {k: [] for k in ["waypoint_delta", "traj_risk", "interaction", "occupancy", "physical"]}
    with torch.no_grad():
        for start in range(0, tensors["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["agent_tokens"].shape[0]))
            out = model(tensors["agent_tokens"][sl], tensors["agent_mask"][sl], tensors["static"][sl])
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
    pred = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}
    ds = tensors["raw"]
    tr = tensors["traj"]
    labels = {
        "floor_fde": ds["floor_fde"].astype(np.float64),
        "candidate_fde": ds["candidate_fde"].astype(np.float64),
        "current_xy": ds["current_xy"].astype(np.float64),
        "future_xy": ds["future_xy"].astype(np.float64),
        "normalizer": ds["normalizer"].astype(np.float64),
        "cand_delta": ds["cand_delta"].astype(np.float64),
        "waypoint_xy": tr["waypoint_xy"].astype(np.float64),
        "waypoint_valid": tr["waypoint_valid"].astype(bool),
        "interaction": tr["interaction_future_close"].astype(bool),
        "occupancy": tr["occupancy_future_dense"].astype(bool),
        "physical": tr["physical_valid"].astype(bool),
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
        raise ValueError("full trajectory ensemble requires at least one checkpoint")
    return {k: np.mean([p[k] for p in preds], axis=0).astype(np.float32) for k in preds[0].keys()}, labels_ref


def _floor_waypoints(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    floor_endpoint = labels["current_xy"] + labels["cand_delta"][:, 0, :] * labels["normalizer"][:, None]
    return labels["current_xy"][:, None, :] + WAYPOINT_FRAC[None, :, None] * (floor_endpoint - labels["current_xy"])[:, None, :]


def _pred_waypoints(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    return labels["current_xy"][:, None, :] + pred["waypoint_delta"].astype(np.float64) * labels["normalizer"][:, None, None]


def _trajectory_errors(xy: np.ndarray, labels: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = labels["waypoint_valid"].astype(bool)
    err = np.linalg.norm(xy - labels["waypoint_xy"], axis=2)
    ade = (err * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    fde = err[:, -1]
    return ade.astype(np.float64), fde.astype(np.float64)


def _binary_metrics(score: np.ndarray, label: np.ndarray) -> Dict[str, Any]:
    y = label.astype(bool)
    if len(y) == 0 or np.sum(y) == 0 or np.sum(~y) == 0:
        return {"auroc": None, "auprc": None, "positive_rate": float(np.mean(y)) if len(y) else 0.0}
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64) + 1.0
    pos_ranks = ranks[y].sum()
    n_pos = float(np.sum(y))
    n_neg = float(np.sum(~y))
    auroc = (pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, EPS)
    desc = np.argsort(score)[::-1]
    tp = np.cumsum(y[desc])
    precision = tp / (np.arange(len(y), dtype=np.float64) + 1.0)
    auprc = float((precision * y[desc]).sum() / max(n_pos, EPS))
    return {"auroc": float(auroc), "auprc": auprc, "positive_rate": float(np.mean(y))}


def _metric(selected_ade: np.ndarray, floor_ade: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> Dict[str, Any]:
    base = {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    return s41._metrics(selected_ade, floor_ade, base, switch)


def _apply_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    floor_xy = _floor_waypoints(labels)
    neural_xy = _pred_waypoints(pred, labels)
    floor_ade, floor_fde = _trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = _trajectory_errors(neural_xy, labels)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        if not np.any(mask):
            continue
        local = (
            mask
            & (pred["traj_risk"] <= float(params.get("traj_risk_max", 1e9)))
            & (pred["physical"] >= float(params.get("physical_prob_min", 0.0)))
        )
        if params.get("hard_only", False):
            local &= hard
        if params.get("easy_block", True):
            local &= ~easy
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch <= 0.0:
            local[:] = False
        elif max_switch < 1.0 and np.any(local):
            ids = np.where(local)[0]
            # Lower predicted trajectory risk is safer.
            keep_n = max(1, int(max_switch * int(np.sum(mask))))
            keep = np.zeros(len(local), dtype=bool)
            keep[ids[np.argsort(pred["traj_risk"][ids])[:keep_n]]] = True
            local &= keep
        selected_ade[local] = neural_ade[local]
        selected_fde[local] = neural_fde[local]
        switch |= local
    return selected_ade, selected_fde, switch, floor_ade


def _policy_grid(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], mask: np.ndarray) -> list[Dict[str, Any]]:
    risk_values = pred["traj_risk"][mask]
    if len(risk_values) == 0:
        thresholds = [0.0]
    else:
        thresholds = [float(v) for v in np.quantile(risk_values, [0.05, 0.10, 0.20, 0.35, 0.50, 0.70])]
    out = []
    for traj_risk_max in thresholds:
        for physical_prob_min in [0.0, 0.35, 0.55]:
            for max_switch in [0.0, 0.05, 0.10, 0.20, 0.40, 0.70]:
                base = {
                    "traj_risk_max": traj_risk_max,
                    "physical_prob_min": physical_prob_min,
                    "max_switch": max_switch,
                    "easy_block": True,
                }
                out.append({**base, "hard_only": False})
                out.append({**base, "hard_only": True})
    return out


def _score(metrics: Mapping[str, Any]) -> float:
    max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 1.5 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 30.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 35.0 * max(0.0, max_domain_easy - 0.02)
    )


def _fit_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    floor_xy = _floor_waypoints(labels)
    floor_ade, _floor_fde = _trajectory_errors(floor_xy, labels)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy = {"type": "full_trajectory_domain_horizon_policy", "slices": {}}
    selected = floor_ade.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    diagnostics: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            masked = {k: v[mask] for k, v in labels.items()}
            masked_pred = {k: v[mask] for k, v in pred.items()}
            best_params: Dict[str, Any] | None = None
            best_score = 0.0
            best_metrics: Dict[str, Any] | None = None
            for params in _policy_grid(pred, labels, mask):
                sel, _fde, sw, floor = _apply_policy(masked_pred, masked, {"slices": {f"{d}|{h}": params}})
                metrics = _metric(sel, floor, masked, sw)
                if metrics.get("all_improvement", 0.0) <= 0.0 or metrics.get("easy_degradation", 0.0) > 0.02:
                    continue
                score = _score(metrics)
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = metrics
            if best_params is not None:
                policy["slices"][f"{d}|{h}"] = best_params
                sel, _fde, sw, floor = _apply_policy(masked_pred, masked, {"slices": {f"{d}|{h}": best_params}})
                selected[mask] = sel
                switch[mask] = sw
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "val_score": float(best_score), "val_metrics": best_metrics or {"rows": int(np.sum(mask)), "all_improvement": 0.0}}
    metrics = _metric(selected, floor_ade, labels, switch)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _eval_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any], bootstrap: bool = False) -> Dict[str, Any]:
    selected_ade, selected_fde, switch, floor_ade = _apply_policy(pred, labels, policy)
    floor_xy = _floor_waypoints(labels)
    neural_xy = _pred_waypoints(pred, labels)
    neural_ade, neural_fde = _trajectory_errors(neural_xy, labels)
    metrics = _metric(selected_ade, floor_ade, labels, switch)
    metrics["trajectory_ade_improvement"] = metrics.get("all_improvement", 0.0)
    metrics["endpoint_fde_metrics"] = _metric(selected_fde, _trajectory_errors(floor_xy, labels)[1], labels, switch)
    metrics["neural_without_fallback_ade"] = _metric(neural_ade, floor_ade, labels, np.ones(len(floor_ade), dtype=bool))
    metrics["neural_without_fallback_fde"] = _metric(neural_fde, _trajectory_errors(floor_xy, labels)[1], labels, np.ones(len(floor_ade), dtype=bool))
    metrics["interaction_risk"] = _binary_metrics(pred["interaction"], labels["interaction"])
    metrics["occupancy_risk"] = _binary_metrics(pred["occupancy"], labels["occupancy"])
    metrics["physical_validity"] = _binary_metrics(pred["physical"], labels["physical"])
    if bootstrap:
        base = {
            "horizon": labels["horizon"],
            "hard": labels["hard"],
            "failure": labels["failure"],
            "easy": labels["easy"],
            "domain": labels["domain"],
            "candidate_fde": labels["candidate_fde"],
        }
        metrics["all_ci"] = s41._bootstrap_ci(selected_ade, floor_ade, base, "all", n=2000)
        metrics["t50_ci"] = s41._bootstrap_ci(selected_ade, floor_ade, base, "t50", n=2000)
        metrics["hard_failure_ci"] = s41._bootstrap_ci(selected_ade, floor_ade, base, "hard_failure", n=1000)
    return metrics


def train_full_trajectory_world_state() -> Dict[str, Any]:
    build_full_trajectory_labels()
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
        if score > best_score:
            best_score = score
            best_name = trial["name"]
            best_policy = policy
            best_paths = [train["checkpoint"]]
    paths = [row["train"]["checkpoint"] for row in trials.values() if row.get("train", {}).get("checkpoint")]
    if paths:
        pred_val, labels_val = _predict_ensemble(paths, "val")
        policy, val_metrics = _fit_policy(pred_val, labels_val)
        score = _score(val_metrics)
        trials["full_trajectory_ensemble"] = {"source": "fresh_run", "paths": paths, "policy": policy, "val_metrics": val_metrics, "val_score": score}
        if score > best_score:
            best_score = score
            best_name = "full_trajectory_ensemble"
            best_policy = policy
            best_paths = paths
    pred_test, labels_test = _predict_ensemble(best_paths, "test")
    test_metrics = _eval_policy(pred_test, labels_test, best_policy, bootstrap=True)
    positive_domains = sum(
        1
        for row in test_metrics.get("by_domain", {}).values()
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
    )
    pass_gate = (
        test_metrics.get("easy_degradation", 1.0) <= 0.02
        and positive_domains >= 2
        and (
            test_metrics.get("all_improvement", 0.0) > 0
            or test_metrics.get("t50_improvement", 0.0) > 0
            or test_metrics.get("hard_failure_improvement", 0.0) > 0
        )
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "fresh_full_trajectory_world_state_probe",
        "best_name": best_name,
        "best_score": best_score,
        "best_policy": best_policy,
        "best_metrics": test_metrics,
        "positive_external_domains": int(positive_domains),
        "full_trajectory_world_state_pass": bool(pass_gate),
        "deployment_decision": "candidate_full_trajectory_world_state_probe" if pass_gate else "diagnostic_keep_endpoint_candidate",
        "trials": trials,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "This is a supervised waypoint/interaction/occupancy dynamics probe under dataset-local raw-frame coordinates; it is not Stage5C, not SMC, not metric, and not true 3D.",
    }
    _write_json(OUT_DIR / "stage41_full_trajectory_world_state.json", result)
    lines = [
        "# Stage41 Full-Trajectory World-State Probe",
        "",
        "- source: `fresh_run`",
        f"- deployment decision: `{result['deployment_decision']}`",
        f"- best: `{best_name}`",
        f"- positive external domains: `{positive_domains}`",
        f"- full trajectory pass: `{result['full_trajectory_world_state_pass']}`",
        f"- metrics: `{test_metrics}`",
        "",
        "This probe reconstructs actual future waypoint labels from raw external trajectories and trains trajectory, interaction-risk, occupancy, and physical-validity heads. Future waypoints remain labels/eval only.",
    ]
    write_md(OUT_DIR / "stage41_full_trajectory_world_state.md", lines)
    return result


def main_full_trajectory_world_state() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        train_full_trajectory_world_state()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_full_trajectory_world_state",
            status,
            started,
            [DATA_DIR / "all_agent_train.npz", DATA_DIR / "full_trajectory_train.npz"],
            [OUT_DIR / "stage41_full_trajectory_world_state.md", OUT_DIR / "stage41_full_trajectory_world_state.json"],
        )

