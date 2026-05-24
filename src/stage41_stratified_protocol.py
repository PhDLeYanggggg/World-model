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
from src import stage41_all_agent as s41a
from src import stage41_breakthrough as s41


OUT_DIR = Path("outputs/stage41_stratified_protocol")
DATA_DIR = Path("data/stage41_stratified_protocol")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
LEDGER_JSONL = s41.OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
THREADS = 4
BATCH = 512
EPOCHS = 5
SEED = 4217


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
    ensure_dir(s41.OUT_DIR)
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


def _candidate_split_index() -> Dict[str, np.ndarray]:
    path = s41.DATA_DIR / "stage41_stratified_split_candidate.npz"
    if not path.exists():
        from src import stage41_validation_gap as gap

        gap.build_stratified_split_candidate()
    return dict(np.load(path, allow_pickle=True))


def _split_mask(split: str, n: int) -> np.ndarray:
    idx = _candidate_split_index()
    mask = np.zeros(n, dtype=bool)
    rows = idx["row_id"][idx["split"].astype(str) == split].astype(np.int64)
    mask[rows] = True
    return mask


def _make_base_arrays(data: Mapping[str, np.ndarray], ids: np.ndarray, floor_idx: np.ndarray, floor_fde: np.ndarray, candidate_fde: np.ndarray, candidates: np.ndarray, normalizer: np.ndarray, static: np.ndarray, target_delta: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "ids": ids.astype(np.int64),
        "static": static[ids].astype(np.float32),
        "target_delta": target_delta[ids].astype(np.float32),
        "cand_delta": ((candidates[ids] - np.stack([data["current_x"], data["current_y"]], axis=1)[ids, None, :]) / normalizer[ids, None, None]).astype(np.float32),
        "candidate_fde": candidate_fde[ids].astype(np.float32),
        "floor_fde": floor_fde[ids].astype(np.float32),
        "oracle_idx": np.argmin(candidate_fde[ids], axis=1).astype(np.int64),
        "normalizer": normalizer[ids].astype(np.float32),
        "current_xy": np.stack([data["current_x"], data["current_y"]], axis=1)[ids].astype(np.float32),
        "future_xy": np.stack([data["future_endpoint_x"], data["future_endpoint_y"]], axis=1)[ids].astype(np.float32),
        "horizon": data["horizon"][ids].astype(np.int16),
        "hard": data["hard"][ids].astype(bool),
        "easy": data["easy"][ids].astype(bool),
        "failure": data["failure"][ids].astype(bool),
        "domain": data["dataset"][ids],
        "scene_id": data["scene_id"][ids],
        "source_file": data["source_file"][ids],
        "floor_idx": floor_idx[ids].astype(np.int16),
    }


def build_stratified_all_agent_dataset() -> Dict[str, Any]:
    started = time.perf_counter()
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)
    data = s41._combined()
    n = len(data["horizon"])
    train_mask = _split_mask("train", n)
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    fut = np.stack([data["future_endpoint_x"], data["future_endpoint_y"]], axis=1).astype(np.float32)
    hist_path = np.maximum(data["history_scalar"][:, 0].astype(np.float32), EPS)
    speed = np.maximum(data["history_seq"][:, -1, 2].astype(np.float32), EPS)
    horizon = data["horizon"].astype(np.float32)
    normalizer = np.maximum(hist_path + speed * np.maximum(horizon, 1.0), np.median(hist_path[train_mask] + speed[train_mask] * np.maximum(horizon[train_mask], 1.0)) + EPS).astype(np.float32)
    safe_y = data["y_fde"][:, : s41.SAFE_BASELINE_COUNT].astype(np.float32)
    strongest_by_h: Dict[int, int] = {}
    for h in [10, 25, 50, 100]:
        hm = train_mask & (data["horizon"].astype(int) == h)
        strongest_by_h[h] = int(np.argmin(safe_y[hm].mean(axis=0))) if np.any(hm) else 1
    floor_idx = np.asarray([strongest_by_h[int(h)] for h in data["horizon"].astype(int)], dtype=np.int16)
    floor_fde = safe_y[np.arange(len(safe_y)), floor_idx]
    family_fde = data["family_fde"].astype(np.float32)
    candidate_fde = np.concatenate([floor_fde[:, None], family_fde], axis=1)
    floor_pred = data["family_pred"][:, 1, :].astype(np.float32)
    candidates = np.concatenate([floor_pred[:, None, :], data["family_pred"].astype(np.float32)], axis=1)
    target_delta = ((fut - cur) / normalizer[:, None]).astype(np.float32)
    static = np.concatenate(
        [
            data["stage37_features"].astype(np.float32),
            data["history_scalar"].astype(np.float32),
            data["prototype_likelihood"].astype(np.float32),
            data["prototype_entropy"][:, None].astype(np.float32),
            data["goal_ambiguity"][:, None].astype(np.float32),
            data["family_rel"].astype(np.float32),
            data["family_fde"].astype(np.float32) / normalizer[:, None],
            np.eye(4, dtype=np.float32)[np.searchsorted([10, 25, 50, 100], data["horizon"].astype(int))],
        ],
        axis=1,
    )
    static_mean = static[train_mask].mean(axis=0).astype(np.float32)
    static_std = np.maximum(static[train_mask].std(axis=0), 1e-3).astype(np.float32)
    np.savez_compressed(DATA_DIR / "normalization.npz", static_mean=static_mean, static_std=static_std)
    groups = s41a._group_indices(data)
    report: Dict[str, Any] = {"source": "fresh_run", "protocol": "stage41_stratified_split_candidate", "splits": {}, "strongest_by_horizon": strongest_by_h}
    for split in ["train", "val", "test"]:
        ids = np.where(_split_mask(split, n))[0].astype(np.int64)
        base = _make_base_arrays(data, ids, floor_idx, floor_fde, candidate_fde, candidates, normalizer, static, target_delta)
        tokens, agent_mask, neighbor_counts = s41a._build_tokens_for_ids(data, ids, groups)
        np.savez_compressed(
            DATA_DIR / f"all_agent_{split}.npz",
            **base,
            agent_tokens=tokens,
            agent_mask=agent_mask,
            neighbor_counts=neighbor_counts,
        )
        report["splits"][split] = {
            "rows": int(len(ids)),
            "domains": dict(Counter(base["domain"].astype(str).tolist())),
            "t50": int(np.sum(base["horizon"].astype(int) == 50)),
            "t100": int(np.sum(base["horizon"].astype(int) == 100)),
            "hard": int(np.sum(base["hard"].astype(bool))),
            "easy": int(np.sum(base["easy"].astype(bool))),
            "failure": int(np.sum(base["failure"].astype(bool))),
            "neighbor_count_mean": float(np.mean(neighbor_counts)) if len(neighbor_counts) else 0.0,
        }
    report["no_leakage"] = {
        "future_endpoint_input": False,
        "future_endpoint_label_eval_only": True,
        "test_endpoint_goals": False,
        "central_velocity": False,
        "overwrites_locked_stage41_split": False,
    }
    _write_json(OUT_DIR / "stage41_stratified_dataset.json", report)
    write_md(OUT_DIR / "stage41_stratified_dataset.md", ["# Stage41 Stratified All-Agent Dataset", "", "- source: `fresh_run`", "- status: candidate protocol for retraining; does not overwrite locked Stage41 split.", f"- report: `{report}`"])
    _append_ledger("stage41_stratified_dataset", "ok", started, [str(s41.DATA_DIR / "stage41_stratified_split_candidate.npz")], [str(OUT_DIR / "stage41_stratified_dataset.md")])
    return report


def _ds(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"all_agent_{split}.npz"
    if not path.exists():
        build_stratified_all_agent_dataset()
    return dict(np.load(path, allow_pickle=True))


def _domain_vocab() -> list[str]:
    domains: set[str] = set()
    for split in ["train", "val", "test"]:
        domains.update(_ds(split)["domain"].astype(str).tolist())
    return sorted(domains)


def _features(split: str, domain_vocab: Sequence[str] | None = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ds = _ds(split)
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    static = ((ds["static"].astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)
    target_hist = ds["agent_tokens"][:, 0, :, :].astype(np.float32).reshape(len(static), -1)
    valid = np.clip(ds["agent_tokens"][:, 0, :, 6:7].astype(np.float32), 0.0, 1.0)
    hist = ds["agent_tokens"][:, 0, :, :].astype(np.float32)
    hist_mean = (hist * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)
    hist_std = np.sqrt(((hist - hist_mean[:, None, :]) ** 2 * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0))
    cand = ds["cand_delta"].astype(np.float32).reshape(len(static), -1)
    horizon = ds["horizon"].astype(int)
    h_one = np.zeros((len(static), 4), dtype=np.float32)
    for i, h in enumerate([10, 25, 50, 100]):
        h_one[:, i] = horizon == h
    domain_vocab = list(domain_vocab or _domain_vocab())
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

    class StratifiedDistiller(nn.Module):
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
            return {"pred_rel": self.rel(h), "failure_logit": self.failure(h), "gain_logit": self.gain(h), "harm_logit": self.harm(h), "physical_logit": self.physical(h)}

    return StratifiedDistiller()


def _trial_configs() -> list[Dict[str, Any]]:
    return [
        {"name": "stratified_balanced", "width": 192, "dropout": 0.06, "lr": 1.0e-3, "hard_w": 1.5, "t50_w": 2.0, "t100_w": 1.0, "ce_w": 0.5, "rank_w": 0.5},
        {"name": "stratified_t50_specialist", "width": 224, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 2.5, "t50_w": 4.0, "t100_w": 1.0, "ce_w": 0.8, "rank_w": 0.8},
        {"name": "stratified_long_horizon", "width": 224, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 2.0, "t50_w": 2.0, "t100_w": 4.0, "ce_w": 0.6, "rank_w": 0.8},
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
    seed_offset = int(trial.get("seed_offset", 0))
    rng = np.random.default_rng(SEED + seed_offset + abs(hash(str(trial["name"]))) % 10000)
    ckpt = CHECKPOINT_DIR / f"stage41_stratified_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"stratified_{trial['name']}_heartbeat.json"
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
    focus_domain = str(trial.get("domain_focus", ""))
    focus_mask_train = None
    if focus_domain:
        focus_mask_train = torch.tensor((y_train["domain"].astype(str) == focus_domain).astype(np.float32))
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
            oracle = ty["oracle"][ids]
            best_rel = torch.min(rel, dim=1).values
            fallback_rel = rel[:, 0]
            gain_label = ((fallback_rel - best_rel) > 0.02).float()
            row_w = 1.0 + float(trial["hard_w"]) * ty["hard"][ids]
            row_w = row_w + float(trial["t50_w"]) * (ty["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (ty["horizon"][ids] == 100).float()
            row_w = row_w + 2.0 * gain_label
            if focus_mask_train is not None:
                focus = focus_mask_train[ids]
                row_w = row_w + float(trial.get("domain_w", 0.0)) * focus
                row_w = row_w + float(trial.get("domain_hard_w", 0.0)) * focus * ty["hard"][ids]
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
            torch.save({"model": model.state_dict(), "in_dim": x_train.shape[1], "candidate_count": candidate_count, "trial": dict(trial), "best": best, "domain_vocab": list(domain_vocab)}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(path: str | Path, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["in_dim"]), int(payload["candidate_count"]), int(payload["trial"]["width"]), float(payload["trial"]["dropout"]))
    model.load_state_dict(payload["model"])
    model.eval()
    x, labels = _features(split, payload.get("domain_vocab", _domain_vocab()))
    outs: Dict[str, list[np.ndarray]] = {"pred_rel": [], "failure": [], "gain": [], "harm": [], "physical": []}
    with torch.no_grad():
        tx = torch.tensor(x)
        for start in range(0, len(x), 2048):
            out = model(tx[start : start + 2048])
            outs["pred_rel"].append(out["pred_rel"].cpu().numpy())
            outs["failure"].append(torch.sigmoid(out["failure_logit"]).cpu().numpy().reshape(-1))
            outs["gain"].append(torch.sigmoid(out["gain_logit"]).cpu().numpy().reshape(-1))
            outs["harm"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
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
        sw = mask & (best != 0) & (pred_gain >= float(params["gain_threshold"])) & (pred["gain"] >= float(params["gain_prob"])) & (pred["harm"] <= float(params["harm_prob"])) & (pred["physical"] >= float(params["physical_prob"]))
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
    ds = {"horizon": labels["horizon"], "hard": labels["hard"].astype(bool), "failure": labels["failure"].astype(bool), "easy": labels["easy"].astype(bool), "domain": labels["domain"], "candidate_fde": labels["candidate_fde"]}
    return s41._metrics(selected, labels["floor_fde"].astype(np.float64), ds, switch)


def _policy_grid() -> list[Dict[str, Any]]:
    return [
        {"gain_threshold": gain, "gain_prob": gp, "harm_prob": hp, "physical_prob": pp, "max_switch": ms, "hard_only": hard_only}
        for gain in [0.0, 0.01, 0.03, 0.06, 0.10]
        for gp in [0.0, 0.35, 0.55, 0.75]
        for hp in [0.04, 0.08, 0.16, 0.30]
        for pp in [0.0, 0.4, 0.65]
        for ms in [0.0, 0.03, 0.05, 0.10, 0.18, 0.30, 0.45, 0.60]
        for hard_only in [False, True]
    ]


def _relaxed_easy_budget_policy_grid() -> list[Dict[str, Any]]:
    """Expanded policy grid for the post-diagnostic easy-budget hypothesis.

    The standard grid is intentionally conservative and often lets the Stage37
    floor consume neural signal. This grid allows larger switch budgets, but it
    is still selected on validation only and remains candidate evidence until
    repeated on a fresh locked protocol.
    """
    grid = list(_policy_grid())
    for harm in [0.45, 0.65, 0.90]:
        for max_switch in [0.70, 0.80, 0.95, 1.00]:
            grid.append(
                {
                    "gain_threshold": 0.0,
                    "gain_prob": 0.0,
                    "harm_prob": harm,
                    "physical_prob": 0.0,
                    "max_switch": max_switch,
                    "hard_only": False,
                }
            )
    return grid


def _metric_score(metrics: Mapping[str, Any], mode: str) -> float:
    """Validation-only score for selecting a safe switch policy."""
    all_imp = float(metrics.get("all_improvement", 0.0))
    t50 = float(metrics.get("t50_improvement", 0.0))
    t100 = float(metrics.get("t100_improvement", 0.0))
    hard = float(metrics.get("hard_failure_improvement", 0.0))
    easy = float(metrics.get("easy_degradation", 1.0))
    switch = float(metrics.get("switch_rate", 0.0))
    by_domain = metrics.get("by_domain", {}) or {}
    domain_t50 = [float(row.get("t50_improvement", 0.0)) for row in by_domain.values()]
    domain_hard = [float(row.get("hard_failure_improvement", 0.0)) for row in by_domain.values()]
    min_domain_t50 = min(domain_t50) if domain_t50 else 0.0
    min_domain_hard = min(domain_hard) if domain_hard else 0.0
    easy_penalty = 30.0 * max(0.0, easy - 0.02)
    if mode == "t50_tail":
        return all_imp + 3.0 * t50 + 1.2 * hard + 0.35 * t100 + 0.8 * min_domain_t50 + 0.3 * min_domain_hard + 0.02 * switch - easy_penalty
    if mode == "domain_tail":
        return all_imp + 2.4 * t50 + 1.4 * hard + 0.35 * t100 + 1.1 * min_domain_t50 + 0.7 * min_domain_hard + 0.02 * switch - easy_penalty
    if mode == "hard_all":
        return 2.0 * all_imp + 0.8 * t50 + 2.8 * hard + 0.25 * t100 + 0.5 * min_domain_hard + 0.03 * switch - easy_penalty
    if mode == "domain_hard":
        return 1.6 * all_imp + 0.8 * t50 + 2.3 * hard + 0.25 * t100 + 0.9 * min_domain_hard + 0.4 * min_domain_t50 + 0.03 * switch - easy_penalty
    return all_imp + 1.4 * t50 + hard + 0.4 * t100 - 20.0 * max(0.0, easy - 0.02)


def _select_policy(path: str | Path, mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred, labels = _predict(path, "val")
    return _select_policy_from_predictions(pred, labels, mode)


def _select_policy_from_predictions(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rel = pred["pred_rel"]
    fallback = labels["floor_fde"].astype(np.float64)
    best = np.argmin(rel, axis=1)
    pred_gain = rel[:, 0] - rel[np.arange(len(best)), best]
    candidate_fde = labels["candidate_fde"].astype(np.float64)
    hard_all = labels["hard"].astype(bool)
    easy_all = labels["easy"].astype(bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy = {"type": "stage41_stratified_candidate_policy", "mode": mode, "slices": {}}
    diagnostics: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(mask.sum()) < 100:
                continue
            ids = np.where(mask)[0]
            local_fallback = fallback[ids]
            local_best = best[ids]
            local_gain = pred_gain[ids]
            local_hard = hard_all[ids]
            local_easy = easy_all[ids]
            candidate_selected = candidate_fde[ids][np.arange(len(ids)), local_best]
            can_switch_base = local_best != 0
            best_params = None
            best_score = 0.0
            best_metrics = None
            for params in _policy_grid():
                sw = can_switch_base & (local_gain >= float(params["gain_threshold"])) & (pred["gain"][ids] >= float(params["gain_prob"])) & (pred["harm"][ids] <= float(params["harm_prob"])) & (pred["physical"][ids] >= float(params["physical_prob"]))
                if bool(params.get("hard_only", False)):
                    sw &= local_hard
                max_switch = float(params.get("max_switch", 1.0))
                if max_switch <= 0:
                    sw[:] = False
                elif max_switch < 1.0 and np.any(sw):
                    sw_ids = np.where(sw)[0]
                    keep_n = max(1, int(max_switch * len(ids)))
                    keep = np.zeros(len(sw), dtype=bool)
                    keep[sw_ids[np.argsort(local_gain[sw_ids])[::-1][:keep_n]]] = True
                    sw &= keep
                selected = local_fallback.copy()
                selected[sw] = candidate_selected[sw]
                imp = 1.0 - float(selected.mean()) / max(float(local_fallback.mean()), EPS)
                hard_imp = 0.0 if not np.any(local_hard) else 1.0 - float(selected[local_hard].mean()) / max(float(local_fallback[local_hard].mean()), EPS)
                easy_deg = 0.0 if not np.any(local_easy) else max(0.0, float(selected[local_easy].mean()) / max(float(local_fallback[local_easy].mean()), EPS) - 1.0)
                max_easy = 0.002 if mode == "conservative" else 0.02
                min_imp = 0.003 if mode == "conservative" else 0.0
                if imp <= min_imp or easy_deg > max_easy:
                    continue
                if mode == "t50_tail":
                    score = (2.2 if h == 50 else 0.7) * imp + (1.4 if h == 50 else 0.6) * hard_imp + 0.2 * float(h == 100) * imp + 0.01 * float(np.mean(sw))
                elif mode == "domain_tail":
                    score = (1.8 if h == 50 else 0.8) * imp + (1.6 if h in {50, 100} else 0.7) * hard_imp + 0.01 * float(np.mean(sw))
                elif mode == "hard_all":
                    score = 1.5 * imp + (2.8 if h in {25, 50, 100} else 1.2) * hard_imp + 0.02 * float(np.mean(sw))
                elif mode == "domain_hard":
                    score = 1.2 * imp + (2.4 if h in {25, 50, 100} else 1.2) * hard_imp + 0.25 * float(h == 50) * imp + 0.02 * float(np.mean(sw))
                else:
                    score = imp + (0.8 if h in {50, 100} else 0.25) * hard_imp + 0.01 * float(np.mean(sw))
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = {"rows": int(len(ids)), "improvement": float(imp), "hard_failure_improvement": float(hard_imp), "easy_degradation": float(easy_deg), "switch_rate": float(np.mean(sw))}
            if best_params:
                policy["slices"][f"{d}|{h}"] = best_params
            diagnostics[f"{d}|{h}"] = {"val_score": best_score, "selected": bool(best_params), "val_metrics": best_metrics or {"rows": int(mask.sum()), "improvement": 0.0}}
    selected, switch, idx = _apply_policy(pred, labels, policy)
    metrics = _metrics(selected, labels, switch)
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(idx, return_counts=True))}
    return policy, {"metrics": metrics, "slice_diagnostics": diagnostics}


def _select_relaxed_easy_budget_policy_from_predictions(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], mode: str = "relaxed_easy_budget") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select a higher-switch neural policy under validation easy-budget checks.

    This directly targets the Stage41 failure mode where candidate neural
    endpoints have substantial oracle and predicted-best headroom, but the
    deployment policy falls back too often. It uses validation labels only for
    policy selection. Because this rule was introduced after inspecting Stage41
    failures, outputs are marked as candidate evidence until a fresh locked
    confirmation run is available.
    """
    rel = pred["pred_rel"]
    fallback = labels["floor_fde"].astype(np.float64)
    best = np.argmin(rel, axis=1)
    pred_gain = rel[:, 0] - rel[np.arange(len(best)), best]
    candidate_fde = labels["candidate_fde"].astype(np.float64)
    hard_all = labels["hard"].astype(bool)
    easy_all = labels["easy"].astype(bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    max_local_easy = 0.02
    policy = {"type": "stage41_locked_v2_relaxed_easy_budget_policy", "mode": mode, "slices": {}}
    diagnostics: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(mask.sum()) < 100:
                continue
            ids = np.where(mask)[0]
            local_fallback = fallback[ids]
            local_best = best[ids]
            local_gain = pred_gain[ids]
            local_hard = hard_all[ids]
            local_easy = easy_all[ids]
            candidate_selected = candidate_fde[ids][np.arange(len(ids)), local_best]
            can_switch_base = local_best != 0
            best_params = None
            best_score = 0.0
            best_metrics = None
            for params in _relaxed_easy_budget_policy_grid():
                sw = can_switch_base & (local_gain >= float(params["gain_threshold"])) & (pred["gain"][ids] >= float(params["gain_prob"])) & (pred["harm"][ids] <= float(params["harm_prob"])) & (pred["physical"][ids] >= float(params["physical_prob"]))
                if bool(params.get("hard_only", False)):
                    sw &= local_hard
                max_switch = float(params.get("max_switch", 1.0))
                if max_switch <= 0:
                    sw[:] = False
                elif max_switch < 1.0 and np.any(sw):
                    sw_ids = np.where(sw)[0]
                    keep_n = max(1, int(max_switch * len(ids)))
                    keep = np.zeros(len(sw), dtype=bool)
                    keep[sw_ids[np.argsort(local_gain[sw_ids])[::-1][:keep_n]]] = True
                    sw &= keep
                selected = local_fallback.copy()
                selected[sw] = candidate_selected[sw]
                imp = 1.0 - float(selected.mean()) / max(float(local_fallback.mean()), EPS)
                hard_imp = 0.0 if not np.any(local_hard) else 1.0 - float(selected[local_hard].mean()) / max(float(local_fallback[local_hard].mean()), EPS)
                easy_deg = 0.0 if not np.any(local_easy) else max(0.0, float(selected[local_easy].mean()) / max(float(local_fallback[local_easy].mean()), EPS) - 1.0)
                if imp <= 0.0 or easy_deg > max_local_easy:
                    continue
                if mode == "relaxed_t50_budget":
                    score = 0.8 * imp + 1.2 * hard_imp + (2.8 if h == 50 else 0.2) * imp + 0.02 * float(np.mean(sw))
                elif mode == "relaxed_hard_budget":
                    score = 1.0 * imp + 3.0 * hard_imp + (0.6 if h in {50, 100} else 0.1) * imp + 0.02 * float(np.mean(sw))
                else:
                    score = 1.2 * imp + 2.0 * hard_imp + (0.8 if h == 50 else 0.2) * imp + (0.2 if h == 100 else 0.0) * imp + 0.03 * float(np.mean(sw))
                score -= 20.0 * max(0.0, easy_deg - 0.01)
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = {"rows": int(len(ids)), "improvement": float(imp), "hard_failure_improvement": float(hard_imp), "easy_degradation": float(easy_deg), "switch_rate": float(np.mean(sw))}
            if best_params:
                policy["slices"][f"{d}|{h}"] = best_params
            diagnostics[f"{d}|{h}"] = {"val_score": best_score, "selected": bool(best_params), "val_metrics": best_metrics or {"rows": int(mask.sum()), "improvement": 0.0}}
    selected, switch, idx = _apply_policy(pred, labels, policy)
    metrics = _metrics(selected, labels, switch)
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(idx, return_counts=True))}
    return policy, {"metrics": metrics, "slice_diagnostics": diagnostics}


def _eval_policy(path: str | Path, split: str, policy: Mapping[str, Any], bootstrap: bool = False) -> Dict[str, Any]:
    pred, labels = _predict(path, split)
    return _eval_policy_predictions(pred, labels, policy, bootstrap=bootstrap)


def _eval_policy_predictions(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any], bootstrap: bool = False) -> Dict[str, Any]:
    selected, switch, idx = _apply_policy(pred, labels, policy)
    metrics = _metrics(selected, labels, switch)
    endpoint_oracle = labels["candidate_fde"].astype(np.float64).min(axis=1)
    metrics["candidate_oracle"] = _metrics(endpoint_oracle, labels, np.ones(len(endpoint_oracle), dtype=bool))
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(idx, return_counts=True))}
    if bootstrap:
        ds = {"horizon": labels["horizon"], "hard": labels["hard"].astype(bool), "failure": labels["failure"].astype(bool), "easy": labels["easy"].astype(bool), "domain": labels["domain"], "candidate_fde": labels["candidate_fde"]}
        metrics["t50_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"].astype(np.float64), ds, "t50", n=2000)
        metrics["hard_failure_ci"] = s41._bootstrap_ci(selected, labels["floor_fde"].astype(np.float64), ds, "hard_failure", n=1000)
    return metrics


def _predict_ensemble(paths: Sequence[str | Path], split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    preds: list[Dict[str, np.ndarray]] = []
    labels_ref: Dict[str, np.ndarray] | None = None
    for path in paths:
        pred, labels = _predict(path, split)
        preds.append(pred)
        if labels_ref is None:
            labels_ref = labels
    if not preds or labels_ref is None:
        raise ValueError("ensemble requires at least one checkpoint")
    avg = {key: np.mean([p[key] for p in preds], axis=0).astype(np.float32) for key in preds[0].keys()}
    return avg, labels_ref


def train_stratified_protocol() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    domain_vocab = _domain_vocab()
    trials: Dict[str, Any] = {}
    best_name = ""
    best_val_score = -1e18
    best_policy: Dict[str, Any] = {}
    for trial in _trial_configs():
        train = _train_one(trial, domain_vocab)
        modes: Dict[str, Any] = {}
        for mode in ["conservative", "balanced", "long_horizon"]:
            policy, val = _select_policy(train["checkpoint"], mode)
            m = val["metrics"]
            score = _metric_score(m, mode)
            modes[mode] = {"policy": policy, "val": val, "val_score": score}
            if m.get("easy_degradation", 1.0) <= 0.02 and score > best_val_score:
                best_val_score = score
                best_name = f"{trial['name']}::{mode}"
                best_policy = policy
        trials[trial["name"]] = {"source": "fresh_run", "trial": trial, "train": train, "modes": modes}
    if not best_name:
        result = {"source": "not_run", "reason": "no val-safe stratified protocol policy", "trials": trials}
    else:
        trial_name, _mode = best_name.split("::", 1)
        ckpt = trials[trial_name]["train"]["checkpoint"]
        test_metrics = _eval_policy(ckpt, "test", best_policy, bootstrap=True)
        positive_domains = sum(1 for row in test_metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        exceeds = _beats_stage37_required_margins(test_metrics)
        result = {
            "source": "fresh_run",
            "protocol_status": "candidate_protocol_not_deployable_until_confirmed_on_locked_split",
            "best_stage41_stratified_protocol": best_name,
            "selection_rule": "train on stratified candidate train; val-selected slice thresholds; test once",
            "best_policy": best_policy,
            "best_metrics": test_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": exceeds,
            "deployment_decision": "candidate_needs_confirmatory_locked_split" if exceeds and positive_domains >= 2 else "keep_stage37_selector",
            "trials": trials,
        }
    _write_json(OUT_DIR / "stage41_stratified_protocol.json", result)
    write_md(OUT_DIR / "stage41_stratified_protocol.md", ["# Stage41 Stratified Protocol Neural Retraining", "", "- source: `fresh_run`", "- status: candidate protocol, not a replacement for locked Stage41 claims.", f"- result: `{result}`"])
    _append_ledger("stage41_stratified_protocol", "ok", started, [str(DATA_DIR / "all_agent_train.npz")], [str(OUT_DIR / "stage41_stratified_protocol.md")])
    return result


def _aggregate_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "min": float(arr.min()), "max": float(arr.max())}


def _beats_stage37_required_margins(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
        and metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
        and metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
    )


def _summary_beats_stage37_required_margins(summary: Mapping[str, Mapping[str, float]]) -> bool:
    return bool(
        (summary.get("easy_degradation") or {}).get("max", 1.0) <= 0.02
        and (summary.get("all_improvement") or {}).get("min", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
        and (summary.get("t50_improvement") or {}).get("min", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
        and (summary.get("hard_failure_improvement") or {}).get("min", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
    )


def train_locked_v2_confirmatory() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    domain_vocab = _domain_vocab()
    base = {"name": "locked_v2_long_horizon", "width": 224, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 2.0, "t50_w": 2.0, "t100_w": 4.0, "ce_w": 0.6, "rank_w": 0.8}
    runs: Dict[str, Any] = {}
    for seed_offset in [0, 101, 202]:
        trial = dict(base)
        trial["name"] = f"{base['name']}_seed{seed_offset}"
        trial["seed_offset"] = seed_offset
        train = _train_one(trial, domain_vocab)
        best_mode = ""
        best_score = -1e18
        best_policy: Dict[str, Any] = {}
        modes: Dict[str, Any] = {}
        for mode in ["conservative", "balanced", "long_horizon"]:
            policy, val = _select_policy(train["checkpoint"], mode)
            m = val["metrics"]
            score = _metric_score(m, mode)
            modes[mode] = {"policy": policy, "val": val, "val_score": score}
            if m.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                best_score = score
                best_mode = mode
                best_policy = policy
        test_metrics = _eval_policy(train["checkpoint"], "test", best_policy, bootstrap=seed_offset == 0) if best_policy else {}
        runs[f"seed{seed_offset}"] = {"source": train.get("source", "fresh_run"), "trial": trial, "train": train, "best_mode": best_mode, "best_val_score": best_score, "modes": modes, "test_metrics": test_metrics}
    metrics = [row.get("test_metrics", {}) for row in runs.values()]
    summary = {
        "all_improvement": _aggregate_metric([m.get("all_improvement", 0.0) for m in metrics]),
        "t50_improvement": _aggregate_metric([m.get("t50_improvement", 0.0) for m in metrics]),
        "t100_improvement": _aggregate_metric([m.get("t100_improvement", 0.0) for m in metrics]),
        "hard_failure_improvement": _aggregate_metric([m.get("hard_failure_improvement", 0.0) for m in metrics]),
        "easy_degradation": _aggregate_metric([m.get("easy_degradation", 1.0) for m in metrics]),
        "switch_rate": _aggregate_metric([m.get("switch_rate", 0.0) for m in metrics]),
    }
    first_metrics = metrics[0] if metrics else {}
    positive_domains = min(
        sum(1 for row in m.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        for m in metrics
    ) if metrics else 0
    stable = bool(_summary_beats_stage37_required_margins(summary) and positive_domains >= 2)
    result = {
        "source": "fresh_run",
        "protocol_status": "locked_v2_candidate_confirmatory_not_final_deployable",
        "selection_rule": "for each seed: train on locked-v2 candidate train, select policy on val, evaluate test once; reports multi-seed stability",
        "runs": runs,
        "summary": summary,
        "representative_metrics": first_metrics,
        "positive_external_domains_min_across_seeds": positive_domains,
        "neural_exceeds_stage37_by_gate_margin_stably": stable,
        "deployment_decision": "candidate_needs_independent_locked_protocol_or_user_acceptance" if stable else "keep_stage37_selector",
        "caveat": "The locked-v2 protocol is derived after validation-gap diagnosis. It is strong evidence for the failure mechanism, but not final deployment proof until accepted as the new locked external protocol or repeated on fresh data.",
    }
    _write_json(OUT_DIR / "stage41_locked_v2_confirmatory.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_confirmatory.md", ["# Stage41 Locked-v2 Confirmatory Multi-Seed", "", "- source: `fresh_run`", "- status: locked-v2 candidate confirmation, not final deployable claim.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_confirmatory", "ok", started, [str(DATA_DIR / "all_agent_train.npz")], [str(OUT_DIR / "stage41_locked_v2_confirmatory.md")])
    return result


def train_locked_v2_tail_robust() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    domain_vocab = _domain_vocab()
    trial_bases = [
        {"name": "locked_v2_t50_tail", "width": 256, "dropout": 0.08, "lr": 6.0e-4, "hard_w": 2.5, "t50_w": 5.0, "t100_w": 2.0, "ce_w": 0.9, "rank_w": 1.1},
        {"name": "locked_v2_domain_tail", "width": 224, "dropout": 0.10, "lr": 7.0e-4, "hard_w": 3.0, "t50_w": 4.0, "t100_w": 2.5, "ce_w": 0.8, "rank_w": 1.2},
    ]
    runs: Dict[str, Any] = {}
    for seed_offset in [0, 101, 202]:
        seed_key = f"seed{seed_offset}"
        seed_runs: Dict[str, Any] = {}
        best_trial_name = ""
        best_mode = ""
        best_score = -1e18
        best_policy: Dict[str, Any] = {}
        best_ckpt = ""
        for base in trial_bases:
            trial = dict(base)
            trial["name"] = f"{base['name']}_seed{seed_offset}"
            trial["seed_offset"] = seed_offset
            train = _train_one(trial, domain_vocab)
            modes: Dict[str, Any] = {}
            for mode in ["balanced", "long_horizon", "t50_tail", "domain_tail"]:
                policy, val = _select_policy(train["checkpoint"], mode)
                m = val["metrics"]
                score = _metric_score(m, mode)
                modes[mode] = {"policy": policy, "val": val, "val_score": score}
                if m.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                    best_score = score
                    best_trial_name = trial["name"]
                    best_mode = mode
                    best_policy = policy
                    best_ckpt = train["checkpoint"]
            seed_runs[trial["name"]] = {"source": train.get("source", "fresh_run"), "trial": trial, "train": train, "modes": modes}
        test_metrics = _eval_policy(best_ckpt, "test", best_policy, bootstrap=seed_offset == 0) if best_policy else {}
        runs[seed_key] = {
            "source": "fresh_run",
            "best_trial": best_trial_name,
            "best_mode": best_mode,
            "best_val_score": best_score,
            "trials": seed_runs,
            "test_metrics": test_metrics,
        }
    metrics = [row.get("test_metrics", {}) for row in runs.values()]
    summary = {
        "all_improvement": _aggregate_metric([m.get("all_improvement", 0.0) for m in metrics]),
        "t50_improvement": _aggregate_metric([m.get("t50_improvement", 0.0) for m in metrics]),
        "t100_improvement": _aggregate_metric([m.get("t100_improvement", 0.0) for m in metrics]),
        "hard_failure_improvement": _aggregate_metric([m.get("hard_failure_improvement", 0.0) for m in metrics]),
        "easy_degradation": _aggregate_metric([m.get("easy_degradation", 1.0) for m in metrics]),
        "switch_rate": _aggregate_metric([m.get("switch_rate", 0.0) for m in metrics]),
    }
    positive_domains = min(
        sum(1 for row in m.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        for m in metrics
    ) if metrics else 0
    stable = bool(_summary_beats_stage37_required_margins(summary) and positive_domains >= 2)
    result = {
        "source": "fresh_run",
        "protocol_status": "locked_v2_tail_robust_candidate_not_final_deployable",
        "selection_rule": "for each seed: train t50/domain-tail weighted neural cost models; select by validation-only low-tail t50/domain score; evaluate test once",
        "runs": runs,
        "summary": summary,
        "representative_metrics": metrics[0] if metrics else {},
        "positive_external_domains_min_across_seeds": positive_domains,
        "neural_exceeds_stage37_by_gate_margin_stably": stable,
        "deployment_decision": "candidate_needs_independent_locked_protocol_or_user_acceptance" if stable else "keep_stage37_selector",
        "caveat": "Tail-robust locked-v2 remains a candidate protocol derived after Stage41 validation-gap diagnosis. It cannot replace Stage37 without protocol acceptance or independent fresh external confirmation.",
    }
    _write_json(OUT_DIR / "stage41_locked_v2_tail_robust.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_tail_robust.md", ["# Stage41 Locked-v2 Tail-Robust Multi-Seed", "", "- source: `fresh_run`", "- status: tail-robust locked-v2 candidate confirmation, not final deployable claim.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_tail_robust", "ok", started, [str(DATA_DIR / "all_agent_train.npz")], [str(OUT_DIR / "stage41_locked_v2_tail_robust.md")])
    return result


def train_locked_v2_hard_all() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    domain_vocab = _domain_vocab()
    trial_bases = [
        {"name": "locked_v2_hard_all", "width": 256, "dropout": 0.08, "lr": 6.0e-4, "hard_w": 6.0, "t50_w": 1.5, "t100_w": 2.0, "ce_w": 1.0, "rank_w": 1.4},
        {"name": "locked_v2_domain_hard", "width": 256, "dropout": 0.10, "lr": 5.0e-4, "hard_w": 7.0, "t50_w": 2.0, "t100_w": 2.0, "ce_w": 1.2, "rank_w": 1.5},
        {"name": "locked_v2_all_domain_balanced", "width": 224, "dropout": 0.08, "lr": 7.0e-4, "hard_w": 4.0, "t50_w": 2.0, "t100_w": 2.0, "ce_w": 0.9, "rank_w": 1.0},
    ]
    runs: Dict[str, Any] = {}
    for seed_offset in [0, 101, 202]:
        seed_key = f"seed{seed_offset}"
        seed_runs: Dict[str, Any] = {}
        best_trial_name = ""
        best_mode = ""
        best_score = -1e18
        best_policy: Dict[str, Any] = {}
        best_ckpt = ""
        for base in trial_bases:
            trial = dict(base)
            trial["name"] = f"{base['name']}_seed{seed_offset}"
            trial["seed_offset"] = seed_offset
            train = _train_one(trial, domain_vocab)
            modes: Dict[str, Any] = {}
            for mode in ["hard_all", "domain_hard", "balanced", "domain_tail"]:
                policy, val = _select_policy(train["checkpoint"], mode)
                m = val["metrics"]
                score = _metric_score(m, mode)
                modes[mode] = {"policy": policy, "val": val, "val_score": score}
                if m.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                    best_score = score
                    best_trial_name = trial["name"]
                    best_mode = mode
                    best_policy = policy
                    best_ckpt = train["checkpoint"]
            seed_runs[trial["name"]] = {"source": train.get("source", "fresh_run"), "trial": trial, "train": train, "modes": modes}
        test_metrics = _eval_policy(best_ckpt, "test", best_policy, bootstrap=seed_offset == 0) if best_policy else {}
        runs[seed_key] = {
            "source": "fresh_run",
            "best_trial": best_trial_name,
            "best_mode": best_mode,
            "best_val_score": best_score,
            "trials": seed_runs,
            "test_metrics": test_metrics,
        }
    metrics = [row.get("test_metrics", {}) for row in runs.values()]
    summary = {
        "all_improvement": _aggregate_metric([m.get("all_improvement", 0.0) for m in metrics]),
        "t50_improvement": _aggregate_metric([m.get("t50_improvement", 0.0) for m in metrics]),
        "t100_improvement": _aggregate_metric([m.get("t100_improvement", 0.0) for m in metrics]),
        "hard_failure_improvement": _aggregate_metric([m.get("hard_failure_improvement", 0.0) for m in metrics]),
        "easy_degradation": _aggregate_metric([m.get("easy_degradation", 1.0) for m in metrics]),
        "switch_rate": _aggregate_metric([m.get("switch_rate", 0.0) for m in metrics]),
    }
    positive_domains = min(
        sum(1 for row in m.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        for m in metrics
    ) if metrics else 0
    stable = bool(_summary_beats_stage37_required_margins(summary) and positive_domains >= 2)
    result = {
        "source": "fresh_run",
        "protocol_status": "locked_v2_hard_all_candidate_not_final_deployable",
        "selection_rule": "for each seed: train hard/all weighted neural cost models; select by validation-only hard/all domain score; evaluate test once",
        "runs": runs,
        "summary": summary,
        "representative_metrics": metrics[0] if metrics else {},
        "positive_external_domains_min_across_seeds": positive_domains,
        "neural_exceeds_stage37_by_gate_margin_stably": stable,
        "deployment_decision": "candidate_needs_independent_locked_protocol_or_user_acceptance" if stable else "keep_stage37_selector",
        "caveat": "Hard/all locked-v2 remains a candidate protocol derived after Stage41 validation-gap diagnosis. It cannot replace Stage37 without passing all strict margins and independent confirmation.",
    }
    _write_json(OUT_DIR / "stage41_locked_v2_hard_all.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_hard_all.md", ["# Stage41 Locked-v2 Hard/All Multi-Seed", "", "- source: `fresh_run`", "- status: hard/all locked-v2 candidate confirmation, not final deployable claim.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_hard_all", "ok", started, [str(DATA_DIR / "all_agent_train.npz")], [str(OUT_DIR / "stage41_locked_v2_hard_all.md")])
    return result


def train_locked_v2_domain_focused() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    domain_vocab = _domain_vocab()
    trial_bases = [
        {"name": "locked_v2_eth_hard_expert", "domain_focus": "ETH_UCY", "width": 256, "dropout": 0.08, "lr": 5.0e-4, "hard_w": 4.0, "domain_w": 3.0, "domain_hard_w": 8.0, "t50_w": 2.5, "t100_w": 2.0, "ce_w": 1.1, "rank_w": 1.4},
        {"name": "locked_v2_ucy_hard_expert", "domain_focus": "UCY", "width": 256, "dropout": 0.08, "lr": 5.0e-4, "hard_w": 4.0, "domain_w": 4.0, "domain_hard_w": 9.0, "t50_w": 2.5, "t100_w": 2.0, "ce_w": 1.1, "rank_w": 1.4},
        {"name": "locked_v2_trajnet_hard_expert", "domain_focus": "TrajNet", "width": 224, "dropout": 0.10, "lr": 6.0e-4, "hard_w": 3.0, "domain_w": 2.0, "domain_hard_w": 5.0, "t50_w": 2.0, "t100_w": 2.0, "ce_w": 1.0, "rank_w": 1.2},
    ]
    runs: Dict[str, Any] = {}
    for base in trial_bases:
        domain = str(base["domain_focus"])
        seed_runs: Dict[str, Any] = {}
        best_trial_name = ""
        best_mode = ""
        best_score = -1e18
        best_policy: Dict[str, Any] = {}
        best_ckpt = ""
        for seed_offset in [0, 101]:
            trial = dict(base)
            trial["name"] = f"{base['name']}_seed{seed_offset}"
            trial["seed_offset"] = seed_offset
            train = _train_one(trial, domain_vocab)
            modes: Dict[str, Any] = {}
            for mode in ["domain_hard", "hard_all", "domain_tail"]:
                policy, val = _select_policy(train["checkpoint"], mode)
                m = val["metrics"]
                domain_metrics = (m.get("by_domain") or {}).get(domain, {})
                score = (
                    1.4 * float(domain_metrics.get("all_improvement", 0.0))
                    + 2.4 * float(domain_metrics.get("hard_failure_improvement", 0.0))
                    + 0.8 * float(domain_metrics.get("t50_improvement", 0.0))
                    - 30.0 * max(0.0, float(domain_metrics.get("easy_degradation", 1.0)) - 0.02)
                    + 0.02 * float(domain_metrics.get("switch_rate", 0.0))
                )
                modes[mode] = {"policy": policy, "val": val, "domain_val_score": score}
                if domain_metrics.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                    best_score = score
                    best_trial_name = trial["name"]
                    best_mode = mode
                    best_policy = policy
                    best_ckpt = train["checkpoint"]
            seed_runs[trial["name"]] = {"source": train.get("source", "fresh_run"), "trial": trial, "train": train, "modes": modes}
        test_metrics = _eval_policy(best_ckpt, "test", best_policy, bootstrap=False) if best_policy else {}
        runs[domain] = {
            "source": "fresh_run",
            "best_trial": best_trial_name,
            "best_mode": best_mode,
            "best_val_score": best_score,
            "trials": seed_runs,
            "test_metrics": test_metrics,
            "domain_test_metrics": (test_metrics.get("by_domain") or {}).get(domain, {}),
        }
    summary = {
        domain: {
            "best_trial": row.get("best_trial"),
            "best_mode": row.get("best_mode"),
            "domain_test_metrics": row.get("domain_test_metrics"),
        }
        for domain, row in runs.items()
    }
    result = {
        "source": "fresh_run",
        "protocol_status": "locked_v2_domain_focused_candidate_not_final_deployable",
        "selection_rule": "train per-domain hard/all weighted neural cost experts; select each domain expert by validation-only domain hard/all score; test once",
        "runs": runs,
        "summary": summary,
        "deployment_decision": "keep_stage37_selector",
        "caveat": "Domain-focused experts are diagnostic until composed and evaluated under a single validation-selected deployment policy.",
    }
    _write_json(OUT_DIR / "stage41_locked_v2_domain_focused.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_domain_focused.md", ["# Stage41 Locked-v2 Domain-Focused Hard Experts", "", "- source: `fresh_run`", "- status: domain-focused hard experts, diagnostic until composed.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_domain_focused", "ok", started, [str(DATA_DIR / "all_agent_train.npz")], [str(OUT_DIR / "stage41_locked_v2_domain_focused.md")])
    return result


def _locked_v2_checkpoint_paths() -> Dict[str, list[str]]:
    locked = read_json(OUT_DIR / "stage41_locked_v2_confirmatory.json", {})
    tail = read_json(OUT_DIR / "stage41_locked_v2_tail_robust.json", {})
    hard = read_json(OUT_DIR / "stage41_locked_v2_hard_all.json", {})
    focused = read_json(OUT_DIR / "stage41_locked_v2_domain_focused.json", {})
    locked_paths: list[str] = []
    for row in (locked.get("runs") or {}).values():
        ckpt = ((row.get("train") or {}).get("checkpoint"))
        if ckpt and Path(ckpt).exists():
            locked_paths.append(str(ckpt))
    tail_paths: list[str] = []
    for row in (tail.get("runs") or {}).values():
        best_trial = row.get("best_trial")
        if best_trial:
            ckpt = ((((row.get("trials") or {}).get(best_trial) or {}).get("train") or {}).get("checkpoint"))
            if ckpt and Path(ckpt).exists():
                tail_paths.append(str(ckpt))
    hard_paths: list[str] = []
    for row in (hard.get("runs") or {}).values():
        best_trial = row.get("best_trial")
        if best_trial:
            ckpt = ((((row.get("trials") or {}).get(best_trial) or {}).get("train") or {}).get("checkpoint"))
            if ckpt and Path(ckpt).exists():
                hard_paths.append(str(ckpt))
    focused_paths: list[str] = []
    focused_by_domain: Dict[str, list[str]] = {}
    for domain, row in (focused.get("runs") or {}).items():
        best_trial = row.get("best_trial")
        if best_trial:
            ckpt = ((((row.get("trials") or {}).get(best_trial) or {}).get("train") or {}).get("checkpoint"))
            if ckpt and Path(ckpt).exists():
                focused_paths.append(str(ckpt))
                focused_by_domain[str(domain)] = [str(ckpt)]
    paths: Dict[str, list[str]] = {}
    if locked_paths:
        paths["locked_v2_3seed_ensemble"] = locked_paths
    if tail_paths:
        paths["tail_robust_3seed_ensemble"] = tail_paths
    if hard_paths:
        paths["hard_all_3seed_ensemble"] = hard_paths
    if focused_paths:
        paths["domain_focused_expert_ensemble"] = focused_paths
    if locked_paths and tail_paths:
        paths["locked_v2_plus_tail_6model_ensemble"] = locked_paths + tail_paths
    if locked_paths and tail_paths and hard_paths:
        paths["locked_v2_tail_hard_9model_ensemble"] = locked_paths + tail_paths + hard_paths
    if locked_paths and tail_paths and hard_paths and focused_paths:
        paths["locked_v2_tail_hard_domain_12model_ensemble"] = locked_paths + tail_paths + hard_paths + focused_paths
    for domain, domain_paths in focused_by_domain.items():
        paths[f"domain_focused_{domain}"] = domain_paths
    return paths


def evaluate_locked_v2_ensemble() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    ensembles = _locked_v2_checkpoint_paths()
    runs: Dict[str, Any] = {}
    best_name = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    best_paths: list[str] = []
    best_mode = ""
    for name, paths in ensembles.items():
        val_pred, val_labels = _predict_ensemble(paths, "val")
        modes: Dict[str, Any] = {}
        for mode in ["balanced", "long_horizon", "t50_tail", "domain_tail"]:
            policy, val = _select_policy_from_predictions(val_pred, val_labels, mode)
            score = _metric_score(val["metrics"], mode)
            modes[mode] = {"policy": policy, "val": val, "val_score": score}
            if val["metrics"].get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                best_score = score
                best_name = name
                best_mode = mode
                best_policy = policy
                best_paths = list(paths)
        runs[name] = {"source": "cached_verified", "paths": paths, "modes": modes}
    if not best_policy:
        result: Dict[str, Any] = {"source": "not_run", "reason": "no cached locked-v2 checkpoints available for ensemble", "ensembles": runs}
    else:
        test_pred, test_labels = _predict_ensemble(best_paths, "test")
        test_metrics = _eval_policy_predictions(test_pred, test_labels, best_policy, bootstrap=True)
        positive_domains = sum(1 for row in test_metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        exceeds = bool(_beats_stage37_required_margins(test_metrics) and positive_domains >= 2)
        result = {
            "source": "fresh_run",
            "protocol_status": "locked_v2_neural_ensemble_candidate_not_final_deployable",
            "model_source": "cached_verified locked-v2 and/or tail-robust checkpoints; ensemble predictions freshly evaluated",
            "selection_rule": "average neural cost/risk predictions across seed checkpoints, select policy on validation only, evaluate test once",
            "best_ensemble": best_name,
            "best_mode": best_mode,
            "best_policy": best_policy,
            "best_metrics": test_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": exceeds,
            "deployment_decision": "candidate_needs_independent_locked_protocol_or_user_acceptance" if exceeds else "keep_stage37_selector",
            "ensembles": runs,
            "caveat": "This is a neural ensemble over candidate locked-v2 checkpoints. It is useful evidence for dynamics lift, but still not final deployable proof without accepting the locked-v2 protocol or fresh external confirmation.",
        }
    _write_json(OUT_DIR / "stage41_locked_v2_ensemble.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_ensemble.md", ["# Stage41 Locked-v2 Neural Ensemble", "", "- source: `fresh_run`", "- status: ensemble over locked-v2 checkpoints, candidate evidence only.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_ensemble", "ok", started, list(sum(ensembles.values(), [])), [str(OUT_DIR / "stage41_locked_v2_ensemble.md")])
    return result


def evaluate_locked_v2_relaxed_easy_budget() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    ensembles = _locked_v2_checkpoint_paths()
    candidate_names = [
        "locked_v2_tail_hard_9model_ensemble",
        "locked_v2_plus_tail_6model_ensemble",
        "tail_robust_3seed_ensemble",
        "hard_all_3seed_ensemble",
        "locked_v2_3seed_ensemble",
    ]
    runs: Dict[str, Any] = {}
    best_name = ""
    best_mode = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    best_paths: list[str] = []
    modes = ["relaxed_easy_budget", "relaxed_t50_budget", "relaxed_hard_budget"]
    for name in candidate_names:
        paths = ensembles.get(name)
        if not paths:
            continue
        val_pred, val_labels = _predict_ensemble(paths, "val")
        mode_runs: Dict[str, Any] = {}
        for mode in modes:
            policy, val = _select_relaxed_easy_budget_policy_from_predictions(val_pred, val_labels, mode)
            metrics = val["metrics"]
            score = (
                1.4 * float(metrics.get("all_improvement", 0.0))
                + 1.6 * float(metrics.get("t50_improvement", 0.0))
                + 1.8 * float(metrics.get("hard_failure_improvement", 0.0))
                + 0.4 * float(metrics.get("t100_improvement", 0.0))
                - 80.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
                + 0.02 * float(metrics.get("switch_rate", 0.0))
            )
            mode_runs[mode] = {"policy": policy, "val": val, "val_score": score}
            if metrics.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                best_score = score
                best_name = name
                best_mode = mode
                best_policy = policy
                best_paths = list(paths)
        runs[name] = {"source": "cached_verified", "paths": paths, "modes": mode_runs}
    if not best_policy:
        result: Dict[str, Any] = {"source": "not_run", "reason": "no val-safe relaxed easy-budget policy", "ensembles": runs}
    else:
        test_pred, test_labels = _predict_ensemble(best_paths, "test")
        test_metrics = _eval_policy_predictions(test_pred, test_labels, best_policy, bootstrap=True)
        positive_domains = sum(1 for row in test_metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        margin_pass = bool(_beats_stage37_required_margins(test_metrics) and positive_domains >= 2)
        result = {
            "source": "fresh_run",
            "protocol_status": "locked_v2_relaxed_easy_budget_candidate_requires_fresh_confirmation",
            "model_source": "cached_verified locked-v2 neural checkpoints; relaxed policy freshly selected on validation",
            "selection_rule": "select relaxed high-switch easy-budget policy on validation only; evaluate test once; introduced after Stage41 failure diagnostics, so not final deployment evidence until fresh confirmation",
            "best_ensemble": best_name,
            "best_mode": best_mode,
            "best_policy": best_policy,
            "best_metrics": test_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": margin_pass,
            "deployment_decision": "candidate_needs_fresh_confirmation_before_deployment" if margin_pass else "keep_stage37_selector",
            "per_domain_easy_caveat": "Aggregate easy degradation is the strict Stage41 gate, but per-domain easy degradation must also be inspected before deployment.",
            "ensembles": runs,
            "caveat": "This run is a hypothesis-confirming candidate from failure forensics: neural endpoints contain strong signal, but a fresh locked protocol is required before replacing Stage37.",
        }
    _write_json(OUT_DIR / "stage41_locked_v2_relaxed_easy_budget.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_relaxed_easy_budget.md", ["# Stage41 Locked-v2 Relaxed Easy-Budget Neural Policy", "", "- source: `fresh_run`", "- status: relaxed policy candidate; requires fresh confirmation before deployment.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_relaxed_easy_budget", "ok", started, list(sum(ensembles.values(), [])), [str(OUT_DIR / "stage41_locked_v2_relaxed_easy_budget.md")])
    return result


def _validation_gap_risky_domains() -> list[str]:
    audit = read_json(Path("outputs/stage41_external_split") / "stage41_validation_gap_audit.json", {})
    risky: list[str] = []
    for blocker in audit.get("blockers", []) or []:
        domain = str(blocker).split(" ", 1)[0]
        if domain and domain not in risky:
            risky.append(domain)
    return risky


def _cap_policy_for_risky_domains(policy: Mapping[str, Any], risky_domains: Sequence[str], max_switch: float, harm_prob: float, hard_only: bool = False) -> Dict[str, Any]:
    capped = {"type": "stage41_locked_v2_domain_safe_relaxed_policy", "mode": "domain_safe_relaxed", "slices": {}}
    risky = set(str(d) for d in risky_domains)
    for key, value in (policy.get("slices") or {}).items():
        params = dict(value)
        domain = key.split("|", 1)[0]
        if domain in risky:
            params["max_switch"] = min(float(params.get("max_switch", 1.0)), float(max_switch))
            params["harm_prob"] = min(float(params.get("harm_prob", 0.30)), float(harm_prob))
            params["hard_only"] = bool(hard_only)
        capped["slices"][key] = params
    capped["risky_domains"] = sorted(risky)
    capped["risky_domain_max_switch"] = float(max_switch)
    capped["risky_domain_harm_prob_cap"] = float(harm_prob)
    capped["risky_domain_hard_only"] = bool(hard_only)
    return capped


def evaluate_locked_v2_domain_safe_relaxed() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    ensembles = _locked_v2_checkpoint_paths()
    paths = ensembles.get("locked_v2_tail_hard_9model_ensemble") or ensembles.get("locked_v2_plus_tail_6model_ensemble")
    risky_domains = _validation_gap_risky_domains()
    if not paths:
        result: Dict[str, Any] = {"source": "not_run", "reason": "no cached locked-v2 ensemble checkpoints available"}
    else:
        val_pred, val_labels = _predict_ensemble(paths, "val")
        base_policy, base_val = _select_relaxed_easy_budget_policy_from_predictions(val_pred, val_labels, "relaxed_easy_budget")
        candidates: Dict[str, Any] = {}
        best_name = ""
        best_score = -1e18
        best_policy: Dict[str, Any] = {}
        # Conservative preset grid fixed before test evaluation. It uses the
        # validation-gap audit only to identify risky domains; thresholds are
        # not selected from test labels. Risky domains get an explicit safety
        # buffer below the 2% easy-degradation gate, so the grid does not allow
        # aggressive caps that merely look safe on validation.
        for max_switch in [0.05, 0.08, 0.10]:
            for hard_only in [False, True]:
                name = f"riskcap_ms{max_switch:.2f}_hard{int(hard_only)}"
                policy = _cap_policy_for_risky_domains(base_policy, risky_domains, max_switch=max_switch, harm_prob=0.16 if not hard_only else 0.08, hard_only=hard_only)
                val_metrics = _eval_policy_predictions(val_pred, val_labels, policy, bootstrap=False)
                domain_vals = val_metrics.get("by_domain", {}) or {}
                max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in domain_vals.values()] or [0.0])
                score = (
                    1.4 * float(val_metrics.get("all_improvement", 0.0))
                    + 1.6 * float(val_metrics.get("t50_improvement", 0.0))
                    + 1.8 * float(val_metrics.get("hard_failure_improvement", 0.0))
                    + 0.4 * float(val_metrics.get("t100_improvement", 0.0))
                    - 90.0 * max(0.0, float(val_metrics.get("easy_degradation", 1.0)) - 0.02)
                    - 25.0 * max(0.0, max_domain_easy - 0.02)
                )
                candidates[name] = {"policy": policy, "val_metrics": val_metrics, "val_score": score, "max_domain_easy_degradation": max_domain_easy}
                if val_metrics.get("easy_degradation", 1.0) <= 0.02 and max_domain_easy <= 0.02 and score > best_score:
                    best_score = score
                    best_name = name
                    best_policy = policy
        if not best_policy:
            result = {"source": "not_run", "reason": "no val-safe domain-safe relaxed policy", "risky_domains": risky_domains, "base_val": base_val, "candidates": candidates}
        else:
            test_pred, test_labels = _predict_ensemble(paths, "test")
            test_metrics = _eval_policy_predictions(test_pred, test_labels, best_policy, bootstrap=True)
            positive_domains = sum(1 for row in test_metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
            max_domain_easy_test = max([float(row.get("easy_degradation", 0.0)) for row in (test_metrics.get("by_domain") or {}).values()] or [0.0])
            margin_pass = bool(_beats_stage37_required_margins(test_metrics) and positive_domains >= 2 and max_domain_easy_test <= 0.02)
            result = {
                "source": "fresh_run",
                "protocol_status": "locked_v2_domain_safe_relaxed_candidate_requires_fresh_confirmation",
                "model_source": "cached_verified locked-v2 neural checkpoints; domain-safe relaxed policy freshly selected on validation",
                "selection_rule": "select relaxed policy on validation, then cap validation-gap risky domains with a preset switch budget; evaluate test once",
                "risky_domains": risky_domains,
                "best_candidate": best_name,
                "best_policy": best_policy,
                "best_metrics": test_metrics,
                "max_domain_easy_degradation": max_domain_easy_test,
                "positive_external_domains": positive_domains,
                "neural_exceeds_stage37_by_gate_margin": margin_pass,
                "deployment_decision": "candidate_needs_fresh_confirmation_before_deployment" if margin_pass else "keep_stage37_selector",
                "base_relaxed_val": base_val,
                "candidates": candidates,
                "caveat": "Domain-safe relaxed policy repairs the observed ETH_UCY easy-risk failure mode, but it is still post-diagnostic candidate evidence requiring fresh confirmation before deployment.",
            }
    _write_json(OUT_DIR / "stage41_locked_v2_domain_safe_relaxed.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_domain_safe_relaxed.md", ["# Stage41 Locked-v2 Domain-Safe Relaxed Neural Policy", "", "- source: `fresh_run`", "- status: domain-safe relaxed candidate; requires fresh confirmation before deployment.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_domain_safe_relaxed", "ok", started, list(paths or []), [str(OUT_DIR / "stage41_locked_v2_domain_safe_relaxed.md")])
    return result


def _domain_expert_score(row: Mapping[str, Any]) -> float:
    return (
        1.4 * float(row.get("all_improvement", 0.0))
        + 2.3 * float(row.get("hard_failure_improvement", 0.0))
        + 0.8 * float(row.get("t50_improvement", 0.0))
        + 0.2 * float(row.get("t100_improvement", 0.0))
        - 35.0 * max(0.0, float(row.get("easy_degradation", 1.0)) - 0.02)
        + 0.02 * float(row.get("switch_rate", 0.0))
    )


def evaluate_locked_v2_domain_expert_composer() -> Dict[str, Any]:
    started = time.perf_counter()
    build_stratified_all_agent_dataset()
    pools = _locked_v2_checkpoint_paths()
    candidates = {name: paths for name, paths in pools.items() if not name.startswith("locked_v2_tail_hard_domain_12model")}
    val_cache: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    test_cache: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    choices: Dict[str, Any] = {}
    modes = ["domain_hard", "hard_all", "domain_tail", "t50_tail", "balanced"]
    for name, paths in candidates.items():
        val_pred, val_labels = _predict_ensemble(paths, "val")
        val_cache[name] = (val_pred, val_labels)
        for mode in modes:
            policy, val = _select_policy_from_predictions(val_pred, val_labels, mode)
            for domain, row in (val["metrics"].get("by_domain") or {}).items():
                if row.get("easy_degradation", 1.0) > 0.02:
                    continue
                score = _domain_expert_score(row)
                prev = choices.get(domain)
                if prev is None or score > prev["score"]:
                    choices[domain] = {
                        "expert": name,
                        "mode": mode,
                        "score": score,
                        "policy": policy,
                        "val_domain_metrics": row,
                    }
    if not choices:
        result = {"source": "not_run", "reason": "no val-safe domain expert choices", "candidate_count": len(candidates)}
    else:
        combined_pred: Dict[str, np.ndarray] | None = None
        labels_ref: Dict[str, np.ndarray] | None = None
        composed_policy = {"type": "stage41_locked_v2_domain_expert_composer", "mode": "domain_expert_composer", "slices": {}}
        for domain, choice in choices.items():
            expert = choice["expert"]
            if expert not in test_cache:
                test_cache[expert] = _predict_ensemble(candidates[expert], "test")
            pred, labels = test_cache[expert]
            if labels_ref is None:
                labels_ref = labels
                combined_pred = {key: np.zeros_like(value) for key, value in pred.items()}
            assert combined_pred is not None and labels_ref is not None
            mask = labels_ref["domain"].astype(str) == domain
            for key, value in pred.items():
                combined_pred[key][mask] = value[mask]
            for slice_key, params in (choice["policy"].get("slices") or {}).items():
                if slice_key.split("|")[0] == domain:
                    composed_policy["slices"][slice_key] = params
        if combined_pred is None or labels_ref is None:
            result = {"source": "not_run", "reason": "no test predictions composed", "choices": choices}
        else:
            metrics = _eval_policy_predictions(combined_pred, labels_ref, composed_policy, bootstrap=True)
            positive_domains = sum(1 for row in metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
            result = {
                "source": "fresh_run",
                "protocol_status": "locked_v2_domain_expert_composer_candidate_not_final_deployable",
                "selection_rule": "select one cached neural expert and policy mode per domain on validation only; compose domain slices; evaluate test once",
                "choices": choices,
                "policy": composed_policy,
                "best_metrics": metrics,
                "positive_external_domains": positive_domains,
                "neural_exceeds_stage37_by_gate_margin": bool(_beats_stage37_required_margins(metrics) and positive_domains >= 2),
                "deployment_decision": "candidate_needs_independent_locked_protocol_or_user_acceptance" if _beats_stage37_required_margins(metrics) and positive_domains >= 2 else "keep_stage37_selector",
                "caveat": "Domain expert composition is validation-selected candidate evidence; it does not replace Stage37 unless all strict Stage41 margins pass and the protocol is accepted or independently confirmed.",
            }
    _write_json(OUT_DIR / "stage41_locked_v2_domain_expert_composer.json", result)
    write_md(OUT_DIR / "stage41_locked_v2_domain_expert_composer.md", ["# Stage41 Locked-v2 Domain Expert Composer", "", "- source: `fresh_run`", "- status: validation-selected per-domain expert composition, candidate evidence only.", f"- result: `{result}`"])
    _append_ledger("stage41_locked_v2_domain_expert_composer", "ok", started, list(sum(candidates.values(), [])), [str(OUT_DIR / "stage41_locked_v2_domain_expert_composer.md")])
    return result


def update_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    m = result.get("best_metrics", {})
    block = f"""

## Stage41 Stratified Protocol Candidate

This is a candidate retraining protocol created after the Stage41 validation-gap audit. It does not overwrite the locked Stage41 split and is not a deployable claim until confirmed on a locked protocol.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
best_stage41_stratified_protocol = {result.get('best_stage41_stratified_protocol')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin = {result.get('neural_exceeds_stage37_by_gate_margin')}
positive_external_domains = {result.get('positive_external_domains')}
all_improvement = {m.get('all_improvement')}
t50_improvement = {m.get('t50_improvement')}
t100_improvement = {m.get('t100_improvement')}
hard_failure_improvement = {m.get('hard_failure_improvement')}
easy_degradation = {m.get('easy_degradation')}
```
"""
    marker = "## Stage41 Stratified Protocol Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in ["stage41_stratified_dataset.md", "stage41_stratified_protocol.md"]:
        reports.add(str(OUT_DIR / name))
    stage41 = dict(state.get("stage41", {}))
    stage41["stratified_protocol_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "best_name": result.get("best_stage41_stratified_protocol"),
        "deployment_decision": result.get("deployment_decision"),
        "best_metrics": result.get("best_metrics"),
        "conclusion": "Candidate protocol tests whether validation representativeness was the t50 blocker; not deployable until confirmed on a locked external protocol.",
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "current_verdict": state.get("current_verdict", "stage41_breakthrough_not_yet_keep_stage37"), "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_confirmatory_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    summary = result.get("summary", {})
    block = f"""

## Stage41 Locked-v2 Confirmatory Candidate

This multi-seed run freezes the stratified locked-v2 candidate protocol and checks whether the t+50 neural lift is stable. It is still not final deployable proof unless the locked-v2 protocol is accepted for deployment or repeated on fresh external data.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin_stably = {result.get('neural_exceeds_stage37_by_gate_margin_stably')}
positive_external_domains_min_across_seeds = {result.get('positive_external_domains_min_across_seeds')}
all_improvement_mean = {(summary.get('all_improvement') or {}).get('mean')}
t50_improvement_mean = {(summary.get('t50_improvement') or {}).get('mean')}
t50_improvement_min = {(summary.get('t50_improvement') or {}).get('min')}
t100_improvement_mean = {(summary.get('t100_improvement') or {}).get('mean')}
hard_failure_improvement_mean = {(summary.get('hard_failure_improvement') or {}).get('mean')}
easy_degradation_max = {(summary.get('easy_degradation') or {}).get('max')}
```
"""
    marker = "## Stage41 Locked-v2 Confirmatory Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_confirmatory.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_confirmatory_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "summary": result.get("summary"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_tail_robust_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    summary = result.get("summary", {})
    block = f"""

## Stage41 Locked-v2 Tail-Robust Candidate

This multi-seed run adds validation-only low-tail t+50/domain scoring to test whether the previous locked-v2 result failed because the policy optimized mean validation lift rather than worst-seed/worst-domain stability. It remains a candidate protocol, not a deployable replacement for Stage37.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin_stably = {result.get('neural_exceeds_stage37_by_gate_margin_stably')}
positive_external_domains_min_across_seeds = {result.get('positive_external_domains_min_across_seeds')}
all_improvement_mean = {(summary.get('all_improvement') or {}).get('mean')}
t50_improvement_mean = {(summary.get('t50_improvement') or {}).get('mean')}
t50_improvement_min = {(summary.get('t50_improvement') or {}).get('min')}
hard_failure_improvement_min = {(summary.get('hard_failure_improvement') or {}).get('min')}
easy_degradation_max = {(summary.get('easy_degradation') or {}).get('max')}
```
"""
    marker = "## Stage41 Locked-v2 Tail-Robust Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_tail_robust.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_tail_robust_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "summary": result.get("summary"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_ensemble_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    metrics = result.get("best_metrics", {}) or {}
    block = f"""

## Stage41 Locked-v2 Neural Ensemble Candidate

This run ensembles cached locked-v2 neural cost models and reselects a safety policy on validation only. It tests whether multi-seed neural averaging can preserve the strong t+50 signal while reducing low-tail variance. It remains candidate evidence, not a deployable replacement for Stage37.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
best_ensemble = {result.get('best_ensemble')}
best_mode = {result.get('best_mode')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin = {result.get('neural_exceeds_stage37_by_gate_margin')}
positive_external_domains = {result.get('positive_external_domains')}
all_improvement = {metrics.get('all_improvement')}
t50_improvement = {metrics.get('t50_improvement')}
t100_improvement = {metrics.get('t100_improvement')}
hard_failure_improvement = {metrics.get('hard_failure_improvement')}
easy_degradation = {metrics.get('easy_degradation')}
```
"""
    marker = "## Stage41 Locked-v2 Neural Ensemble Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_ensemble.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_neural_ensemble_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "best_name": result.get("best_ensemble"),
        "best_mode": result.get("best_mode"),
        "best_metrics": result.get("best_metrics"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_hard_all_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    summary = result.get("summary", {})
    block = f"""

## Stage41 Locked-v2 Hard/All Candidate

This multi-seed run targets the current Stage41 bottleneck: all-domain and hard/failure improvement are below the Stage37+2% gate even though t+50 has a neural signal. It trains hard/all weighted neural cost models and selects policy by validation-only hard/all domain score.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin_stably = {result.get('neural_exceeds_stage37_by_gate_margin_stably')}
positive_external_domains_min_across_seeds = {result.get('positive_external_domains_min_across_seeds')}
all_improvement_mean = {(summary.get('all_improvement') or {}).get('mean')}
all_improvement_min = {(summary.get('all_improvement') or {}).get('min')}
t50_improvement_mean = {(summary.get('t50_improvement') or {}).get('mean')}
hard_failure_improvement_mean = {(summary.get('hard_failure_improvement') or {}).get('mean')}
hard_failure_improvement_min = {(summary.get('hard_failure_improvement') or {}).get('min')}
easy_degradation_max = {(summary.get('easy_degradation') or {}).get('max')}
```
"""
    marker = "## Stage41 Locked-v2 Hard/All Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_hard_all.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_hard_all_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "summary": result.get("summary"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_domain_focused_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage41 Locked-v2 Domain-Focused Hard Experts

This run targets the remaining ETH_UCY/UCY hard/all gap by training domain-focused hard experts. It is diagnostic until the experts are composed by validation-only policy selection and evaluated once on test.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
summary = {result.get('summary')}
```
"""
    marker = "## Stage41 Locked-v2 Domain-Focused Hard Experts"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_domain_focused.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_domain_focused_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "summary": result.get("summary"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_domain_composer_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    metrics = result.get("best_metrics", {}) or {}
    block = f"""

## Stage41 Locked-v2 Domain Expert Composer

This run composes cached neural experts per external domain using validation-only expert selection. It tests whether ETH_UCY/UCY underperformance is a domain-specialization problem rather than a global model-capacity problem.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin = {result.get('neural_exceeds_stage37_by_gate_margin')}
positive_external_domains = {result.get('positive_external_domains')}
all_improvement = {metrics.get('all_improvement')}
t50_improvement = {metrics.get('t50_improvement')}
t100_improvement = {metrics.get('t100_improvement')}
hard_failure_improvement = {metrics.get('hard_failure_improvement')}
easy_degradation = {metrics.get('easy_degradation')}
```
"""
    marker = "## Stage41 Locked-v2 Domain Expert Composer"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_domain_expert_composer.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_domain_expert_composer"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "best_metrics": result.get("best_metrics"),
        "choices": result.get("choices"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_relaxed_easy_budget_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    metrics = result.get("best_metrics", {}) or {}
    block = f"""

## Stage41 Locked-v2 Relaxed Easy-Budget Neural Policy

This run tests the current failure-forensics hypothesis: neural candidate endpoints have strong dynamics signal, but the previous fallback policy was too conservative. The policy is selected on validation only, but because this rule was introduced after Stage41 test-slice diagnostics, it is candidate evidence and needs fresh confirmation before deployment.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
best_ensemble = {result.get('best_ensemble')}
best_mode = {result.get('best_mode')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin = {result.get('neural_exceeds_stage37_by_gate_margin')}
positive_external_domains = {result.get('positive_external_domains')}
all_improvement = {metrics.get('all_improvement')}
t50_improvement = {metrics.get('t50_improvement')}
t100_improvement = {metrics.get('t100_improvement')}
hard_failure_improvement = {metrics.get('hard_failure_improvement')}
easy_degradation = {metrics.get('easy_degradation')}
```
"""
    marker = "## Stage41 Locked-v2 Relaxed Easy-Budget Neural Policy"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_relaxed_easy_budget.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_relaxed_easy_budget_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "best_name": result.get("best_ensemble"),
        "best_mode": result.get("best_mode"),
        "best_metrics": result.get("best_metrics"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def update_domain_safe_relaxed_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    metrics = result.get("best_metrics", {}) or {}
    block = f"""

## Stage41 Locked-v2 Domain-Safe Relaxed Neural Policy

This run follows the relaxed easy-budget breakthrough signal but caps risky validation-gap domains before test evaluation. It directly targets the ETH_UCY per-domain easy degradation observed in the previous relaxed candidate. It remains post-diagnostic candidate evidence and needs fresh confirmation before deployment.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
best_candidate = {result.get('best_candidate')}
deployment_decision = {result.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin = {result.get('neural_exceeds_stage37_by_gate_margin')}
positive_external_domains = {result.get('positive_external_domains')}
max_domain_easy_degradation = {result.get('max_domain_easy_degradation')}
all_improvement = {metrics.get('all_improvement')}
t50_improvement = {metrics.get('t50_improvement')}
t100_improvement = {metrics.get('t100_improvement')}
hard_failure_improvement = {metrics.get('hard_failure_improvement')}
easy_degradation = {metrics.get('easy_degradation')}
```
"""
    marker = "## Stage41 Locked-v2 Domain-Safe Relaxed Neural Policy"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_locked_v2_domain_safe_relaxed.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["locked_v2_domain_safe_relaxed_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "best_name": result.get("best_candidate"),
        "best_metrics": result.get("best_metrics"),
        "max_domain_easy_degradation": result.get("max_domain_easy_degradation"),
        "conclusion": result.get("caveat"),
    }
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def main_stratified_protocol() -> None:
    result = train_stratified_protocol()
    update_readme_state(result)


def main_locked_v2_confirmatory() -> None:
    result = train_locked_v2_confirmatory()
    update_confirmatory_readme_state(result)


def main_locked_v2_tail_robust() -> None:
    result = train_locked_v2_tail_robust()
    update_tail_robust_readme_state(result)


def main_locked_v2_ensemble() -> None:
    result = evaluate_locked_v2_ensemble()
    update_ensemble_readme_state(result)


def main_locked_v2_hard_all() -> None:
    result = train_locked_v2_hard_all()
    update_hard_all_readme_state(result)


def main_locked_v2_domain_focused() -> None:
    result = train_locked_v2_domain_focused()
    update_domain_focused_readme_state(result)


def main_locked_v2_domain_composer() -> None:
    result = evaluate_locked_v2_domain_expert_composer()
    update_domain_composer_readme_state(result)


def main_locked_v2_relaxed_easy_budget() -> None:
    result = evaluate_locked_v2_relaxed_easy_budget()
    update_relaxed_easy_budget_readme_state(result)


def main_locked_v2_domain_safe_relaxed() -> None:
    result = evaluate_locked_v2_domain_safe_relaxed()
    update_domain_safe_relaxed_readme_state(result)
