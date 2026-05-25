from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_source_level_validation_repair as slv


DATA_DIR = Path("data/stage41_world_model")
OUT_DIR = Path("outputs/stage41_domain_local")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage41_domain_local_neural_retrain.json"
REPORT_MD = OUT_DIR / "stage41_domain_local_neural_retrain.md"
THREADS = 4
BATCH = 2048
EPOCHS = 4
SEED = 41531
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


def _torch():
    torch = s41._torch()
    torch.set_num_threads(THREADS)
    return torch


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
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _load_split(split: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"seq2seq_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage41 seq2seq dataset: {path}")
    return dict(np.load(path))


def _load_combined() -> dict[str, np.ndarray]:
    path = DATA_DIR / "combined_external.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage41 combined external data: {path}")
    return dict(np.load(path))


def _seq_summary(seq: np.ndarray) -> np.ndarray:
    valid = seq[:, :, -1] > 0.5
    denom = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    masked = np.where(valid[:, :, None], seq[:, :, :6], 0.0)
    mean = masked.sum(axis=1) / denom
    last_idx = np.maximum(valid.sum(axis=1) - 1, 0).astype(int)
    last = seq[np.arange(len(seq)), last_idx, :6]
    first = seq[:, 0, :6]
    span = last - first
    return np.concatenate([mean, last, span, valid.sum(axis=1, keepdims=True).astype(np.float32) / seq.shape[1]], axis=1).astype(np.float32)


def _domain_mask(data: Mapping[str, np.ndarray], domain: str) -> np.ndarray:
    return data["domain"].astype(str) == domain


def _feature_matrix(data: Mapping[str, np.ndarray]) -> np.ndarray:
    horizon = data["horizon"].astype(np.float32)
    horizon_onehot = np.stack([(horizon == h).astype(np.float32) for h in [10, 25, 50, 100]], axis=1)
    cand = data["cand_delta"].astype(np.float32).reshape(len(horizon), -1)
    return np.concatenate(
        [
            _seq_summary(data["seq"].astype(np.float32)),
            data["static"].astype(np.float32),
            cand,
            horizon_onehot,
        ],
        axis=1,
    ).astype(np.float32)


def _subset(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v) for k, v in data.items()}


def _source_key_array(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([slv._source_key(src) for src in data["source_file"].astype(str)], dtype="U256")


def _normalizer_from_combined(data: Mapping[str, np.ndarray]) -> np.ndarray:
    hist_path = np.maximum(data["history_scalar"][:, 0].astype(np.float32), EPS)
    speed = np.maximum(data["history_seq"][:, -1, 2].astype(np.float32), EPS)
    horizon = np.maximum(data["horizon"].astype(np.float32), 1.0)
    return np.maximum(hist_path + speed * horizon, 1e-3).astype(np.float32)


def _combined_to_endpoint_dataset(data: Mapping[str, np.ndarray], mask: np.ndarray, train_mask: np.ndarray) -> dict[str, np.ndarray]:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    fut = np.stack([data["future_endpoint_x"], data["future_endpoint_y"]], axis=1).astype(np.float32)
    normalizer = _normalizer_from_combined(data)
    _floor_idx, floor_fde, floor_pred, _strongest_by_h, _diag = s41._train_horizon_floor(data, train_mask)
    family_pred = data["family_pred"].astype(np.float32)
    family_fde = data["family_fde"].astype(np.float32)
    candidates = np.concatenate([floor_pred[:, None, :], family_pred], axis=1)
    cand_delta = ((candidates - cur[:, None, :]) / normalizer[:, None, None]).astype(np.float32)
    static = np.concatenate(
        [
            data["stage37_features"].astype(np.float32),
            data["history_scalar"].astype(np.float32),
            data["prototype_likelihood"].astype(np.float32),
            data["prototype_entropy"][:, None].astype(np.float32),
            data["goal_ambiguity"][:, None].astype(np.float32),
            data["horizon"].astype(np.float32)[:, None] / 100.0,
        ],
        axis=1,
    ).astype(np.float32)
    out = {
        "seq": data["history_seq"].astype(np.float32),
        "static": static,
        "target_delta": ((fut - cur) / normalizer[:, None]).astype(np.float32),
        "cand_delta": cand_delta,
        "candidate_fde": np.concatenate([floor_fde[:, None], family_fde], axis=1).astype(np.float32),
        "floor_fde": floor_fde.astype(np.float32),
        "normalizer": normalizer.astype(np.float32),
        "current_xy": cur,
        "future_xy": fut,
        "horizon": data["horizon"].astype(np.int16),
        "hard": data["hard"].astype(bool),
        "easy": data["easy"].astype(bool),
        "failure": data["failure"].astype(bool),
        "domain": data["dataset"].astype("U32"),
        "scene_id": data["scene_id"].astype("U80"),
        "source_file": data["source_file"].astype("U256"),
    }
    return _subset(out, mask)


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-3).astype(np.float32)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def _make_endpoint_model(dim: int, width: int = 96, dropout: float = 0.05):
    torch = _torch()
    import torch.nn as nn

    class DomainEndpointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.GELU(),
                nn.LayerNorm(width),
            )
            self.delta = nn.Linear(width, 2)
            self.log_uncertainty = nn.Linear(width, 1)

        def forward(self, x):
            h = self.net(x)
            return self.delta(h), self.log_uncertainty(h).squeeze(-1)

    return DomainEndpointModel()


def _train_endpoint(domain: str, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    x_raw = _feature_matrix(train)
    mean, std = _fit_standardizer(x_raw)
    x = torch.tensor(_standardize(x_raw, mean, std))
    y = torch.tensor(train["target_delta"].astype(np.float32))
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    vx = torch.tensor(_standardize(_feature_matrix(val), mean, std))
    vy = torch.tensor(val["target_delta"].astype(np.float32))
    domain_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(domain)) % 1000
    torch.manual_seed(SEED + domain_seed)
    model = _make_endpoint_model(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + domain_seed)
    ckpt = CHECKPOINT_DIR / f"stage41_domain_local_{domain}.pt"
    heartbeat = OUT_DIR / f"stage41_domain_local_{domain}_heartbeat.json"
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            pred, log_u = model(x[ids])
            row_w = 1.0 + 1.5 * hard[ids]
            mse = (F.smooth_l1_loss(pred, y[ids], reduction="none").mean(dim=1) * row_w).mean()
            unc_target = torch.clamp(torch.linalg.norm(pred.detach() - y[ids], dim=1), min=1e-4)
            unc_loss = F.smooth_l1_loss(torch.exp(log_u), unc_target)
            loss = mse + 0.15 * unc_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred, log_u = model(vx)
            val_loss = float((F.smooth_l1_loss(pred, vy) + 0.05 * torch.mean(torch.exp(log_u))).cpu())
        heartbeat.write_text(
            json.dumps({"domain": domain, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "dim": int(x.shape[1]), "mean": mean, "std": std, "domain": domain, "best": best}, ckpt)
    return {"checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best, "feature_dim": int(x.shape[1])}


def _predict_endpoint(path: str | Path, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _make_endpoint_model(int(payload["dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    x = torch.tensor(_standardize(_feature_matrix(data), payload["mean"], payload["std"]))
    deltas: list[np.ndarray] = []
    uncert: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            pred, log_u = model(x[start : start + 4096])
            deltas.append(pred.cpu().numpy())
            uncert.append(np.exp(log_u.cpu().numpy()))
    return {"delta": np.concatenate(deltas).astype(np.float32), "uncertainty": np.concatenate(uncert).astype(np.float32)}


def _endpoint_fde(delta: np.ndarray, data: Mapping[str, np.ndarray]) -> np.ndarray:
    pred_xy = data["current_xy"].astype(np.float64) + delta.astype(np.float64) * data["normalizer"].astype(np.float64)[:, None]
    return np.linalg.norm(pred_xy - data["future_xy"].astype(np.float64), axis=1)


def _metrics(selected_fde: np.ndarray, floor_fde: np.ndarray, data: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    ds = {
        "horizon": data["horizon"],
        "hard": data["hard"],
        "failure": data["failure"],
        "easy": data["easy"],
        "domain": data["domain"],
        "candidate_fde": data["candidate_fde"],
    }
    return s41._metrics(selected_fde.astype(np.float64), floor_fde.astype(np.float64), ds, switch.astype(bool))


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    return (xb @ w).astype(np.float64)


def _gate_features(data: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], model_fde: np.ndarray) -> np.ndarray:
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float32)
    delta = pred["delta"].astype(np.float32)
    uncertainty = pred["uncertainty"][:, None].astype(np.float32)
    distance_from_floor = np.linalg.norm((delta - floor_delta).astype(np.float32), axis=1, keepdims=True)
    horizon = data["horizon"].astype(np.float32)
    horizon_onehot = np.stack([(horizon == h).astype(np.float32) for h in [10, 25, 50, 100]], axis=1)
    rel_model_fde = (model_fde / np.maximum(data["normalizer"], EPS))[:, None].astype(np.float32)
    return np.concatenate(
        [
            _seq_summary(data["seq"].astype(np.float32)),
            data["static"].astype(np.float32),
            delta,
            floor_delta,
            uncertainty,
            distance_from_floor,
            rel_model_fde,
            horizon_onehot,
        ],
        axis=1,
    ).astype(np.float32)


def _train_gate(train: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], model_fde: np.ndarray) -> dict[str, Any]:
    x_raw = _gate_features(train, pred, model_fde)
    mean, std = _fit_standardizer(x_raw)
    x = _standardize(x_raw, mean, std)
    gain = train["floor_fde"].astype(np.float64) - model_fde.astype(np.float64)
    harm = model_fde.astype(np.float64) - train["floor_fde"].astype(np.float64)
    return {"mean": mean, "std": std, "gain_w": _ridge_fit(x, gain), "harm_w": _ridge_fit(x, harm)}


def _predict_gate(gate: Mapping[str, Any], data: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], model_fde: np.ndarray) -> dict[str, np.ndarray]:
    x = _standardize(_gate_features(data, pred, model_fde), gate["mean"], gate["std"])
    return {"pred_gain": _ridge_predict(x, gate["gain_w"]), "pred_harm": _ridge_predict(x, gate["harm_w"])}


def _policy_grid(gate_pred: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray]) -> list[dict[str, float]]:
    gain_grid = [0.0]
    gain_grid.extend(float(v) for v in np.quantile(gate_pred["pred_gain"], [0.50, 0.65, 0.80]))
    harm_grid = [0.0]
    harm_grid.extend(float(v) for v in np.quantile(gate_pred["pred_harm"], [0.20, 0.40, 0.60]))
    uncertainty_grid = [float(v) for v in np.quantile(pred["uncertainty"], [0.35, 0.50, 0.65, 0.80])]
    policies = []
    for gain_min in gain_grid:
        for harm_max in harm_grid:
            for uncertainty_max in uncertainty_grid:
                for alpha in [0.25, 0.50, 0.75, 1.00]:
                    policies.append({"gain_min": gain_min, "harm_max": harm_max, "uncertainty_max": uncertainty_max, "alpha": alpha})
    return policies


def _apply_policy(data: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
    switch = (
        (gate_pred["pred_gain"] >= float(policy["gain_min"]))
        & (gate_pred["pred_harm"] <= float(policy["harm_max"]))
        & (pred["uncertainty"] <= float(policy["uncertainty_max"]))
    )
    alpha = float(policy["alpha"])
    selected_delta = floor_delta.copy()
    selected_delta[switch] = floor_delta[switch] + alpha * (pred["delta"].astype(np.float64)[switch] - floor_delta[switch])
    return selected_delta, switch


def _eligible(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("switch_rate", 0.0) > 0
    )


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.4 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
    )


def _select_gate_policy(val: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy in _policy_grid(gate_pred, pred):
        selected_delta, switch = _apply_policy(val, pred, gate_pred, policy)
        selected_fde = _endpoint_fde(selected_delta, val)
        metrics = _metrics(selected_fde, val["floor_fde"], val, switch)
        rows.append({"policy": _jsonable(policy), "metrics": metrics, "eligible": _eligible(metrics), "score": _score(metrics)})
    pool = [row for row in rows if row["eligible"]] or rows
    selected = max(pool, key=lambda row: row["score"])
    return {"selected": selected, "candidate_count": len(rows), "eligible_count": int(sum(row["eligible"] for row in rows)), "top_candidates": sorted(rows, key=lambda row: row["score"], reverse=True)[:20]}


def _evaluate_domain(
    domain: str,
    train_all: Mapping[str, np.ndarray],
    val_all: Mapping[str, np.ndarray],
    test_all: Mapping[str, np.ndarray],
    *,
    prefiltered: bool = False,
) -> dict[str, Any]:
    if prefiltered:
        train, val, test = dict(train_all), dict(val_all), dict(test_all)
    else:
        train = _subset(train_all, _domain_mask(train_all, domain))
        val = _subset(val_all, _domain_mask(val_all, domain))
        test = _subset(test_all, _domain_mask(test_all, domain))
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 100:
        return {"domain": domain, "status": "not_run", "reason": "not enough train/val/test rows for domain-local neural retrain"}
    training = _train_endpoint(domain, train, val)
    pred_train = _predict_endpoint(training["checkpoint"], train)
    pred_val = _predict_endpoint(training["checkpoint"], val)
    pred_test = _predict_endpoint(training["checkpoint"], test)
    fde_train = _endpoint_fde(pred_train["delta"], train)
    fde_val = _endpoint_fde(pred_val["delta"], val)
    fde_test = _endpoint_fde(pred_test["delta"], test)
    direct_val = _metrics(fde_val, val["floor_fde"], val, np.ones(len(fde_val), dtype=bool))
    direct_test = _metrics(fde_test, test["floor_fde"], test, np.ones(len(fde_test), dtype=bool))
    gate = _train_gate(train, pred_train, fde_train)
    gate_val = _predict_gate(gate, val, pred_val, fde_val)
    selection = _select_gate_policy(val, pred_val, gate_val)
    gate_test = _predict_gate(gate, test, pred_test, fde_test)
    selected_delta, switch = _apply_policy(test, pred_test, gate_test, selection["selected"]["policy"])
    selected_fde = _endpoint_fde(selected_delta, test)
    gated_test = _metrics(selected_fde, test["floor_fde"], test, switch)
    pass_gate = bool(_eligible(gated_test))
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "t50_rows": {"train": int(np.sum(train["horizon"] == 50)), "val": int(np.sum(val["horizon"] == 50)), "test": int(np.sum(test["horizon"] == 50))},
        "t100_rows": {"train": int(np.sum(train["horizon"] == 100)), "val": int(np.sum(val["horizon"] == 100)), "test": int(np.sum(test["horizon"] == 100))},
        "training": training,
        "direct_neural_without_fallback_val": direct_val,
        "direct_neural_without_fallback_test": direct_test,
        "gate_selection": selection,
        "gated_neural_with_floor_test": gated_test,
        "domain_local_endpoint_gate": pass_gate,
        "caveat": "Domain-local endpoint model is trained from causal seq2seq features only, but this report is endpoint-FDE-only and does not replace the all-agent composite policy without joint proximity/world-state audit.",
    }


def _pure_ucy_expanded_datasets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = _load_combined()
    keys = _source_key_array(data)
    train_mask = np.isin(keys, ["UCY/students01/students001-trajnet.txt", "UCY/students03/obsmat.txt"])
    val_mask = keys == "UCY/zara01/obsmat.txt"
    test_mask = np.isin(keys, ["UCY/zara02/obsmat.txt", "UCY/zara03/crowds_zara03.txt"])
    return (
        _combined_to_endpoint_dataset(data, train_mask, train_mask),
        _combined_to_endpoint_dataset(data, val_mask, train_mask),
        _combined_to_endpoint_dataset(data, test_mask, train_mask),
    )


def _evaluate_pure_ucy_expanded() -> dict[str, Any]:
    train, val, test = _pure_ucy_expanded_datasets()
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 100:
        return {"domain": "UCY_expanded", "status": "not_run", "reason": "not enough pure UCY expanded rows"}
    row = _evaluate_domain("UCY_expanded", train, val, test, prefiltered=True)
    row["protocol"] = "pure_ucy_expanded_students_train_zara_val_test_neural_endpoint_retrain"
    row["strict_pure_ucy_endpoint_neural_retrain_gate"] = bool(row.get("domain_local_endpoint_gate"))
    row["source_split"] = {
        "train": ["UCY/students01/students001-trajnet.txt", "UCY/students03/obsmat.txt"],
        "val": ["UCY/zara01/obsmat.txt"],
        "test": ["UCY/zara02/obsmat.txt", "UCY/zara03/crowds_zara03.txt"],
    }
    return row


def run_domain_local_neural_retrain() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")
    domains = [str(d) for d in sorted(set(train["domain"].astype(str)) & set(val["domain"].astype(str)) & set(test["domain"].astype(str)))]
    results = {domain: _evaluate_domain(domain, train, val, test) for domain in domains}
    pure_ucy_expanded = _evaluate_pure_ucy_expanded()
    positive_domains = [
        domain
        for domain, row in results.items()
        if row.get("domain_local_endpoint_gate")
    ]
    if pure_ucy_expanded.get("domain_local_endpoint_gate"):
        positive_domains.append("UCY_expanded")
    result = {
        "source": "fresh_run",
        "protocol": "domain_local_causal_neural_endpoint_retrain",
        "domains": domains,
        "positive_domains": positive_domains,
        "positive_domain_count": int(len(positive_domains)),
        "two_domain_endpoint_gate": bool(len(positive_domains) >= 2),
        "domain_results": results,
        "pure_ucy_expanded_neural_retrain": pure_ucy_expanded,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is a strict domain-local neural endpoint retrain probe over Stage41 seq2seq causal features. It is not metric, not seconds-level, not true 3D, not foundation, and not a Stage5C/SMC execution.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Domain-Local Neural Endpoint Retrain",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive_domains}`",
        f"- two-domain endpoint gate: `{result['two_domain_endpoint_gate']}`",
        "",
        "| domain | rows train/val/test | direct all | direct t50 | gated all | gated t50 | gated t100 | gated hard | easy | switch | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `{row.get('reason')}` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `False` |")
            continue
        dm = row["direct_neural_without_fallback_test"]
        gm = row["gated_neural_with_floor_test"]
        lines.append(
            f"| `{domain}` | `{row['rows']['train']}/{row['rows']['val']}/{row['rows']['test']}` | "
            f"{float(dm.get('all_improvement', 0.0)):.4f} | {float(dm.get('t50_improvement', 0.0)):.4f} | "
            f"{float(gm.get('all_improvement', 0.0)):.4f} | {float(gm.get('t50_improvement', 0.0)):.4f} | "
            f"{float(gm.get('t100_improvement', 0.0)):.4f} | {float(gm.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(gm.get('easy_degradation', 0.0)):.4f} | {float(gm.get('switch_rate', 0.0)):.4f} | "
            f"`{row['domain_local_endpoint_gate']}` |"
        )
    if pure_ucy_expanded.get("status") == "ok":
        dm = pure_ucy_expanded["direct_neural_without_fallback_test"]
        gm = pure_ucy_expanded["gated_neural_with_floor_test"]
        lines.append(
            f"| `UCY_expanded` | `{pure_ucy_expanded['rows']['train']}/{pure_ucy_expanded['rows']['val']}/{pure_ucy_expanded['rows']['test']}` | "
            f"{float(dm.get('all_improvement', 0.0)):.4f} | {float(dm.get('t50_improvement', 0.0)):.4f} | "
            f"{float(gm.get('all_improvement', 0.0)):.4f} | {float(gm.get('t50_improvement', 0.0)):.4f} | "
            f"{float(gm.get('t100_improvement', 0.0)):.4f} | {float(gm.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(gm.get('easy_degradation', 0.0)):.4f} | {float(gm.get('switch_rate', 0.0)):.4f} | "
            f"`{pure_ucy_expanded['domain_local_endpoint_gate']}` |"
        )
    lines.extend(
        [
            "",
            "This directly trains neural endpoint dynamics per domain using causal seq2seq inputs. Future endpoints are labels/evaluation only. Because this is endpoint-FDE-only, deployment still requires the protected all-agent composite world-state path.",
            "",
            f"- no leakage: `{result['no_leakage']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_domain_local_neural_retrain() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_domain_local_neural_retrain()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_domain_local_neural_retrain",
            status,
            started,
            [DATA_DIR / "seq2seq_train.npz", DATA_DIR / "seq2seq_val.npz", DATA_DIR / "seq2seq_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_domain_local_neural_retrain()
