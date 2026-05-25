from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_source_level_validation_repair as slv


OUT_DIR = Path("outputs/stage41_external_split")
DATA_DIR = Path("data/stage41_pure_ucy_neural_retrain")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage41_pure_ucy_neural_retrain.json"
REPORT_MD = OUT_DIR / "stage41_pure_ucy_neural_retrain.md"
EPS = 1e-6
SEED = 415101
THREADS = 4
BATCH = 512
EPOCHS = 4


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
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _torch():
    torch = s41._torch()
    torch.set_num_threads(THREADS)
    return torch


def _source_key(source_file: str) -> str:
    return slv._source_key(source_file)


def _pure_ucy_split(source_file: str) -> str:
    source = _source_key(source_file)
    if source in {"UCY/students01/students001-trajnet.txt", "UCY/students03/obsmat.txt"}:
        return "train"
    if source == "UCY/zara01/obsmat.txt":
        return "val"
    if source in {"UCY/zara02/obsmat.txt", "UCY/zara03/crowds_zara03.txt"}:
        return "test"
    return "unused"


def _split_index(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([_pure_ucy_split(src) for src in data["source_file"].astype(str)], dtype="U8")


def _source_inventory(data: Mapping[str, np.ndarray], split: np.ndarray) -> dict[str, Any]:
    source = np.asarray([_source_key(src) for src in data["source_file"].astype(str)], dtype="U256")
    horizon = data["horizon"].astype(int)
    out: dict[str, Any] = {}
    for sp in ["train", "val", "test", "unused"]:
        rows = []
        for src in sorted(set(source[split == sp].tolist())):
            mask = (split == sp) & (source == src)
            if src.startswith("UCY/") or sp != "unused":
                rows.append(
                    {
                        "source": src,
                        "rows": int(np.sum(mask)),
                        "t10": int(np.sum(mask & (horizon == 10))),
                        "t25": int(np.sum(mask & (horizon == 25))),
                        "t50": int(np.sum(mask & (horizon == 50))),
                        "t100": int(np.sum(mask & (horizon == 100))),
                        "is_pure_ucy_source": src.startswith("UCY/"),
                    }
                )
        out[sp] = rows
    return out


def _stats(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    horizon = data["horizon"].astype(int)[mask]
    valid_len = data["history_seq"][mask, :, -1].sum(axis=1) if np.any(mask) else np.zeros(0)
    source = np.asarray([_source_key(src) for src in data["source_file"].astype(str)[mask]], dtype="U256")
    return {
        "rows": int(np.sum(mask)),
        "sources": int(len(set(source.tolist()))),
        "t10": int(np.sum(horizon == 10)),
        "t25": int(np.sum(horizon == 25)),
        "t50": int(np.sum(horizon == 50)),
        "t100": int(np.sum(horizon == 100)),
        "hard": int(np.sum(data["hard"].astype(bool)[mask])),
        "easy": int(np.sum(data["easy"].astype(bool)[mask])),
        "failure": int(np.sum(data["failure"].astype(bool)[mask])),
        "history_len_mean": float(np.mean(valid_len)) if len(valid_len) else 0.0,
        "history_ge_32": int(np.sum(valid_len >= 32)),
        "history_ge_64": int(np.sum(valid_len >= 64)),
    }


def _make_seq2seq_dataset() -> dict[str, Any]:
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)
    data = s41._combined()
    split = _split_index(data)
    train_mask = split == "train"
    if not np.any(train_mask) or not np.any(split == "val") or not np.any(split == "test"):
        raise RuntimeError("Strict pure-UCY train/val/test sources are required.")

    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    fut = np.stack([data["future_endpoint_x"], data["future_endpoint_y"]], axis=1).astype(np.float32)
    hist_path = np.maximum(data["history_scalar"][:, 0].astype(np.float32), EPS)
    speed = np.maximum(data["history_seq"][:, -1, 2].astype(np.float32), EPS)
    horizon = data["horizon"].astype(np.float32)
    train_scale = hist_path[train_mask] + speed[train_mask] * np.maximum(horizon[train_mask], 1.0)
    train_median_scale = float(np.median(train_scale)) if len(train_scale) else 1.0
    normalizer = np.maximum(hist_path + speed * np.maximum(horizon, 1.0), train_median_scale + EPS).astype(np.float32)
    floor_idx, floor_fde, floor_pred, strongest_by_h, geometry = s41._train_horizon_floor(data, train_mask)
    family_pred = data["family_pred"].astype(np.float32)
    family_fde = data["family_fde"].astype(np.float32)
    candidates = np.concatenate([floor_pred[:, None, :], family_pred], axis=1)
    candidate_fde = np.concatenate([floor_fde[:, None], family_fde], axis=1)
    cand_delta = ((candidates - cur[:, None, :]) / normalizer[:, None, None]).astype(np.float32)
    target_delta = ((fut - cur) / normalizer[:, None]).astype(np.float32)
    oracle_idx = np.argmin(candidate_fde, axis=1).astype(np.int64)
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

    split_index_path = DATA_DIR / "pure_ucy_split_index.npz"
    np.savez_compressed(
        split_index_path,
        row_id=data["row_id"].astype(np.int64),
        split=split,
        source_file=data["source_file"].astype("U256"),
        scene_id=data["scene_id"].astype("U80"),
        domain=data["dataset"].astype("U32"),
    )
    reports: dict[str, Any] = {}
    for sp in ["train", "val", "test"]:
        ids = np.where(split == sp)[0].astype(np.int64)
        np.savez_compressed(
            DATA_DIR / f"seq2seq_{sp}.npz",
            ids=ids,
            seq=data["history_seq"][ids].astype(np.float32),
            static=static[ids],
            target_delta=target_delta[ids],
            cand_delta=cand_delta[ids],
            candidate_fde=candidate_fde[ids],
            floor_fde=floor_fde[ids].astype(np.float32),
            oracle_idx=oracle_idx[ids],
            normalizer=normalizer[ids],
            current_xy=cur[ids],
            future_xy=fut[ids],
            horizon=data["horizon"].astype(np.int16)[ids],
            hard=data["hard"].astype(bool)[ids],
            easy=data["easy"].astype(bool)[ids],
            failure=data["failure"].astype(bool)[ids],
            domain=data["dataset"].astype("U32")[ids],
            scene_id=data["scene_id"].astype("U80")[ids],
            source_file=data["source_file"].astype("U256")[ids],
        )
        reports[sp] = _stats(data, split == sp)
    train = dict(np.load(DATA_DIR / "seq2seq_train.npz", allow_pickle=True))
    mean = train["static"].mean(axis=0).astype(np.float32)
    std = np.maximum(train["static"].std(axis=0), 1e-3).astype(np.float32)
    np.savez_compressed(DATA_DIR / "normalization.npz", static_mean=mean, static_std=std)
    result = {
        "source": "fresh_run",
        "protocol": "strict_pure_ucy_seq2seq_neural_world_model_dataset",
        "splits": reports,
        "source_inventory": _source_inventory(data, split),
        "strongest_by_horizon_train_only": strongest_by_h,
        "floor_geometry": geometry,
        "train_only_normalizer": {"median_history_speed_horizon_scale": train_median_scale},
        "input_modalities": [
            "past K=64 target-agent history",
            "past-only density/TTC/neighbor scalar features",
            "scene-agnostic goal prototype proxy",
            "causal baseline rollouts",
            "horizon metadata",
        ],
        "outputs": ["future endpoint delta", "candidate baseline relative error", "failure/gain/harm/interaction/physical proxy heads"],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "candidate_fde_input": False,
            "candidate_fde_label_only": True,
            "train_only_floor_selection": True,
            "train_only_normalization_statistics": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    _write_json(OUT_DIR / "stage41_pure_ucy_neural_dataset.json", result)
    write_md(
        OUT_DIR / "stage41_pure_ucy_neural_dataset.md",
        [
            "# Stage41 Strict Pure-UCY Neural Dataset",
            "",
            "- source: `fresh_run`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "- metric/seconds claim: `False`",
            f"- splits: `{reports}`",
            f"- train-only strongest floor: `{strongest_by_h}`",
            f"- no leakage: `{result['no_leakage']}`",
        ],
    )
    return result


def _ds(split: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"seq2seq_{split}.npz"
    if not path.exists():
        _make_seq2seq_dataset()
    return dict(np.load(path, allow_pickle=True))


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz", allow_pickle=True))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _load_tensors(split: str) -> dict[str, Any]:
    torch = _torch()
    ds = _ds(split)
    return {
        "seq": torch.tensor(ds["seq"].astype(np.float32)),
        "static": torch.tensor(_norm_static(ds["static"])),
        "cand_delta": torch.tensor(ds["cand_delta"].astype(np.float32)),
        "target_delta": torch.tensor(ds["target_delta"].astype(np.float32)),
        "candidate_rel": torch.tensor(np.log1p(np.clip(ds["candidate_fde"].astype(np.float32) / np.maximum(ds["normalizer"].astype(np.float32)[:, None], EPS), 0.0, 1e6))),
        "oracle": torch.tensor(ds["oracle_idx"].astype(np.int64)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"].astype(bool).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
        "failure": torch.tensor(ds["failure"].astype(bool).astype(np.float32)),
    }


def _trial_configs() -> list[dict[str, Any]]:
    return [
        {"trial_id": 1, "name": "pure_ucy_transformer", "kind": "transformer", "width": 64, "layers": 1, "lr": 1.5e-3, "endpoint_weight": 0.8, "score_weight": 1.2, "hard_weight": 1.5, "t50_weight": 2.0, "t100_weight": 0.8, "jepa_weight": 0.05},
        {"trial_id": 2, "name": "pure_ucy_t50_hard_transformer", "kind": "transformer", "width": 72, "layers": 1, "lr": 1.2e-3, "endpoint_weight": 0.7, "score_weight": 1.4, "hard_weight": 3.0, "t50_weight": 4.0, "t100_weight": 1.0, "jepa_weight": 0.05},
        {"trial_id": 3, "name": "pure_ucy_hybrid_jepa", "kind": "hybrid", "width": 72, "layers": 2, "lr": 8.0e-4, "endpoint_weight": 0.8, "score_weight": 1.3, "hard_weight": 2.0, "t50_weight": 2.5, "t100_weight": 1.5, "jepa_weight": 0.35},
    ]


def _train_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train")
    val = _load_tensors("val")
    model = s41._make_world_model(train["static"].shape[1], train["cand_delta"].shape[1], width=int(trial["width"]), layers=int(trial["layers"]), kind=str(trial["kind"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["trial_id"]))
    ckpt = CHECKPOINT_DIR / f"stage41_pure_ucy_neural_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"stage41_pure_ucy_neural_{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        try:
            payload = json.loads(heartbeat.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {
                "source": "cached_verified",
                "checkpoint": str(ckpt),
                "heartbeat": str(heartbeat),
                "best": {
                    "val_loss": float(payload.get("val_loss", 0.0)),
                    "epoch": int(payload.get("epoch", EPOCHS)),
                    "train_loss": float(payload.get("train_loss", 0.0)),
                    "latent_variance": float(payload.get("latent_variance", 0.0)),
                },
                "resume_note": "completed checkpoint and heartbeat verified; skipped retraining this trial",
            }
    best = {"val_loss": float("inf"), "epoch": 0, "latent_variance": 0.0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["seq"].shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["seq"][ids], train["static"][ids], train["cand_delta"][ids])
            oracle = train["oracle"][ids]
            row_w = 1.0 + float(trial["hard_weight"]) * train["hard"][ids]
            row_w = row_w + float(trial["t50_weight"]) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_weight"]) * (train["horizon"][ids] == 100).float()
            endpoint = (F.smooth_l1_loss(out["endpoint_delta"], train["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            score = (F.smooth_l1_loss(out["candidate_score"], train["candidate_rel"][ids], reduction="none").mean(dim=1) * row_w).mean()
            ce = (F.cross_entropy(out["candidate_score"], oracle, reduction="none") * row_w).mean()
            gain_label = (oracle != 0).float()[:, None]
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], train["failure"][ids, None])
            gain = F.binary_cross_entropy_with_logits(out["gain_logit"], gain_label)
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], train["easy"][ids, None])
            interaction = F.binary_cross_entropy_with_logits(out["interaction_logit"], train["hard"][ids, None])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], 1.0 - train["failure"][ids, None])
            occupancy = F.binary_cross_entropy_with_logits(out["occupancy_logit"], train["hard"][ids, None])
            goal = F.cross_entropy(out["goal_logits"], torch.clamp(oracle, max=7))
            jepa = F.smooth_l1_loss(out["jepa_pred"], out["jepa_z"].detach())
            loss = (
                float(trial["endpoint_weight"]) * endpoint
                + float(trial["score_weight"]) * score
                + 0.5 * ce
                + 0.2 * failure
                + 0.2 * gain
                + 0.2 * harm
                + 0.05 * interaction
                + 0.05 * physical
                + 0.05 * occupancy
                + 0.05 * goal
                + float(trial["jepa_weight"]) * jepa
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["seq"], val["static"], val["cand_delta"])
            val_loss = float((F.smooth_l1_loss(out["candidate_score"], val["candidate_rel"]) + F.smooth_l1_loss(out["endpoint_delta"], val["target_delta"])).cpu())
            z = out["jepa_z"].detach().cpu().numpy()
            latent_variance = float(np.mean(np.var(z, axis=0)))
        heartbeat.write_text(
            json.dumps(
                _jsonable(
                    {
                        "source": "fresh_run",
                        "trial": dict(trial),
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss,
                        "latent_variance": latent_variance,
                        "checkpoint": str(ckpt),
                    }
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses)), "latent_variance": latent_variance}
            torch.save({"model": model.state_dict(), "static_dim": train["static"].shape[1], "candidate_count": train["cand_delta"].shape[1], "trial": dict(trial), "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = s41._make_world_model(int(payload["static_dim"]), int(payload["candidate_count"]), width=int(trial["width"]), layers=int(trial["layers"]), kind=str(trial["kind"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _predict(path: str | Path, split: str) -> dict[str, np.ndarray]:
    torch = _torch()
    model, _payload = _load_model(path)
    tensors = _load_tensors(split)
    outs: dict[str, list[np.ndarray]] = {k: [] for k in ["endpoint_delta", "candidate_score", "failure", "gain", "harm", "interaction", "physical", "jepa_z"]}
    with torch.no_grad():
        for start in range(0, tensors["seq"].shape[0], 4096):
            sl = slice(start, min(start + 4096, tensors["seq"].shape[0]))
            pred = model(tensors["seq"][sl], tensors["static"][sl], tensors["cand_delta"][sl])
            outs["endpoint_delta"].append(pred["endpoint_delta"].cpu().numpy())
            outs["candidate_score"].append(pred["candidate_score"].cpu().numpy())
            outs["failure"].append(torch.sigmoid(pred["failure_logit"]).cpu().numpy().reshape(-1))
            outs["gain"].append(torch.sigmoid(pred["gain_logit"]).cpu().numpy().reshape(-1))
            outs["harm"].append(torch.sigmoid(pred["harm_logit"]).cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(pred["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(pred["physical_logit"]).cpu().numpy().reshape(-1))
            outs["jepa_z"].append(pred["jepa_z"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def _metric_context(ds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "horizon": ds["horizon"],
        "hard": ds["hard"].astype(bool),
        "failure": ds["failure"].astype(bool),
        "easy": ds["easy"].astype(bool),
        "domain": ds["domain"].astype(str),
        "candidate_fde": ds["candidate_fde"].astype(np.float64),
    }


def _select(policy: Mapping[str, Any], pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if bool(policy.get("hard_only", False)):
        switch &= ds["hard"].astype(bool) | ds["failure"].astype(bool)
    if bool(policy.get("t50_only", False)):
        switch &= ds["horizon"].astype(int) == 50
    if bool(policy.get("t100_only", False)):
        switch &= ds["horizon"].astype(int) == 100
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch <= 0:
        switch[:] = False
    elif max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        order = ids[np.argsort(pred_gain[ids])[::-1]]
        keep = np.zeros(len(switch), dtype=bool)
        keep[order[:keep_n]] = True
        switch &= keep
    selected_idx = np.zeros(len(best), dtype=np.int64)
    selected_idx[switch] = best[switch]
    selected = ds["candidate_fde"].astype(np.float64)[np.arange(len(best)), selected_idx]
    return selected, switch, selected_idx


def _policy_grid() -> list[dict[str, Any]]:
    policies = []
    # UCY-only validation has one target domain, so a compact grid is enough to
    # select a conservative policy without spending minutes on equivalent
    # threshold combinations.
    for gain in [0.0, 0.005, 0.02, 0.05]:
        for gp in [0.0, 0.5]:
            for hp in [0.1, 0.35]:
                for pp in [0.0]:
                    for max_switch in [0.0, 0.03, 0.10, 0.30]:
                        policies.append({"gain_threshold": gain, "gain_prob": gp, "harm_prob": hp, "physical_prob": pp, "max_switch": max_switch})
    policies.extend(
        [
            {"gain_threshold": 0.005, "gain_prob": 0.25, "harm_prob": 0.2, "physical_prob": 0.0, "max_switch": 0.25, "hard_only": True},
            {"gain_threshold": 0.005, "gain_prob": 0.25, "harm_prob": 0.2, "physical_prob": 0.0, "max_switch": 0.25, "t50_only": True},
            {"gain_threshold": 0.005, "gain_prob": 0.25, "harm_prob": 0.2, "physical_prob": 0.0, "max_switch": 0.20, "t100_only": True},
        ]
    )
    return policies


def _score_metrics(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.4 * float(metrics.get("t50_improvement", 0.0))
        + 0.6 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 0.25 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
    )


def _eval_policy(pred: Mapping[str, np.ndarray], split: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    ds = _ds(split)
    selected, switch, selected_idx = _select(policy, pred, ds)
    floor = ds["floor_fde"].astype(np.float64)
    metrics = s41._metrics(selected, floor, _metric_context(ds), switch)
    endpoint_pred = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    endpoint_fde = np.linalg.norm(endpoint_pred - ds["future_xy"].astype(np.float64), axis=1)
    candidate_without_fallback = ds["candidate_fde"].astype(np.float64)[np.arange(len(selected_idx)), np.argmin(pred["candidate_score"], axis=1)]
    metrics["neural_endpoint_without_fallback"] = s41._metrics(endpoint_fde, floor, _metric_context(ds))
    metrics["neural_candidate_without_fallback"] = s41._metrics(candidate_without_fallback, floor, _metric_context(ds))
    metrics["selected_candidate_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(selected_idx, return_counts=True))}
    metrics["latent_variance"] = float(np.mean(np.var(pred["jepa_z"], axis=0))) if len(pred.get("jepa_z", [])) else 0.0
    return metrics


def _endpoint_residual_alpha(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> np.ndarray:
    mode = str(policy.get("mode", "all"))
    alpha = np.zeros(len(ds["horizon"]), dtype=np.float64)
    allow = np.ones(len(alpha), dtype=bool)
    if mode == "hard_only":
        allow &= ds["hard"].astype(bool) | ds["failure"].astype(bool)
    elif mode == "t50_hard":
        allow &= (ds["horizon"].astype(int) == 50) | ds["hard"].astype(bool) | ds["failure"].astype(bool)
    elif mode == "gain_harm":
        allow &= pred["gain"] >= float(policy.get("gain_prob", 0.0))
        allow &= pred["harm"] <= float(policy.get("harm_prob", 1.0))
        allow &= pred["physical"] >= float(policy.get("physical_prob", 0.0))
    elif mode != "all":
        raise ValueError(f"unknown residual mode: {mode}")
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch <= 0.0:
        allow[:] = False
    elif max_switch < 1.0 and np.any(allow):
        score = pred["gain"] - pred["harm"]
        ids = np.where(allow)[0]
        keep_n = max(1, int(max_switch * len(alpha)))
        keep = np.zeros(len(alpha), dtype=bool)
        keep[ids[np.argsort(score[ids])[::-1][:keep_n]]] = True
        allow &= keep
    alpha[allow] = float(policy.get("alpha", 0.0))
    return alpha


def _eval_endpoint_residual(pred: Mapping[str, np.ndarray], split: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    ds = _ds(split)
    floor_xy = ds["current_xy"].astype(np.float64) + ds["cand_delta"].astype(np.float64)[:, 0, :] * ds["normalizer"].astype(np.float64)[:, None]
    neural_xy = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    alpha = _endpoint_residual_alpha(pred, ds, policy)
    selected_xy = floor_xy + alpha[:, None] * (neural_xy - floor_xy)
    selected = np.linalg.norm(selected_xy - ds["future_xy"].astype(np.float64), axis=1)
    metrics = s41._metrics(selected, ds["floor_fde"].astype(np.float64), _metric_context(ds), alpha > EPS)
    endpoint_fde = np.linalg.norm(neural_xy - ds["future_xy"].astype(np.float64), axis=1)
    metrics["neural_endpoint_without_fallback"] = s41._metrics(endpoint_fde, ds["floor_fde"].astype(np.float64), _metric_context(ds))
    metrics["alpha_mean"] = float(np.mean(alpha))
    metrics["alpha_positive_rate"] = float(np.mean(alpha > EPS))
    metrics["latent_variance"] = float(np.mean(np.var(pred["jepa_z"], axis=0))) if len(pred.get("jepa_z", [])) else 0.0
    return metrics


def _endpoint_residual_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for alpha in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        for mode in ["all", "hard_only", "t50_hard"]:
            for max_switch in [0.10, 0.25, 0.50, 1.00]:
                policies.append({"type": "bounded_endpoint_residual", "mode": mode, "alpha": alpha, "max_switch": max_switch})
        for gain_prob in [0.0, 0.35, 0.55]:
            for harm_prob in [0.10, 0.25, 0.50]:
                for max_switch in [0.10, 0.25, 0.50, 1.00]:
                    policies.append(
                        {
                            "type": "bounded_endpoint_residual",
                            "mode": "gain_harm",
                            "alpha": alpha,
                            "gain_prob": gain_prob,
                            "harm_prob": harm_prob,
                            "physical_prob": 0.0,
                            "max_switch": max_switch,
                        }
                    )
    return policies


def _select_endpoint_residual_policy(pred: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_policy: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1e18
    for policy in _endpoint_residual_grid():
        metrics = _eval_endpoint_residual(pred, "val", policy)
        # Conservative tie-break: the original UCY retrain had positive
        # residual signal but selected an almost-equal validation policy that
        # intervened on every row. Prefer lower-switch, harm-gated policies
        # when validation utility is effectively tied, so source-shifted easy
        # rows are less likely to be harmed. This uses validation metrics and
        # policy shape only; test metrics are still evaluated once below.
        mode = str(policy.get("mode", ""))
        switch_penalty = 0.02 * float(metrics.get("switch_rate", 0.0))
        ungated_penalty = 0.015 if mode == "all" else 0.0
        easy_penalty = 50.0 * max(0.0, float(metrics.get("easy_degradation", 0.0)) - 0.02)
        score = _score_metrics(metrics) - switch_penalty - ungated_penalty - easy_penalty
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_metrics = metrics
    assert best_policy is not None and best_metrics is not None
    best_policy["val_score"] = float(best_score)
    best_policy["selection_rule"] = "validation_score_minus_switch_ungated_easy_risk_penalty"
    return best_policy, best_metrics


def _quick_eval_policy(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    selected, switch, _selected_idx = _select(policy, pred, ds)
    return s41._metrics(selected, ds["floor_fde"].astype(np.float64), _metric_context(ds), switch)


def _select_policy(pred: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    best_policy: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1e18
    val_ds = _ds("val")
    for policy in _policy_grid():
        metrics = _quick_eval_policy(pred, val_ds, policy)
        score = _score_metrics(metrics)
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_metrics = metrics
    assert best_policy is not None and best_metrics is not None
    best_policy["val_score"] = float(best_score)
    return best_policy, best_metrics


def _strict_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0.0
        and metrics.get("t50_improvement", 0.0) > 0.0
        and metrics.get("hard_failure_improvement", 0.0) > 0.0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("switch_rate", 0.0) > 0.0
    )


def run_strict_pure_ucy_neural_retrain() -> dict[str, Any]:
    started = time.perf_counter()
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    dataset = _make_seq2seq_dataset()
    trial_results: dict[str, Any] = {}
    for trial in _trial_configs():
        train_result = _train_trial(trial)
        val_pred = _predict(train_result["checkpoint"], "val")
        policy, val_metrics = _select_policy(val_pred)
        residual_policy, residual_val_metrics = _select_endpoint_residual_policy(val_pred)
        test_pred = _predict(train_result["checkpoint"], "test")
        test_metrics = _eval_policy(test_pred, "test", policy)
        residual_test_metrics = _eval_endpoint_residual(test_pred, "test", residual_policy)
        candidate_val_score = _score_metrics(val_metrics)
        residual_val_score = float(residual_policy.get("val_score", _score_metrics(residual_val_metrics)))
        if residual_val_score > candidate_val_score:
            best_mode = "bounded_endpoint_residual"
            combined_policy = residual_policy
            combined_val_metrics = residual_val_metrics
            combined_test_metrics = residual_test_metrics
        else:
            best_mode = "candidate_switch"
            combined_policy = policy
            combined_val_metrics = val_metrics
            combined_test_metrics = test_metrics
        trial_results[str(trial["name"])] = {
            "source": "fresh_run",
            "trial": dict(trial),
            "train": train_result,
            "candidate_switch_policy": policy,
            "candidate_switch_val_metrics": val_metrics,
            "candidate_switch_test_metrics": test_metrics,
            "bounded_endpoint_residual_policy": residual_policy,
            "bounded_endpoint_residual_val_metrics": residual_val_metrics,
            "bounded_endpoint_residual_test_metrics": residual_test_metrics,
            "candidate_switch_val_score": candidate_val_score,
            "bounded_endpoint_residual_val_score": residual_val_score,
            "best_mode": best_mode,
            "selected_policy": combined_policy,
            "val_metrics": combined_val_metrics,
            "test_metrics": combined_test_metrics,
            "strict_gate": _strict_gate(combined_test_metrics),
        }
    best_name = max(
        trial_results,
        key=lambda name: max(
            float(trial_results[name].get("candidate_switch_val_score", -1e18)),
            float(trial_results[name].get("bounded_endpoint_residual_val_score", -1e18)),
        ),
    )
    best = trial_results[best_name]
    strict_gate = bool(best["strict_gate"])
    result = {
        "source": "fresh_run",
        "protocol": "strict_pure_ucy_only_neural_retrain_select_test",
        "dataset": dataset,
        "trials": trial_results,
        "best_trial": best_name,
        "best_mode": best["best_mode"],
        "best_metrics": best["test_metrics"],
        "best_policy": best["selected_policy"],
        "strict_pure_ucy_only_neural_retrain_select_test_gate": strict_gate,
        "remaining_blocker": "" if strict_gate else "Strict pure-UCY neural retrain did not satisfy all/t50/hard positive with easy<=2%; keep mixed-external M3W-Neural v1 candidate and Stage37/teacher floor as deployable path.",
        "no_leakage": {
            **dataset["no_leakage"],
            "validation_policy_selection_only": True,
            "test_threshold_tuning": False,
            "mixed_external_neural_proposal_used": False,
            "mixed_external_floor_used": False,
        },
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
            "torch_threads": THREADS,
            "num_workers": 0,
        },
        "claim_boundary": {
            "strict_pure_ucy_neural_retrain": True,
            "protected_policy_head": True,
            "ungated_neural_replacement": False,
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    _write_json(REPORT_JSON, result)
    lines = [
        "# Stage41 Strict Pure-UCY Neural Retrain",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- best trial: `{best_name}`",
        f"- best mode: `{best['best_mode']}`",
        f"- strict pure UCY-only neural retrain/select/test gate: `{strict_gate}`",
        f"- remaining blocker: `{result['remaining_blocker']}`",
        "",
        "## Best Test Metrics",
        "",
        f"- all improvement: `{float(best['test_metrics'].get('all_improvement', 0.0)):.6f}`",
        f"- t50 improvement: `{float(best['test_metrics'].get('t50_improvement', 0.0)):.6f}`",
        f"- t100 diagnostic improvement: `{float(best['test_metrics'].get('t100_improvement', 0.0)):.6f}`",
        f"- hard/failure improvement: `{float(best['test_metrics'].get('hard_failure_improvement', 0.0)):.6f}`",
        f"- easy degradation: `{float(best['test_metrics'].get('easy_degradation', 0.0)):.6f}`",
        f"- switch rate: `{float(best['test_metrics'].get('switch_rate', 0.0)):.6f}`",
        f"- neural endpoint without fallback: `{best['test_metrics'].get('neural_endpoint_without_fallback')}`",
        f"- neural candidate without fallback: `{best['test_metrics'].get('neural_candidate_without_fallback')}`",
        "",
        "## Trial Table",
        "",
        "| trial | best mode | all | t50 | t100 | hard/failure | easy | switch | strict gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in trial_results.items():
        m = row["test_metrics"]
        lines.append(
            f"| `{name}` | `{row['best_mode']}` | {float(m.get('all_improvement', 0.0)):.4f} | {float(m.get('t50_improvement', 0.0)):.4f} | "
            f"{float(m.get('t100_improvement', 0.0)):.4f} | {float(m.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(m.get('easy_degradation', 0.0)):.4f} | {float(m.get('switch_rate', 0.0)):.4f} | `{row['strict_gate']}` |"
        )
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- splits: `{dataset['splits']}`",
            f"- train-only strongest floor: `{dataset['strongest_by_horizon_train_only']}`",
            f"- source inventory: `{dataset['source_inventory']}`",
            "",
            "## No Leakage",
            "",
            f"`{result['no_leakage']}`",
            "",
            "This is strict UCY-source neural retraining with train-only floor selection and train-only normalization. Future endpoints and candidate FDE are labels/evaluation only.",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_strict_pure_ucy_neural_retrain() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_strict_pure_ucy_neural_retrain()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_strict_pure_ucy_neural_retrain",
            status,
            started,
            [s41.DATA_DIR / "combined_external.npz"],
            [REPORT_JSON, REPORT_MD],
        )


if __name__ == "__main__":
    main_strict_pure_ucy_neural_retrain()
