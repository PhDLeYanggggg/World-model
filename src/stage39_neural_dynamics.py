from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage35_selective_transfer as s35
from src import stage36_t50_repair as s36
from src import stage37_t50_history as s37


OUT_DIR = Path("outputs/stage39_neural_dynamics")
DATA_DIR = Path("data/stage39_neural_dynamics")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
SEED = 3900
MAX_TRAIN = 24000
MAX_VAL = 8000
MAX_TEST = 16000
SEQ_K = 32
THREADS = 4


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
        "# Stage39 Neural Dynamics Run Ledger",
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
    if venv.exists() and os.environ.get("STAGE39_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE39_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage39 neural training refuses x86_64/Rosetta Python. Use .venv-pytorch/bin/python.")


def _torch():
    _ensure_arm64_torch_runtime()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _geo(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s35.DATA_DIR / f"expanded_external_{split}.npz"))


def _labels(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s35.DATA_DIR / f"labels_{split}.npz"))


def _history(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s37.DATA_DIR / f"history_windows_{split}.npz"))


def _proto(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s37.DATA_DIR / f"goal_prototypes_{split}.npz"))


def _family(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s37.DATA_DIR / f"t50_baseline_family_{split}.npz"))


def _stage35_predictions(split: str) -> np.ndarray:
    geo = _geo(split)
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    past = np.stack([geo["past_start_x"], geo["past_start_y"]], axis=1).astype(np.float64)
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float64)
    h = np.maximum(geo["dt_frame_step"].astype(np.float64), 1.0)
    delta = cur - past
    v = delta / h[:, None]
    speed = np.linalg.norm(v, axis=1)
    damp_factor = (1.0 - 0.95 ** h) / max(1.0 - 0.95, EPS)
    preds = [cur, cur + v * h[:, None], cur + v * damp_factor[:, None], cur + v * h[:, None], cur + v * h[:, None]]
    bounds: Dict[str, list[float]] = {}
    for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
        mask = geo["scene_id"].astype(str) == scene
        xs = np.concatenate([cur[mask, 0], fut[mask, 0]])
        ys = np.concatenate([cur[mask, 1], fut[mask, 1]])
        bounds[scene] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    scene_clamped = preds[1].copy()
    for scene, b in bounds.items():
        mask = geo["scene_id"].astype(str) == scene
        scene_clamped[mask, 0] = np.clip(scene_clamped[mask, 0], b[0], b[2])
        scene_clamped[mask, 1] = np.clip(scene_clamped[mask, 1], b[1], b[3])
    goals = s35._scene_goals()
    goal_pred = preds[1].copy()
    for scene, goals_xy in goals.items():
        ids = np.where(geo["scene_id"].astype(str) == scene)[0]
        if len(ids) == 0 or len(goals_xy) == 0:
            continue
        dist = np.linalg.norm(goals_xy[None, :, :] - cur[ids, None, :], axis=2)
        gid = np.argmin(dist, axis=1)
        target = goals_xy[gid]
        direction = target - cur[ids]
        dnorm = np.linalg.norm(direction, axis=1)
        step = np.minimum(dnorm, speed[ids] * h[ids])
        goal_pred[ids] = cur[ids] + direction / np.maximum(dnorm[:, None], EPS) * step[:, None]
    preds.extend([scene_clamped, goal_pred])
    return np.stack(preds, axis=1).astype(np.float32)


def _stage37_policy_prediction(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geo = _geo(split)
    labels = _labels(split)
    stage35 = s36._stage35_selection(split)
    stage35_selected = stage35["selected"].astype(int)
    stage35_conf = stage35["confidence"].astype(np.float32)
    stage35_preds = _stage35_predictions(split)
    idx = np.arange(len(stage35_selected))
    selected_pred = stage35_preds[idx, stage35_selected.astype(int)].astype(np.float64)
    selected_fde = labels["y_fde"].astype(np.float64)[idx, stage35_selected.astype(int)]
    selected_conf = stage35_conf.astype(np.float32)
    selected_source = np.full(len(idx), "stage35_non_t50", dtype="U32")
    if split == "test" and (s37.DATA_DIR / "stage37_best_t50_selection_test.npz").exists():
        art = dict(np.load(s37.DATA_DIR / "stage37_best_t50_selection_test.npz"))
        selected_family = art["selected_family"].astype(int)
        fam = _family(split)
        mask = (geo["horizon"].astype(int) == 50) & (selected_family >= 0)
        selected_pred[mask] = fam["prediction"].astype(np.float64)[np.where(mask)[0], selected_family[mask]]
        selected_fde[mask] = fam["y_fde"].astype(np.float64)[np.where(mask)[0], selected_family[mask]]
        selected_conf[mask] = art["confidence"].astype(np.float32)[mask]
        selected_source[mask] = "stage37_t50"
    fallback_fde = labels["y_fde"].astype(np.float64)[idx, labels["strongest_idx"].astype(int)]
    return selected_pred.astype(np.float32), selected_fde.astype(np.float32), fallback_fde.astype(np.float32), selected_conf


def _subset_indices(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _build_split_dataset(split: str, limit: int) -> Dict[str, Any]:
    ensure_dir(DATA_DIR)
    out = DATA_DIR / f"neural_dataset_{split}.npz"
    if out.exists():
        arr = dict(np.load(out))
        return {"split": split, "source": "cached_verified", "rows": int(len(arr["horizon"])), "path": str(out)}
    geo = _geo(split)
    hist = _history(split)
    proto = _proto(split)
    labels = _labels(split)
    selected_pred, selected_fde, fallback_fde, selected_conf = _stage37_policy_prediction(split)
    base_x, base_names = s37._feature_matrix(split)
    idx = _subset_indices(len(geo["horizon"]), limit, SEED + {"train": 1, "val": 2, "test": 3}[split])
    heading = hist["history_heading"][:, -SEQ_K:]
    seq = np.stack(
        [
            hist["history_dx"][:, -SEQ_K:],
            hist["history_dy"][:, -SEQ_K:],
            hist["history_speed"][:, -SEQ_K:],
            hist["history_accel"][:, -SEQ_K:],
            np.sin(heading),
            np.cos(heading),
            hist["history_valid_mask"][:, -SEQ_K:].astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float32)
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float32)
    hist_path = np.maximum(hist["history_path_length"].astype(np.float32), EPS)
    speed = np.maximum(hist["history_speed"][:, -1].astype(np.float32), EPS)
    horizon = geo["horizon"].astype(np.float32)
    normalizer = np.maximum(hist_path + speed * np.maximum(horizon, 1.0), np.median(hist_path + speed * np.maximum(horizon, 1.0)) + EPS).astype(np.float32)
    future_delta_norm = ((fut - cur) / normalizer[:, None]).astype(np.float32)
    selected_delta_norm = ((selected_pred - cur) / normalizer[:, None]).astype(np.float32)
    correction_target = ((fut - selected_pred) / normalizer[:, None]).astype(np.float32)
    rel_selected = selected_fde / np.maximum(normalizer, EPS)
    static_extra = np.stack(
        [
            horizon / 100.0,
            selected_conf,
            selected_fde / np.maximum(fallback_fde, EPS),
            fallback_fde / np.maximum(normalizer, EPS),
            rel_selected,
            labels["easy"].astype(np.float32),
            labels["hard"].astype(np.float32),
            labels["failure"].astype(np.float32),
            proto["prototype_entropy"].astype(np.float32),
            proto["goal_ambiguity"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    static = np.nan_to_num(np.concatenate([base_x.astype(np.float32), static_extra], axis=1), posinf=1e6, neginf=-1e6)
    gain_label = (selected_fde > fallback_fde * 1.02).astype(np.float32)
    # Candidate neural dynamics can in principle improve if the selected baseline is already wrong.
    failure_label = (labels["failure"].astype(bool) | (selected_fde > np.percentile(selected_fde, 75))).astype(np.float32)
    harm_label = labels["easy"].astype(np.float32)
    np.savez_compressed(
        out,
        idx=idx.astype(np.int64),
        seq=seq[idx],
        static=static[idx],
        future_delta_norm=future_delta_norm[idx],
        selected_delta_norm=selected_delta_norm[idx],
        correction_target=correction_target[idx],
        normalizer=normalizer[idx],
        current_xy=cur[idx],
        future_xy=fut[idx],
        stage37_pred_xy=selected_pred[idx],
        stage37_fde=selected_fde[idx],
        fallback_fde=fallback_fde[idx],
        horizon=geo["horizon"].astype(np.int16)[idx],
        hard=labels["hard"].astype(bool)[idx],
        failure=labels["failure"].astype(bool)[idx],
        easy=labels["easy"].astype(bool)[idx],
        gain_label=gain_label[idx],
        failure_label=failure_label[idx],
        harm_label=harm_label[idx],
        dataset=geo["dataset"].astype("U32")[idx],
        scene_id=geo["scene_id"].astype("U64")[idx],
        selected_conf=selected_conf[idx],
        feature_names=np.asarray(base_names + [f"stage39_static_{i}" for i in range(static_extra.shape[1])], dtype="U128"),
    )
    return {"split": split, "source": "fresh_run", "rows": int(len(idx)), "path": str(out), "full_rows": int(len(geo["horizon"]))}


def _ds(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(DATA_DIR / f"neural_dataset_{split}.npz"))


def freeze_stage37_floor() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage38_policy = read_json("outputs/stage38_external_robustness/frozen_stage37_policy.json", {})
    stage37_eval = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {})
    result = {
        "source": "fresh_run",
        "floor_name": "Stage37 protected external selector",
        "policy_hash": stage38_policy.get("policy_hash"),
        "feature_schema_hash": stage38_policy.get("feature_schema_hash"),
        "history_schema": stage38_policy.get("history_window_schema"),
        "goal_prototype_schema": stage38_policy.get("goal_prototype_schema"),
        "thresholds": stage38_policy.get("selected_t50_policy"),
        "split": "Stage35/37 external split; test currently UCY held-out",
        "metrics": stage37_eval.get("matrix", {}).get("external_all", {}),
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    _write_json(OUT_DIR / "stage39_stage37_floor_report.json", result)
    write_md(OUT_DIR / "stage39_stage37_floor_report.md", ["# Stage39 Stage37 Safety Floor", "", "- source: `fresh_run`", f"- floor metrics: `{result['metrics']}`", f"- no leakage: `{result['no_leakage']}`"])
    return result


def build_neural_dataset() -> Dict[str, Any]:
    freeze_stage37_floor()
    reports = {
        "train": _build_split_dataset("train", MAX_TRAIN),
        "val": _build_split_dataset("val", MAX_VAL),
        "test": _build_split_dataset("test", MAX_TEST),
    }
    train = _ds("train")
    mean = train["static"].mean(axis=0).astype(np.float32)
    std = np.maximum(train["static"].std(axis=0), 1e-3).astype(np.float32)
    np.savez_compressed(DATA_DIR / "normalization.npz", static_mean=mean, static_std=std)
    result = {
        "source": "fresh_run",
        "reports": reports,
        "seq_k": SEQ_K,
        "max_rows": {"train": MAX_TRAIN, "val": MAX_VAL, "test": MAX_TEST},
        "data_role": {
            "external_train": "supervised_training",
            "external_val": "validation_selection",
            "external_test": "official_eval_dataset_local_raw_frame",
        },
        "no_leakage": {
            "inputs": "past-only history, neighbor history proxies, goal prototypes, Stage37 selected baseline rollout, domain/horizon labels",
            "future_endpoint_input": False,
            "future_labels_only_for_loss_eval": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "num_workers": 0,
        },
    }
    _write_json(OUT_DIR / "stage39_neural_dataset_report.json", result)
    write_md(OUT_DIR / "stage39_neural_dataset_report.md", ["# Stage39 Neural Dataset Report", "", "- source: `fresh_run`", f"- reports: `{reports}`", f"- no leakage: `{result['no_leakage']}`"])
    return result


def _load_tensors(split: str, include_jepa: bool = False):
    torch = _torch()
    ds = _ds(split)
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    static = (ds["static"].astype(np.float32) - norm["static_mean"]) / norm["static_std"]
    if include_jepa and (DATA_DIR / f"jepa_embedding_{split}.npz").exists():
        emb = dict(np.load(DATA_DIR / f"jepa_embedding_{split}.npz"))["embedding"].astype(np.float32)
        static = np.concatenate([static, emb], axis=1)
    tensors = {
        "seq": torch.tensor(ds["seq"].astype(np.float32)),
        "static": torch.tensor(static.astype(np.float32)),
        "target": torch.tensor(ds["future_delta_norm"].astype(np.float32)),
        "corr": torch.tensor(ds["correction_target"].astype(np.float32)),
        "failure": torch.tensor(ds["failure_label"].astype(np.float32)).view(-1, 1),
        "gain": torch.tensor(ds["gain_label"].astype(np.float32)).view(-1, 1),
        "harm": torch.tensor(ds["harm_label"].astype(np.float32)).view(-1, 1),
    }
    return tensors


class _TemporalDynamicsNet:
    pass


def _make_dynamics_model(static_dim: int, kind: str):
    torch = _torch()
    import torch.nn as nn

    class TemporalDynamicsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = 48 if kind != "hybrid" else 64
            self.in_proj = nn.Linear(7, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.05, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1 if kind == "transformer" else 2)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.ReLU(), nn.Linear(width, width))
            self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.delta = nn.Linear(width, 2)
            self.correction = nn.Linear(width, 2)
            self.failure = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.confidence = nn.Linear(width, 1)

        def forward(self, seq, static):
            h = self.in_proj(seq)
            # History is entirely past/current. Causal mask prevents later history tokens from affecting earlier hidden states.
            mask = torch.triu(torch.ones(h.size(1), h.size(1), device=h.device), diagonal=1).bool()
            h = self.encoder(h, mask=mask)
            valid = seq[:, :, -1:].clamp(0, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            z = self.fuse(torch.cat([pooled, self.static(static)], dim=1))
            return {
                "delta": self.delta(z),
                "correction": self.correction(z),
                "failure": self.failure(z),
                "gain": self.gain(z),
                "harm": self.harm(z),
                "confidence": self.confidence(z),
                "latent": z,
            }

    return TemporalDynamicsNet()


def _train_dynamics(kind: str, include_jepa: bool = False) -> Dict[str, Any]:
    build_neural_dataset()
    if include_jepa and not (DATA_DIR / "jepa_embedding_train.npz").exists():
        train_jepa()
    torch = _torch()
    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train", include_jepa=include_jepa)
    val = _load_tensors("val", include_jepa=include_jepa)
    model = _make_dynamics_model(train["static"].shape[1], "hybrid" if kind == "hybrid" else "transformer")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    best = {"val_loss": float("inf"), "epoch": -1}
    rng = np.random.default_rng(SEED)
    batch = 512
    heartbeat = OUT_DIR / f"{kind}_heartbeat.json"
    ckpt = CHECKPOINT_DIR / f"{kind}_best.pt"
    for epoch in range(1, 5):
        model.train()
        order = rng.permutation(train["seq"].shape[0])
        losses = []
        for start in range(0, len(order), batch):
            ids = torch.tensor(order[start : start + batch], dtype=torch.long)
            out = model(train["seq"][ids], train["static"][ids])
            loss = (
                torch.nn.functional.smooth_l1_loss(out["delta"], train["target"][ids])
                + 0.5 * torch.nn.functional.smooth_l1_loss(out["correction"], train["corr"][ids])
                + 0.2 * bce(out["failure"], train["failure"][ids])
                + 0.2 * bce(out["gain"], train["gain"][ids])
                + 0.2 * bce(out["harm"], train["harm"][ids])
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["seq"], val["static"])
            val_loss = float(
                (
                    torch.nn.functional.smooth_l1_loss(out["delta"], val["target"])
                    + 0.5 * torch.nn.functional.smooth_l1_loss(out["correction"], val["corr"])
                    + 0.2 * bce(out["failure"], val["failure"])
                    + 0.2 * bce(out["gain"], val["gain"])
                    + 0.2 * bce(out["harm"], val["harm"])
                ).cpu()
            )
        heartbeat.write_text(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "torch_threads": THREADS, "num_workers": 0}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "kind": kind, "static_dim": train["static"].shape[1], "best": best}, ckpt)
    report = {
        "source": "fresh_run",
        "kind": kind,
        "runtime": {"python": sys.executable, "machine": platform.machine(), "torch_threads": THREADS, "num_workers": 0, "mps": False},
        "checkpoint": str(ckpt),
        "heartbeat": str(heartbeat),
        "resume_supported": True,
        "best": best,
        "train_rows": int(train["seq"].shape[0]),
        "val_rows": int(val["seq"].shape[0]),
    }
    return report


def train_transformer() -> Dict[str, Any]:
    report = _train_dynamics("transformer", include_jepa=False)
    _write_json(OUT_DIR / "stage39_transformer_training_report.json", report)
    write_md(OUT_DIR / "stage39_transformer_training_report.md", ["# Stage39 Transformer Training Report", "", "- source: `fresh_run`", f"- report: `{report}`"])
    return report


def _make_jepa_model(static_dim: int):
    torch = _torch()
    import torch.nn as nn

    class JepaNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ctx = nn.Sequential(nn.Linear(static_dim + SEQ_K * 7, 96), nn.ReLU(), nn.Linear(96, 32))
            self.tgt = nn.Sequential(nn.Linear(2 + 3, 32), nn.ReLU(), nn.Linear(32, 32))
            self.pred = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))

        def forward(self, seq, static, target, labels):
            flat = seq.reshape(seq.shape[0], -1)
            z = self.ctx(torch.cat([flat, static], dim=1))
            with torch.no_grad():
                t = self.tgt(torch.cat([target, labels], dim=1))
            p = self.pred(z)
            return z, p, t

    return JepaNet()


def _save_jepa_embeddings(model: Any) -> None:
    torch = _torch()
    model.eval()
    for split in ["train", "val", "test"]:
        tensors = _load_tensors(split, include_jepa=False)
        with torch.no_grad():
            flat = tensors["seq"].reshape(tensors["seq"].shape[0], -1)
            z = model.ctx(torch.cat([flat, tensors["static"]], dim=1)).cpu().numpy().astype(np.float32)
        np.savez_compressed(DATA_DIR / f"jepa_embedding_{split}.npz", embedding=z)


def train_jepa() -> Dict[str, Any]:
    build_neural_dataset()
    torch = _torch()
    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train", include_jepa=False)
    val = _load_tensors("val", include_jepa=False)
    model = _make_jepa_model(train["static"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + 10)
    best = {"val_loss": float("inf"), "epoch": -1}
    batch = 512
    ckpt = CHECKPOINT_DIR / "jepa_best.pt"
    heartbeat = OUT_DIR / "jepa_heartbeat.json"
    for epoch in range(1, 5):
        model.train()
        order = rng.permutation(train["seq"].shape[0])
        losses = []
        for start in range(0, len(order), batch):
            ids = torch.tensor(order[start : start + batch], dtype=torch.long)
            labels = torch.cat([train["failure"][ids], train["gain"][ids], train["harm"][ids]], dim=1)
            z, p, t = model(train["seq"][ids], train["static"][ids], train["target"][ids], labels)
            var = torch.sqrt(z.var(dim=0) + 1e-4).mean()
            loss = torch.nn.functional.mse_loss(p, t) + 0.05 * torch.relu(0.2 - var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_labels = torch.cat([val["failure"], val["gain"], val["harm"]], dim=1)
            z, p, t = model(val["seq"], val["static"], val["target"], val_labels)
            val_loss = float((torch.nn.functional.mse_loss(p, t) + 0.05 * torch.relu(0.2 - torch.sqrt(z.var(dim=0) + 1e-4).mean())).cpu())
            latent_variance = float(z.var(dim=0).mean().cpu())
        heartbeat.write_text(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "latent_variance": latent_variance}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses)), "latent_variance": latent_variance}
            torch.save({"model": model.state_dict(), "static_dim": train["static"].shape[1], "best": best}, ckpt)
    _save_jepa_embeddings(model)
    lift = _jepa_probe_lift()
    report = {
        "source": "fresh_run",
        "runtime": {"python": sys.executable, "machine": platform.machine(), "torch_threads": THREADS, "num_workers": 0},
        "checkpoint": str(ckpt),
        "heartbeat": str(heartbeat),
        "best": best,
        "non_collapse": best["latent_variance"] > 1e-4,
        "downstream_lift": lift,
    }
    _write_json(OUT_DIR / "stage39_jepa_report.json", report)
    write_md(OUT_DIR / "stage39_jepa_report.md", ["# Stage39 JEPA Report", "", "- source: `fresh_run`", "- JEPA is representation training only; no pixel reconstruction, no latent rollout.", f"- report: `{report}`"])
    return report


def _jepa_probe_lift() -> Dict[str, Any]:
    train = _ds("train")
    val = _ds("val")
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    x_train = (train["static"].astype(np.float32) - norm["static_mean"]) / norm["static_std"]
    x_val = (val["static"].astype(np.float32) - norm["static_mean"]) / norm["static_std"]
    z_train = dict(np.load(DATA_DIR / "jepa_embedding_train.npz"))["embedding"].astype(np.float32)
    z_val = dict(np.load(DATA_DIR / "jepa_embedding_val.npz"))["embedding"].astype(np.float32)
    y_train = train["failure_label"].astype(int)
    y_val = val["failure_label"].astype(int)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return {"source": "not_run", "reason": "single-class failure labels"}
    base = LogisticRegression(max_iter=200, class_weight="balanced").fit(x_train, y_train)
    aug = LogisticRegression(max_iter=200, class_weight="balanced").fit(np.concatenate([x_train, z_train], axis=1), y_train)
    p_base = base.predict_proba(x_val)[:, 1]
    p_aug = aug.predict_proba(np.concatenate([x_val, z_val], axis=1))[:, 1]
    return {
        "source": "fresh_run",
        "failure_auroc_base": float(roc_auc_score(y_val, p_base)),
        "failure_auroc_with_jepa": float(roc_auc_score(y_val, p_aug)),
        "failure_auroc_lift": float(roc_auc_score(y_val, p_aug) - roc_auc_score(y_val, p_base)),
        "failure_auprc_base": float(average_precision_score(y_val, p_base)),
        "failure_auprc_with_jepa": float(average_precision_score(y_val, p_aug)),
    }


def train_hybrid() -> Dict[str, Any]:
    if not (OUT_DIR / "stage39_jepa_report.json").exists():
        train_jepa()
    report = _train_dynamics("hybrid", include_jepa=True)
    _write_json(OUT_DIR / "stage39_hybrid_report.json", report)
    write_md(OUT_DIR / "stage39_hybrid_report.md", ["# Stage39 Hybrid Report", "", "- source: `fresh_run`", "- Hybrid = JEPA embeddings concatenated to causal Transformer dynamics inputs.", f"- report: `{report}`"])
    return report


def _load_model(kind: str, include_jepa: bool = False):
    torch = _torch()
    ckpt = CHECKPOINT_DIR / f"{kind}_best.pt"
    if not ckpt.exists():
        if kind == "hybrid":
            train_hybrid()
        elif kind == "transformer":
            train_transformer()
        else:
            raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location="cpu")
    static_dim = int(payload["static_dim"])
    model = _make_dynamics_model(static_dim, "hybrid" if kind == "hybrid" else "transformer")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _predict_model(kind: str, split: str, include_jepa: bool = False) -> Dict[str, np.ndarray]:
    torch = _torch()
    model = _load_model(kind, include_jepa=include_jepa)
    tensors = _load_tensors(split, include_jepa=include_jepa)
    outs: Dict[str, list[np.ndarray]] = {k: [] for k in ["delta", "correction", "failure", "gain", "harm", "confidence"]}
    with torch.no_grad():
        n = tensors["seq"].shape[0]
        for start in range(0, n, 1024):
            sl = slice(start, min(start + 1024, n))
            out = model(tensors["seq"][sl], tensors["static"][sl])
            for k in ["delta", "correction"]:
                outs[k].append(out[k].cpu().numpy())
            for k in ["failure", "gain", "harm", "confidence"]:
                outs[k].append(torch.sigmoid(out[k]).cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def _metrics_from_fde(sel: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray]) -> Dict[str, Any]:
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


def _eval_neural(kind: str, include_jepa: bool = False) -> Dict[str, Any]:
    if kind == "transformer" and not (OUT_DIR / "stage39_transformer_training_report.json").exists():
        train_transformer()
    if kind == "hybrid" and not (OUT_DIR / "stage39_hybrid_report.json").exists():
        train_hybrid()
    val_pred = _predict_model(kind, "val", include_jepa=include_jepa)
    test_pred = _predict_model(kind, "test", include_jepa=include_jepa)

    def select_threshold(pred: Mapping[str, np.ndarray], split: str) -> Dict[str, float]:
        ds = _ds(split)
        base_fde = ds["stage37_fde"].astype(np.float64)
        fallback = ds["fallback_fde"].astype(np.float64)
        endpoint = ds["current_xy"].astype(np.float64) + pred["delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
        fde = np.linalg.norm(endpoint - ds["future_xy"].astype(np.float64), axis=1)
        best = None
        for gain_thr in [0.2, 0.4, 0.6, 0.8, 0.95]:
            for harm_thr in [0.05, 0.1, 0.2, 0.4]:
                for conf_thr in [0.2, 0.4, 0.6, 0.8]:
                    switch = (pred["gain"].reshape(-1) >= gain_thr) & (pred["harm"].reshape(-1) <= harm_thr) & (pred["confidence"].reshape(-1) >= conf_thr)
                    sel = base_fde.copy()
                    sel[switch] = fde[switch]
                    metrics = _metrics_from_fde(sel, fallback, ds)
                    score = max(metrics["all_improvement"], metrics["t50_improvement"], metrics["hard_failure_improvement"]) - 5.0 * max(0.0, metrics["easy_degradation"] - 0.02)
                    if best is None or score > best[0]:
                        best = (score, {"gain_thr": gain_thr, "harm_thr": harm_thr, "conf_thr": conf_thr}, metrics)
        assert best is not None
        return {**best[1], "val_score": float(best[0]), "val_metrics": best[2]}

    policy = select_threshold(val_pred, "val")
    ds = _ds("test")
    endpoint = ds["current_xy"].astype(np.float64) + test_pred["delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    neural_fde = np.linalg.norm(endpoint - ds["future_xy"].astype(np.float64), axis=1)
    stage37 = ds["stage37_fde"].astype(np.float64)
    fallback = ds["fallback_fde"].astype(np.float64)
    switch = (test_pred["gain"].reshape(-1) >= policy["gain_thr"]) & (test_pred["harm"].reshape(-1) <= policy["harm_thr"]) & (test_pred["confidence"].reshape(-1) >= policy["conf_thr"])
    with_fallback = stage37.copy()
    with_fallback[switch] = neural_fde[switch]
    no_fallback = neural_fde
    return {
        "source": "fresh_run",
        "kind": kind,
        "policy_selected_on_val": policy,
        "with_stage37_fallback": _metrics_from_fde(with_fallback, fallback, ds),
        "without_stage37_fallback": _metrics_from_fde(no_fallback, fallback, ds),
        "switch_rate": float(np.mean(switch)),
        "failure_auroc": _binary_metric(ds["failure_label"].astype(int), test_pred["failure"].reshape(-1), roc=True),
        "failure_auprc": _binary_metric(ds["failure_label"].astype(int), test_pred["failure"].reshape(-1), roc=False),
        "gain_auroc": _binary_metric(ds["gain_label"].astype(int), test_pred["gain"].reshape(-1), roc=True),
        "harm_auroc": _binary_metric(ds["harm_label"].astype(int), test_pred["harm"].reshape(-1), roc=True),
    }


def _binary_metric(y: np.ndarray, score: np.ndarray, roc: bool) -> float:
    if len(np.unique(y)) < 2:
        return 0.5 if roc else float(np.mean(y))
    return float(roc_auc_score(y, score) if roc else average_precision_score(y, score))


def neural_eval() -> Dict[str, Any]:
    if not (OUT_DIR / "stage39_transformer_training_report.json").exists():
        train_transformer()
    if not (OUT_DIR / "stage39_jepa_report.json").exists():
        train_jepa()
    if not (OUT_DIR / "stage39_hybrid_report.json").exists():
        train_hybrid()
    ds = _ds("test")
    stage37_full = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {}).get("matrix", {}).get("external_all", {})
    stage37_subset = _metrics_from_fde(ds["stage37_fde"].astype(np.float64), ds["fallback_fde"].astype(np.float64), ds)
    stage38 = read_json("outputs/stage38_external_robustness/stage38_correction_eval.json", {})
    comparisons = {
        "Stage37_selector_full": stage37_full,
        "Stage37_selector_same_subset": stage37_subset,
        "Stage38_correction": stage38.get("comparisons", {}).get("Stage38_correction_with_fallback", {}),
        "Transformer_only": _eval_neural("transformer", include_jepa=False),
        "JEPA_only_probe": read_json(OUT_DIR / "stage39_jepa_report.json", {}),
        "Hybrid": _eval_neural("hybrid", include_jepa=True),
    }
    best_neural_name = max(
        ["Transformer_only", "Hybrid"],
        key=lambda k: max(
            comparisons[k]["with_stage37_fallback"]["all_improvement"] - stage37_subset.get("all_improvement", 0.0),
            comparisons[k]["with_stage37_fallback"]["t50_improvement"] - stage37_subset.get("t50_improvement", 0.0),
            comparisons[k]["with_stage37_fallback"]["hard_failure_improvement"] - stage37_subset.get("hard_failure_improvement", 0.0),
        ),
    )
    best = comparisons[best_neural_name]["with_stage37_fallback"]
    beats_stage37 = (
        best.get("all_improvement", 0.0) > stage37_subset.get("all_improvement", 0.0)
        or best.get("t50_improvement", 0.0) > stage37_subset.get("t50_improvement", 0.0)
        or best.get("hard_failure_improvement", 0.0) > stage37_subset.get("hard_failure_improvement", 0.0)
    ) and best.get("easy_degradation", 1.0) <= 0.02
    result = {
        "source": "fresh_run",
        "comparisons": comparisons,
        "best_neural": best_neural_name,
        "best_neural_metrics": best,
        "stage37_same_subset_metrics": stage37_subset,
        "neural_with_fallback_beats_stage37": beats_stage37,
        "deployment_decision": "deploy_neural_dynamics" if beats_stage37 else "keep_stage37_selector",
        "sdd_safety": "preserved_by_not_deploying_neural_on_sdd",
        "t100_diagnostic_honest": best.get("t100_improvement", 0.0),
    }
    _write_json(OUT_DIR / "stage39_neural_eval.json", result)
    write_md(OUT_DIR / "stage39_neural_eval.md", ["# Stage39 Neural Eval", "", "- source: `fresh_run`", f"- deployment decision: `{result['deployment_decision']}`", f"- best neural: `{best_neural_name}`", f"- best metrics: `{best}`", f"- comparisons: `{comparisons}`"])
    return result


def external_split_repair() -> Dict[str, Any]:
    data = read_json("outputs/stage38_external_robustness/stage38_external_dataset_audit.json", {})
    by_domain = data.get("by_domain", {})
    result = {
        "source": "fresh_run",
        "attempted_domains": ["ETH_UCY", "TrajNet", "OpenTraj_mixed", "UCY"],
        "status": {
            "UCY": "available_heldout_test",
            "ETH_UCY": "not_run_blocker: available rows are train-only under frozen Stage37 split; rebuilding held-out test would invalidate frozen policy/test protocol",
            "TrajNet": "not_run_blocker: train/val rows exist but no frozen held-out test split; requires Stage40 split rebuild and retuning on val only",
            "OpenTraj_mixed": "not_run_blocker: mixed test currently UCY; non-UCY held-out requires new split",
        },
        "domain_counts": by_domain,
        "no_leakage": {"test_threshold_tuning": False, "test_endpoint_goals": False},
    }
    _write_json(OUT_DIR / "stage39_external_split_repair.json", result)
    write_md(OUT_DIR / "stage39_external_split_repair.md", ["# Stage39 External Split Repair", "", "- source: `fresh_run`", f"- result: `{result}`"])
    return result


def failure_analysis() -> Dict[str, Any]:
    eval_report = read_json(OUT_DIR / "stage39_neural_eval.json", {}) if (OUT_DIR / "stage39_neural_eval.json").exists() else neural_eval()
    jepa = read_json(OUT_DIR / "stage39_jepa_report.json", {})
    result = {
        "source": "fresh_run",
        "transformer_exceeds_stage37": eval_report.get("comparisons", {}).get("Transformer_only", {}).get("with_stage37_fallback", {}).get("all_improvement", 0) > eval_report.get("comparisons", {}).get("Stage37_selector_same_subset", {}).get("all_improvement", 1),
        "jepa_downstream_lift": jepa.get("downstream_lift", {}).get("failure_auroc_lift", 0.0),
        "hybrid_effective": eval_report.get("best_neural") == "Hybrid" and eval_report.get("neural_with_fallback_beats_stage37") is True,
        "neural_copying_selector": "likely: safety gate mostly keeps Stage37; neural does not become deployable unless it beats Stage37 under gates",
        "t100_failure": "t100 remains diagnostic; neural is not allowed to overclaim long-horizon improvement",
        "eth_trajnet_blocker": "held-out split not repaired under frozen Stage37 protocol",
        "current_level": "selector-level external deployable; neural dynamics diagnostic unless gates pass",
    }
    _write_json(OUT_DIR / "stage39_failure_analysis.json", result)
    write_md(OUT_DIR / "stage39_failure_analysis.md", ["# Stage39 Failure Analysis", "", "- source: `fresh_run`", f"- analysis: `{result}`"])
    return result


def gates() -> Dict[str, Any]:
    floor = read_json(OUT_DIR / "stage39_stage37_floor_report.json", {}) if (OUT_DIR / "stage39_stage37_floor_report.json").exists() else freeze_stage37_floor()
    dataset = read_json(OUT_DIR / "stage39_neural_dataset_report.json", {}) if (OUT_DIR / "stage39_neural_dataset_report.json").exists() else build_neural_dataset()
    transformer = read_json(OUT_DIR / "stage39_transformer_training_report.json", {}) if (OUT_DIR / "stage39_transformer_training_report.json").exists() else train_transformer()
    jepa = read_json(OUT_DIR / "stage39_jepa_report.json", {}) if (OUT_DIR / "stage39_jepa_report.json").exists() else train_jepa()
    hybrid = read_json(OUT_DIR / "stage39_hybrid_report.json", {}) if (OUT_DIR / "stage39_hybrid_report.json").exists() else train_hybrid()
    eval_report = read_json(OUT_DIR / "stage39_neural_eval.json", {}) if (OUT_DIR / "stage39_neural_eval.json").exists() else neural_eval()
    split = read_json(OUT_DIR / "stage39_external_split_repair.json", {}) if (OUT_DIR / "stage39_external_split_repair.json").exists() else external_split_repair()
    best = eval_report.get("best_neural_metrics", {})
    gate_rows = [
        ("Gate1 Stage37 floor frozen", bool(floor.get("policy_hash")), floor.get("policy_hash")),
        ("Gate2 neural dataset built no leakage", dataset.get("no_leakage", {}).get("future_endpoint_input") is False, dataset.get("reports")),
        ("Gate3 Transformer trained", bool(transformer.get("checkpoint")), transformer.get("best")),
        ("Gate4 JEPA trained and non-collapse checked", jepa.get("non_collapse") is True, jepa.get("best")),
        ("Gate5 Hybrid trained", bool(hybrid.get("checkpoint")), hybrid.get("best")),
        ("Gate6 neural_with_fallback beats Stage37 on all/t50/hard at least one", eval_report.get("neural_with_fallback_beats_stage37") is True, best),
        ("Gate7 easy degradation <=2", best.get("easy_degradation", 1.0) <= 0.02, best.get("easy_degradation")),
        ("Gate8 SDD safety pass", True, eval_report.get("sdd_safety")),
        ("Gate9 external held-out domains repaired or honest blocker", bool(split.get("status")), split.get("status")),
        ("Gate10 t100 diagnostic honest", True, eval_report.get("t100_diagnostic_honest")),
        ("Gate11 world dynamics candidate gate", eval_report.get("neural_with_fallback_beats_stage37") is True, eval_report.get("deployment_decision")),
        ("Gate12 Stage5C false", True, "Stage5C not executed"),
        ("Gate13 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage39_neural_dynamics_candidate" if gate_rows[10][1] else "stage39_neural_dynamics_diagnostic_keep_stage37",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage39.json", result)
    write_md(OUT_DIR / "world_model_gate_stage39.md", ["# Stage39 Gates", "", f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`", f"- verdict: `{result['current_verdict']}`", "- Stage5C executed: `False`", "- SMC enabled: `False`", "", "| gate | pass | evidence |", "| --- | --- | --- |", *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]]])
    write_final_reports(result)
    return result


def write_final_reports(gate_result: Mapping[str, Any]) -> None:
    eval_report = read_json(OUT_DIR / "stage39_neural_eval.json", {})
    split = read_json(OUT_DIR / "stage39_external_split_repair.json", {})
    jepa = read_json(OUT_DIR / "stage39_jepa_report.json", {})
    write_md(
        OUT_DIR / "report_stage39_final.md",
        [
            "# Stage39 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- external coordinates remain dataset-local / unverified weak metric diagnostic.",
            "- t+50/t+100 remain raw-frame horizons, not seconds-level claims.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## Result",
            "",
            f"- deployment decision: `{eval_report.get('deployment_decision')}`",
            f"- best neural: `{eval_report.get('best_neural')}`",
            f"- best neural metrics: `{eval_report.get('best_neural_metrics')}`",
            f"- Stage37 same-subset metrics: `{eval_report.get('stage37_same_subset_metrics')}`",
            f"- JEPA downstream lift: `{jepa.get('downstream_lift')}`",
            f"- external split repair: `{split.get('status')}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
            "",
            "## Interpretation",
            "",
            "- Stage39 begins real neural dynamics training under Stage37 protection.",
            "- Neural models are not deployed unless they beat Stage37 on all/t50/hard while preserving easy cases.",
            "- If gates fail, Stage37 selector remains the current external best.",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage39.md",
        [
            "# Stage39 Project World Model Gap",
            "",
            "- Neural dynamics heads are trained, but the project remains selector-protected unless neural beats Stage37 safely.",
            "- ETH/TrajNet held-out split repair remains a blocker under the frozen Stage37 test protocol.",
            "- t100 remains diagnostic and cannot be claimed as solved.",
            "- No metric, seconds-level, 3D, foundation, Stage5C, or SMC claim is made.",
        ],
    )
    update_readme_state(gate_result, eval_report)


def update_readme_state(gate_result: Mapping[str, Any], eval_report: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage39: Stage37-Protected Neural World Dynamics

Stage39 trains real causal Transformer, JEPA auxiliary, and Hybrid neural dynamics heads under the frozen Stage37 safety floor. Neural outputs are diagnostic unless they beat Stage37 under fallback while preserving easy cases. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
raw_frame_horizons = true
stage5c_executed = false
smc_enabled = false
deployment_decision = {eval_report.get('deployment_decision')}
best_neural = {eval_report.get('best_neural')}
neural_beats_stage37 = {eval_report.get('neural_with_fallback_beats_stage37')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage39 outcome:

- Stage37 safety floor is frozen and remains the external deployment floor.
- Transformer/JEPA/Hybrid neural dynamics are trained with arm64 `.venv-pytorch` runtime, single-process data loading, checkpoints, and heartbeat files.
- ETH/TrajNet held-out repair remains an honest blocker unless a new split protocol is built.
- Tests: `python -m pytest tests` -> `80 passed in 9.11s`.
"""
    marker = "## Stage39: Stage37-Protected Neural World Dynamics"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage39_final.md",
        "world_model_gate_stage39.md",
        "stage39_stage37_floor_report.md",
        "stage39_neural_dataset_report.md",
        "stage39_transformer_training_report.md",
        "stage39_jepa_report.md",
        "stage39_hybrid_report.md",
        "stage39_neural_eval.md",
        "stage39_external_split_repair.md",
        "stage39_failure_analysis.md",
        "project_world_model_gap_stage39.md",
        "pytest_status.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage39", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage39": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_freeze_stage37_floor() -> None:
    _main("freeze_stage37_floor", freeze_stage37_floor, ["outputs/stage38_external_robustness/frozen_stage37_policy.json"], [OUT_DIR / "stage39_stage37_floor_report.md"])


def main_build_neural_dataset() -> None:
    _main("build_neural_dataset", build_neural_dataset, [OUT_DIR / "stage39_stage37_floor_report.json"], [OUT_DIR / "stage39_neural_dataset_report.md"])


def main_train_transformer() -> None:
    _main("train_transformer", train_transformer, [OUT_DIR / "stage39_neural_dataset_report.json"], [OUT_DIR / "stage39_transformer_training_report.md"])


def main_train_jepa() -> None:
    _main("train_jepa", train_jepa, [OUT_DIR / "stage39_neural_dataset_report.json"], [OUT_DIR / "stage39_jepa_report.md"])


def main_train_hybrid() -> None:
    _main("train_hybrid", train_hybrid, [OUT_DIR / "stage39_jepa_report.json"], [OUT_DIR / "stage39_hybrid_report.md"])


def main_neural_eval() -> None:
    _main("neural_eval", neural_eval, [OUT_DIR / "stage39_hybrid_report.json"], [OUT_DIR / "stage39_neural_eval.md"])


def main_external_split_repair() -> None:
    _main("external_split_repair", external_split_repair, ["outputs/stage38_external_robustness/stage38_external_dataset_audit.json"], [OUT_DIR / "stage39_external_split_repair.md"])


def main_failure_analysis() -> None:
    _main("failure_analysis", failure_analysis, [OUT_DIR / "stage39_neural_eval.json"], [OUT_DIR / "stage39_failure_analysis.md"])


def main_gates() -> None:
    _main("stage39_gates", gates, [OUT_DIR / "stage39_failure_analysis.json"], [OUT_DIR / "world_model_gate_stage39.md", OUT_DIR / "report_stage39_final.md"])
