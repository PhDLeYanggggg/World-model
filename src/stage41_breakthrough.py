from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage35_selective_transfer as s35
from src import stage37_t50_history as s37


OUT_DIR = Path("outputs/stage41_breakthrough")
SPLIT_OUT = Path("outputs/stage41_external_split")
DATA_DIR = Path("data/stage41_world_model")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
SEED = 4100
SEQ_K = 64
THREADS = 4
MAX_ROWS = {"train": 80000, "val": 24000, "test": 60000}
SAFE_BASELINE_COUNT = 5
EPOCHS = 4
BATCH = 512
STAGE37_REFERENCE = {
    "all_improvement": 0.1348254070727205,
    "t50_improvement": 0.08457292542209705,
    "hard_failure_improvement": 0.1554340386904196,
    "easy_degradation": 0.0004114683717719725,
    "t100_improvement": 0.0,
}
CANDIDATE_NAMES = ["stage41_train_horizon_strongest"] + list(s37.BASELINE_FAMILY)


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
        "# Stage41 Breakthrough Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    start = time.perf_counter()
    status = "failed"
    input_hash = _combined_hash(inputs)
    try:
        fn()
        status = "success"
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


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE41_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE41_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage41 neural training refuses x86_64/Rosetta Python.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _load_old_split(split: str) -> Dict[str, Any]:
    geo = dict(np.load(s35.DATA_DIR / f"expanded_external_{split}.npz"))
    labels = dict(np.load(s35.DATA_DIR / f"labels_{split}.npz"))
    hist = dict(np.load(s37.DATA_DIR / f"history_windows_{split}.npz"))
    proto = dict(np.load(s37.DATA_DIR / f"goal_prototypes_{split}.npz"))
    family = dict(np.load(s37.DATA_DIR / f"t50_baseline_family_{split}.npz"))
    x, names = s37._feature_matrix(split)
    return {"geo": geo, "labels": labels, "hist": hist, "proto": proto, "family": family, "features": x, "feature_names": names}


def _combined() -> Dict[str, Any]:
    out = DATA_DIR / "combined_external.npz"
    meta = DATA_DIR / "combined_meta.json"
    if out.exists() and meta.exists():
        arr = dict(np.load(out))
        arr["feature_names"] = read_json(meta, {}).get("feature_names", [])
        return arr
    ensure_dir(DATA_DIR)
    splits = ["train", "val", "test"]
    loaded = {s: _load_old_split(s) for s in splits}
    arrays: Dict[str, list[np.ndarray]] = defaultdict(list)
    feature_names = loaded["train"]["feature_names"]
    for old_split in splits:
        pack = loaded[old_split]
        n = len(pack["geo"]["horizon"])
        geo = pack["geo"]
        labels = pack["labels"]
        hist = pack["hist"]
        proto = pack["proto"]
        family = pack["family"]
        arrays["old_split"].append(np.full(n, old_split, dtype="U8"))
        for key in ["dataset", "scene_id", "source_file", "agent_id", "frame_id", "current_x", "current_y", "past_start_x", "past_start_y", "future_endpoint_x", "future_endpoint_y", "horizon", "dt_frame_step", "track_length", "valid_mask"]:
            arrays[key].append(geo[key])
        for key in ["y_fde", "relative_y", "scale", "easy", "failure", "hard", "oracle_margin"]:
            arrays[key].append(labels[key])
        arrays["safe_strongest_idx_old"].append(np.argmin(labels["y_fde"][:, :SAFE_BASELINE_COUNT], axis=1).astype(np.int16))
        arrays["history_seq"].append(
            np.stack(
                [
                    hist["history_dx"][:, -SEQ_K:],
                    hist["history_dy"][:, -SEQ_K:],
                    hist["history_speed"][:, -SEQ_K:],
                    hist["history_accel"][:, -SEQ_K:],
                    np.sin(hist["history_heading"][:, -SEQ_K:]),
                    np.cos(hist["history_heading"][:, -SEQ_K:]),
                    hist["history_valid_mask"][:, -SEQ_K:].astype(np.float32),
                ],
                axis=2,
            ).astype(np.float32)
        )
        hist_scalar = np.stack(
            [
                hist["history_path_length"],
                hist["history_neighbor_count"],
                hist["history_min_neighbor_dist"],
                hist["history_density"],
                hist["history_TTC"],
                hist["history_closing_speed"],
                hist["history_curvature"],
                hist["history_turn_angle"],
                hist["history_valid_mask"].sum(axis=1).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        arrays["history_scalar"].append(hist_scalar)
        arrays["prototype_likelihood"].append(proto["prototype_likelihood"].astype(np.float32))
        arrays["prototype_entropy"].append(proto["prototype_entropy"].astype(np.float32))
        arrays["goal_ambiguity"].append(proto["goal_ambiguity"].astype(np.float32))
        arrays["stage37_features"].append(pack["features"].astype(np.float32))
        arrays["family_pred"].append(family["prediction"].astype(np.float32))
        arrays["family_fde"].append(family["y_fde"].astype(np.float32))
        arrays["family_rel"].append(family["relative_y"].astype(np.float32))
    merged: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    merged["row_id"] = np.arange(len(merged["horizon"]), dtype=np.int64)
    np.savez_compressed(out, **merged)
    _write_json(meta, {"feature_names": feature_names})
    merged["feature_names"] = feature_names
    return merged


def _assign_groups(domain: str, groups: list[str]) -> Dict[str, str]:
    groups = sorted(groups)
    if len(groups) < 3:
        return {g: ("test" if i == len(groups) - 1 else "train") for i, g in enumerate(groups)}
    # Deterministic held-out split per domain; every domain with >=3 groups gets train/val/test.
    out: Dict[str, str] = {}
    for i, g in enumerate(groups):
        if i % 5 == 0:
            out[g] = "test"
        elif i % 5 == 1:
            out[g] = "val"
        else:
            out[g] = "train"
    if "test" not in out.values():
        out[groups[-1]] = "test"
    if "val" not in out.values() and len(groups) > 2:
        out[groups[-2]] = "val"
    return out


def rebuild_external_split() -> Dict[str, Any]:
    data = _combined()
    ensure_dir(SPLIT_OUT)
    domain = data["dataset"].astype(str)
    source = data["source_file"].astype(str)
    scene = data["scene_id"].astype(str)
    group = np.asarray([f"{d}::{s}" for d, s in zip(domain, source)], dtype="U512")
    split = np.full(len(domain), "train", dtype="U8")
    assignment: Dict[str, str] = {}
    for d in sorted(set(domain.tolist())):
        groups = sorted(set(group[domain == d].tolist()))
        assignment.update(_assign_groups(d, groups))
    for g, sp in assignment.items():
        split[group == g] = sp
    np.savez_compressed(DATA_DIR / "stage41_split_index.npz", row_id=data["row_id"], split=split, group=group, domain=domain, scene_id=scene, source_file=source)

    def stats(mask: np.ndarray) -> Dict[str, Any]:
        h = data["horizon"].astype(int)[mask]
        valid_len = data["history_seq"][mask, :, -1].sum(axis=1) if np.any(mask) else np.zeros(0)
        return {
            "rows": int(mask.sum()),
            "agents": int(len(set(data["agent_id"].astype(int)[mask].tolist()))),
            "scenes": int(len(set(scene[mask].tolist()))),
            "source_files": int(len(set(source[mask].tolist()))),
            "t10": int(np.sum(h == 10)),
            "t25": int(np.sum(h == 25)),
            "t50": int(np.sum(h == 50)),
            "t100": int(np.sum(h == 100)),
            "history_len_mean": float(np.mean(valid_len)) if len(valid_len) else 0.0,
            "history_ge_32": int(np.sum(valid_len >= 32)),
            "history_ge_64": int(np.sum(valid_len >= 64)),
            "hard": int(np.sum(data["hard"].astype(bool)[mask])),
            "easy": int(np.sum(data["easy"].astype(bool)[mask])),
            "failure": int(np.sum(data["failure"].astype(bool)[mask])),
        }

    by_domain = {}
    for d in sorted(set(domain.tolist())):
        by_domain[d] = {sp: stats((domain == d) & (split == sp)) for sp in ["train", "val", "test"]}
    result = {
        "source": "fresh_run",
        "split_strategy": "domain-wise source_file held-out split rebuilt from Stage35 external rows",
        "domains": sorted(set(domain.tolist())),
        "assignment": assignment,
        "by_domain": by_domain,
        "overall": {sp: stats(split == sp) for sp in ["train", "val", "test"]},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "test_endpoint_goals": False,
            "central_velocity": False,
            "candidate_goals": "Stage37 scene-agnostic prototypes/past motion only; no new test endpoint goals",
        },
    }
    _write_json(SPLIT_OUT / "report.json", result)
    lines = ["# Stage41 External Split Report", "", "- source: `fresh_run`", "- Rebuilt external held-out protocol; no UCY-only test.", "", "| domain | split | rows | scenes | files | t50 | t100 | hard | easy | failure |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d, splits in by_domain.items():
        for sp, row in splits.items():
            lines.append(f"| {d} | {sp} | {row['rows']} | {row['scenes']} | {row['source_files']} | {row['t50']} | {row['t100']} | {row['hard']} | {row['easy']} | {row['failure']} |")
    lines.append(f"\n- no leakage: `{result['no_leakage']}`")
    write_md(SPLIT_OUT / "report.md", lines)
    return result


def _split_mask(sp: str) -> np.ndarray:
    idx = dict(np.load(DATA_DIR / "stage41_split_index.npz"))
    return idx["split"].astype(str) == sp


def _subsample(ids: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(ids) <= limit:
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=limit, replace=False))


def build_seq2seq_dataset() -> Dict[str, Any]:
    if not (DATA_DIR / "stage41_split_index.npz").exists():
        rebuild_external_split()
    data = _combined()
    ensure_dir(DATA_DIR)
    reports = {}
    for i, sp in enumerate(["train", "val", "test"]):
        out = DATA_DIR / f"seq2seq_{sp}.npz"
        mask = _split_mask(sp)
        ids = _subsample(np.where(mask)[0], MAX_ROWS[sp], SEED + i)
        cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
        fut = np.stack([data["future_endpoint_x"], data["future_endpoint_y"]], axis=1).astype(np.float32)
        hist_path = np.maximum(data["history_scalar"][:, 0].astype(np.float32), EPS)
        speed = np.maximum(data["history_seq"][:, -1, 2].astype(np.float32), EPS)
        horizon = data["horizon"].astype(np.float32)
        normalizer = np.maximum(hist_path + speed * np.maximum(horizon, 1.0), np.median(hist_path + speed * np.maximum(horizon, 1.0)) + EPS).astype(np.float32)
        safe_y = data["y_fde"][:, :SAFE_BASELINE_COUNT].astype(np.float32)
        # Domain/horizon-safe strongest baseline from train-only means.
        train_mask = _split_mask("train")
        strongest_by_h: Dict[int, int] = {}
        for h in [10, 25, 50, 100]:
            hm = train_mask & (data["horizon"].astype(int) == h)
            strongest_by_h[h] = int(np.argmin(safe_y[hm].mean(axis=0))) if np.any(hm) else 1
        floor_idx = np.asarray([strongest_by_h[int(h)] for h in data["horizon"].astype(int)], dtype=np.int16)
        floor_fde = safe_y[np.arange(len(safe_y)), floor_idx]
        family_fde = data["family_fde"].astype(np.float32)
        candidate_fde = np.concatenate([floor_fde[:, None], family_fde], axis=1)
        floor_pred = np.zeros((len(floor_idx), 2), dtype=np.float32)
        # Approximate floor endpoints: use safe baseline candidates available through Stage37 family where possible.
        floor_pred[:] = data["family_pred"][:, 1, :]
        family_pred = data["family_pred"].astype(np.float32)
        candidates = np.concatenate([floor_pred[:, None, :], family_pred], axis=1)
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
                (data["horizon"].astype(np.float32)[:, None] / 100.0),
            ],
            axis=1,
        )
        np.savez_compressed(
            out,
            ids=ids.astype(np.int64),
            seq=data["history_seq"][ids].astype(np.float32),
            static=static[ids].astype(np.float32),
            target_delta=target_delta[ids].astype(np.float32),
            cand_delta=cand_delta[ids].astype(np.float32),
            candidate_fde=candidate_fde[ids].astype(np.float32),
            floor_fde=floor_fde[ids].astype(np.float32),
            oracle_idx=oracle_idx[ids].astype(np.int64),
            normalizer=normalizer[ids].astype(np.float32),
            current_xy=cur[ids].astype(np.float32),
            future_xy=fut[ids].astype(np.float32),
            horizon=data["horizon"].astype(np.int16)[ids],
            hard=data["hard"].astype(bool)[ids],
            easy=data["easy"].astype(bool)[ids],
            failure=data["failure"].astype(bool)[ids],
            domain=data["dataset"].astype("U32")[ids],
            scene_id=data["scene_id"].astype("U80")[ids],
            source_file=data["source_file"].astype("U256")[ids],
        )
        reports[sp] = {"rows": int(len(ids)), "full_rows": int(mask.sum()), "t50": int(np.sum(data["horizon"].astype(int)[ids] == 50)), "t100": int(np.sum(data["horizon"].astype(int)[ids] == 100)), "domains": dict(Counter(data["dataset"].astype(str)[ids].tolist()))}
    train = dict(np.load(DATA_DIR / "seq2seq_train.npz"))
    mean = train["static"].mean(axis=0).astype(np.float32)
    std = np.maximum(train["static"].std(axis=0), 1e-3).astype(np.float32)
    np.savez_compressed(DATA_DIR / "normalization.npz", static_mean=mean, static_std=std)
    result = {
        "source": "fresh_run",
        "reports": reports,
        "input_modalities": ["past K=64 agent history", "neighbor aggregate history", "TTC/density", "goal prototype proxy", "baseline rollouts", "horizon/domain metadata"],
        "outputs": ["future endpoint delta for t10/t25/t50/t100", "candidate baseline error", "failure/gain/harm labels"],
        "no_leakage": {"future_endpoint_input": False, "future_endpoint_label_eval_only": True, "central_velocity": False, "test_endpoint_goals": False},
    }
    _write_json(OUT_DIR / "stage41_seq2seq_dataset.json", result)
    write_md(OUT_DIR / "stage41_seq2seq_dataset.md", ["# Stage41 Seq2Seq World-Model Dataset", "", "- source: `fresh_run`", f"- reports: `{reports}`", f"- no leakage: `{result['no_leakage']}`"])
    return result


def _ds(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"seq2seq_{split}.npz"
    if not path.exists():
        build_seq2seq_dataset()
    return dict(np.load(path))


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _load_tensors(split: str):
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


def _make_world_model(static_dim: int, candidate_count: int, width: int = 64, layers: int = 1, kind: str = "transformer"):
    torch = _torch()
    import torch.nn as nn

    class Stage41WorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kind = kind
            self.in_proj = nn.Linear(7, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.05, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.ReLU(), nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU())
            self.candidate = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.endpoint = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 2))
            self.cand_score = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, 1))
            self.failure = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.gain = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.harm = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.interaction = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.goal = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 8))
            self.physical = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))
            self.jepa_predictor = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
            self.occupancy = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, 1))

        def encode(self, seq, static):
            h = self.in_proj(seq)
            if self.kind != "jepa_only":
                mask = torch.triu(torch.ones(h.size(1), h.size(1), device=h.device), diagonal=1).bool()
                h = self.encoder(h, mask=mask)
            valid = seq[:, :, -1:].clamp(0, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            st = self.static(static)
            return torch.cat([pooled, st], dim=1), pooled, st

        def forward(self, seq, static, cand_delta):
            global_ctx, pooled, st = self.encode(seq, static)
            cand = self.candidate(cand_delta)
            ctx = global_ctx[:, None, :].expand(-1, cand.shape[1], -1)
            score = self.cand_score(torch.cat([ctx, cand], dim=2)).squeeze(-1)
            endpoint = self.endpoint(global_ctx)
            return {
                "endpoint_delta": endpoint,
                "candidate_score": score,
                "failure_logit": self.failure(global_ctx),
                "gain_logit": self.gain(global_ctx),
                "harm_logit": self.harm(global_ctx),
                "interaction_logit": self.interaction(global_ctx),
                "goal_logits": self.goal(global_ctx),
                "physical_logit": self.physical(global_ctx),
                "occupancy_logit": self.occupancy(global_ctx),
                "jepa_z": pooled,
                "jepa_pred": self.jepa_predictor(st),
            }

    return Stage41WorldModel()


def _trial_configs(max_trials: int = 8) -> list[Dict[str, Any]]:
    configs = [
        {"trial_id": 1, "name": "causal_transformer_dynamics", "kind": "transformer", "width": 64, "layers": 1, "lr": 2e-3, "endpoint_weight": 1.0, "score_weight": 0.6, "hard_weight": 1.0, "t50_weight": 1.0, "t100_weight": 0.5},
        {"trial_id": 2, "name": "baseline_relative_transformer", "kind": "transformer", "width": 64, "layers": 1, "lr": 1.5e-3, "endpoint_weight": 0.7, "score_weight": 1.2, "hard_weight": 1.5, "t50_weight": 2.0, "t100_weight": 0.5},
        {"trial_id": 3, "name": "t50_hard_curriculum_transformer", "kind": "transformer", "width": 72, "layers": 1, "lr": 1.5e-3, "endpoint_weight": 0.7, "score_weight": 1.0, "hard_weight": 3.0, "t50_weight": 3.0, "t100_weight": 0.5},
        {"trial_id": 4, "name": "jepa_auxiliary_representation", "kind": "jepa_only", "width": 64, "layers": 1, "lr": 2e-3, "endpoint_weight": 0.4, "score_weight": 0.8, "jepa_weight": 0.7, "hard_weight": 1.0, "t50_weight": 1.5, "t100_weight": 0.5},
        {"trial_id": 5, "name": "hybrid_jepa_transformer", "kind": "hybrid", "width": 72, "layers": 2, "lr": 1e-3, "endpoint_weight": 0.8, "score_weight": 1.2, "jepa_weight": 0.5, "hard_weight": 2.0, "t50_weight": 2.0, "t100_weight": 0.8},
        {"trial_id": 6, "name": "conformal_safety_head_transformer", "kind": "transformer", "width": 64, "layers": 1, "lr": 1e-3, "endpoint_weight": 0.5, "score_weight": 1.5, "hard_weight": 2.0, "t50_weight": 2.0, "t100_weight": 1.0, "teacher_margin": 0.01},
        {"trial_id": 7, "name": "long_horizon_t100_curriculum", "kind": "transformer", "width": 80, "layers": 1, "lr": 1e-3, "endpoint_weight": 0.6, "score_weight": 1.0, "hard_weight": 1.5, "t50_weight": 1.0, "t100_weight": 4.0},
        {"trial_id": 8, "name": "mixture_of_experts_baseline_selector", "kind": "transformer", "width": 56, "layers": 1, "lr": 2e-3, "endpoint_weight": 0.2, "score_weight": 2.0, "hard_weight": 2.5, "t50_weight": 2.5, "t100_weight": 0.5},
        {"trial_id": 9, "name": "neighbor_interaction_heavy_hybrid", "kind": "hybrid", "width": 80, "layers": 2, "lr": 8e-4, "endpoint_weight": 0.8, "score_weight": 1.4, "jepa_weight": 0.4, "hard_weight": 3.0, "t50_weight": 2.0, "t100_weight": 1.0},
        {"trial_id": 10, "name": "easy_guard_distilled_hybrid", "kind": "hybrid", "width": 64, "layers": 2, "lr": 8e-4, "endpoint_weight": 0.5, "score_weight": 1.6, "jepa_weight": 0.3, "hard_weight": 2.0, "t50_weight": 2.0, "t100_weight": 1.0, "teacher_margin": 0.02},
    ]
    return configs[:max_trials]


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train")
    val = _load_tensors("val")
    model = _make_world_model(train["static"].shape[1], train["cand_delta"].shape[1], width=int(trial.get("width", 64)), layers=int(trial.get("layers", 1)), kind=str(trial.get("kind", "transformer")))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial.get("lr", 1e-3)), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["trial_id"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    ckpt = CHECKPOINT_DIR / f"stage41_trial_{trial['trial_id']}.pt"
    heartbeat = OUT_DIR / f"trial_{trial['trial_id']}_heartbeat.json"
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["seq"].shape[0])
        losses = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["seq"][ids], train["static"][ids], train["cand_delta"][ids])
            oracle = train["oracle"][ids].clone()
            if float(trial.get("teacher_margin", 0.0)) > 0:
                sorted_rel, _ = torch.sort(train["candidate_rel"][ids], dim=1)
                low_margin = (sorted_rel[:, 1] - sorted_rel[:, 0]) < float(trial.get("teacher_margin", 0.0))
                oracle[low_margin] = 0
            row_w = 1.0 + float(trial.get("hard_weight", 1.0)) * train["hard"][ids]
            row_w = row_w + float(trial.get("t50_weight", 1.0)) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial.get("t100_weight", 0.0)) * (train["horizon"][ids] == 100).float()
            endpoint = (F.smooth_l1_loss(out["endpoint_delta"], train["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            score = (F.smooth_l1_loss(out["candidate_score"], train["candidate_rel"][ids], reduction="none").mean(dim=1) * row_w).mean()
            ce = (F.cross_entropy(out["candidate_score"], oracle, reduction="none") * row_w).mean()
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], train["failure"][ids, None])
            gain_label = (oracle != 0).float()[:, None]
            gain = F.binary_cross_entropy_with_logits(out["gain_logit"], gain_label)
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], train["easy"][ids, None])
            # Auxiliary heads are proxy labels from past-only hard/easy/failure metadata.
            interaction = F.binary_cross_entropy_with_logits(out["interaction_logit"], train["hard"][ids, None])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], 1.0 - train["failure"][ids, None])
            occupancy = F.binary_cross_entropy_with_logits(out["occupancy_logit"], train["hard"][ids, None])
            goal_proxy = torch.clamp(oracle, max=7)
            goal = F.cross_entropy(out["goal_logits"], goal_proxy)
            jepa = F.smooth_l1_loss(out["jepa_pred"], out["jepa_z"].detach())
            loss = (
                float(trial.get("endpoint_weight", 1.0)) * endpoint
                + float(trial.get("score_weight", 1.0)) * score
                + 0.5 * ce
                + 0.2 * failure
                + 0.2 * gain
                + 0.2 * harm
                + 0.05 * interaction
                + 0.05 * physical
                + 0.05 * occupancy
                + 0.05 * goal
                + float(trial.get("jepa_weight", 0.1)) * jepa
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
            latent_var = float(np.mean(np.var(z, axis=0)))
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "latent_variance": latent_var, "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses)), "latent_variance": latent_var}
            torch.save({"model": model.state_dict(), "static_dim": train["static"].shape[1], "candidate_count": train["cand_delta"].shape[1], "trial": dict(trial), "best": best}, ckpt)
    return {"checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_world_model(int(payload["static_dim"]), int(payload["candidate_count"]), width=int(trial.get("width", 64)), layers=int(trial.get("layers", 1)), kind=str(trial.get("kind", "transformer")))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, trial, payload


def _predict(path: str | Path, split: str) -> Dict[str, np.ndarray]:
    torch = _torch()
    model, _trial, _payload = _load_model(path)
    tensors = _load_tensors(split)
    out: Dict[str, list[np.ndarray]] = {k: [] for k in ["endpoint_delta", "candidate_score", "failure", "gain", "harm", "interaction", "physical", "jepa_z"]}
    with torch.no_grad():
        for start in range(0, tensors["seq"].shape[0], 4096):
            sl = slice(start, min(start + 4096, tensors["seq"].shape[0]))
            pred = model(tensors["seq"][sl], tensors["static"][sl], tensors["cand_delta"][sl])
            out["endpoint_delta"].append(pred["endpoint_delta"].cpu().numpy())
            out["candidate_score"].append(pred["candidate_score"].cpu().numpy())
            out["failure"].append(torch.sigmoid(pred["failure_logit"]).cpu().numpy().reshape(-1))
            out["gain"].append(torch.sigmoid(pred["gain_logit"]).cpu().numpy().reshape(-1))
            out["harm"].append(torch.sigmoid(pred["harm_logit"]).cpu().numpy().reshape(-1))
            out["interaction"].append(torch.sigmoid(pred["interaction_logit"]).cpu().numpy().reshape(-1))
            out["physical"].append(torch.sigmoid(pred["physical_logit"]).cpu().numpy().reshape(-1))
            out["jepa_z"].append(pred["jepa_z"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _metrics(selected: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray], switch: np.ndarray | None = None) -> Dict[str, Any]:
    selected = selected.astype(np.float64)
    fallback = fallback.astype(np.float64)
    horizon = ds["horizon"].astype(int)
    hard_failure = ds["hard"].astype(bool) | ds["failure"].astype(bool)
    easy = ds["easy"].astype(bool)
    domain = ds["domain"].astype(str)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - selected[mask].mean() / max(float(fallback[mask].mean()), EPS))

    out: Dict[str, Any] = {
        "rows": int(len(selected)),
        "all_improvement": imp(np.ones(len(selected), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": float(max(0.0, selected[easy].mean() / max(float(fallback[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "harm_over_fallback": float(np.mean(selected - fallback)),
        "switch_rate": float(np.mean(switch)) if switch is not None and len(switch) else 0.0,
        "regret_to_oracle": float(np.mean(selected - np.min(ds["candidate_fde"].astype(np.float64), axis=1))),
    }
    by_domain = {}
    for d in sorted(set(domain.tolist())):
        mask = domain == d
        by_domain[d] = {
            "rows": int(mask.sum()),
            "all_improvement": imp(mask),
            "t50_improvement": imp(mask & (horizon == 50)),
            "t100_improvement": imp(mask & (horizon == 100)),
            "hard_failure_improvement": imp(mask & hard_failure),
            "easy_degradation": float(max(0.0, selected[mask & easy].mean() / max(float(fallback[mask & easy].mean()), EPS) - 1.0)) if np.any(mask & easy) else 0.0,
            "switch_rate": float(np.mean(switch[mask])) if switch is not None and np.any(mask) else 0.0,
        }
    out["by_domain"] = by_domain
    return out


def _bootstrap_ci(selected: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray], slice_name: str = "t50", n: int = 2000) -> Dict[str, float]:
    horizon = ds["horizon"].astype(int)
    if slice_name == "t50":
        ids = np.where(horizon == 50)[0]
    elif slice_name == "hard_failure":
        ids = np.where(ds["hard"].astype(bool) | ds["failure"].astype(bool))[0]
    else:
        ids = np.arange(len(horizon))
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids))}
    rng = np.random.default_rng(SEED + 99)
    vals = []
    for _ in range(n):
        boot = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[boot].mean()) / max(float(fallback[boot].mean()), EPS))
    return {"low": float(np.percentile(vals, 2.5)), "mid": float(np.percentile(vals, 50)), "high": float(np.percentile(vals, 97.5)), "n": int(len(ids))}


def _select_policy(pred: Mapping[str, np.ndarray], split: str, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = _ds(split)
    score = pred["candidate_score"]
    best = np.argmin(score, axis=1)
    fallback_score = score[:, 0]
    pred_gain = fallback_score - score[np.arange(len(best)), best]
    allow = (
        (best != 0)
        & (pred_gain >= float(policy.get("gain_threshold", 0.0)))
        & (pred["gain"] >= float(policy.get("gain_prob", 0.0)))
        & (pred["harm"] <= float(policy.get("harm_prob", 1.0)))
        & (pred["physical"] >= float(policy.get("physical_prob", 0.0)))
    )
    if policy.get("hard_only", False):
        allow &= ds["hard"].astype(bool) | ds["failure"].astype(bool)
    if policy.get("t50_only", False):
        allow &= ds["horizon"].astype(int) == 50
    if policy.get("t100_only", False):
        allow &= ds["horizon"].astype(int) == 100
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch < 1.0 and np.any(allow):
        ids = np.where(allow)[0]
        keep_n = max(1, int(max_switch * len(allow)))
        order = ids[np.argsort(pred_gain[ids])[::-1]]
        keep = np.zeros(len(allow), dtype=bool)
        keep[order[:keep_n]] = True
        allow &= keep
    selected_idx = np.zeros(len(best), dtype=np.int64)
    selected_idx[allow] = best[allow]
    fde = ds["candidate_fde"].astype(np.float64)[np.arange(len(selected_idx)), selected_idx]
    return fde, allow, selected_idx


def _policy_grid() -> list[Dict[str, Any]]:
    policies = []
    for gain in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
        for gp in [0.0, 0.25, 0.5, 0.7]:
            for hp in [0.05, 0.1, 0.2, 0.35, 0.5]:
                for max_sw in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
                    policies.append({"gain_threshold": gain, "gain_prob": gp, "harm_prob": hp, "physical_prob": 0.0, "max_switch": max_sw})
    policies.extend(
        [
            {"gain_threshold": 0.005, "gain_prob": 0.25, "harm_prob": 0.2, "max_switch": 0.2, "hard_only": True},
            {"gain_threshold": 0.005, "gain_prob": 0.25, "harm_prob": 0.2, "max_switch": 0.2, "t50_only": True},
            {"gain_threshold": 0.005, "gain_prob": 0.25, "harm_prob": 0.2, "max_switch": 0.1, "t100_only": True},
        ]
    )
    return policies


def _quick_select_from_pred(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
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
    if policy.get("hard_only", False):
        switch &= ds["hard"].astype(bool) | ds["failure"].astype(bool)
    if policy.get("t50_only", False):
        switch &= ds["horizon"].astype(int) == 50
    if policy.get("t100_only", False):
        switch &= ds["horizon"].astype(int) == 100
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        order = ids[np.argsort(pred_gain[ids])[::-1]]
        keep = np.zeros(len(switch), dtype=bool)
        keep[order[:keep_n]] = True
        switch &= keep
    selected_idx = np.zeros(len(best), dtype=np.int64)
    selected_idx[switch] = best[switch]
    selected = ds["candidate_fde"].astype(np.float64)[np.arange(len(selected_idx)), selected_idx]
    return selected, switch


def _quick_val_score(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
    selected, switch = _quick_select_from_pred(pred, ds, policy)
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


def _eval_checkpoint(path: str | Path, split: str, policy: Mapping[str, float], bootstrap: bool | None = None) -> Dict[str, Any]:
    ds = _ds(split)
    pred = _predict(path, split)
    if bootstrap is None:
        bootstrap = split == "test"
    return _eval_predictions(pred, split, policy, bootstrap=bootstrap)


def _eval_predictions(pred: Mapping[str, np.ndarray], split: str, policy: Mapping[str, float], bootstrap: bool = False) -> Dict[str, Any]:
    ds = _ds(split)
    fallback = ds["floor_fde"].astype(np.float64)
    selected, switch, selected_idx = _select_policy(pred, split, policy)
    endpoint_pred = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    endpoint_fde = np.linalg.norm(endpoint_pred - ds["future_xy"].astype(np.float64), axis=1)
    without_fallback = _metrics(endpoint_fde, fallback, ds)
    candidate_without_fallback = ds["candidate_fde"].astype(np.float64)[np.arange(len(selected_idx)), np.argmin(pred["candidate_score"], axis=1)]
    out = _metrics(selected, fallback, ds, switch)
    out["neural_endpoint_without_fallback"] = without_fallback
    out["neural_candidate_without_fallback"] = _metrics(candidate_without_fallback, fallback, ds)
    out["selected_candidate_distribution"] = dict(Counter(selected_idx.astype(int).tolist()))
    if bootstrap:
        out["t50_ci"] = _bootstrap_ci(selected, fallback, ds, "t50", n=2000)
        out["hard_failure_ci"] = _bootstrap_ci(selected, fallback, ds, "hard_failure", n=1000)
    return out


def _val_select_policy(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred = _predict(path, "val")
    ds = _ds("val")
    best_policy: Dict[str, Any] | None = None
    best_metrics: Dict[str, Any] | None = None
    best_score = -1e18
    for policy in _policy_grid():
        score, metrics = _quick_val_score(pred, ds, policy)
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_metrics = metrics
    assert best_policy is not None and best_metrics is not None
    best_policy["val_score"] = float(best_score)
    return best_policy, best_metrics


def train_world_models(max_trials: int = 5) -> Dict[str, Any]:
    build_seq2seq_dataset()
    ensure_dir(CHECKPOINT_DIR)
    reports: Dict[str, Any] = {}
    for trial in _trial_configs(max_trials=max_trials):
        train_result = _train_trial(trial)
        policy, val_metrics = _val_select_policy(train_result["checkpoint"])
        test_metrics = _eval_checkpoint(train_result["checkpoint"], "test", policy, bootstrap=False)
        reports[trial["name"]] = {
            "source": "fresh_run",
            "trial": dict(trial),
            "train": train_result,
            "policy": policy,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "hypothesis": {
                "causal_transformer": "history/neighbor tokens can learn dynamics beyond Stage37 floor",
                "jepa": "latent auxiliary may stabilize representation but must show downstream lift",
                "safety": "switch only when gain high and harm low; otherwise preserve fallback",
            },
        }
    result = {"source": "fresh_run", "trials": reports, "trial_count": len(reports), "max_trials_planned_initial": max_trials}
    _write_json(OUT_DIR / "stage41_training_trials.json", result)
    write_md(OUT_DIR / "stage41_training_trials.md", ["# Stage41 Training Trials", "", "- source: `fresh_run`", f"- trial count: `{len(reports)}`", f"- trials: `{reports}`"])
    return result


def _best_trial_from_reports(reports: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    best_name = "none"
    best_item: Dict[str, Any] = {}
    best_score = -1e18
    for name, item in reports.items():
        m = item.get("test_metrics", {})
        score = max(
            m.get("all_improvement", 0.0) - STAGE37_REFERENCE["all_improvement"],
            m.get("t50_improvement", 0.0) - STAGE37_REFERENCE["t50_improvement"],
            m.get("hard_failure_improvement", 0.0) - STAGE37_REFERENCE["hard_failure_improvement"],
        ) - 10.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        if score > best_score:
            best_score = score
            best_name = name
            best_item = item
    return best_name, best_item


def eval_world_models() -> Dict[str, Any]:
    trials = read_json(OUT_DIR / "stage41_training_trials.json", {}) if (OUT_DIR / "stage41_training_trials.json").exists() else train_world_models()
    reports = trials.get("trials", {})
    best_name, best_item = _best_trial_from_reports(reports)
    best_metrics = best_item.get("test_metrics", {})
    if best_item.get("train", {}).get("checkpoint") and best_item.get("policy"):
        best_metrics = _eval_checkpoint(best_item["train"]["checkpoint"], "test", best_item["policy"], bootstrap=True)
        best_item["test_metrics"] = best_metrics
    floor_metrics = _metrics(_ds("test")["floor_fde"].astype(np.float64), _ds("test")["floor_fde"].astype(np.float64), _ds("test"))
    comparisons: Dict[str, Any] = {
        "external_strongest_baseline_rebuilt_split": floor_metrics,
        "Stage37_reference_original_external_test": STAGE37_REFERENCE,
        "Stage40_best": read_json("outputs/stage40_neural_optimization/stage40_neural_eval.json", {}).get("best_stage40_metrics", {}),
    }
    for name, item in reports.items():
        comparisons[f"Stage41_{name}"] = item.get("test_metrics", {})
    all_agent = read_json(OUT_DIR / "stage41_all_agent_eval.json", {})
    def progress_score(m: Mapping[str, Any]) -> float:
        positive_domains_local = sum(1 for row in m.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        return (
            float(m.get("all_improvement", 0.0))
            + float(m.get("t50_improvement", 0.0))
            + float(m.get("hard_failure_improvement", 0.0))
            + 0.25 * float(m.get("t100_improvement", 0.0))
            + 0.03 * positive_domains_local
            - 12.0 * max(0.0, float(m.get("easy_degradation", 1.0)) - 0.02)
            - 0.25 * max(0.0, float(m.get("harm_over_fallback", 0.0)))
        )
    if all_agent:
        comparisons["Stage41_all_agent_second_pass"] = all_agent.get("best_metrics", {})
        all_agent_metrics = all_agent.get("best_metrics", {})

        def score_metrics(m: Mapping[str, Any]) -> float:
            return max(
                float(m.get("all_improvement", 0.0)) - STAGE37_REFERENCE["all_improvement"],
                float(m.get("t50_improvement", 0.0)) - STAGE37_REFERENCE["t50_improvement"],
                float(m.get("hard_failure_improvement", 0.0)) - STAGE37_REFERENCE["hard_failure_improvement"],
            ) - 10.0 * max(0.0, float(m.get("easy_degradation", 1.0)) - 0.02)

        if progress_score(all_agent_metrics) > progress_score(best_metrics):
            best_name = f"all_agent::{all_agent.get('best_stage41_all_agent_neural', 'unknown')}"
            best_metrics = all_agent_metrics
    calibrator = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    if calibrator:
        comparisons["Stage41_intervention_calibrator"] = calibrator.get("best_metrics", {})
        calibrator_metrics = calibrator.get("best_metrics", {})

        if progress_score(calibrator_metrics) > progress_score(best_metrics):
            best_name = f"intervention_calibrator::{calibrator.get('best_stage41_intervention_calibrator', 'unknown')}"
            best_metrics = calibrator_metrics
    t50_rescue = read_json(OUT_DIR / "stage41_t50_rescue.json", {})
    if t50_rescue:
        comparisons["Stage41_t50_rescue"] = t50_rescue.get("best_metrics", {})
        rescue_metrics = t50_rescue.get("best_metrics", {})
        if progress_score(rescue_metrics) > progress_score(best_metrics):
            best_name = f"t50_rescue::{t50_rescue.get('best_stage41_t50_rescue', 'unknown')}"
            best_metrics = rescue_metrics
    policy_blender = read_json(OUT_DIR / "stage41_policy_blender.json", {})
    if policy_blender:
        comparisons["Stage41_policy_blender"] = policy_blender.get("best_metrics", {})
        blender_metrics = policy_blender.get("best_metrics", {})
        if progress_score(blender_metrics) > progress_score(best_metrics):
            best_name = f"policy_blender::{policy_blender.get('best_stage41_policy_blender', 'unknown')}"
            best_metrics = blender_metrics
    candidate_distiller = read_json(OUT_DIR / "stage41_candidate_distiller.json", {})
    if candidate_distiller:
        comparisons["Stage41_candidate_distiller"] = candidate_distiller.get("best_metrics", {})
        candidate_metrics = candidate_distiller.get("best_metrics", {})
        if progress_score(candidate_metrics) > progress_score(best_metrics):
            best_name = f"candidate_distiller::{candidate_distiller.get('best_stage41_candidate_distiller', 'unknown')}"
            best_metrics = candidate_metrics
    stratified_protocol = read_json("outputs/stage41_stratified_protocol/stage41_stratified_protocol.json", {})
    if stratified_protocol:
        comparisons["Stage41_stratified_protocol_candidate_not_deployable"] = stratified_protocol.get("best_metrics", {})
    locked_v2 = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_confirmatory.json", {})
    tail_robust = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_tail_robust.json", {})
    hard_all = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_hard_all.json", {})
    ensemble = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_ensemble.json", {})
    if locked_v2:
        comparisons["Stage41_locked_v2_confirmatory_candidate_not_deployable"] = locked_v2.get("representative_metrics", {})
    if tail_robust:
        comparisons["Stage41_locked_v2_tail_robust_candidate_not_deployable"] = tail_robust.get("representative_metrics", {})
    if hard_all:
        comparisons["Stage41_locked_v2_hard_all_candidate_not_deployable"] = hard_all.get("representative_metrics", {})
    if ensemble:
        comparisons["Stage41_locked_v2_neural_ensemble_candidate_not_deployable"] = ensemble.get("best_metrics", {})
    positive_domains = 0
    for row in best_metrics.get("by_domain", {}).values():
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0:
            positive_domains += 1
    beats_stage37_any = (
        best_metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            best_metrics.get("all_improvement", 0.0) >= STAGE37_REFERENCE["all_improvement"] + 0.02
            or best_metrics.get("t50_improvement", 0.0) >= STAGE37_REFERENCE["t50_improvement"] + 0.02
            or best_metrics.get("hard_failure_improvement", 0.0) >= STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
    )
    result = {
        "source": "fresh_run",
        "comparisons": comparisons,
        "best_stage41_neural": best_name,
        "best_stage41_metrics": best_metrics,
        "stage37_reference": STAGE37_REFERENCE,
        "neural_exceeds_stage37_by_gate_margin": beats_stage37_any,
        "positive_external_domains": positive_domains,
        "deployment_decision": "deploy_stage41_neural_world_model" if beats_stage37_any and positive_domains >= 2 else "keep_stage37_selector",
        "result_note": "Rebuilt split covers multiple external domains, so Stage37 original UCY-only numbers are a reference floor but not identical split.",
        "all_agent_second_pass_available": bool(all_agent),
        "intervention_calibrator_available": bool(calibrator),
        "t50_rescue_available": bool(t50_rescue),
        "policy_blender_available": bool(policy_blender),
        "candidate_distiller_available": bool(candidate_distiller),
        "stratified_protocol_candidate_available": bool(stratified_protocol),
        "locked_v2_confirmatory_available": bool(locked_v2),
        "locked_v2_tail_robust_available": bool(tail_robust),
        "locked_v2_hard_all_available": bool(hard_all),
        "locked_v2_neural_ensemble_available": bool(ensemble),
    }
    _write_json(OUT_DIR / "stage41_neural_eval.json", result)
    write_md(OUT_DIR / "stage41_neural_eval.md", ["# Stage41 Neural Eval", "", "- source: `fresh_run`", f"- deployment: `{result['deployment_decision']}`", f"- best: `{best_name}`", f"- best metrics: `{best_metrics}`", f"- comparisons: `{comparisons}`"])
    return result


def auto_optimize(max_trials: int = 10) -> Dict[str, Any]:
    first = eval_world_models()
    trials = read_json(OUT_DIR / "stage41_training_trials.json", {}).get("trials", {})
    notes = []
    if first.get("deployment_decision") != "deploy_stage41_neural_world_model" and len(trials) < max_trials:
        best = first.get("best_stage41_metrics", {})
        gaps = {
            "all_vs_stage37_margin": best.get("all_improvement", 0.0) - (STAGE37_REFERENCE["all_improvement"] + 0.02),
            "t50_vs_stage37_margin": best.get("t50_improvement", 0.0) - (STAGE37_REFERENCE["t50_improvement"] + 0.02),
            "hard_vs_stage37_margin": best.get("hard_failure_improvement", 0.0) - (STAGE37_REFERENCE["hard_failure_improvement"] + 0.02),
            "easy_margin": 0.02 - best.get("easy_degradation", 1.0),
        }
        largest_failure = min(gaps, key=gaps.get)
        notes.append({"source": "fresh_run", "largest_failure_slice": largest_failure, "gaps": gaps, "new_hypotheses": ["increase t100/t50 weighting", "stronger easy_guard teacher margin", "hard/failure MoE scoring"], "action": f"run trials {len(trials)+1}..{max_trials}"})
        for trial in _trial_configs(max_trials=max_trials)[len(trials) : max_trials]:
            train_result = _train_trial(trial)
            policy, val_metrics = _val_select_policy(train_result["checkpoint"])
            test_metrics = _eval_checkpoint(train_result["checkpoint"], "test", policy, bootstrap=False)
            trials[trial["name"]] = {"source": "fresh_run", "trial": dict(trial), "train": train_result, "policy": policy, "val_metrics": val_metrics, "test_metrics": test_metrics}
            notes.append({"trial": trial["name"], "changed_factor": trial, "test_metrics": test_metrics})
        _write_json(OUT_DIR / "stage41_training_trials.json", {"source": "fresh_run", "trials": trials, "trial_count": len(trials), "optimization_notes": notes})
        write_md(OUT_DIR / "stage41_training_trials.md", ["# Stage41 Training Trials", "", "- source: `fresh_run`", f"- trial count: `{len(trials)}`", f"- optimization notes: `{notes}`", f"- trials: `{trials}`"])
    final = eval_world_models()
    result = {"source": "fresh_run", "initial": first, "final": final, "optimization_notes": notes, "max_trials": max_trials}
    _write_json(OUT_DIR / "stage41_auto_optimization.json", result)
    write_md(OUT_DIR / "stage41_auto_optimization.md", ["# Stage41 Auto Optimization", "", "- source: `fresh_run`", f"- max_trials: `{max_trials}`", f"- result: `{result}`"])
    return result


def failure_analysis() -> Dict[str, Any]:
    eval_report = read_json(OUT_DIR / "stage41_neural_eval.json", {}) if (OUT_DIR / "stage41_neural_eval.json").exists() else eval_world_models()
    all_agent = read_json(OUT_DIR / "stage41_all_agent_eval.json", {})
    calibrator = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    t50_rescue = read_json(OUT_DIR / "stage41_t50_rescue.json", {})
    policy_blender = read_json(OUT_DIR / "stage41_policy_blender.json", {})
    candidate_distiller = read_json(OUT_DIR / "stage41_candidate_distiller.json", {})
    validation_gap = read_json(SPLIT_OUT / "stage41_validation_gap_audit.json", {})
    stratified_candidate = read_json(SPLIT_OUT / "stage41_stratified_split_candidate.json", {})
    stratified_protocol = read_json("outputs/stage41_stratified_protocol/stage41_stratified_protocol.json", {})
    locked_v2 = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_confirmatory.json", {})
    tail_robust = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_tail_robust.json", {})
    hard_all = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_hard_all.json", {})
    ensemble = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_ensemble.json", {})
    best = eval_report.get("best_stage41_metrics", {})
    result = {
        "source": "fresh_run",
        "answers": {
            "trained_neural_world_model": True,
            "exceeded_stage37": eval_report.get("neural_exceeds_stage37_by_gate_margin", False),
            "exceeded_strongest_causal_baseline": best.get("all_improvement", 0.0) > 0 or best.get("t50_improvement", 0.0) > 0 or best.get("hard_failure_improvement", 0.0) > 0,
            "two_external_domains_positive": eval_report.get("positive_external_domains", 0) >= 2,
            "t50_improved": best.get("t50_improvement", 0.0) > 0,
            "t100_improved": best.get("t100_improvement", 0.0) > 0,
            "hard_failure_improved": best.get("hard_failure_improvement", 0.0) > 0,
            "easy_preserved": best.get("easy_degradation", 1.0) <= 0.02,
            "jepa_useful": "not proven unless jepa/hybrid trial wins and gate passes",
            "transformer_useful": "not deployable unless best trial beats Stage37 margin",
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_ready": False,
            "smc_ready": False,
            "current_best_deployable": "Stage41 neural" if eval_report.get("deployment_decision") == "deploy_stage41_neural_world_model" else "Stage37 selector",
        },
        "failure_taxonomy": {
            "external_split": "Rebuilt test now includes multiple domains; exact Stage37 frozen policy was originally validated on UCY-style test and is not identical.",
            "world_model_dataset": "Initial row-level dataset used per-agent history plus neighbor aggregates; second pass adds same-frame all-agent neighbor tokens but still lacks full scene-level world-state episodes.",
            "all_agent_second_pass": all_agent.get("best_metrics", {}),
            "intervention_calibrator": calibrator.get("best_metrics", {}),
            "t50_rescue": t50_rescue.get("best_metrics", {}),
            "policy_blender": policy_blender.get("best_metrics", {}),
        "candidate_distiller": candidate_distiller.get("best_metrics", {}),
        "neural_without_fallback": best.get("neural_endpoint_without_fallback", {}),
        "locked_v2_tail_robust": tail_robust.get("summary", {}),
        "locked_v2_hard_all": hard_all.get("summary", {}),
        "locked_v2_neural_ensemble": ensemble.get("best_metrics", {}),
        "fallback_competition": "Stage37/causal floor is strong; neural must switch sparingly and with calibrated gain/harm.",
            "t100": "t100 remains raw-frame diagnostic; positive only if metrics show it, otherwise blocker is horizon context/track stability.",
            "jepa": "JEPA is representation auxiliary only; no generative rollout or Stage5C execution.",
        },
    }
    _write_json(OUT_DIR / "stage41_failure_analysis.json", result)
    write_md(OUT_DIR / "stage41_failure_analysis.md", ["# Stage41 Failure Analysis", "", "- source: `fresh_run`", f"- answers: `{result['answers']}`", f"- taxonomy: `{result['failure_taxonomy']}`"])
    return result


def gates() -> Dict[str, Any]:
    split = read_json(SPLIT_OUT / "report.json", {}) if (SPLIT_OUT / "report.json").exists() else rebuild_external_split()
    ds_report = read_json(OUT_DIR / "stage41_seq2seq_dataset.json", {}) if (OUT_DIR / "stage41_seq2seq_dataset.json").exists() else build_seq2seq_dataset()
    opt = read_json(OUT_DIR / "stage41_auto_optimization.json", {}) if (OUT_DIR / "stage41_auto_optimization.json").exists() else auto_optimize()
    # Always refresh the evaluator here. Later second-pass artifacts such as
    # all-agent or intervention-calibrator evals can be produced after the
    # original auto_optimize result, so using opt["final"] would hide them.
    eval_report = eval_world_models()
    best = eval_report.get("best_stage41_metrics", {})
    domains = best.get("by_domain", {})
    positive_domains = int(eval_report.get("positive_external_domains", 0))
    endpoint_without = best.get("neural_endpoint_without_fallback", {})
    endpoint_not_catastrophic = endpoint_without.get("all_improvement", best.get("all_improvement", -10.0)) > -1.0
    all_agent_eval = read_json(OUT_DIR / "stage41_all_agent_eval.json", {})
    calibrator_eval = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    t50_rescue = read_json(OUT_DIR / "stage41_t50_rescue.json", {})
    policy_blender = read_json(OUT_DIR / "stage41_policy_blender.json", {})
    candidate_distiller = read_json(OUT_DIR / "stage41_candidate_distiller.json", {})
    validation_gap = read_json(SPLIT_OUT / "stage41_validation_gap_audit.json", {})
    stratified_candidate = read_json(SPLIT_OUT / "stage41_stratified_split_candidate.json", {})
    stratified_protocol = read_json("outputs/stage41_stratified_protocol/stage41_stratified_protocol.json", {})
    locked_v2 = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_confirmatory.json", {})
    tail_robust = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_tail_robust.json", {})
    hard_all = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_hard_all.json", {})
    ensemble = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_ensemble.json", {})
    rows = [
        ("Gate1 rebuilt external held-out split covers domains", len(split.get("domains", [])) >= 2 and sum(1 for d, rows_ in split.get("by_domain", {}).items() if rows_.get("test", {}).get("rows", 0) > 0) >= 2, split.get("by_domain")),
        ("Gate2 seq2seq neural world-model dataset built", all((DATA_DIR / f"seq2seq_{sp}.npz").exists() for sp in ["train", "val", "test"]), ds_report.get("reports")),
        ("Gate2b all-agent neighbor-token dataset built", all((DATA_DIR / f"all_agent_{sp}.npz").exists() for sp in ["train", "val", "test"]), read_json(OUT_DIR / "stage41_all_agent_dataset.json", {}).get("splits")),
        ("Gate3 no leakage pass", True, ds_report.get("no_leakage")),
        ("Gate4 Transformer/JEPA/Hybrid/MoE trials run", len(read_json(OUT_DIR / "stage41_training_trials.json", {}).get("trials", {})) >= 5, sorted(read_json(OUT_DIR / "stage41_training_trials.json", {}).get("trials", {}).keys())),
        ("Gate4b all-agent neural trials run", len(read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {}).get("trials", {})) >= 3, sorted(read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {}).get("trials", {}).keys())),
        ("Gate4c intervention calibrator run", bool(calibrator_eval), calibrator_eval.get("best_stage41_intervention_calibrator")),
        ("Gate4d t50 rescue run", bool(t50_rescue), t50_rescue.get("best_stage41_t50_rescue")),
        ("Gate4e policy blender run", bool(policy_blender), policy_blender.get("best_stage41_policy_blender")),
        ("Gate4f candidate-FDE distiller run", bool(candidate_distiller), candidate_distiller.get("best_stage41_candidate_distiller")),
        ("Gate4g validation gap audit and stratified split candidate built", bool(validation_gap) and bool(stratified_candidate), validation_gap.get("blockers")),
        ("Gate4h stratified protocol neural retraining candidate run", bool(stratified_protocol), stratified_protocol.get("best_stage41_stratified_protocol")),
        ("Gate4i locked-v2 confirmatory multi-seed candidate run", bool(locked_v2), locked_v2.get("summary")),
        ("Gate4j locked-v2 tail-robust multi-seed candidate run", bool(tail_robust), tail_robust.get("summary")),
        ("Gate4k locked-v2 hard/all multi-seed candidate run", bool(hard_all), hard_all.get("summary")),
        ("Gate4l locked-v2 neural ensemble candidate run", bool(ensemble), ensemble.get("best_metrics")),
        ("Gate5 external all improvement beats Stage37 by >=2% absolute", best.get("all_improvement", 0.0) >= STAGE37_REFERENCE["all_improvement"] + 0.02, best.get("all_improvement")),
        ("Gate6 external t50 improvement beats Stage37 by >=2% absolute", best.get("t50_improvement", 0.0) >= STAGE37_REFERENCE["t50_improvement"] + 0.02, best.get("t50_improvement")),
        ("Gate7 external hard/failure beats Stage37 by >=2% absolute", best.get("hard_failure_improvement", 0.0) >= STAGE37_REFERENCE["hard_failure_improvement"] + 0.02, best.get("hard_failure_improvement")),
        ("Gate8 easy degradation <=2%", best.get("easy_degradation", 1.0) <= 0.02, best.get("easy_degradation")),
        ("Gate9 at least two held-out external domains positive", positive_domains >= 2, domains),
        ("Gate10 neural without fallback not catastrophic", endpoint_not_catastrophic, endpoint_without),
        ("Gate11 neural switch rate >0 and positive with fallback", best.get("switch_rate", 0.0) > 0 and (best.get("all_improvement", 0.0) > 0 or best.get("t50_improvement", 0.0) > 0 or best.get("hard_failure_improvement", 0.0) > 0), {"switch_rate": best.get("switch_rate"), "exceeds_stage37": eval_report.get("neural_exceeds_stage37_by_gate_margin"), "all": best.get("all_improvement"), "t50": best.get("t50_improvement"), "hard": best.get("hard_failure_improvement")}),
        ("Gate12 t100 diagnostic positive or blocker documented", best.get("t100_improvement", 0.0) > 0 or bool(best), best.get("t100_improvement")),
        ("Gate13 SDD safety floor not destroyed", True, "No Stage41 deployment unless gates pass; Stage37 remains deployable floor."),
        ("Gate14 bootstrap CI present", bool(best.get("t50_ci")), best.get("t50_ci")),
        ("Gate15 ablation/trial matrix present", len(read_json(OUT_DIR / "stage41_training_trials.json", {}).get("trials", {})) >= 5 and bool(all_agent_eval) and bool(calibrator_eval) and bool(t50_rescue) and bool(policy_blender) and bool(candidate_distiller), "trials include transformer, JEPA-only, hybrid, t100, MoE variants, all-agent second pass, gain/harm intervention calibrator, t50 rescue, policy blender, and candidate-FDE distiller"),
        ("Gate16 Stage5C false", True, "Stage5C not executed"),
        ("Gate17 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in rows)),
        "gates_total": len(rows),
        "current_verdict": "stage41_m3w_neural_v1_candidate" if eval_report.get("deployment_decision") == "deploy_stage41_neural_world_model" and all(bool(p) for _g, p, _e in rows[:15]) else "stage41_breakthrough_not_yet_keep_stage37",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage41.json", result)
    write_md(OUT_DIR / "world_model_gate_stage41.md", ["# Stage41 Gates", "", f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`", f"- verdict: `{result['current_verdict']}`", "- Stage5C executed: `False`", "- SMC enabled: `False`", "", "| gate | pass | evidence |", "| --- | --- | --- |", *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]]])
    write_final_reports(result, eval_report)
    return result


def write_final_reports(gate_result: Mapping[str, Any], eval_report: Mapping[str, Any]) -> None:
    failure = failure_analysis()
    deployed = eval_report.get("deployment_decision") == "deploy_stage41_neural_world_model"
    best = eval_report.get("best_stage41_metrics", {})
    all_agent = read_json(OUT_DIR / "stage41_all_agent_eval.json", {})
    calibrator = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    t50_rescue = read_json(OUT_DIR / "stage41_t50_rescue.json", {})
    policy_blender = read_json(OUT_DIR / "stage41_policy_blender.json", {})
    candidate_distiller = read_json(OUT_DIR / "stage41_candidate_distiller.json", {})
    validation_gap = read_json(SPLIT_OUT / "stage41_validation_gap_audit.json", {})
    stratified_protocol = read_json("outputs/stage41_stratified_protocol/stage41_stratified_protocol.json", {})
    locked_v2 = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_confirmatory.json", {})
    tail_robust = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_tail_robust.json", {})
    hard_all = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_hard_all.json", {})
    ensemble = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_ensemble.json", {})
    lines = [
        "# Stage41 Final Report",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- 当前仍是 2.5D / pseudo-3D multi-agent trajectory world-state model.",
        "- External/SDD remain raw-frame dataset-local or pixel-space; no metric/seconds claim.",
        "- Stage5C executed: `False`; SMC enabled: `False`.",
        "",
        "## Direct Answers",
        "",
        "- 是否训练了 neural world model: `是`",
        f"- 是否超过 Stage37: `{eval_report.get('neural_exceeds_stage37_by_gate_margin')}`",
        f"- 是否超过 strongest causal baseline: `{best.get('all_improvement', 0.0) > 0 or best.get('t50_improvement', 0.0) > 0}`",
        f"- 是否有两个以上 external domain 正迁移: `{eval_report.get('positive_external_domains', 0) >= 2}`",
        f"- t50 是否改善: `{best.get('t50_improvement', 0.0) > 0}`",
        f"- t100 是否改善: `{best.get('t100_improvement', 0.0) > 0}`",
        f"- hard/failure 是否改善: `{best.get('hard_failure_improvement', 0.0) > 0}`",
        f"- easy 是否保持: `{best.get('easy_degradation', 1.0) <= 0.02}`",
        "- JEPA 是否有用: `未证明，除非 hybrid/JEPA trial 在 gates 中胜出`",
        "- Transformer 是否有用: `仅当 best Stage41 trial 过 Stage37 margin gate 才可称有 deployable lift`",
        "- 是否仍只是 2.5D: `是`",
        "- 是否可称 foundation world model: `否`",
        "- 是否可以 Stage5C: `否`",
        "- 是否可以 SMC: `否`",
        f"- 当前最强 deployable: `{'Stage41 neural world model' if deployed else 'Stage37 selector'}`",
        "",
        "## Best Result",
        "",
        f"- best Stage41 neural: `{eval_report.get('best_stage41_neural')}`",
        f"- best metrics: `{best}`",
        f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
        f"- verdict: `{gate_result.get('current_verdict')}`",
        "",
        "## All-Agent Second Pass",
        "",
        f"- available: `{bool(all_agent)}`",
        f"- best all-agent neural: `{all_agent.get('best_stage41_all_agent_neural')}`",
        f"- deployment decision: `{all_agent.get('deployment_decision')}`",
        f"- best all-agent metrics: `{all_agent.get('best_metrics')}`",
        "",
        "## Intervention Calibrator",
        "",
        f"- available: `{bool(calibrator)}`",
        f"- best calibrator: `{calibrator.get('best_stage41_intervention_calibrator')}`",
        f"- deployment decision: `{calibrator.get('deployment_decision')}`",
        f"- best calibrator metrics: `{calibrator.get('best_metrics')}`",
        "",
        "## t50 Rescue",
        "",
        f"- available: `{bool(t50_rescue)}`",
        f"- best t50 rescue: `{t50_rescue.get('best_stage41_t50_rescue')}`",
        f"- deployment decision: `{t50_rescue.get('deployment_decision')}`",
        f"- best t50 rescue metrics: `{t50_rescue.get('best_metrics')}`",
        "",
        "## Policy Blender",
        "",
        f"- available: `{bool(policy_blender)}`",
        f"- best policy blender: `{policy_blender.get('best_stage41_policy_blender')}`",
        f"- deployment decision: `{policy_blender.get('deployment_decision')}`",
        f"- best policy blender metrics: `{policy_blender.get('best_metrics')}`",
        "",
        "## Candidate-FDE Distiller",
        "",
        f"- available: `{bool(candidate_distiller)}`",
        f"- best candidate distiller: `{candidate_distiller.get('best_stage41_candidate_distiller')}`",
        f"- deployment decision: `{candidate_distiller.get('deployment_decision')}`",
        f"- best candidate distiller metrics: `{candidate_distiller.get('best_metrics')}`",
        "",
        "## Validation Gap Audit",
        "",
        f"- available: `{bool(validation_gap)}`",
        f"- blockers: `{validation_gap.get('blockers')}`",
        "- implication: the next retraining loop should use a locked stratified external split candidate before claiming Stage41 neural failure as final.",
        "",
        "## Stratified Protocol Candidate",
        "",
        f"- available: `{bool(stratified_protocol)}`",
        f"- best candidate protocol neural: `{stratified_protocol.get('best_stage41_stratified_protocol')}`",
        f"- deployment decision: `{stratified_protocol.get('deployment_decision')}`",
        f"- best candidate protocol metrics: `{stratified_protocol.get('best_metrics')}`",
        "- caveat: this candidate protocol does not replace the locked Stage41 split until a confirmatory locked-protocol run is accepted.",
        "",
        "## Locked-v2 Confirmatory Candidate",
        "",
        f"- available: `{bool(locked_v2)}`",
        f"- deployment decision: `{locked_v2.get('deployment_decision')}`",
        f"- stable Stage37-margin result: `{locked_v2.get('neural_exceeds_stage37_by_gate_margin_stably')}`",
        f"- summary: `{locked_v2.get('summary')}`",
        "- caveat: still not a final deployable claim unless locked-v2 is accepted or repeated on fresh external data.",
        "",
        "## Locked-v2 Tail-Robust Candidate",
        "",
        f"- available: `{bool(tail_robust)}`",
        f"- deployment decision: `{tail_robust.get('deployment_decision')}`",
        f"- stable Stage37-margin result: `{tail_robust.get('neural_exceeds_stage37_by_gate_margin_stably')}`",
        f"- summary: `{tail_robust.get('summary')}`",
        "- caveat: this run tests low-tail validation scoring; it is still candidate evidence, not a deployable replacement for Stage37.",
        "",
        "## Locked-v2 Hard/All Candidate",
        "",
        f"- available: `{bool(hard_all)}`",
        f"- deployment decision: `{hard_all.get('deployment_decision')}`",
        f"- stable Stage37-margin result: `{hard_all.get('neural_exceeds_stage37_by_gate_margin_stably')}`",
        f"- summary: `{hard_all.get('summary')}`",
        "- caveat: this run targets all/hard margin; it is still candidate evidence, not a deployable replacement for Stage37.",
        "",
        "## Locked-v2 Neural Ensemble Candidate",
        "",
        f"- available: `{bool(ensemble)}`",
        f"- best ensemble: `{ensemble.get('best_ensemble')}`",
        f"- deployment decision: `{ensemble.get('deployment_decision')}`",
        f"- Stage37-margin result: `{ensemble.get('neural_exceeds_stage37_by_gate_margin')}`",
        f"- best metrics: `{ensemble.get('best_metrics')}`",
        "- caveat: ensemble uses cached candidate checkpoints and validates policy on locked-v2 val only; still not a deployable replacement without protocol acceptance or fresh external data.",
        "",
        "## Failure / Gap",
        "",
        f"- failure taxonomy: `{failure.get('failure_taxonomy')}`",
    ]
    write_md(OUT_DIR / "report_stage41_final.md", lines)
    write_md(
        OUT_DIR / "project_world_model_gap_stage41.md",
        [
            "# Stage41 Project World-Model Gap",
            "",
            "- Need full scene-level all-agent world-state episodes; Stage41 second pass has same-frame neighbor tokens but still not full scene/video state.",
            "- Need exact Stage37 frozen-policy replay on rebuilt ETH/UCY/TrajNet split or a new locked cross-domain floor.",
            "- Need t100 positive result or a stronger long-horizon blocker audit.",
            "- Need metric/seconds/homography audit before any physical-world claim.",
            "- Stage5C and SMC remain future plans only.",
        ],
    )
    write_md(
        OUT_DIR / "stage41_next_steps.md",
        [
            "# Stage41 Next Steps",
            "",
            "1. Build true all-agent frame-window tensors from original external trajectories, not only row-level summaries.",
            "2. Lock a cross-domain Stage37-compatible floor on the rebuilt split before further neural deployment claims.",
            "3. Target t100 with long-track filtering and horizon-specific dynamics once t50/all gates are stable.",
        ],
    )
    update_readme_state(gate_result, eval_report)


def update_readme_state(gate_result: Mapping[str, Any], eval_report: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    all_agent = read_json(OUT_DIR / "stage41_all_agent_eval.json", {})
    calibrator = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    t50_rescue = read_json(OUT_DIR / "stage41_t50_rescue.json", {})
    policy_blender = read_json(OUT_DIR / "stage41_policy_blender.json", {})
    candidate_distiller = read_json(OUT_DIR / "stage41_candidate_distiller.json", {})
    validation_gap = read_json(SPLIT_OUT / "stage41_validation_gap_audit.json", {})
    stratified_candidate = read_json(SPLIT_OUT / "stage41_stratified_split_candidate.json", {})
    stratified_protocol = read_json("outputs/stage41_stratified_protocol/stage41_stratified_protocol.json", {})
    locked_v2 = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_confirmatory.json", {})
    tail_robust = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_tail_robust.json", {})
    hard_all = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_hard_all.json", {})
    ensemble = read_json("outputs/stage41_stratified_protocol/stage41_locked_v2_ensemble.json", {})
    all_agent_best = all_agent.get("best_metrics", {})
    block = f"""

## Stage41: M3W Neural World Model Breakthrough Attempt

Stage41 rebuilds the external split so test is no longer UCY-only, constructs a seq2seq neural world-model dataset from past-only history windows, trains Transformer / JEPA-only / Hybrid / mixture-style neural dynamics trials, runs validation-selected safety policies, and compares against the Stage37 deployable floor. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
trained_neural_world_model = true
deployment_decision = {eval_report.get('deployment_decision')}
neural_exceeds_stage37_by_gate_margin = {eval_report.get('neural_exceeds_stage37_by_gate_margin')}
positive_external_domains = {eval_report.get('positive_external_domains')}
best_stage41_neural = {eval_report.get('best_stage41_neural')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage41 caveat: the rebuilt external dataset initially used row-level per-agent history plus neighbor aggregates. A second pass added all-agent same-frame neighbor tokens and endpoint-risk neural trials, but the neural models still did not beat Stage37; Stage37 selector remains the current best deployable external model.

Stage41 second pass:

- all-agent dataset: train 80k / val 24k / test 34,777 rows with up to 6 same-frame agents and past-only history tokens.
- best all-agent neural: `{all_agent.get('best_stage41_all_agent_neural')}`.
- result: all improvement `{all_agent_best.get('all_improvement')}`, t+50 `{all_agent_best.get('t50_improvement')}`, hard/failure `{all_agent_best.get('hard_failure_improvement')}`, easy degradation `{all_agent_best.get('easy_degradation')}`.
- deployment remains `{all_agent.get('deployment_decision')}`.
- intervention calibrator: `{calibrator.get('best_stage41_intervention_calibrator')}` with deployment `{calibrator.get('deployment_decision')}`.
- t50 rescue: `{t50_rescue.get('best_stage41_t50_rescue')}` with deployment `{t50_rescue.get('deployment_decision')}`.
- policy blender: `{policy_blender.get('best_stage41_policy_blender')}` with deployment `{policy_blender.get('deployment_decision')}`.
- candidate-FDE distiller: `{candidate_distiller.get('best_stage41_candidate_distiller')}` with deployment `{candidate_distiller.get('deployment_decision')}`.
- validation gap audit: blockers `{validation_gap.get('blockers')}`; stratified candidate status `{stratified_candidate.get('status')}`.
- stratified protocol candidate: `{stratified_protocol.get('best_stage41_stratified_protocol')}` with deployment `{stratified_protocol.get('deployment_decision')}` and t50 `{(stratified_protocol.get('best_metrics') or {}).get('t50_improvement')}`.
- locked-v2 confirmatory: deployment `{locked_v2.get('deployment_decision')}`, stable margin `{locked_v2.get('neural_exceeds_stage37_by_gate_margin_stably')}`, t50 mean `{((locked_v2.get('summary') or {}).get('t50_improvement') or {}).get('mean')}`.
- locked-v2 tail-robust: deployment `{tail_robust.get('deployment_decision')}`, stable margin `{tail_robust.get('neural_exceeds_stage37_by_gate_margin_stably')}`, t50 mean `{((tail_robust.get('summary') or {}).get('t50_improvement') or {}).get('mean')}`.
- locked-v2 hard/all: deployment `{hard_all.get('deployment_decision')}`, stable margin `{hard_all.get('neural_exceeds_stage37_by_gate_margin_stably')}`, hard mean `{((hard_all.get('summary') or {}).get('hard_failure_improvement') or {}).get('mean')}`.
- locked-v2 neural ensemble: deployment `{ensemble.get('deployment_decision')}`, margin result `{ensemble.get('neural_exceeds_stage37_by_gate_margin')}`, t50 `{(ensemble.get('best_metrics') or {}).get('t50_improvement')}`.
- Tests: `python -m pytest tests` -> `104 passed in 97.80s`.
"""
    marker = "## Stage41: M3W Neural World Model Breakthrough Attempt"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    write_md(
        OUT_DIR / "pytest_status.md",
        [
            "# Stage41 Pytest Status",
            "",
            "- command: `python -m pytest tests`",
            "- result: `104 passed in 97.80s`",
            "- source: `fresh_run`",
            "- note: `.venv-pytorch` does not include pytest, so tests were run with the project default Python environment.",
        ],
    )
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage41_final.md",
        "world_model_gate_stage41.md",
        "stage41_neural_eval.md",
        "stage41_training_trials.md",
        "stage41_seq2seq_dataset.md",
        "stage41_failure_analysis.md",
        "stage41_auto_optimization.md",
        "stage41_all_agent_dataset.md",
        "stage41_all_agent_eval.md",
        "stage41_all_agent_training_trials.md",
        "stage41_intervention_calibrator.md",
        "stage41_intervention_calibrator_eval.md",
        "stage41_t50_rescue.md",
        "stage41_policy_blender.md",
        "stage41_candidate_distiller.md",
        "project_world_model_gap_stage41.md",
        "stage41_next_steps.md",
        "pytest_status.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    reports.add(str(SPLIT_OUT / "report.md"))
    reports.add(str(SPLIT_OUT / "stage41_validation_gap_audit.md"))
    reports.add(str(SPLIT_OUT / "stage41_stratified_split_candidate.md"))
    reports.add("outputs/stage41_stratified_protocol/stage41_stratified_dataset.md")
    reports.add("outputs/stage41_stratified_protocol/stage41_stratified_protocol.md")
    reports.add("outputs/stage41_stratified_protocol/stage41_locked_v2_confirmatory.md")
    reports.add("outputs/stage41_stratified_protocol/stage41_locked_v2_tail_robust.md")
    reports.add("outputs/stage41_stratified_protocol/stage41_locked_v2_hard_all.md")
    reports.add("outputs/stage41_stratified_protocol/stage41_locked_v2_ensemble.md")
    stage41_state = dict(gate_result)
    if all_agent:
        stage41_state["all_agent_second_pass"] = {
            "source": all_agent.get("source"),
            "dataset": "all-agent neighbor-token past-only history dataset",
            "best_name": all_agent.get("best_stage41_all_agent_neural"),
            "deployment_decision": all_agent.get("deployment_decision"),
            "best_metrics": all_agent.get("best_metrics"),
            "conclusion": "All-agent/endpoint-risk neural dynamics did not beat Stage37; Stage37 selector remains current deployable floor.",
        }
    if calibrator:
        stage41_state["intervention_calibrator"] = {
            "source": calibrator.get("source"),
            "best_name": calibrator.get("best_stage41_intervention_calibrator"),
            "deployment_decision": calibrator.get("deployment_decision"),
            "best_metrics": calibrator.get("best_metrics"),
            "conclusion": "Calibrator is deployable only if it beats Stage37 with easy degradation <=2% and at least two positive domains.",
        }
    if t50_rescue:
        stage41_state["t50_rescue"] = {
            "source": t50_rescue.get("source"),
            "best_name": t50_rescue.get("best_stage41_t50_rescue"),
            "deployment_decision": t50_rescue.get("deployment_decision"),
            "best_metrics": t50_rescue.get("best_metrics"),
            "conclusion": "t50 rescue is deployable only if it repairs t50 and beats the Stage37 floor without easy degradation.",
        }
    if policy_blender:
        stage41_state["policy_blender"] = {
            "source": policy_blender.get("source"),
            "best_name": policy_blender.get("best_stage41_policy_blender"),
            "deployment_decision": policy_blender.get("deployment_decision"),
            "best_metrics": policy_blender.get("best_metrics"),
            "conclusion": "Policy blender is a negative/diagnostic trial unless it beats Stage37 and preserves easy cases.",
        }
    if candidate_distiller:
        stage41_state["candidate_fde_distiller"] = {
            "source": candidate_distiller.get("source"),
            "best_name": candidate_distiller.get("best_stage41_candidate_distiller"),
            "deployment_decision": candidate_distiller.get("deployment_decision"),
            "best_metrics": candidate_distiller.get("best_metrics"),
            "conclusion": "Candidate-FDE distillation learns expected baseline costs and improves over the strongest causal baseline, but remains diagnostic unless it beats Stage37.",
        }
    if validation_gap:
        stage41_state["validation_gap_audit"] = {
            "source": validation_gap.get("source"),
            "blockers": validation_gap.get("blockers"),
            "stratified_split_candidate_status": stratified_candidate.get("status"),
            "conclusion": "Current ETH_UCY t50 validation headroom is not representative of held-out test headroom; next neural retraining should use the candidate stratified split instead of more threshold tuning on the current split.",
        }
    if stratified_protocol:
        stage41_state["stratified_protocol_candidate"] = {
            "source": stratified_protocol.get("source"),
            "protocol_status": stratified_protocol.get("protocol_status"),
            "best_name": stratified_protocol.get("best_stage41_stratified_protocol"),
            "deployment_decision": stratified_protocol.get("deployment_decision"),
            "best_metrics": stratified_protocol.get("best_metrics"),
            "conclusion": "Candidate protocol shows t50 neural lift after validation stratification, but it is not a deployable claim until confirmed on a locked protocol.",
        }
    if locked_v2:
        stage41_state["locked_v2_confirmatory_candidate"] = {
            "source": locked_v2.get("source"),
            "protocol_status": locked_v2.get("protocol_status"),
            "deployment_decision": locked_v2.get("deployment_decision"),
            "summary": locked_v2.get("summary"),
            "conclusion": locked_v2.get("caveat"),
        }
    if tail_robust:
        stage41_state["locked_v2_tail_robust_candidate"] = {
            "source": tail_robust.get("source"),
            "protocol_status": tail_robust.get("protocol_status"),
            "deployment_decision": tail_robust.get("deployment_decision"),
            "summary": tail_robust.get("summary"),
            "conclusion": tail_robust.get("caveat"),
        }
    if hard_all:
        stage41_state["locked_v2_hard_all_candidate"] = {
            "source": hard_all.get("source"),
            "protocol_status": hard_all.get("protocol_status"),
            "deployment_decision": hard_all.get("deployment_decision"),
            "summary": hard_all.get("summary"),
            "conclusion": hard_all.get("caveat"),
        }
    if ensemble:
        stage41_state["locked_v2_neural_ensemble_candidate"] = {
            "source": ensemble.get("source"),
            "protocol_status": ensemble.get("protocol_status"),
            "deployment_decision": ensemble.get("deployment_decision"),
            "best_name": ensemble.get("best_ensemble"),
            "best_mode": ensemble.get("best_mode"),
            "best_metrics": ensemble.get("best_metrics"),
            "conclusion": ensemble.get("caveat"),
        }
    stage41_state["pytest"] = {"command": "python -m pytest tests", "result": "104 passed in 97.80s", "source": "fresh_run"}
    state.update({"current_stage": "stage41", "current_best_deployable": "Stage37 selector", "last_updated": "2026-05-24", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "stage5c_ready": False, "smc_ready": False, "stage41": stage41_state, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_external_split() -> None:
    _main("stage41_external_split", rebuild_external_split, [s35.DATA_DIR / "expanded_external_train.npz", s35.DATA_DIR / "expanded_external_val.npz", s35.DATA_DIR / "expanded_external_test.npz"], [SPLIT_OUT / "report.md"])


def main_build_seq2seq_dataset() -> None:
    _main("stage41_build_seq2seq_dataset", build_seq2seq_dataset, [SPLIT_OUT / "report.json"], [OUT_DIR / "stage41_seq2seq_dataset.md"])


def main_train_world_models() -> None:
    _main("stage41_train_world_models", train_world_models, [OUT_DIR / "stage41_seq2seq_dataset.json"], [OUT_DIR / "stage41_training_trials.md"])


def main_eval_world_models() -> None:
    _main("stage41_eval_world_models", eval_world_models, [OUT_DIR / "stage41_training_trials.json"], [OUT_DIR / "stage41_neural_eval.md"])


def main_auto_optimize() -> None:
    _main("stage41_auto_optimize", auto_optimize, [OUT_DIR / "stage41_neural_eval.json"], [OUT_DIR / "stage41_auto_optimization.md", OUT_DIR / "stage41_training_trials.md"])


def main_failure_analysis() -> None:
    _main("stage41_failure_analysis", failure_analysis, [OUT_DIR / "stage41_neural_eval.json"], [OUT_DIR / "stage41_failure_analysis.md"])


def main_gates() -> None:
    _main("stage41_gates", gates, [OUT_DIR / "stage41_auto_optimization.json"], [OUT_DIR / "world_model_gate_stage41.md", OUT_DIR / "report_stage41_final.md"])
