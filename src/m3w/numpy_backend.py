from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import yaml

from src.m3w.token_schema import TOKEN_NAMES, build_token_schema
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


BACKEND_NAME = "numpy_safe_fallback_due_torch_openmp_shm_blocker"
BASELINE_NAMES = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity",
    "constant_acceleration_causal",
    "constant_turn_rate_velocity",
    "scene_clamped_baseline",
    "goal_directed_baseline",
]
STAGE26_T50 = 0.14583655843823773
STAGE26_HARD = 0.11232167634621226
STAGE26_EASY_DEG = 0.01808836280803794


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_split(store: str | Path, split: str, limit: int | None = None) -> Dict[str, np.ndarray]:
    data = np.load(Path(store) / f"{split}.npz")
    n = len(data["x"]) if limit is None else min(int(limit), len(data["x"]))
    return {
        "x": data["x"][:n].astype(np.float32),
        "y_fde": data["y_fde"][:n].astype(np.float32),
        "horizon": data["horizon"][:n].astype(np.int32),
        "split_type": data["split_type"][:n],
        "strongest_idx": data["strongest_idx"][:n].astype(np.int64),
        "oracle_idx": data["oracle_idx"][:n].astype(np.int64),
        "hard_candidate": data["hard_candidate"][:n].astype(np.float32),
    }


def _feature_names(store: str | Path, dim: int) -> list[str]:
    manifest = read_json(Path(store) / "manifest.json", {})
    return list(manifest.get("feature_names", [f"f{i}" for i in range(dim)]))


def _failure_labels(split: Dict[str, np.ndarray], threshold: float | None = None) -> Tuple[np.ndarray, float]:
    strong = split["y_fde"][np.arange(len(split["y_fde"])), split["strongest_idx"]]
    thr = float(np.percentile(strong, 90)) if threshold is None else float(threshold)
    return (strong >= thr).astype(np.float32), thr


def _occupancy(split: Dict[str, np.ndarray], feature_names: list[str]) -> np.ndarray:
    idx = feature_names.index("density_r50") if "density_r50" in feature_names else 0
    return np.clip(split["x"][:, idx] / 10.0, 0.0, 1.0).astype(np.float32)


def _normalize(train_x: np.ndarray, *xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = (train_x.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std, [((x - mean) / std).astype(np.float32) for x in xs]


def _safe_svd_projection(x: np.ndarray, latent_dim: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sample = x[: min(len(x), 12000)]
    cov = (sample.T @ sample) / max(1, len(sample) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    k = min(latent_dim, x.shape[1])
    proj = vecs[:, order[:k]].astype(np.float32)
    if proj.shape[1] < latent_dim:
        pad = rng.normal(0, 0.05, size=(x.shape[1], latent_dim - proj.shape[1])).astype(np.float32)
        proj = np.concatenate([proj, pad], axis=1)
    explained = float(vals[order[:k]].sum() / max(float(vals.sum()), 1e-6))
    return proj.astype(np.float32), {"latent_variance": float(vals[order[:k]].mean()), "explained_variance": explained}


def _token_pool_features(x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    schema = build_token_schema(feature_names)
    pooled = []
    for token in TOKEN_NAMES:
        idx = schema.token_to_features[token]
        pooled.append(x[:, idx].mean(axis=1, keepdims=True))
        pooled.append(x[:, idx].std(axis=1, keepdims=True))
    return np.concatenate(pooled, axis=1).astype(np.float32)


def _variant_features(variant: str, x: np.ndarray, feature_names: list[str], projection: np.ndarray | None) -> np.ndarray:
    token_pool = _token_pool_features(x, feature_names)
    if variant == "jepa_only":
        latent = np.tanh(x @ projection).astype(np.float32) if projection is not None else token_pool
        return np.concatenate([latent, token_pool], axis=1).astype(np.float32)
    if variant == "transformer_only":
        interactions = []
        for token_ix in range(0, token_pool.shape[1], 2):
            interactions.append((token_pool[:, token_ix : token_ix + 1] * token_pool[:, :1]).astype(np.float32))
        return np.concatenate([x, token_pool, *interactions], axis=1).astype(np.float32)
    latent = np.tanh(x @ projection).astype(np.float32) if projection is not None else token_pool
    transformer = _variant_features("transformer_only", x, feature_names, projection)
    return np.concatenate([latent, transformer], axis=1).astype(np.float32)


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 5.0) -> Dict[str, np.ndarray]:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(alpha)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(xtx + reg, x_aug.T @ y).astype(np.float32)
    return {"weights": weights}


def _predict_ridge(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    return x_aug @ model["weights"]


def _fit_variant(
    variant: str,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    feature_names: list[str],
    projection: np.ndarray | None,
    failure_threshold: float,
) -> Dict[str, Any]:
    train_feat = _variant_features(variant, train["x_norm"], feature_names, projection)
    val_feat = _variant_features(variant, val["x_norm"], feature_names, projection)
    train_failure, _ = _failure_labels(train, failure_threshold)
    val_failure, _ = _failure_labels(val, failure_threshold)
    y_log_fde = np.log1p(train["y_fde"]).astype(np.float32)
    fde_model = _fit_ridge(train_feat, y_log_fde, alpha=8.0)
    failure_model = _fit_ridge(train_feat, train_failure[:, None], alpha=3.0)
    interaction_model = _fit_ridge(train_feat, train["hard_candidate"][:, None], alpha=3.0)
    occupancy_model = _fit_ridge(train_feat, train["occupancy"][:, None], alpha=3.0)
    val_log_fde = _predict_ridge(fde_model, val_feat)
    val_failure = _sigmoid(_predict_ridge(failure_model, val_feat).reshape(-1))
    val_interaction = _sigmoid(_predict_ridge(interaction_model, val_feat).reshape(-1))
    val_occupancy = np.clip(_predict_ridge(occupancy_model, val_feat).reshape(-1), 0.0, 1.0)
    val_loss = float(np.mean((val_log_fde - np.log1p(val["y_fde"])) ** 2))
    val_loss += 0.4 * float(np.mean((val_failure - val_failure.mean()) ** 2))
    val_loss += 0.2 * float(np.mean((val_interaction - val["hard_candidate"]) ** 2))
    val_loss += 0.1 * float(np.mean((val_occupancy - val["occupancy"]) ** 2))
    return {
        "variant": variant,
        "feature_dim": int(train_feat.shape[1]),
        "val_loss": val_loss,
        "models": {
            "fde": fde_model,
            "failure": failure_model,
            "interaction": interaction_model,
            "occupancy": occupancy_model,
        },
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def train_numpy_m3w(
    config_path: str | Path,
    quick: bool = False,
    medium: bool = False,
    long: bool = False,
    resume: bool = False,
    mps: bool = False,
    cpu: bool = False,
    checkpoint_every: int | None = None,
) -> Dict[str, Any]:
    start = time.time()
    config = load_config(config_path)
    if quick:
        config["max_train_samples"] = min(int(config.get("max_train_samples", 40000)), 20000)
        config["max_val_samples"] = min(int(config.get("max_val_samples", 20000)), 10000)
    if medium:
        config["mode"] = "medium"
    out_dir = ensure_dir(config.get("output_dir", "outputs/m3w"))
    ckpt_dir = ensure_dir(Path(out_dir) / "checkpoints")
    _heartbeat(out_dir, "loading Stage26 causal feature store", start, {})
    store = config["feature_store"]
    train = _load_split(store, "train", config.get("max_train_samples"))
    val = _load_split(store, "val", config.get("max_val_samples"))
    test = _load_split(store, "test", config.get("max_test_samples"))
    feature_names = _feature_names(store, train["x"].shape[1])
    train["occupancy"] = _occupancy(train, feature_names)
    val["occupancy"] = _occupancy(val, feature_names)
    test["occupancy"] = _occupancy(test, feature_names)
    mean, std, normed = _normalize(train["x"], train["x"], val["x"], test["x"])
    train["x_norm"], val["x_norm"], test["x_norm"] = normed
    train_failure, failure_threshold = _failure_labels(train)
    _heartbeat(out_dir, "fitting lightweight JEPA/Transformer/hybrid deterministic heads", start, {})
    projection, jepa_diag = _safe_svd_projection(train["x_norm"], int(config.get("latent_dim", 32)), int(config.get("seed", 27)))
    results = {}
    best: Dict[str, Any] | None = None
    for variant in ["jepa_only", "transformer_only", "hybrid"]:
        result = _fit_variant(variant, train, val, feature_names, projection, failure_threshold)
        result["jepa_stats"] = [jepa_diag] if variant in {"jepa_only", "hybrid"} else []
        results[variant] = {k: v for k, v in result.items() if k != "models"}
        if best is None or result["val_loss"] < best["val_loss"]:
            best = result
        _heartbeat(out_dir, f"completed {variant}", start, results)
    assert best is not None
    checkpoint = {
        "backend": BACKEND_NAME,
        "config": config,
        "variant": best["variant"],
        "val_loss": best["val_loss"],
        "feature_mean": mean,
        "feature_std": std,
        "feature_names": feature_names,
        "baseline_names": BASELINE_NAMES,
        "projection": projection,
        "failure_threshold": failure_threshold,
        "models": best["models"],
        "jepa_stats": best["jepa_stats"],
        "note": "Torch JEPA/Transformer code exists, but this checkpoint was produced by a CPU-safe NumPy surrogate because local PyTorch/OpenMP SHM execution blocked.",
    }
    ckpt_path = Path(ckpt_dir) / "best_small.pt"
    with ckpt_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    report = {
        "project_name": "M3W: Real-World Multimodal Agent-Scene World Model",
        "mode": config.get("mode", "small"),
        "backend": BACKEND_NAME,
        "torch_execution_status": "blocked_by_uninterruptible_openmp_shm_process",
        "variants": results,
        "best": {"variant": best["variant"], "val_loss": best["val_loss"], "checkpoint": str(ckpt_path)},
        "true_3d": False,
        "foundation_world_model": False,
        "latent_generative": False,
        "smc": False,
        "ordinary_residual_trained": False,
        "elapsed_s": time.time() - start,
    }
    write_json(Path(out_dir) / "training_report.json", _jsonable(report))
    write_md(
        Path(out_dir) / "training_report.md",
        [
            "# M3W Training Report",
            "",
            "- This run used the CPU-safe NumPy backend because local PyTorch/OpenMP SHM execution blocked before heartbeat.",
            "- The PyTorch JEPA/Transformer implementation is present, but this checkpoint is not a full torch JEPA-Transformer success.",
            "- No Stage5C latent generative execution, no SMC, no ordinary residual training.",
            f"- backend: `{BACKEND_NAME}`",
            f"- best variant: `{best['variant']}`",
            f"- best checkpoint: `{ckpt_path}`",
        ],
    )
    return report


def evaluate_numpy_m3w(checkpoint: str | Path) -> Dict[str, Any]:
    with Path(checkpoint).open("rb") as handle:
        ckpt = pickle.load(handle)
    config = ckpt["config"]
    out_dir = ensure_dir(config.get("output_dir", "outputs/m3w"))
    store = config["feature_store"]
    val = _load_split(store, "val", config.get("max_val_samples"))
    test = _load_split(store, "test", config.get("max_test_samples"))
    feature_names = list(ckpt["feature_names"])
    for split in [val, test]:
        split["x_norm"] = ((split["x"] - ckpt["feature_mean"]) / ckpt["feature_std"]).astype(np.float32)
        split["occupancy"] = _occupancy(split, feature_names)
    val_pred = _predict_checkpoint(ckpt, val)
    test_pred = _predict_checkpoint(ckpt, test)
    selected_policy = _search_policy(val, val_pred["fde"], val_pred["failure"], config)
    test_selected, test_conf = _select(test, test_pred["fde"], test_pred["failure"], selected_policy["policy"])
    test_metrics = _metrics(test, test_selected, test_conf)
    failure_labels, _ = _failure_labels(test, ckpt["failure_threshold"])
    failure_auroc = float(_roc_auc_score(failure_labels, test_pred["failure"])) if len(set(failure_labels.tolist())) > 1 else 0.5
    failure_auprc = float(_average_precision_score(failure_labels, test_pred["failure"])) if len(set(failure_labels.tolist())) > 1 else float(np.mean(failure_labels))
    interaction_labels = test["hard_candidate"]
    interaction_auroc = float(_roc_auc_score(interaction_labels, test_pred["interaction"])) if len(set(interaction_labels.tolist())) > 1 else 0.5
    occupancy_mse = float(np.mean((test_pred["occupancy"] - test["occupancy"]) ** 2))
    stage26 = read_json("outputs/reports/report_stage26_final.json", {})
    stage26_t50 = float(stage26.get("t50_improvement", STAGE26_T50) or STAGE26_T50)
    result = {
        "checkpoint": str(checkpoint),
        "backend": ckpt.get("backend", BACKEND_NAME),
        "variant": ckpt["variant"],
        "selected_policy": selected_policy["policy"],
        "validation_metrics": selected_policy["metrics"],
        "test_metrics": test_metrics,
        "stage26_selector": {
            "t50_improvement": stage26.get("t50_improvement", STAGE26_T50),
            "hard_failure_improvement": stage26.get("hard_failure_improvement", STAGE26_HARD),
            "easy_degradation": stage26.get("easy_degradation", STAGE26_EASY_DEG),
        },
        "beats_stage26_selector": test_metrics["official_t50_improvement"] > stage26_t50,
        "failure_AUROC": failure_auroc,
        "failure_AUPRC": failure_auprc,
        "failure_ECE": _ece(failure_labels, test_pred["failure"]),
        "interaction_AUROC": interaction_auroc,
        "occupancy_MSE": occupancy_mse,
        "jepa_latent_variance": float(ckpt.get("jepa_stats", [{}])[-1].get("latent_variance", 0.0)) if ckpt.get("jepa_stats") else 0.0,
        "jepa_non_collapse": bool((ckpt.get("jepa_stats") or [{}])[-1].get("latent_variance", 0.0) > 0.01),
        "jepa_downstream_lift": False,
        "transformer_dynamics_lift": test_metrics["official_t50_improvement"] > 0.0,
        "goal_metrics": "diagnostic_unavailable_no_human_goal_labels",
        "physical_validity": "selected physical baseline only; no residual/correction",
        "t100_status": "raw-frame diagnostic, not seconds-level",
        "metric_status": "pixel-space only",
    }
    write_json(Path(out_dir) / "metrics_m3w.json", _jsonable(result))
    write_md(
        Path(out_dir) / "metrics_m3w.md",
        [
            "# M3W Metrics",
            "",
            f"- backend: `{result['backend']}`",
            f"- variant: `{result['variant']}`",
            f"- t+50 improvement: `{test_metrics['official_t50_improvement']}`",
            f"- hard/failure improvement: `{test_metrics['hard_failure_improvement']}`",
            f"- easy degradation: `{test_metrics['easy_degradation']}`",
            f"- beats Stage26 selector: `{result['beats_stage26_selector']}`",
            f"- failure AUROC/AUPRC/ECE: `{failure_auroc}` / `{failure_auprc}` / `{result['failure_ECE']}`",
            f"- interaction AUROC: `{interaction_auroc}`",
            f"- JEPA non-collapse: `{result['jepa_non_collapse']}`",
            "",
            "This is not a latent generative rollout, not true 3D, and not a foundation world model.",
        ],
    )
    return result


def _predict_checkpoint(ckpt: Dict[str, Any], split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    feat = _variant_features(ckpt["variant"], split["x_norm"], list(ckpt["feature_names"]), ckpt.get("projection"))
    log_fde = _predict_ridge(ckpt["models"]["fde"], feat)
    return {
        "fde": np.maximum(0.0, np.expm1(np.clip(log_fde, -20.0, 8.5))),
        "failure": _sigmoid(_predict_ridge(ckpt["models"]["failure"], feat).reshape(-1)),
        "interaction": _sigmoid(_predict_ridge(ckpt["models"]["interaction"], feat).reshape(-1)),
        "occupancy": np.clip(_predict_ridge(ckpt["models"]["occupancy"], feat).reshape(-1), 0.0, 1.0),
    }


def _select(split: Dict[str, np.ndarray], pred_fde: np.ndarray, failure_prob: np.ndarray, policy: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    selected = []
    confs = []
    for i in range(len(split["y_fde"])):
        strong = int(split["strongest_idx"][i])
        order = np.argsort(pred_fde[i])
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        gain = float(pred_fde[i, strong] - pred_fde[i, best])
        conf = float((pred_fde[i, second] - pred_fde[i, best]) / max(pred_fde[i, strong], 1e-6))
        fallback = best == strong or gain < policy["gain_threshold"] or conf < policy["confidence_threshold"] or failure_prob[i] < policy["failure_threshold"]
        selected.append(strong if fallback else best)
        confs.append(conf)
    return np.asarray(selected, dtype=np.int64), np.asarray(confs, dtype=np.float32)


def _metrics(split: Dict[str, np.ndarray], selected: np.ndarray, confs: np.ndarray) -> Dict[str, Any]:
    y = split["y_fde"]
    idx = np.arange(len(y))
    selected_err = y[idx, selected]
    strongest_err = y[idx, split["strongest_idx"]]
    oracle_err = y[idx, np.argmin(y, axis=1)]
    easy = strongest_err <= 10.0
    failure_thr = np.percentile(strongest_err, 90)
    hard = np.logical_or(split["hard_candidate"] > 0.5, strongest_err >= failure_thr)
    h50 = split["horizon"] == 50
    h100 = split["horizon"] == 100

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - selected_err[mask].mean() / max(float(strongest_err[mask].mean()), 1e-6))

    easy_deg = float(max(0.0, selected_err[easy].mean() / max(float(strongest_err[easy].mean()), 1e-6) - 1.0)) if np.any(easy) else 0.0
    return {
        "improvement_over_strongest": imp(np.ones(len(y), dtype=bool)),
        "official_t50_improvement": imp(h50),
        "diagnostic_t100_raw_frame_improvement": imp(h100),
        "hard_failure_improvement": imp(hard),
        "easy_degradation": easy_deg,
        "selector_regret": float(np.mean(selected_err - oracle_err)),
        "harm_over_fallback": float(np.mean(selected_err - strongest_err)),
        "switch_rate": float(np.mean(selected != split["strongest_idx"])),
        "mean_confidence": float(np.mean(confs)),
        "selected_distribution": {BASELINE_NAMES[i]: int((selected == i).sum()) for i in range(len(BASELINE_NAMES))},
    }


def _search_policy(split: Dict[str, np.ndarray], pred_fde: np.ndarray, failure_prob: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    search = config.get("fallback_search", {})
    best = None
    best_safe = None
    for conf in search.get("confidence_thresholds", [0.0, 0.05, 0.1, 0.2]):
        for gain in search.get("gain_thresholds", [0.0, 2.0, 5.0, 10.0]):
            for fail in search.get("failure_thresholds", [0.0, 0.1, 0.2]):
                policy = {"confidence_threshold": float(conf), "gain_threshold": float(gain), "failure_threshold": float(fail)}
                selected, confs = _select(split, pred_fde, failure_prob, policy)
                met = _metrics(split, selected, confs)
                safe = met["easy_degradation"] <= 0.02
                objective = met["official_t50_improvement"] + 0.5 * met["hard_failure_improvement"] - 25.0 * max(0.0, met["easy_degradation"] - 0.02)
                if best is None or objective > best["objective"]:
                    best = {"policy": policy, "metrics": met, "objective": objective}
                if safe:
                    safe_objective = met["official_t50_improvement"] + 0.5 * met["hard_failure_improvement"] - 0.1 * met["switch_rate"]
                    if best_safe is None or safe_objective > best_safe["objective"]:
                        best_safe = {"policy": policy, "metrics": met, "objective": safe_objective}
    assert best is not None
    return best_safe if best_safe is not None else best


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (probs >= lo) & (probs < hi)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return total


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    total_pos = int(sorted_labels.sum())
    if total_pos == 0:
        return 0.0
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / total_pos)


def _heartbeat(out_dir: Path, task: str, start: float, results: Dict[str, Any]) -> None:
    write_md(
        Path(out_dir) / "heartbeat.md",
        [
            "# M3W Heartbeat",
            "",
            f"- current task: `{task}`",
            f"- elapsed seconds: `{time.time() - start:.1f}`",
            f"- completed variants: `{list(results.keys())}`",
            f"- backend: `{BACKEND_NAME}`",
            "- latent generative: `False`",
            "- SMC: `False`",
        ],
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def train_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--torch-backend", action="store_true")
    args = parser.parse_args(argv)
    if args.torch_backend:
        from src.m3w.train import train_m3w

        train_m3w(args.config, quick=args.quick, medium=args.medium, long=args.long, resume=args.resume, mps=args.mps, cpu=args.cpu, checkpoint_every=args.checkpoint_every)
        return
    train_numpy_m3w(args.config, quick=args.quick, medium=args.medium, long=args.long, resume=args.resume, mps=args.mps, cpu=args.cpu, checkpoint_every=args.checkpoint_every)


def eval_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--torch-backend", action="store_true")
    args = parser.parse_args(argv)
    if args.torch_backend:
        from src.m3w.eval import evaluate_m3w

        evaluate_m3w(args.checkpoint)
        return
    evaluate_numpy_m3w(args.checkpoint)
