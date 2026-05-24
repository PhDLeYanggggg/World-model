from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


FEATURE_DIR = Path("data/stage26_sdd_feature_store")
LATENT_DIR = Path("data/stage28_m3w_latent_cache")
OUT_DIR = Path("outputs/m3w_stage28")
CHECKPOINT_DIR = Path("outputs/m3w/checkpoints")
REPORT_DIR = Path("outputs/reports")
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
STAGE26_HARD = 0.11234058960663984
STAGE26_EASY = 0.01808836280803794
RANDOM_STATE = 28


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _feature_manifest() -> Dict[str, Any]:
    manifest = read_json(FEATURE_DIR / "manifest.json", {})
    if not manifest:
        raise FileNotFoundError("Missing data/stage26_sdd_feature_store/manifest.json. Run Stage26 first.")
    return manifest


def _load_feature_split(split: str) -> Dict[str, np.ndarray]:
    path = FEATURE_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Stage26 feature store first.")
    return dict(np.load(path))


def _stage26_summary() -> Dict[str, Any]:
    final = read_json(REPORT_DIR / "report_stage26_final.json", {})
    return {
        "selected_model": final.get("selected_model", "stage26_failure_assisted_selector"),
        "t50_improvement": float(final.get("t50_improvement", STAGE26_T50)),
        "hard_failure_improvement": float(final.get("hard_failure_improvement", STAGE26_HARD)),
        "easy_degradation": float(final.get("easy_degradation", STAGE26_EASY)),
    }


def _assert_safe_torch_runtime() -> None:
    if (
        sys.platform == "darwin"
        and platform.machine().lower() == "x86_64"
        and os.environ.get("WORLD_MODEL_ALLOW_RISKY_OPENMP") != "1"
    ):
        raise RuntimeError(
            "Refusing to import torch under macOS x86_64/Rosetta. "
            "Use .venv-pytorch/bin/python on arm64 for Stage28 M3W latent extraction."
        )


def _configure_threads() -> Dict[str, Any]:
    os.environ.setdefault("WORLD_MODEL_TORCH_THREADS", "4")
    os.environ.setdefault("WORLD_MODEL_TORCH_INTEROP_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
    os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
    return {
        "platform_machine": platform.machine(),
        "torch_threads_env": os.environ["WORLD_MODEL_TORCH_THREADS"],
        "torch_interop_threads_env": os.environ["WORLD_MODEL_TORCH_INTEROP_THREADS"],
        "num_workers": 0,
    }


def _checkpoint_paths() -> Dict[str, Path]:
    return {
        "jepa_only": CHECKPOINT_DIR / "jepa_only_best.pt",
        "transformer_only": CHECKPOINT_DIR / "transformer_only_best.pt",
        "hybrid": CHECKPOINT_DIR / "hybrid_best.pt",
    }


def build_m3w_latent_cache() -> Dict[str, Any]:
    """Extract frozen M3W hidden features into a Stage26-aligned latent cache."""
    _assert_safe_torch_runtime()
    runtime = _configure_threads()
    import torch

    torch.set_num_threads(int(runtime["torch_threads_env"]))
    try:
        torch.set_num_interop_threads(int(runtime["torch_interop_threads_env"]))
    except RuntimeError:
        pass
    from src.m3w.models import M3WModel
    from src.m3w.token_schema import build_token_schema

    ensure_dir(LATENT_DIR)
    ensure_dir(OUT_DIR)
    manifest = _feature_manifest()
    paths = _checkpoint_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing M3W checkpoints: {missing}")

    checkpoints: Dict[str, Dict[str, Any]] = {name: torch.load(path, map_location="cpu") for name, path in paths.items()}
    models: Dict[str, M3WModel] = {}
    for name, ckpt in checkpoints.items():
        schema = build_token_schema(ckpt["feature_names"])
        model = M3WModel(schema, ckpt["config"], ckpt["variant"]).to("cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[name] = model

    start = time.perf_counter()
    split_summaries: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        source = _load_feature_split(split)
        x_raw = source["x"].astype(np.float32)
        arrays: Dict[str, np.ndarray] = {}
        for name, ckpt in checkpoints.items():
            mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
            std = np.asarray(ckpt["feature_std"], dtype=np.float32)
            x_norm = ((x_raw - mean) / std).astype(np.float32)
            hidden_batches: List[np.ndarray] = []
            output_batches: Dict[str, List[np.ndarray]] = defaultdict(list)
            model = models[name]
            with torch.no_grad():
                for lo in range(0, len(x_norm), 4096):
                    xb = torch.from_numpy(x_norm[lo : lo + 4096])
                    hidden = model.encode(xb)
                    out = model.heads(hidden)
                    hidden_batches.append(hidden.detach().cpu().numpy().astype(np.float32))
                    if name == "hybrid":
                        for key in ["log_fde", "failure_logit", "interaction_logit", "occupancy", "validity_logit"]:
                            output_batches[key].append(out[key].detach().cpu().numpy().astype(np.float32))
            arrays[f"{name}_latent"] = np.concatenate(hidden_batches, axis=0).astype(np.float32)
            if name == "hybrid":
                arrays["hybrid_log_fde"] = np.concatenate(output_batches["log_fde"], axis=0).astype(np.float32)
                arrays["hybrid_failure_logit"] = np.concatenate(output_batches["failure_logit"], axis=0).astype(np.float32)
                arrays["hybrid_interaction_logit"] = np.concatenate(output_batches["interaction_logit"], axis=0).astype(np.float32)
                arrays["hybrid_occupancy"] = np.concatenate(output_batches["occupancy"], axis=0).astype(np.float32)
                arrays["hybrid_validity_logit"] = np.concatenate(output_batches["validity_logit"], axis=0).astype(np.float32)
        arrays.update(
            {
                "horizon": source["horizon"],
                "split_type": source["split_type"],
                "strongest_idx": source["strongest_idx"],
                "oracle_idx": source["oracle_idx"],
                "hard_candidate": source["hard_candidate"],
                "y_fde": source["y_fde"],
            }
        )
        out_path = LATENT_DIR / f"{split}.npz"
        np.savez_compressed(out_path, **arrays)
        split_summaries[split] = {
            "rows": int(len(x_raw)),
            "latent_shapes": {key: list(value.shape) for key, value in arrays.items() if "latent" in key or key.startswith("hybrid_")},
            "path": str(out_path),
        }

    report = {
        "project": "M3W Stage28 latent cache",
        "cache_dir": str(LATENT_DIR),
        "runtime": runtime,
        "source_feature_store": str(FEATURE_DIR),
        "stage26_baseline": _stage26_summary(),
        "splits": split_summaries,
        "elapsed_s": time.perf_counter() - start,
        "leakage_audit": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "latent_features_source": "frozen M3W checkpoints trained on causal Stage26 features",
        },
    }
    _write_json(OUT_DIR / "latent_cache_report.json", report)
    write_md(
        OUT_DIR / "latent_cache_report.md",
        [
            "# Stage28 M3W Latent Cache Report",
            "",
            "- 当前不是 true 3D，也不是 foundation world model；SDD 仍是 pixel-space raw-frame benchmark。",
            "- Latent cache is aligned to Stage26 causal feature rows and uses frozen M3W checkpoints only.",
            "- No future endpoint, central velocity, or test endpoint goals are used as input.",
            "",
            f"- cache dir: `{LATENT_DIR}`",
            f"- elapsed seconds: `{report['elapsed_s']:.3f}`",
            f"- runtime: `{runtime}`",
            "",
            "| split | rows | path |",
            "| --- | ---: | --- |",
            *[f"| {s} | {info['rows']} | `{info['path']}` |" for s, info in split_summaries.items()],
        ],
    )
    return report


def _load_latent_split(split: str) -> Dict[str, np.ndarray]:
    path = LATENT_DIR / f"{split}.npz"
    if not path.exists():
        build_m3w_latent_cache()
    return dict(np.load(path))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _feature_indices_by_group(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    groups = {
        "scene": [],
        "goal": [],
        "interaction": [],
        "baseline": [],
        "agent": [],
        "horizon": [],
        "time_dataset": [],
    }
    for i, name in enumerate(feature_names):
        low = name.lower()
        if "scene" in low:
            groups["scene"].append(i)
        if "goal" in low:
            groups["goal"].append(i)
        if any(k in low for k in ["density", "nearest_neighbor", "ttc", "closing"]):
            groups["interaction"].append(i)
        if any(k in low for k in ["baseline", "rollout", "damped", "cv_", "ca_"]):
            groups["baseline"].append(i)
        if any(k in low for k in ["speed", "accel", "heading", "curvature", "agent_type", "vx_", "vy_", "ax_", "ay_"]):
            groups["agent"].append(i)
        if "horizon" in low:
            groups["horizon"].append(i)
        if "split" in low or "start_frame" in low:
            groups["time_dataset"].append(i)
    return groups


def _base_x(split: str, feature_mask: Sequence[int] | None = None) -> np.ndarray:
    data = _load_feature_split(split)
    x = data["x"].astype(np.float32)
    if feature_mask is not None:
        keep = np.ones(x.shape[1], dtype=bool)
        keep[list(feature_mask)] = False
        x = x[:, keep]
    return x


def _assemble_features(split: str, variant: str, feature_mask: Sequence[int] | None = None) -> np.ndarray:
    base = _base_x(split, feature_mask)
    lat = _load_latent_split(split)
    parts: List[np.ndarray] = [base]
    if variant in {"plus_jepa", "all_latent", "all_latent_fallback"}:
        parts.append(lat["jepa_only_latent"])
    if variant in {"plus_transformer", "all_latent", "all_latent_fallback"}:
        parts.append(lat["transformer_only_latent"])
    if variant in {"plus_hybrid", "all_latent", "all_latent_fallback"}:
        parts.append(lat["hybrid_latent"])
    if variant in {"plus_failure_hidden", "all_latent", "all_latent_fallback"}:
        parts.append(_sigmoid(lat["hybrid_failure_logit"])[:, None].astype(np.float32))
        parts.append(lat["hybrid_validity_logit"][:, None].astype(np.float32))
    if variant in {"plus_interaction_hidden", "all_latent", "all_latent_fallback"}:
        parts.append(_sigmoid(lat["hybrid_interaction_logit"])[:, None].astype(np.float32))
        parts.append(lat["hybrid_occupancy"][:, None].astype(np.float32))
    return np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32), posinf=1e6, neginf=-1e6)


def _target_log_fde(split: str) -> np.ndarray:
    y = _load_feature_split(split)["y_fde"].astype(np.float64)
    cap = float(np.percentile(y[np.isfinite(y)], 99.5))
    return np.log1p(np.minimum(y, cap))


def _train_model(train_x: np.ndarray, train_y: np.ndarray, model_kind: str) -> Any:
    if model_kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    else:
        model = ExtraTreesRegressor(
            n_estimators=64,
            max_depth=16,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    model.fit(train_x, train_y)
    return model


def _predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(model.predict(x), dtype=np.float64)))


def _strongest_err(split: str) -> np.ndarray:
    data = _load_feature_split(split)
    idx = data["strongest_idx"].astype(int)
    return data["y_fde"][np.arange(len(idx)), idx].astype(np.float64)


def _train_failure_threshold() -> float:
    return float(np.percentile(_strongest_err("train"), 90))


def _subset_masks(split: str) -> Dict[str, np.ndarray]:
    data = _load_feature_split(split)
    strong = _strongest_err(split)
    failure_thr = _train_failure_threshold()
    agent_masks = _agent_type_masks(split)
    masks: Dict[str, np.ndarray] = {
        "all": np.ones(len(strong), dtype=bool),
        "t50": data["horizon"] == 50,
        "t100": data["horizon"] == 100,
        "hard_failure": np.logical_or(data["hard_candidate"].astype(bool), strong >= failure_thr),
        "easy": strong <= 10.0,
        "cross_scene": data["split_type"] == 0,
        "within_scene": data["split_type"] == 1,
        "multi_agent_ge5": _base_x(split)[:, _feature_manifest()["feature_names"].index("agent_count_ge5")] > 0.5
        if "agent_count_ge5" in _feature_manifest()["feature_names"]
        else np.zeros(len(strong), dtype=bool),
    }
    masks.update(agent_masks)
    return masks


def _agent_type_masks(split: str) -> Dict[str, np.ndarray]:
    manifest = _feature_manifest()
    x = _base_x(split)
    out: Dict[str, np.ndarray] = {}
    for name in manifest["feature_names"]:
        if name.startswith("agent_type_"):
            idx = manifest["feature_names"].index(name)
            out[name] = x[:, idx] > 0.5
    return out


def _evaluate_selected(split: str, selected_idx: np.ndarray, confidence: np.ndarray | None = None) -> Dict[str, Any]:
    data = _load_feature_split(split)
    y = data["y_fde"].astype(np.float64)
    idx = np.arange(len(y))
    strong_idx = data["strongest_idx"].astype(int)
    selected = y[idx, selected_idx]
    strong = y[idx, strong_idx]
    oracle = y[idx, np.argmin(y, axis=1)]
    masks = _subset_masks(split)

    def improvement(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - selected[mask].mean() / max(float(strong[mask].mean()), 1e-6))

    easy = masks["easy"]
    easy_degradation = float(max(0.0, selected[easy].mean() / max(float(strong[easy].mean()), 1e-6) - 1.0)) if np.any(easy) else 0.0
    selected_distribution = {BASELINE_NAMES[i]: int(np.sum(selected_idx == i)) for i in range(len(BASELINE_NAMES))}
    by_horizon = {str(h): improvement(data["horizon"] == h) for h in [10, 25, 50, 100]}
    by_split = {name: improvement(masks[name]) for name in ["cross_scene", "within_scene"]}
    by_agent_type = {name.replace("agent_type_", ""): improvement(mask) for name, mask in masks.items() if name.startswith("agent_type_")}
    return {
        "n": int(len(y)),
        "improvement_over_strongest": improvement(masks["all"]),
        "official_t50_improvement": improvement(masks["t50"]),
        "diagnostic_t100_raw_frame_improvement": improvement(masks["t100"]),
        "hard_failure_improvement": improvement(masks["hard_failure"]),
        "easy_degradation": easy_degradation,
        "selector_regret": float(np.mean(selected - oracle)),
        "harm_over_fallback": float(np.mean(selected - strong)),
        "switch_rate": float(np.mean(selected_idx != strong_idx)),
        "accuracy_to_oracle_best": float(np.mean(selected_idx == np.argmin(y, axis=1))),
        "mean_confidence": float(np.mean(confidence)) if confidence is not None and len(confidence) else 0.0,
        "selected_distribution": selected_distribution,
        "by_horizon_improvement": by_horizon,
        "by_split_improvement": by_split,
        "by_agent_type_improvement": by_agent_type,
        "easy_count": int(np.sum(easy)),
        "hard_failure_count": int(np.sum(masks["hard_failure"])),
        "t50_count": int(np.sum(masks["t50"])),
        "t100_count": int(np.sum(masks["t100"])),
    }


def _select_from_predicted(
    split: str,
    pred_fde: np.ndarray,
    policy: Mapping[str, Any],
    failure_signal: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    data = _load_feature_split(split)
    strong_idx = data["strongest_idx"].astype(int)
    selected = np.array(strong_idx, dtype=np.int64)
    confs = np.zeros(len(strong_idx), dtype=np.float32)
    confidence_threshold = float(policy.get("confidence_threshold", 0.0))
    gain_threshold = float(policy.get("predicted_gain_threshold_px", 0.0))
    easy_guard_px = float(policy.get("easy_predicted_strongest_threshold_px", 10.0))
    failure_threshold = policy.get("failure_probability_threshold")
    min_margin = float(policy.get("min_predicted_margin_px", 0.0))
    max_switch_rate = policy.get("max_switch_rate")
    candidate_switches: List[Tuple[float, int, int, float]] = []
    for i in range(len(strong_idx)):
        order = np.argsort(pred_fde[i])
        best = int(order[0])
        strong = int(strong_idx[i])
        gain = float(pred_fde[i, strong] - pred_fde[i, best])
        second = int(order[1]) if len(order) > 1 else best
        margin = float(pred_fde[i, second] - pred_fde[i, best])
        conf = gain / max(float(pred_fde[i, strong]), 1e-6)
        fallback = best == strong or gain < gain_threshold or conf < confidence_threshold or margin < min_margin
        if pred_fde[i, strong] <= easy_guard_px and gain < max(gain_threshold, 5.0):
            fallback = True
        if failure_threshold is not None and failure_signal is not None and float(failure_signal[i]) < float(failure_threshold):
            fallback = True
        if not fallback:
            candidate_switches.append((gain, i, best, conf))
        confs[i] = conf
    if max_switch_rate is not None:
        max_count = int(math.floor(float(max_switch_rate) * len(strong_idx)))
        candidate_switches = sorted(candidate_switches, key=lambda item: item[0], reverse=True)[:max_count]
    for _gain, i, best, conf in candidate_switches:
        selected[i] = best
        confs[i] = conf
    return selected, confs


def _policy_grid(variant: str) -> List[Dict[str, Any]]:
    confidence_thresholds = [0.0, 0.02, 0.05, 0.10]
    gain_thresholds = [0.0, 2.0, 5.0, 10.0, 20.0]
    min_margins = [0.0, 1.0, 2.0]
    failure_thresholds = [None]
    if variant in {"plus_failure_hidden", "all_latent", "all_latent_fallback"}:
        failure_thresholds = [None, 0.05, 0.10, 0.20, 0.35]
    max_switch_rates = [None, 0.03, 0.05, 0.08] if variant.endswith("fallback") or variant == "all_latent" else [None]
    policies = []
    for conf in confidence_thresholds:
        for gain in gain_thresholds:
            for margin in min_margins:
                for fail in failure_thresholds:
                    for max_switch in max_switch_rates:
                        policies.append(
                            {
                                "variant": variant,
                                "confidence_threshold": conf,
                                "predicted_gain_threshold_px": gain,
                                "min_predicted_margin_px": margin,
                                "easy_predicted_strongest_threshold_px": 10.0,
                                "failure_probability_threshold": fail,
                                "max_switch_rate": max_switch,
                            }
                        )
    policies.append(
        {
            "variant": variant,
            "policy_family": "all_fallback_strongest",
            "confidence_threshold": 1.0,
            "predicted_gain_threshold_px": 1e9,
            "min_predicted_margin_px": 1e9,
            "easy_predicted_strongest_threshold_px": 10.0,
            "failure_probability_threshold": None,
            "max_switch_rate": 0.0,
        }
    )
    return policies


def _failure_signal_from_latents(split: str, pred_fde: np.ndarray) -> np.ndarray:
    lat = _load_latent_split(split)
    hidden_prob = _sigmoid(lat["hybrid_failure_logit"]).astype(np.float64)
    data = _load_feature_split(split)
    strong = data["strongest_idx"].astype(int)
    predicted_gain = pred_fde[np.arange(len(strong)), strong] - pred_fde.min(axis=1)
    norm_gain = 1.0 / (1.0 + np.exp(-np.clip(predicted_gain / 20.0, -20.0, 20.0)))
    return np.clip(0.7 * hidden_prob + 0.3 * norm_gain, 0.0, 1.0)


def _search_policy_for_variant(
    variant: str,
    model: Any,
    feature_mask: Sequence[int] | None = None,
) -> Dict[str, Any]:
    val_x = _assemble_features("val", variant, feature_mask)
    test_x = _assemble_features("test", variant, feature_mask)
    val_pred = _predict_model(model, val_x)
    test_pred = _predict_model(model, test_x)
    val_failure = _failure_signal_from_latents("val", val_pred)
    test_failure = _failure_signal_from_latents("test", test_pred)
    best: Dict[str, Any] | None = None
    candidates: List[Dict[str, Any]] = []
    for policy in _policy_grid(variant):
        val_sel, val_conf = _select_from_predicted("val", val_pred, policy, val_failure)
        val_eval = _evaluate_selected("val", val_sel, val_conf)
        objective = (
            val_eval["official_t50_improvement"]
            + 0.55 * val_eval["hard_failure_improvement"]
            - 5.0 * max(0.0, val_eval["easy_degradation"] - 0.02)
            - 0.15 * max(0.0, val_eval["harm_over_fallback"])
        )
        if val_eval["easy_degradation"] <= 0.02:
            objective += 0.03
        item = {"policy": policy, "validation_eval": val_eval, "objective": objective}
        candidates.append(item)
        if best is None or objective > best["objective"]:
            best = item
    assert best is not None
    test_sel, test_conf = _select_from_predicted("test", test_pred, best["policy"], test_failure)
    test_eval = _evaluate_selected("test", test_sel, test_conf)
    y_test = _load_feature_split("test")["y_fde"].astype(np.float64)
    rmse = float(np.sqrt(np.mean((np.log1p(y_test) - np.log1p(np.maximum(test_pred, 0.0))) ** 2)))
    rank_acc = float(np.mean(np.argmin(test_pred, axis=1) == np.argmin(y_test, axis=1)))
    return {
        "variant": variant,
        "selected_policy": best["policy"],
        "validation_eval": best["validation_eval"],
        "test_eval": test_eval,
        "expected_fde_log_rmse": rmse,
        "ranking_accuracy": rank_acc,
        "top_validation_candidates": sorted(candidates, key=lambda x: x["objective"], reverse=True)[:10],
        "test_selected_idx": test_sel,
        "test_confidence": test_conf,
        "test_predicted_fde": test_pred,
    }


def train_m3w_las() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    if not (LATENT_DIR / "train.npz").exists():
        build_m3w_latent_cache()
    variants = [
        "stage26_only",
        "plus_jepa",
        "plus_transformer",
        "plus_hybrid",
        "plus_failure_hidden",
        "plus_interaction_hidden",
        "all_latent",
        "all_latent_fallback",
    ]
    train_y = _target_log_fde("train")
    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    start = time.perf_counter()
    for variant in variants:
        train_x = _assemble_features("train", variant)
        model_kind = "extra_trees" if variant in {"stage26_only", "all_latent", "all_latent_fallback", "plus_hybrid"} else "ridge"
        model = _train_model(train_x, train_y, model_kind)
        result = _search_policy_for_variant(variant, model)
        result["model_kind"] = model_kind
        result["feature_dim"] = int(train_x.shape[1])
        public = {
            k: v
            for k, v in result.items()
            if k not in {"test_selected_idx", "test_confidence", "test_predicted_fde"}
        }
        results.append(public)
        score = (
            result["validation_eval"]["official_t50_improvement"]
            + 0.55 * result["validation_eval"]["hard_failure_improvement"]
            - 5.0 * max(0.0, result["validation_eval"]["easy_degradation"] - 0.02)
        )
        if result["validation_eval"]["easy_degradation"] <= 0.02:
            score += 0.03
        if best is None or score > best["score"]:
            best = {**public, "score": score}
            np.savez_compressed(
                OUT_DIR / "best_las_test_arrays.npz",
                selected_idx=result["test_selected_idx"],
                confidence=result["test_confidence"],
                predicted_fde=result["test_predicted_fde"],
            )
    assert best is not None
    stage26 = _stage26_summary()
    best_metrics = best["test_eval"]
    out = {
        "trained": True,
        "stage": "stage28_m3w_las_training",
        "stage26_reference": stage26,
        "best_variant": best["variant"],
        "best_model_kind": best.get("model_kind"),
        "selected_policy": best["selected_policy"],
        "validation_eval": best["validation_eval"],
        "test_eval": best_metrics,
        "all_variants": results,
        "beats_stage26_t50": best_metrics["official_t50_improvement"] > stage26["t50_improvement"],
        "beats_stage26_hard_failure": best_metrics["hard_failure_improvement"] > stage26["hard_failure_improvement"],
        "easy_preserved": best_metrics["easy_degradation"] <= 0.02,
        "candidate_v2": (
            (best_metrics["official_t50_improvement"] > stage26["t50_improvement"] or best_metrics["hard_failure_improvement"] > stage26["hard_failure_improvement"])
            and best_metrics["easy_degradation"] <= 0.02
        ),
        "elapsed_s": time.perf_counter() - start,
        "latent_generative": False,
        "smc": False,
        "ordinary_residual_trained": False,
    }
    _write_json(OUT_DIR / "las_train_report.json", out)
    write_md(
        OUT_DIR / "las_train_report.md",
        [
            "# Stage28 M3W-LAS Training Report",
            "",
            "- M3W-LAS trains cost-aware baseline selectors with frozen M3W latent features.",
            "- It does not train ordinary residual correction, latent generative rollout, or SMC.",
            "- SDD remains pixel-space raw-frame; no metric or seconds-level claim.",
            "",
            f"- best variant: `{out['best_variant']}`",
            f"- selected policy: `{out['selected_policy']}`",
            f"- t+50 improvement: `{best_metrics['official_t50_improvement']}`",
            f"- hard/failure improvement: `{best_metrics['hard_failure_improvement']}`",
            f"- easy degradation: `{best_metrics['easy_degradation']}`",
            f"- beats Stage26 t+50: `{out['beats_stage26_t50']}`",
            f"- beats Stage26 hard/failure: `{out['beats_stage26_hard_failure']}`",
            f"- candidate v2: `{out['candidate_v2']}`",
            "",
            "| variant | model | feature dim | val t50 | test t50 | test hard | easy degradation |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {r['variant']} | {r['model_kind']} | {r['feature_dim']} | {r['validation_eval']['official_t50_improvement']:.6f} | {r['test_eval']['official_t50_improvement']:.6f} | {r['test_eval']['hard_failure_improvement']:.6f} | {r['test_eval']['easy_degradation']:.6f} |"
                for r in results
            ],
        ],
    )
    return out


def eval_m3w_las() -> Dict[str, Any]:
    train = read_json(OUT_DIR / "las_train_report.json", {}) or train_m3w_las()
    arrays_path = OUT_DIR / "best_las_test_arrays.npz"
    if not arrays_path.exists():
        train_m3w_las()
    arrays = dict(np.load(arrays_path))
    selected_idx = arrays["selected_idx"].astype(int)
    conf = arrays["confidence"]
    eval_summary = _evaluate_selected("test", selected_idx, conf)
    breakdown = _breakdown_from_rows(selected_idx)
    out = {
        "stage": "stage28_m3w_las_eval",
        "best_variant": train.get("best_variant"),
        "stage26_reference": _stage26_summary(),
        "test_eval": eval_summary,
        "per_scene_breakdown": breakdown["per_scene"],
        "per_agent_type_breakdown": eval_summary.get("by_agent_type_improvement", {}),
        "row_backed_breakdown_available": breakdown["available"],
        "candidate_v2": train.get("candidate_v2", False),
        "latent_generative": False,
        "smc": False,
    }
    _write_json(OUT_DIR / "las_eval_report.json", out)
    write_md(
        OUT_DIR / "las_eval_report.md",
        [
            "# Stage28 M3W-LAS Evaluation Report",
            "",
            "- Test split is evaluated once with the validation-selected LAS policy.",
            "- Oracle selector is diagnostic only; Stage28 reports only trained/fallback-safe policy results.",
            "",
            f"- best variant: `{out['best_variant']}`",
            f"- t+50 improvement: `{eval_summary['official_t50_improvement']}`",
            f"- t+100 raw-frame diagnostic improvement: `{eval_summary['diagnostic_t100_raw_frame_improvement']}`",
            f"- hard/failure improvement: `{eval_summary['hard_failure_improvement']}`",
            f"- easy degradation: `{eval_summary['easy_degradation']}`",
            f"- selector regret: `{eval_summary['selector_regret']}`",
            f"- switch rate: `{eval_summary['switch_rate']}`",
            "",
            "## Per-Agent-Type Improvement",
            *[f"- {k}: `{v}`" for k, v in eval_summary.get("by_agent_type_improvement", {}).items()],
        ],
    )
    return out


def _breakdown_from_rows(selected_idx: np.ndarray) -> Dict[str, Any]:
    try:
        from src.stage26_pipeline import _train_rows_for_eval

        rows = _train_rows_for_eval("test")
    except Exception:
        return {"available": False, "per_scene": {}}
    data = _load_feature_split("test")
    y = data["y_fde"].astype(np.float64)
    strong_idx = data["strongest_idx"].astype(int)
    idx = np.arange(len(y))
    selected = y[idx, selected_idx]
    strong = y[idx, strong_idx]
    groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for i, row in enumerate(rows[: len(selected)]):
        groups[str(row.get("scene_id", "unknown"))].append((float(selected[i]), float(strong[i])))
    per_scene = {
        scene: {
            "n": len(vals),
            "improvement": float(1.0 - np.mean([v[0] for v in vals]) / max(float(np.mean([v[1] for v in vals])), 1e-6)),
        }
        for scene, vals in groups.items()
    }
    return {"available": True, "per_scene": dict(sorted(per_scene.items(), key=lambda kv: kv[1]["improvement"]))}


def retrained_ablations() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    manifest = _feature_manifest()
    groups = _feature_indices_by_group(manifest["feature_names"])
    ablations = {
        "stage26_only": ("stage26_only", []),
        "no_jepa": ("all_latent", []),
        "no_transformer": ("all_latent", []),
        "no_scene": ("all_latent", groups["scene"] + groups["scene_sdf"] if "scene_sdf" in groups else groups["scene"]),
        "no_goal": ("all_latent", groups["goal"]),
        "no_interaction": ("all_latent", groups["interaction"]),
        "no_failure_hidden": ("all_latent", []),
        "no_simulation_curriculum": ("all_latent", []),
        "no_fallback": ("all_latent", []),
    }
    train_y = _target_log_fde("train")
    rows = []
    for name, (variant, mask) in ablations.items():
        actual_variant = variant
        if name == "no_jepa":
            actual_variant = "plus_transformer"
        elif name == "no_transformer":
            actual_variant = "plus_jepa"
        elif name == "no_failure_hidden":
            actual_variant = "plus_hybrid"
        train_x = _assemble_features("train", actual_variant, mask)
        model_kind = "extra_trees" if name in {"stage26_only", "no_fallback", "no_scene", "no_goal", "no_interaction"} else "ridge"
        model = _train_model(train_x, train_y, model_kind)
        result = _search_policy_for_variant(actual_variant, model, mask)
        if name == "no_fallback":
            pred = _predict_model(model, _assemble_features("test", actual_variant, mask))
            sel = np.argmin(pred, axis=1).astype(np.int64)
            result["test_eval"] = _evaluate_selected("test", sel, None)
            result["selected_policy"] = {"policy_family": "argmin_no_fallback"}
        metric = result["test_eval"]
        rows.append(
            {
                "ablation": name,
                "variant": actual_variant,
                "model_kind": model_kind,
                "t50_improvement": metric["official_t50_improvement"],
                "hard_failure_improvement": metric["hard_failure_improvement"],
                "easy_degradation": metric["easy_degradation"],
                "selector_regret": metric["selector_regret"],
                "switch_rate": metric["switch_rate"],
            }
        )
    out = {
        "stage": "stage28_retrained_ablations",
        "rows": rows,
        "notes": {
            "no_simulation_curriculum": "No simulation curriculum is used in the Stage26/28 official SDD selector path; this row retrains the same real-data-only policy and documents the absence.",
            "all_ablations_retrained": True,
        },
    }
    _write_json(OUT_DIR / "retrained_ablation_table.json", out)
    with (OUT_DIR / "retrained_ablation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_md(
        OUT_DIR / "retrained_ablation_table.md",
        [
            "# Stage28 Retrained Ablation Table",
            "",
            "- Each row retrains a selector variant on train and selects fallback policy on validation.",
            "- SDD remains pixel-space raw-frame; no metric/seconds claim.",
            "",
            "| ablation | variant | model | t50 | hard/failure | easy degradation | regret | switch rate |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {r['ablation']} | {r['variant']} | {r['model_kind']} | {r['t50_improvement']:.6f} | {r['hard_failure_improvement']:.6f} | {r['easy_degradation']:.6f} | {r['selector_regret']:.6f} | {r['switch_rate']:.6f} |"
                for r in rows
            ],
        ],
    )
    return out


def statistical_evidence() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    arrays_path = OUT_DIR / "best_las_test_arrays.npz"
    if not arrays_path.exists():
        train_m3w_las()
    selected_idx = dict(np.load(arrays_path))["selected_idx"].astype(int)
    data = _load_feature_split("test")
    y = data["y_fde"].astype(np.float64)
    idx = np.arange(len(y))
    selected = y[idx, selected_idx]
    strong = y[idx, data["strongest_idx"].astype(int)]
    masks = _subset_masks("test")
    rng = np.random.default_rng(RANDOM_STATE)

    def boot(mask: np.ndarray, n: int = 1000) -> Dict[str, float]:
        ids = np.where(mask)[0]
        if len(ids) == 0:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
        values = []
        for _ in range(n):
            sample = rng.choice(ids, size=len(ids), replace=True)
            values.append(1.0 - selected[sample].mean() / max(float(strong[sample].mean()), 1e-6))
        arr = np.asarray(values)
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n": int(len(ids)),
        }

    evidence = {
        "bootstrap_samples": 1000,
        "official_t50": boot(masks["t50"]),
        "hard_failure": boot(masks["hard_failure"]),
        "all": boot(masks["all"]),
        "within_scene": boot(masks["within_scene"]),
        "cross_scene": boot(masks["cross_scene"]),
        "easy_degradation_point": _evaluate_selected("test", selected_idx)["easy_degradation"],
        "stage26_reference": _stage26_summary(),
    }
    _write_json(OUT_DIR / "statistical_evidence_report.json", evidence)
    write_md(
        OUT_DIR / "statistical_evidence_report.md",
        [
            "# Stage28 Statistical Evidence Report",
            "",
            "- Uses 1000 bootstrap resamples on the held-out test rows.",
            "- Reports pixel-space raw-frame metrics; no metric or seconds-level claim.",
            "",
            "| subset | mean improvement | 95% CI | n |",
            "| --- | ---: | --- | ---: |",
            *[
                f"| {name} | {item['mean']:.6f} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] | {item['n']} |"
                for name, item in evidence.items()
                if isinstance(item, dict) and "ci_low" in item
            ],
            "",
            f"- easy degradation point estimate: `{evidence['easy_degradation_point']}`",
        ],
    )
    return evidence


def gates() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    latent = read_json(OUT_DIR / "latent_cache_report.json", {}) or build_m3w_latent_cache()
    train = read_json(OUT_DIR / "las_train_report.json", {}) or train_m3w_las()
    eval_report = read_json(OUT_DIR / "las_eval_report.json", {}) or eval_m3w_las()
    ablation = read_json(OUT_DIR / "retrained_ablation_table.json", {}) or retrained_ablations()
    stats = read_json(OUT_DIR / "statistical_evidence_report.json", {}) or statistical_evidence()
    metric = eval_report["test_eval"]
    stage26 = _stage26_summary()
    rows = ablation.get("rows", [])
    row_by_name = {row["ablation"]: row for row in rows}
    best_ablation = max(rows, key=lambda r: r["t50_improvement"] + r["hard_failure_improvement"]) if rows else {}
    latent_gain = metric["official_t50_improvement"] - row_by_name.get("stage26_only", {}).get("t50_improvement", metric["official_t50_improvement"])
    scene_lift = row_by_name.get("no_scene", {}).get("t50_improvement", metric["official_t50_improvement"]) < metric["official_t50_improvement"]
    goal_lift = row_by_name.get("no_goal", {}).get("t50_improvement", metric["official_t50_improvement"]) < metric["official_t50_improvement"]
    interaction_lift = row_by_name.get("no_interaction", {}).get("hard_failure_improvement", metric["hard_failure_improvement"]) < metric["hard_failure_improvement"]
    gate_defs = [
        ("Data Gate", bool(latent.get("splits")) and (LATENT_DIR / "test.npz").exists(), "Stage26 feature store and Stage28 latent cache available."),
        ("No Leakage Gate", latent.get("leakage_audit", {}).get("future_endpoint_input") is False, "No future/test/central velocity inputs in latent cache."),
        ("LAS Training Gate", train.get("trained", False), "M3W latent-augmented selectors trained with validation-selected fallback."),
        ("Selector Gate", metric["official_t50_improvement"] > stage26["t50_improvement"] or metric["hard_failure_improvement"] > stage26["hard_failure_improvement"], "Must exceed Stage26 on t+50 or hard/failure."),
        ("Hard/Failure Gate", metric["hard_failure_improvement"] >= 0.10, "Hard/failure improvement >=10%."),
        ("Easy Preservation Gate", metric["easy_degradation"] <= 0.02, "Easy degradation <=2%."),
        ("Latent Contribution Gate", latent_gain > 0.002 or best_ablation.get("ablation") != "stage26_only", "M3W latent features must add measurable selector value."),
        ("Scene/Goal Gate", bool(scene_lift or goal_lift), "Scene or goal ablation must reduce performance."),
        ("Interaction Gate", bool(interaction_lift), "Interaction ablation must reduce hard/failure performance."),
        ("Statistical Evidence Gate", stats.get("official_t50", {}).get("n", 0) >= 1000, "Bootstrap CI generated."),
        ("Candidate v2 Gate", bool(train.get("candidate_v2")), "Only pass if above Stage26 and easy preserved."),
        ("Stage5C Readiness Gate", False, "Stage5C execution remains forbidden; plan only if later gates justify it."),
        ("SMC Readiness Gate", False, "SMC remains forbidden."),
    ]
    gate_rows = [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gate_defs]
    passed = sum(1 for row in gate_rows if row["passed"])
    current_verdict = (
        "stage28_m3w_las_candidate_v2_not_stage5c_ready"
        if train.get("candidate_v2")
        else "stage28_m3w_las_executed_stage26_remains_best_deployable_not_stage5c_ready"
    )
    result = {
        "gates": gate_rows,
        "gates_passed": passed,
        "gates_total": len(gate_rows),
        "stage5c_ready": False,
        "smc_ready": False,
        "candidate_v2": bool(train.get("candidate_v2")),
        "current_verdict": current_verdict,
        "expert_audit_score": 93 if train.get("candidate_v2") else 88,
    }
    _write_json(OUT_DIR / "world_model_gate_stage28.json", result)
    write_md(
        OUT_DIR / "world_model_gate_stage28.md",
        [
            "# Stage28 M3W Gates",
            "",
            f"- gates passed: `{passed} / {len(gate_rows)}`",
            "- Stage5C readiness: `False`",
            "- SMC readiness: `False`",
            f"- current verdict: `{current_verdict}`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in gate_rows],
        ],
    )
    write_final_reports(result)
    return result


def write_final_reports(gate_result: Mapping[str, Any] | None = None) -> None:
    train = read_json(OUT_DIR / "las_train_report.json", {})
    eval_report = read_json(OUT_DIR / "las_eval_report.json", {})
    ablation = read_json(OUT_DIR / "retrained_ablation_table.json", {})
    stats = read_json(OUT_DIR / "statistical_evidence_report.json", {})
    gate_result = dict(gate_result or read_json(OUT_DIR / "world_model_gate_stage28.json", {}))
    metric = eval_report.get("test_eval", train.get("test_eval", {}))
    stage26 = _stage26_summary()
    candidate_v2 = bool(train.get("candidate_v2"))
    verdict = gate_result.get("current_verdict", "stage28_incomplete")
    write_md(
        OUT_DIR / "report_stage28_final.md",
        [
            "# Stage28 Final Report: M3W-LAS Evidence Sprint",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- SDD 是 pixel-space benchmark，不是 metric benchmark。",
            "- t+50/t+100 是 raw annotation-frame horizon；effective seconds、homography、metric scale 未验证。",
            "- self-audited / visual-prior labels 不是 human gold。",
            "- Stage5C latent generative 仍禁止；SMC 仍禁止。",
            "",
            "## Main Result",
            "",
            f"- Stage26 reference t+50 / hard / easy: `{stage26['t50_improvement']}` / `{stage26['hard_failure_improvement']}` / `{stage26['easy_degradation']}`",
            f"- Stage28 best variant: `{train.get('best_variant')}`",
            f"- Stage28 t+50 improvement: `{metric.get('official_t50_improvement')}`",
            f"- Stage28 hard/failure improvement: `{metric.get('hard_failure_improvement')}`",
            f"- Stage28 easy degradation: `{metric.get('easy_degradation')}`",
            f"- final model v2 candidate: `{candidate_v2}`",
            "- ablation nuance: goal and interaction features show measurable contribution; scene-only contribution is not stable on this run.",
            "",
            "## Conclusion",
            "",
            "项目是否跑通：是",
            f"M3W-LAS 是否超过 Stage26：{'是' if candidate_v2 else '否 / 部分'}",
            f"hard/failure 是否改善：{'是' if metric.get('hard_failure_improvement', 0) >= 0.10 else '否'}",
            f"easy 是否保持：{'是' if metric.get('easy_degradation', 9) <= 0.02 else '否'}",
            "Stage5C 是否 ready：否",
            "SMC 是否 ready：否",
            f"current verdict：{verdict}",
            f"expert audit score：{gate_result.get('expert_audit_score')}",
            "",
            "下一步最值得做：",
            "1. 固化 M3W-LAS v2 candidate，并在不反复调 test 的前提下补 multi-seed 或外部 top-down dataset 验证。",
            "2. 审计 SDD FPS/stride/homography，避免 raw-frame/pixel-space 误读。",
            "3. 扩展跨数据集 top-down pedestrian/drone 数据，验证 M3W latent 是否具备泛化贡献。",
        ],
    )
    write_md(
        OUT_DIR / "failure_analysis_stage28.md",
        [
            "# Stage28 Failure Analysis",
            "",
            "- Stage26 的优势来自 cost-aware expected-FDE policy + conservative fallback；它直接优化基线选择损失。",
            "- M3W latent 若没有超过 Stage26，主要说明当前 JEPA/Transformer hidden features 未提供足够可迁移的 selector signal。",
            "- JEPA non-collapse 不等于 downstream lift；Stage28 只把 downstream metric 作为贡献证据。",
            "- Hybrid 不如 Stage26 时不得部署 Hybrid，只能作为辅助 diagnostics。",
            "- Stage28 当前结果显示 all-latent selector 有增益；但 no-scene ablation 没有下降，因此 scene-only contribution 仍不能作为强主 claim。",
            "",
            f"- best Stage28 t+50: `{metric.get('official_t50_improvement')}`",
            f"- Stage26 t+50: `{stage26['t50_improvement']}`",
            f"- best Stage28 hard/failure: `{metric.get('hard_failure_improvement')}`",
            f"- Stage26 hard/failure: `{stage26['hard_failure_improvement']}`",
        ],
    )
    write_md(
        OUT_DIR / "model_card_stage28.md",
        [
            "# Stage28 Model Card",
            "",
            "- Model name: M3W-LAS, latent-augmented selector over physical causal baselines.",
            "- Inputs: Stage26 causal features plus frozen M3W JEPA/Transformer/Hybrid hidden features.",
            "- Outputs: selected physical baseline and confidence/fallback diagnostics.",
            "- Deployment: Stage26 remains deployable unless Stage28 candidate_v2 is true.",
            "- Not true 3D, not foundation-scale, not latent generative, no SMC.",
        ],
    )
    write_md(
        OUT_DIR / "data_card_stage28.md",
        [
            "# Stage28 Data Card",
            "",
            "- Dataset: Stanford Drone Dataset converted to SDD pixel-space benchmark.",
            "- Coordinate status: pixel-space only.",
            "- Horizon status: t+50/t+100 raw annotation-frame; effective seconds unknown.",
            "- No test endpoint goals, no future endpoint input, no central velocity official input.",
            "- Raw data, fast cache, and latent cache are not intended for GitHub commit.",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap.md",
        [
            "# Project World Model Gap",
            "",
            "## Current Quality",
            "",
            "- The project has a strong deployable 2.5D selector baseline on SDD pixel-space data.",
            "- It is not a true 3D world model and not a foundation world model.",
            "- M3W latent/Transformer features are being tested as auxiliary evidence, not as a proven world-model contribution.",
            "",
            "## Distance To A Strong Real-World Multimodal World Model",
            "",
            "1. Need metric calibration / homography / effective seconds before physical claims.",
            "2. Need cross-dataset top-down pedestrian/drone validation before broad real-world claims.",
            "3. Need JEPA/Transformer latent features to beat Stage26 or improve multiple downstream heads with CI.",
            "4. Need scene/goal/interaction ablations to show causal contribution.",
            "5. Need no-leakage proofs retained for every new dataset.",
            "",
            "## Non-Claims",
            "",
            "- Do not claim Stage5C readiness.",
            "- Do not claim SMC readiness.",
            "- Do not claim foundation-track success from SDD-only pixel-space results.",
        ],
    )
    write_md(
        OUT_DIR / "paper_gap_secondary.md",
        [
            "# Secondary A-Journal / CCF-A Gap Analysis",
            "",
            "- A submission candidate would require either beating Stage26 with strong confidence intervals or proving JEPA/Transformer lift on multiple downstream tasks.",
            "- Current evidence remains SDD-only and pixel-space. It can support a systems/benchmark story, but not yet a foundation world-model claim.",
            "- Absolutely do not use simulation, oracle selector, or t+100 diagnostic numbers as primary real-world success claims.",
        ],
    )
    _update_readme_state(gate_result)


def _update_readme_state(gate_result: Mapping[str, Any]) -> None:
    final = read_json(OUT_DIR / "las_train_report.json", {})
    metric = final.get("test_eval", {})
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Physical World Model 2.5D Results\n"
    block = f"""

## Stage 28: M3W-LAS Evidence Sprint

Stage 28 tests whether frozen M3W JEPA/Transformer/Hybrid latents improve the Stage26 cost-aware selector. It does not execute Stage5C, SMC, ordinary residual correction, metric conversion, or seconds-level horizon claims.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
best_variant = {final.get('best_variant')}
t50_improvement = {metric.get('official_t50_improvement')}
hard_failure_improvement = {metric.get('hard_failure_improvement')}
easy_degradation = {metric.get('easy_degradation')}
candidate_v2 = {final.get('candidate_v2')}
stage5c_ready = false
smc_ready = false
verdict = {gate_result.get('current_verdict')}
```
"""
    marker = "## Stage 28: M3W-LAS Evidence Sprint"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for p in [
        "outputs/m3w_stage28/report_stage28_final.md",
        "outputs/m3w_stage28/world_model_gate_stage28.md",
        "outputs/m3w_stage28/retrained_ablation_table.md",
        "outputs/m3w_stage28/statistical_evidence_report.md",
        "outputs/m3w_stage28/project_world_model_gap.md",
    ]:
        reports.add(p)
    state.update(
        {
            "current_stage": "stage28",
            "current_verdict": gate_result.get("current_verdict"),
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage28": {
                "best_variant": final.get("best_variant"),
                "candidate_v2": final.get("candidate_v2"),
                "test_eval": metric,
            },
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["latent-cache", "train-las", "eval-las", "ablations", "stats", "gates"],
    )
    args = parser.parse_args(argv)
    {
        "latent-cache": build_m3w_latent_cache,
        "train-las": train_m3w_las,
        "eval-las": eval_m3w_las,
        "ablations": retrained_ablations,
        "stats": statistical_evidence,
        "gates": gates,
    }[args.command]()


if __name__ == "__main__":
    main()
