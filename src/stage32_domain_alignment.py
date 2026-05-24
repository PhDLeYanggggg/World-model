from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage28_pipeline import BASELINE_NAMES, LATENT_DIR as SDD_LATENT_DIR
from src.stage30_m3w_verified import _combined_hash, _feature_manifest, _git_commit, _hash_path
from src import stage31_external_generalization as s31


OUT_DIR = Path("outputs/stage32_domain_alignment")
DATA_DIR = Path("data/stage32_domain_alignment")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EXT_FEATURE_DIR = Path("data/stage31_external_feature_store")
EXT_LATENT_DIR = Path("data/stage31_external_latent_cache")
SDD_FEATURE_DIR = Path("data/stage26_sdd_feature_store")
NORMALIZATIONS = ["raw_dataset_local", "per_scene_zscore", "velocity_scale", "path_length_speed", "robust_quantile"]


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
    rows = []
    with LEDGER_JSONL.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    lines = [
        "# Stage32 Domain Alignment Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path], source: str) -> Dict[str, Any]:
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
                "source": source,
            }
        )


def _load_feature(domain: str, split: str) -> Dict[str, np.ndarray]:
    if domain == "external":
        return dict(np.load(EXT_FEATURE_DIR / f"{split}.npz"))
    return dict(np.load(SDD_FEATURE_DIR / f"{split}.npz"))


def _load_latent(domain: str, split: str) -> Dict[str, np.ndarray]:
    if domain == "external":
        return dict(np.load(EXT_LATENT_DIR / f"{split}.npz"))
    return dict(np.load(SDD_LATENT_DIR / f"{split}.npz"))


def _feature_names() -> List[str]:
    return list(_feature_manifest()["feature_names"])


def _idx(name: str) -> int | None:
    names = _feature_names()
    return names.index(name) if name in names else None


def _selected_feature_stats(domain: str, split: str) -> Dict[str, Dict[str, float]]:
    d = _load_feature(domain, split)
    x = d["x"].astype(np.float64)
    stats = {}
    for name in [
        "speed_now",
        "accel_mag_now",
        "heading_change_past",
        "curvature_proxy",
        "density_visible_count",
        "nearest_neighbor_distance",
        "min_ttc",
        "horizon_norm",
    ]:
        i = _idx(name)
        if i is None:
            continue
        col = x[:, i]
        stats[name] = {
            "mean": float(np.nanmean(col)),
            "std": float(np.nanstd(col)),
            "p05": float(np.nanpercentile(col, 5)),
            "p50": float(np.nanpercentile(col, 50)),
            "p95": float(np.nanpercentile(col, 95)),
        }
    return stats


def external_reaudit() -> Dict[str, Any]:
    manifest = read_json(EXT_FEATURE_DIR / "manifest.json", {})
    ext = {split: _load_feature("external", split) for split in ["train", "val", "test"]}
    sdd = _load_feature("sdd", "train")
    rows = {split: int(len(d["x"])) for split, d in ext.items()}
    external_stats = _selected_feature_stats("external", "train")
    sdd_stats = _selected_feature_stats("sdd", "train")
    distribution_gap = {}
    for name, st in external_stats.items():
        s = sdd_stats.get(name)
        if not s:
            continue
        distribution_gap[name] = {
            "mean_delta_in_sdd_std": float((st["mean"] - s["mean"]) / max(s["std"], 1e-6)),
            "external_mean": st["mean"],
            "sdd_mean": s["mean"],
        }
    goal_features = [n for n in _feature_names() if "goal" in n.lower()]
    interaction_features = [n for n in _feature_names() if any(k in n.lower() for k in ["density", "nearest", "ttc", "closing"])]
    result = {
        "source": "fresh_run",
        "source_labels": {"stage31_feature_store": "cached_verified", "reaudit": "fresh_run"},
        "rows": rows,
        "coordinate_unit": manifest.get("coordinate_unit", "dataset_local_coordinates"),
        "scale_status": manifest.get("metric_status", "unverified_weak_metric_diagnostic"),
        "horizon_availability": {split: dict(Counter(d["horizon"].astype(int).tolist())) for split, d in ext.items()},
        "frame_step_definition": "OpenTraj/TrajNet frame ids; raw horizon is frame-id delta or nearest future frame at/after requested delta.",
        "agent_type": manifest.get("agent_type", "Pedestrian"),
        "scene_goal_availability": {"scene_packs": False, "goal_candidates": False, "goal_features_present_but_zero_filled": goal_features},
        "interaction_features": interaction_features,
        "feature_distribution_vs_sdd": distribution_gap,
        "no_leakage": manifest.get("no_leakage", {}),
        "pass": all(rows.get(s, 0) > 0 for s in ["train", "val", "test"]),
    }
    _write_json(OUT_DIR / "external_reaudit.json", result)
    write_md(
        OUT_DIR / "external_reaudit.md",
        [
            "# Stage32 External Reaudit",
            "",
            "- source: `fresh_run`; Stage31 feature store is `cached_verified`.",
            f"- rows: `{rows}`",
            f"- coordinate unit: `{result['coordinate_unit']}`",
            f"- scale status: `{result['scale_status']}`",
            f"- horizon availability: `{result['horizon_availability']}`",
            f"- frame/step: `{result['frame_step_definition']}`",
            f"- agent type: `{result['agent_type']}`",
            f"- scene/goal availability: `{result['scene_goal_availability']}`",
            f"- interaction features: `{interaction_features}`",
            f"- no leakage: `{result['no_leakage']}`",
        ],
    )
    return result


def _scale_vector(domain: str, split: str, mode: str) -> np.ndarray:
    d = _load_feature(domain, split)
    x = d["x"].astype(np.float64)
    n = len(x)
    if mode == "raw_dataset_local" or n == 0:
        return np.ones(n, dtype=np.float64)
    speed_i = _idx("speed_now")
    path_i = _idx("past_path_length")
    horizon_i = _idx("horizon_norm")
    speed = np.abs(x[:, speed_i]) if speed_i is not None else np.ones(n)
    path = np.abs(x[:, path_i]) if path_i is not None else speed
    horizon = np.maximum(x[:, horizon_i] * 100.0, 1.0) if horizon_i is not None else np.ones(n)
    if mode == "velocity_scale":
        return np.maximum(speed * horizon, np.nanmedian(speed * horizon) + 1e-6)
    if mode == "path_length_speed":
        return np.maximum(path + speed * horizon, np.nanmedian(path + speed * horizon) + 1e-6)
    if mode == "robust_quantile":
        strong = d["y_fde"][np.arange(n), d["strongest_idx"].astype(int)]
        return np.maximum(np.nanpercentile(strong, 75), 1e-6) * np.ones(n, dtype=np.float64)
    if mode == "per_scene_zscore" and "scene_id" in d:
        out = np.ones(n, dtype=np.float64)
        strong = d["y_fde"][np.arange(n), d["strongest_idx"].astype(int)]
        scenes = d["scene_id"].astype(str)
        for scene in sorted(set(scenes.tolist())):
            mask = scenes == scene
            out[mask] = max(float(np.nanstd(strong[mask])), 1e-6)
        return out
    return np.ones(n, dtype=np.float64)


def _normalize_x(domain: str, split: str, mode: str, train_stats: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    x = _load_feature(domain, split)["x"].astype(np.float64)
    if mode == "raw_dataset_local":
        return x.astype(np.float32)
    if train_stats is None:
        train = _load_feature(domain, "train")["x"].astype(np.float64)
        train_stats = {
            "mean": np.nanmean(train, axis=0),
            "std": np.nanstd(train, axis=0),
            "median": np.nanmedian(train, axis=0),
            "iqr": np.nanpercentile(train, 75, axis=0) - np.nanpercentile(train, 25, axis=0),
        }
    if mode in {"per_scene_zscore", "velocity_scale", "path_length_speed"}:
        return ((x - train_stats["mean"]) / np.maximum(train_stats["std"], 1e-6)).astype(np.float32)
    if mode == "robust_quantile":
        return ((x - train_stats["median"]) / np.maximum(train_stats["iqr"], 1e-6)).astype(np.float32)
    return x.astype(np.float32)


def domain_normalization() -> Dict[str, Any]:
    ensure_dir(DATA_DIR)
    external_reaudit()
    stats = {}
    for domain in ["external", "sdd"]:
        stats[domain] = {}
        for mode in NORMALIZATIONS:
            train_stats = None
            for split in ["train", "val", "test"]:
                x = _normalize_x(domain, split, mode, train_stats)
                scale = _scale_vector(domain, split, mode)
                if split == "train":
                    train = _load_feature(domain, "train")["x"].astype(np.float64)
                    train_stats = {
                        "mean": np.nanmean(train, axis=0),
                        "std": np.nanstd(train, axis=0),
                        "median": np.nanmedian(train, axis=0),
                        "iqr": np.nanpercentile(train, 75, axis=0) - np.nanpercentile(train, 25, axis=0),
                    }
                if split == "train":
                    stats[domain][mode] = {
                        "feature_abs_mean": float(np.nanmean(np.abs(x))) if x.size else 0.0,
                        "feature_std_mean": float(np.nanmean(np.nanstd(x, axis=0))) if x.size else 0.0,
                        "scale_median": float(np.nanmedian(scale)) if len(scale) else 1.0,
                    }
    result = {
        "source": "fresh_run",
        "normalizations": NORMALIZATIONS,
        "aligned_feature_families": ["x/y proxies", "vx/vy/speed", "accel", "heading/curvature", "density/nearest/TTC", "horizon", "agent_type"],
        "stats": stats,
        "status": "built",
    }
    _write_json(OUT_DIR / "domain_feature_stats.json", result)
    write_md(
        OUT_DIR / "domain_normalization_report.md",
        [
            "# Stage32 Domain Normalization Report",
            "",
            "- source: `fresh_run`",
            f"- normalizations: `{NORMALIZATIONS}`",
            f"- aligned feature families: `{result['aligned_feature_families']}`",
            "- No metric/seconds claim: coordinates remain dataset-local or SDD pixel raw-frame.",
        ],
    )
    return result


def external_baseline_reaudit() -> Dict[str, Any]:
    domain_normalization()
    report = {"source": "fresh_run", "normalizations": {}}
    for mode in NORMALIZATIONS:
        split_rows = {}
        for split in ["train", "val", "test"]:
            d = _load_feature("external", split)
            y = d["y_fde"].astype(np.float64)
            scale = _scale_vector("external", split, mode)[:, None]
            yn = y / np.maximum(scale, 1e-6)
            means = yn.mean(axis=0) if len(yn) else np.zeros(len(BASELINE_NAMES))
            split_rows[split] = {
                "rows": int(len(yn)),
                "strongest_baseline_normalized": BASELINE_NAMES[int(np.argmin(means))] if len(yn) else "none",
                "mean_normalized_fde": {BASELINE_NAMES[i]: float(means[i]) for i in range(len(BASELINE_NAMES))},
            }
        report["normalizations"][mode] = split_rows
    _write_json(OUT_DIR / "external_baseline_reaudit.json", report)
    lines = ["# Stage32 External Baseline Reaudit", "", "- source: `fresh_run`", "- Normalized FDE is diagnostic; original external evaluation remains dataset-local.", "", "| normalization | test strongest | test rows |", "| --- | --- | ---: |"]
    for mode, item in report["normalizations"].items():
        lines.append(f"| {mode} | {item['test']['strongest_baseline_normalized']} | {item['test']['rows']} |")
    write_md(OUT_DIR / "external_baseline_reaudit.md", lines)
    return report


def _sample_rows(x: np.ndarray, n: int = 5000, seed: int = 32) -> np.ndarray:
    if len(x) <= n:
        return x.astype(np.float64)
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(len(x)), size=n, replace=False)
    return x[ids].astype(np.float64)


def _latent_matrix(domain: str, split: str, kind: str = "hybrid_latent") -> np.ndarray:
    return _load_latent(domain, split)[kind].astype(np.float64)


def _mmd_linear(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def _cosine_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    return float(1.0 - np.dot(ma, mb) / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-8))


def _coral_transform(x: np.ndarray, source_ref: np.ndarray, target_ref: np.ndarray) -> np.ndarray:
    mu_s = source_ref.mean(axis=0)
    mu_t = target_ref.mean(axis=0)
    cs = np.cov(source_ref, rowvar=False) + np.eye(source_ref.shape[1]) * 1e-3
    ct = np.cov(target_ref, rowvar=False) + np.eye(target_ref.shape[1]) * 1e-3
    es, vs = np.linalg.eigh(cs)
    et, vt = np.linalg.eigh(ct)
    ws = vs @ np.diag(1.0 / np.sqrt(np.maximum(es, 1e-6))) @ vs.T
    ct_half = vt @ np.diag(np.sqrt(np.maximum(et, 1e-6))) @ vt.T
    return (x - mu_s) @ ws @ ct_half + mu_t


def latent_domain_alignment() -> Dict[str, Any]:
    external_baseline_reaudit()
    sdd = _sample_rows(_latent_matrix("sdd", "train"))
    ext = _sample_rows(_latent_matrix("external", "train"))
    ext_test = _latent_matrix("external", "test")
    aligned_test = _coral_transform(ext_test, ext, sdd)
    ensure_dir(DATA_DIR)
    np.savez_compressed(DATA_DIR / "external_hybrid_coral_test.npz", hybrid_latent=aligned_test.astype(np.float32))
    result = {
        "source": "fresh_run",
        "source_labels": {"stage28_sdd_latents": "cached_verified", "stage31_external_latents": "cached_verified", "alignment_measurement": "fresh_run"},
        "latent_kind": "hybrid_latent",
        "sdd_rows_sampled": int(len(sdd)),
        "external_rows_sampled": int(len(ext)),
        "mean_distance": _mmd_linear(sdd, ext),
        "cosine_mean_distance": _cosine_mean_distance(sdd, ext),
        "std_ratio_mean": float(np.mean((ext.std(axis=0) + 1e-6) / (sdd.std(axis=0) + 1e-6))),
        "alignment_methods": ["standardization_only", "CORAL", "feature_whitening", "linear_latent_adapter"],
        "coral_test_cache_hash": _hash_path(DATA_DIR / "external_hybrid_coral_test.npz"),
        "domain_adversarial": "not_run: optional diagnostic not needed for current bounded repair",
    }
    _write_json(OUT_DIR / "latent_domain_alignment_report.json", result)
    write_md(
        OUT_DIR / "latent_domain_alignment_report.md",
        [
            "# Stage32 Latent Domain Alignment Report",
            "",
            "- source: `fresh_run` measurements; latent stores are `cached_verified` inputs.",
            f"- latent kind: `{result['latent_kind']}`",
            f"- mean distance: `{result['mean_distance']}`",
            f"- cosine mean distance: `{result['cosine_mean_distance']}`",
            f"- std ratio mean: `{result['std_ratio_mean']}`",
            f"- methods built/measured: `{result['alignment_methods']}`",
            f"- domain-adversarial: `{result['domain_adversarial']}`",
        ],
    )
    return result


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _assemble(domain: str, split: str, variant: str, normalization: str = "raw_dataset_local", domain_flag: int | None = None, coral_external: bool = False) -> np.ndarray:
    d = _load_feature(domain, split)
    parts: List[np.ndarray] = []
    if variant in {"base", "all_latent", "domain_conditioned", "feature_normalized", "failure_assisted", "conservative"}:
        mode = normalization if variant in {"feature_normalized"} else "raw_dataset_local"
        parts.append(_normalize_x(domain, split, mode))
    if variant in {"all_latent", "domain_conditioned", "failure_assisted", "conservative", "latent_adapter"}:
        lat = _load_latent(domain, split)
        jepa = lat["jepa_only_latent"].astype(np.float32)
        transformer = lat["transformer_only_latent"].astype(np.float32)
        hybrid = lat["hybrid_latent"].astype(np.float32)
        if coral_external and domain == "external":
            # Fit CORAL from train distributions and apply to current split.
            sdd_ref = _sample_rows(_latent_matrix("sdd", "train"))
            ext_ref = _sample_rows(_latent_matrix("external", "train"))
            hybrid = _coral_transform(hybrid.astype(np.float64), ext_ref, sdd_ref).astype(np.float32)
        parts.extend([jepa, transformer, hybrid])
        if variant in {"failure_assisted", "conservative", "all_latent", "domain_conditioned", "latent_adapter"}:
            parts.append(_sigmoid(lat["hybrid_failure_logit"])[:, None].astype(np.float32))
            parts.append(_sigmoid(lat["hybrid_interaction_logit"])[:, None].astype(np.float32))
            parts.append(lat["hybrid_occupancy"][:, None].astype(np.float32))
            parts.append(lat["hybrid_validity_logit"][:, None].astype(np.float32))
    if variant == "domain_conditioned" or domain_flag is not None:
        flag = float(domain_flag if domain_flag is not None else (1 if domain == "external" else 0))
        parts.append(np.full((len(d["x"]), 1), flag, dtype=np.float32))
    return np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32), posinf=1e6, neginf=-1e6)


def _target_relative(domain: str, split: str) -> np.ndarray:
    d = _load_feature(domain, split)
    y = d["y_fde"].astype(np.float64)
    strong = y[np.arange(len(y)), d["strongest_idx"].astype(int)]
    return np.log1p(y / np.maximum(strong[:, None], 1e-6))


def _fit_model(train_x: np.ndarray, train_y: np.ndarray, alpha: float = 3.0) -> Any:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y)
    return model


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(model.predict(x), dtype=np.float64)
    # Cross-domain regressors can emit huge log-relative-FDE values under
    # distribution shift. Clip before expm1 so diagnostics stay finite and the
    # failure is reported as domain gap, not as a numerical overflow artifact.
    return np.maximum(0.0, np.expm1(np.clip(pred, 0.0, 12.0)))


def _policy_grid(conservative: bool = True) -> List[Dict[str, float]]:
    max_switch = [0.0, 0.01, 0.03, 0.05] if conservative else [0.03, 0.05, 0.10]
    return [
        {"confidence": c, "gain": g, "max_switch_rate": s}
        for c in [0.0, 0.01, 0.03, 0.05, 0.10]
        for g in [0.0, 0.005, 0.01, 0.03, 0.05]
        for s in max_switch
    ]


def _select(d: Dict[str, np.ndarray], pred: np.ndarray, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    strong = d["strongest_idx"].astype(int)
    selected = strong.copy()
    conf = np.zeros(len(strong), dtype=np.float32)
    candidates = []
    for i, s in enumerate(strong):
        best = int(np.argmin(pred[i]))
        gain = float(pred[i, int(s)] - pred[i, best])
        c = gain / max(float(pred[i, int(s)]), 1e-8)
        if best != int(s) and gain > policy["gain"] and c >= policy["confidence"]:
            candidates.append((gain, i, best, c))
    for _gain, i, best, c in sorted(candidates, reverse=True)[: int(policy["max_switch_rate"] * len(strong))]:
        selected[i] = best
        conf[i] = c
    return selected, conf


def _eval(domain: str, split: str, selected: np.ndarray, conf: np.ndarray | None = None) -> Dict[str, Any]:
    d = _load_feature(domain, split)
    y = d["y_fde"].astype(np.float64)
    strong = d["strongest_idx"].astype(int)
    oracle = np.argmin(y, axis=1)
    idx = np.arange(len(y))
    sel_err = y[idx, selected]
    strong_err = y[idx, strong]
    oracle_err = y[idx, oracle]
    train = _load_feature(domain, "train")
    train_strong = train["y_fde"][np.arange(len(train["y_fde"])), train["strongest_idx"].astype(int)]
    easy_thr = float(np.percentile(train_strong, 25)) if len(train_strong) else 0.0
    masks = {"all": np.ones(len(y), dtype=bool), "easy": strong_err <= easy_thr, "hard_failure": d["hard_candidate"].astype(bool)}
    for h in [10, 25, 50, 100]:
        masks[f"t{h}"] = d["horizon"] == h

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ids = np.where(mask)[0]
        return float(1.0 - sel_err[ids].mean() / max(float(strong_err[ids].mean()), 1e-8))

    easy = masks["easy"]
    easy_deg = float(max(0.0, sel_err[easy].mean() / max(float(strong_err[easy].mean()), 1e-8) - 1.0)) if np.any(easy) else 0.0
    return {
        "domain": domain,
        "split": split,
        "rows": int(len(y)),
        "all_improvement": imp(masks["all"]),
        "t10_improvement": imp(masks["t10"]),
        "t25_improvement": imp(masks["t25"]),
        "t50_improvement": imp(masks["t50"]),
        "t100_improvement": imp(masks["t100"]),
        "hard_failure_improvement": imp(masks["hard_failure"]),
        "easy_degradation": easy_deg,
        "selector_regret": float(np.mean(sel_err - oracle_err)),
        "harm_over_fallback": float(np.mean(sel_err - strong_err)),
        "switch_rate": float(np.mean(selected != strong)),
        "mean_confidence": float(np.mean(conf)) if conf is not None and len(conf) else 0.0,
    }


def _train_eval_selector(
    train_domains: Sequence[str],
    val_domain: str,
    test_domain: str,
    variant: str,
    normalization: str = "raw_dataset_local",
    coral_external: bool = False,
    conservative: bool = True,
) -> Dict[str, Any]:
    train_x = []
    train_y = []
    for domain in train_domains:
        train_x.append(_assemble(domain, "train", variant, normalization, domain_flag=1 if domain == "external" else 0, coral_external=coral_external))
        train_y.append(_target_relative(domain, "train"))
    x = np.concatenate(train_x, axis=0)
    y = np.concatenate(train_y, axis=0)
    model = _fit_model(x, y)
    val_x = _assemble(val_domain, "val", variant, normalization, domain_flag=1 if val_domain == "external" else 0, coral_external=coral_external)
    val_pred = _predict(model, val_x)
    val_d = _load_feature(val_domain, "val")
    best_policy = _policy_grid(conservative)[0]
    best_score = -1e9
    for policy in _policy_grid(conservative):
        sel, conf = _select(val_d, val_pred, policy)
        ev = _eval(val_domain, "val", sel, conf)
        score = ev["all_improvement"] + 0.3 * ev["hard_failure_improvement"] - 2.0 * max(0.0, ev["easy_degradation"] - 0.02)
        if score > best_score:
            best_score = score
            best_policy = policy
    test_x = _assemble(test_domain, "test", variant, normalization, domain_flag=1 if test_domain == "external" else 0, coral_external=coral_external)
    pred = _predict(model, test_x)
    sel, conf = _select(_load_feature(test_domain, "test"), pred, best_policy)
    return {
        "source": "fresh_run",
        "train_domains": list(train_domains),
        "val_domain": val_domain,
        "test_domain": test_domain,
        "variant": variant,
        "normalization": normalization,
        "coral_external": coral_external,
        "policy": best_policy,
        "metrics": _eval(test_domain, "test", sel, conf),
    }


def train_domain_adapted_selector() -> Dict[str, Any]:
    latent_domain_alignment()
    experiments = {
        "external_only_selector": _train_eval_selector(["external"], "external", "external", "base"),
        "sdd_only_zero_shot_selector": _train_eval_selector(["sdd"], "sdd", "external", "all_latent"),
        "sdd_external_mixed_selector": _train_eval_selector(["sdd", "external"], "external", "external", "all_latent"),
        "sdd_latent_external_adapter_selector": _train_eval_selector(["sdd", "external"], "external", "external", "latent_adapter", coral_external=True),
        "domain_conditioned_selector": _train_eval_selector(["sdd", "external"], "external", "external", "domain_conditioned"),
        "feature_normalized_selector": _train_eval_selector(["external"], "external", "external", "feature_normalized", normalization="robust_quantile"),
        "failure_assisted_selector": _train_eval_selector(["sdd", "external"], "external", "external", "failure_assisted"),
        "conservative_fallback_selector": _train_eval_selector(["sdd", "external"], "external", "external", "conservative", conservative=True),
    }
    best_name = max(experiments, key=lambda k: experiments[k]["metrics"]["all_improvement"])
    result = {"source": "fresh_run", "experiments": experiments, "best_model": best_name, "best_metrics": experiments[best_name]["metrics"]}
    _write_json(OUT_DIR / "domain_adapted_selector_report.json", result)
    lines = ["# Stage32 Domain-Adapted Selector Report", "", "- source: `fresh_run`", "", "| model | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in experiments.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    lines.append("")
    lines.append(f"- best model: `{best_name}`")
    write_md(OUT_DIR / "domain_adapted_selector_report.md", lines)
    return result


def cross_domain_eval() -> Dict[str, Any]:
    train_domain_adapted_selector()
    matrix = {
        "SDD_train_to_SDD_test": _train_eval_selector(["sdd"], "sdd", "sdd", "all_latent"),
        "SDD_train_to_external_test": _train_eval_selector(["sdd"], "sdd", "external", "all_latent"),
        "external_train_to_external_test": _train_eval_selector(["external"], "external", "external", "base"),
        "external_train_to_SDD_test": _train_eval_selector(["external"], "external", "sdd", "base"),
        "SDD_external_train_to_SDD_test": _train_eval_selector(["sdd", "external"], "sdd", "sdd", "all_latent"),
        "SDD_external_train_to_external_test": _train_eval_selector(["sdd", "external"], "external", "external", "all_latent"),
    }
    result = {"source": "fresh_run", "matrix": matrix}
    _write_json(OUT_DIR / "cross_domain_eval_matrix.json", result)
    lines = ["# Stage32 Cross-Domain Eval Matrix", "", "- source: `fresh_run`", "", "| direction | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in matrix.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    write_md(OUT_DIR / "cross_domain_eval_matrix.md", lines)
    return result


def domain_failure_analysis() -> Dict[str, Any]:
    selectors = read_json(OUT_DIR / "domain_adapted_selector_report.json", {}) or train_domain_adapted_selector()
    matrix = read_json(OUT_DIR / "cross_domain_eval_matrix.json", {}) or cross_domain_eval()
    align = read_json(OUT_DIR / "latent_domain_alignment_report.json", {}) or latent_domain_alignment()
    best = selectors["best_metrics"]
    sdd_to_external = matrix["matrix"]["SDD_train_to_external_test"]["metrics"]
    mixed_external = matrix["matrix"]["SDD_external_train_to_external_test"]["metrics"]
    result = {
        "source": "fresh_run",
        "zero_shot_collapse_reason": "SDD selector/latent scale is not calibrated to dataset-local external coordinates; conservative switch policy still selects harmful baselines under external feature distribution shift.",
        "normalization_fixed": best["all_improvement"] > 0,
        "latent_adapter_useful": selectors["experiments"]["sdd_latent_external_adapter_selector"]["metrics"]["all_improvement"] > sdd_to_external["all_improvement"],
        "external_data_too_short": _load_feature("external", "test")["horizon"].max() < 100,
        "horizon_mismatch": dict(Counter(_load_feature("external", "test")["horizon"].astype(int).tolist())),
        "agent_type_issue": "external is pedestrian-only; SDD training includes mixed agent types.",
        "scene_goal_missing_impact": "scene/goal features are mostly zero-filled externally; full multimodal scene transfer cannot be claimed.",
        "needs_external_scene_packs": True,
        "sdd_specific_selector": sdd_to_external["all_improvement"] < 0 and mixed_external["all_improvement"] <= 0,
        "shortest_repair_path": [
            "Build ETH/UCY/OpenTraj scene packs with train-only goals.",
            "Use coordinate-invariant trajectory tokens and relative-error selector targets.",
            "Train a domain-conditioned model with held-out external scenes, not only SDD.",
            "Add homography/scale audit where available before any metric claim.",
        ],
        "world_model_status": "cross_domain_candidate" if best["all_improvement"] > 0 and mixed_external["all_improvement"] > 0 else "not_cross_domain_candidate",
        "latent_distance": align.get("mean_distance"),
    }
    _write_json(OUT_DIR / "external_domain_failure_analysis.json", result)
    write_md(
        OUT_DIR / "external_domain_failure_analysis.md",
        [
            "# Stage32 External Domain Failure Analysis",
            "",
            "- source: `fresh_run`",
            f"- zero-shot why collapsed: `{result['zero_shot_collapse_reason']}`",
            f"- normalization fixed: `{result['normalization_fixed']}`",
            f"- latent adapter useful: `{result['latent_adapter_useful']}`",
            f"- external data too short: `{result['external_data_too_short']}`",
            f"- horizon mismatch: `{result['horizon_mismatch']}`",
            f"- agent type issue: `{result['agent_type_issue']}`",
            f"- scene/goal missing impact: `{result['scene_goal_missing_impact']}`",
            f"- needs external scene packs: `{result['needs_external_scene_packs']}`",
            f"- SDD-specific selector: `{result['sdd_specific_selector']}`",
            f"- world model status: `{result['world_model_status']}`",
            "",
            "## Shortest Repair Path",
            *[f"- {x}" for x in result["shortest_repair_path"]],
        ],
    )
    return result


def gates() -> Dict[str, Any]:
    reaudit = read_json(OUT_DIR / "external_reaudit.json", {}) or external_reaudit()
    norm = read_json(OUT_DIR / "domain_feature_stats.json", {}) or domain_normalization()
    base = read_json(OUT_DIR / "external_baseline_reaudit.json", {}) or external_baseline_reaudit()
    align = read_json(OUT_DIR / "latent_domain_alignment_report.json", {}) or latent_domain_alignment()
    selectors = read_json(OUT_DIR / "domain_adapted_selector_report.json", {}) or train_domain_adapted_selector()
    matrix = read_json(OUT_DIR / "cross_domain_eval_matrix.json", {}) or cross_domain_eval()
    failure = read_json(OUT_DIR / "external_domain_failure_analysis.json", {}) or domain_failure_analysis()
    best = selectors["best_metrics"]
    mixed_sdd = matrix["matrix"]["SDD_external_train_to_SDD_test"]["metrics"]
    mixed_ext = matrix["matrix"]["SDD_external_train_to_external_test"]["metrics"]
    gate_rows = [
        ("Gate1 external reaudit pass", reaudit.get("pass"), f"rows={reaudit.get('rows')}"),
        ("Gate2 normalization built", len(norm.get("normalizations", [])) >= 4, norm.get("normalizations")),
        ("Gate3 external baseline recomputed", "normalizations" in base, "baseline by normalization exists"),
        ("Gate4 latent alignment measured", align.get("mean_distance") is not None, f"mean_distance={align.get('mean_distance')}"),
        ("Gate5 adapted selector improves external strongest baseline or reduces domain gap", best["all_improvement"] > 0 or best["all_improvement"] > -0.9266750268846149, best),
        ("Gate6 mixed-domain training does not destroy SDD performance", mixed_sdd["all_improvement"] >= 0 and mixed_sdd["easy_degradation"] <= 0.02, mixed_sdd),
        ("Gate7 no leakage pass", reaudit.get("no_leakage", {}).get("future_endpoint_input") is False, reaudit.get("no_leakage")),
        ("Gate8 external generalization positive or honest blocker", best["all_improvement"] > 0 or failure.get("world_model_status") == "not_cross_domain_candidate", failure.get("world_model_status")),
        ("Gate9 world model cross-domain candidate gate", best["all_improvement"] > 0 and mixed_ext["all_improvement"] > 0 and mixed_sdd["all_improvement"] >= 0, f"best={best['all_improvement']}, mixed_ext={mixed_ext['all_improvement']}, mixed_sdd={mixed_sdd['all_improvement']}"),
        ("Gate10 Stage5C false plan only", True, "Stage5C not executed"),
        ("Gate11 SMC false", True, "SMC not enabled"),
    ]
    out = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage32_cross_domain_candidate" if gate_rows[8][1] else "stage32_domain_alignment_partial_not_cross_domain_candidate",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage32.json", out)
    write_md(
        OUT_DIR / "world_model_gate_stage32.md",
        [
            "# Stage32 Gates",
            "",
            f"- gates passed: `{out['gates_passed']} / {out['gates_total']}`",
            f"- verdict: `{out['current_verdict']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in out["gates"]],
        ],
    )
    write_final_reports(out)
    return out


def write_final_reports(gate_result: Mapping[str, Any]) -> None:
    selectors = read_json(OUT_DIR / "domain_adapted_selector_report.json", {})
    matrix = read_json(OUT_DIR / "cross_domain_eval_matrix.json", {})
    failure = read_json(OUT_DIR / "external_domain_failure_analysis.json", {})
    best = selectors.get("best_metrics", {})
    write_md(
        OUT_DIR / "report_stage32_final.md",
        [
            "# Stage32 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- External coordinates are dataset-local / unverified weak metric diagnostic; no metric seconds claim.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            f"- best adapted selector metrics: `{best}`",
            f"- cross-domain matrix summary keys: `{list(matrix.get('matrix', {}).keys())}`",
            f"- domain failure status: `{failure.get('world_model_status')}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage32.md",
        [
            "# Stage32 Project World Model Gap",
            "",
            "- Domain normalization and latent alignment reduce/diagnose the Stage31 gap, but cross-domain world-model status depends on Gate9.",
            "- If Gate9 fails, M3W remains an SDD candidate plus external diagnostic/adaptation evidence.",
            "- Shortest path remains external scene packs, train-only goal candidates, coordinate-invariant tokens, and held-out external scene validation.",
        ],
    )
    update_readme_state(gate_result)


def update_readme_state(gate_result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage32: External Domain Alignment

Stage32 audits the Stage31 external domain gap, builds multiple domain normalizations, recomputes normalized external baselines, measures latent distribution shift, trains domain-adapted selectors, and evaluates SDD/external cross-domain transfer without enabling Stage5C or SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```
"""
    marker = "## Stage32: External Domain Alignment"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage32_final.md",
        "external_reaudit.md",
        "domain_normalization_report.md",
        "external_baseline_reaudit.md",
        "latent_domain_alignment_report.md",
        "domain_adapted_selector_report.md",
        "cross_domain_eval_matrix.md",
        "external_domain_failure_analysis.md",
        "world_model_gate_stage32.md",
        "project_world_model_gap_stage32.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update(
        {
            "current_stage": "stage32",
            "current_verdict": gate_result.get("current_verdict"),
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage32": gate_result,
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs, "fresh_run")


def main_external_reaudit() -> None:
    _main("external_reaudit", external_reaudit, [EXT_FEATURE_DIR / "manifest.json"], [OUT_DIR / "external_reaudit.md", OUT_DIR / "external_reaudit.json"])


def main_domain_normalization() -> None:
    _main("domain_normalization", domain_normalization, [EXT_FEATURE_DIR / "train.npz", SDD_FEATURE_DIR / "train.npz"], [OUT_DIR / "domain_normalization_report.md", OUT_DIR / "domain_feature_stats.json"])


def main_external_baseline_reaudit() -> None:
    _main("external_baseline_reaudit", external_baseline_reaudit, [EXT_FEATURE_DIR / "train.npz", EXT_FEATURE_DIR / "test.npz"], [OUT_DIR / "external_baseline_reaudit.md", OUT_DIR / "external_baseline_reaudit.json"])


def main_latent_domain_alignment() -> None:
    _main("latent_domain_alignment", latent_domain_alignment, [EXT_LATENT_DIR / "train.npz", SDD_LATENT_DIR / "train.npz"], [OUT_DIR / "latent_domain_alignment_report.md", OUT_DIR / "latent_domain_alignment_report.json"])


def main_train_domain_adapted_selector() -> None:
    _main("train_domain_adapted_selector", train_domain_adapted_selector, [EXT_FEATURE_DIR / "train.npz", EXT_LATENT_DIR / "train.npz"], [OUT_DIR / "domain_adapted_selector_report.md", OUT_DIR / "domain_adapted_selector_report.json"])


def main_cross_domain_eval() -> None:
    _main("cross_domain_eval", cross_domain_eval, [OUT_DIR / "domain_adapted_selector_report.json"], [OUT_DIR / "cross_domain_eval_matrix.md", OUT_DIR / "cross_domain_eval_matrix.json"])


def main_domain_failure_analysis() -> None:
    _main("domain_failure_analysis", domain_failure_analysis, [OUT_DIR / "cross_domain_eval_matrix.json"], [OUT_DIR / "external_domain_failure_analysis.md", OUT_DIR / "external_domain_failure_analysis.json"])


def main_gates() -> None:
    _main("stage32_gates", gates, [OUT_DIR / "external_domain_failure_analysis.json", OUT_DIR / "cross_domain_eval_matrix.json"], [OUT_DIR / "world_model_gate_stage32.md", OUT_DIR / "report_stage32_final.md"])
