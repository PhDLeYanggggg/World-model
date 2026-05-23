from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.m3w.dataset import BASELINE_NAMES, M3WFeatureDataset
from src.m3w.models import M3WModel
from src.m3w.token_schema import build_token_schema
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


def _load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def _dataset_from_checkpoint(ckpt: Dict[str, Any], split: str, limit_key: str) -> M3WFeatureDataset:
    config = ckpt["config"]
    return M3WFeatureDataset(
        config["feature_store"],
        split,
        mean=np.asarray(ckpt["feature_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_std"], dtype=np.float32),
        limit=config.get(limit_key),
    )


@torch.no_grad()
def _predict(model: M3WModel, dataset: M3WFeatureDataset, device: torch.device) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
    outs: Dict[str, List[np.ndarray]] = defaultdict_list()
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        for key, value in out.items():
            outs[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(vals, axis=0) for key, vals in outs.items()}


def defaultdict_list() -> Dict[str, List[np.ndarray]]:
    return {"log_fde": [], "failure_logit": [], "goal_logits": [], "interaction_logit": [], "occupancy": [], "validity_logit": []}


def _select(dataset: M3WFeatureDataset, pred_fde: np.ndarray, failure_prob: np.ndarray, policy: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    selected = []
    confs = []
    for i in range(len(dataset)):
        strong = int(dataset.strongest_idx[i])
        order = np.argsort(pred_fde[i])
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        gain = float(pred_fde[i, strong] - pred_fde[i, best])
        conf = float((pred_fde[i, second] - pred_fde[i, best]) / max(pred_fde[i, strong], 1e-6))
        fallback = best == strong or gain < policy["gain_threshold"] or conf < policy["confidence_threshold"] or failure_prob[i] < policy["failure_threshold"]
        selected.append(strong if fallback else best)
        confs.append(conf)
    return np.asarray(selected, dtype=np.int64), np.asarray(confs, dtype=np.float32)


def _metrics(dataset: M3WFeatureDataset, selected: np.ndarray, confs: np.ndarray) -> Dict[str, Any]:
    y = dataset.y_fde
    idx = np.arange(len(dataset))
    selected_err = y[idx, selected]
    strongest_err = y[idx, dataset.strongest_idx]
    oracle_err = y[idx, np.argmin(y, axis=1)]
    easy = strongest_err <= 10.0
    failure_thr = np.percentile(strongest_err, 90)
    hard = np.logical_or(dataset.hard_candidate > 0.5, strongest_err >= failure_thr)
    h50 = dataset.horizon == 50
    h100 = dataset.horizon == 100

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - selected_err[mask].mean() / max(float(strongest_err[mask].mean()), 1e-6))

    easy_deg = float(max(0.0, selected_err[easy].mean() / max(float(strongest_err[easy].mean()), 1e-6) - 1.0)) if np.any(easy) else 0.0
    return {
        "improvement_over_strongest": imp(np.ones(len(dataset), dtype=bool)),
        "official_t50_improvement": imp(h50),
        "diagnostic_t100_raw_frame_improvement": imp(h100),
        "hard_failure_improvement": imp(hard),
        "easy_degradation": easy_deg,
        "selector_regret": float(np.mean(selected_err - oracle_err)),
        "harm_over_fallback": float(np.mean(selected_err - strongest_err)),
        "switch_rate": float(np.mean(selected != dataset.strongest_idx)),
        "mean_confidence": float(np.mean(confs)),
        "selected_distribution": {BASELINE_NAMES[i]: int((selected == i).sum()) for i in range(len(BASELINE_NAMES))},
    }


def _search_policy(dataset: M3WFeatureDataset, pred_fde: np.ndarray, failure_prob: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    search = config.get("fallback_search", {})
    best = None
    for conf in search.get("confidence_thresholds", [0.0, 0.05, 0.1]):
        for gain in search.get("gain_thresholds", [0.0, 2.0, 5.0]):
            for fail in search.get("failure_thresholds", [0.0, 0.1]):
                policy = {"confidence_threshold": float(conf), "gain_threshold": float(gain), "failure_threshold": float(fail)}
                selected, confs = _select(dataset, pred_fde, failure_prob, policy)
                met = _metrics(dataset, selected, confs)
                objective = met["official_t50_improvement"] + 0.5 * met["hard_failure_improvement"] - 5.0 * max(0.0, met["easy_degradation"] - 0.02)
                if best is None or objective > best["objective"]:
                    best = {"policy": policy, "metrics": met, "objective": objective}
    assert best is not None
    return best


def evaluate_m3w(checkpoint: str | Path) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint)
    config = ckpt["config"]
    out_dir = ensure_dir(config.get("output_dir", "outputs/m3w"))
    device = torch.device("cpu")
    val = _dataset_from_checkpoint(ckpt, "val", "max_val_samples")
    test = _dataset_from_checkpoint(ckpt, "test", "max_test_samples")
    schema = build_token_schema(ckpt["feature_names"])
    model = M3WModel(schema, config, ckpt["variant"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, device)
    test_pred = _predict(model, test, device)
    val_fde = np.maximum(0.0, np.expm1(val_pred["log_fde"]))
    test_fde = np.maximum(0.0, np.expm1(test_pred["log_fde"]))
    val_failure = 1.0 / (1.0 + np.exp(-val_pred["failure_logit"]))
    test_failure = 1.0 / (1.0 + np.exp(-test_pred["failure_logit"]))
    selected_policy = _search_policy(val, val_fde, val_failure, config)
    test_selected, test_conf = _select(test, test_fde, test_failure, selected_policy["policy"])
    test_metrics = _metrics(test, test_selected, test_conf)
    failure_labels = test.failure_label
    failure_auroc = float(roc_auc_score(failure_labels, test_failure)) if len(set(failure_labels.tolist())) > 1 else 0.5
    failure_auprc = float(average_precision_score(failure_labels, test_failure)) if len(set(failure_labels.tolist())) > 1 else float(np.mean(failure_labels))
    interaction_labels = test.hard_candidate
    interaction_score = 1.0 / (1.0 + np.exp(-test_pred["interaction_logit"]))
    interaction_auroc = float(roc_auc_score(interaction_labels, interaction_score)) if len(set(interaction_labels.tolist())) > 1 else 0.5
    occupancy_mse = float(np.mean((test_pred["occupancy"] - test.occupancy_target) ** 2))
    stage26 = read_json("outputs/reports/report_stage26_final.json", {})
    jepa_stats = ckpt.get("jepa_stats", [])
    latent_variance = float(jepa_stats[-1].get("latent_variance", 0.0)) if jepa_stats else 0.0
    result = {
        "checkpoint": str(checkpoint),
        "variant": ckpt["variant"],
        "selected_policy": selected_policy["policy"],
        "validation_metrics": selected_policy["metrics"],
        "test_metrics": test_metrics,
        "stage26_selector": {
            "t50_improvement": stage26.get("t50_improvement"),
            "hard_failure_improvement": stage26.get("hard_failure_improvement"),
            "easy_degradation": stage26.get("easy_degradation"),
        },
        "beats_stage26_selector": test_metrics["official_t50_improvement"] > float(stage26.get("t50_improvement", 1e9) or 1e9),
        "failure_AUROC": failure_auroc,
        "failure_AUPRC": failure_auprc,
        "failure_ECE": _ece(failure_labels, test_failure),
        "interaction_AUROC": interaction_auroc,
        "occupancy_MSE": occupancy_mse,
        "jepa_latent_variance": latent_variance,
        "jepa_non_collapse": latent_variance > 0.01 if jepa_stats else False,
        "goal_metrics": "diagnostic_unavailable_no_human_goal_labels",
        "physical_validity": "selected physical baseline only; no residual/correction",
        "t100_status": "raw-frame diagnostic, not seconds-level",
        "metric_status": "pixel-space only",
    }
    write_json(Path(out_dir) / "metrics_m3w.json", result)
    write_md(
        Path(out_dir) / "metrics_m3w.md",
        [
            "# M3W Metrics",
            "",
            f"- variant: `{result['variant']}`",
            f"- t+50 improvement: `{test_metrics['official_t50_improvement']}`",
            f"- hard/failure improvement: `{test_metrics['hard_failure_improvement']}`",
            f"- easy degradation: `{test_metrics['easy_degradation']}`",
            f"- beats Stage26 selector: `{result['beats_stage26_selector']}`",
            f"- failure AUROC/AUPRC/ECE: `{failure_auroc}` / `{failure_auprc}` / `{result['failure_ECE']}`",
            f"- interaction AUROC: `{interaction_auroc}`",
            f"- JEPA non-collapse: `{result['jepa_non_collapse']}`",
        ],
    )
    return result


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (probs >= lo) & (probs < hi)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    evaluate_m3w(args.checkpoint)
