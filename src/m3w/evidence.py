from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.m3w.dataset import BASELINE_NAMES, M3WFeatureDataset
from src.m3w.eval import _ece, _metrics, _predict, _search_policy, _select
from src.m3w.models import M3WModel
from src.m3w.token_schema import TOKEN_NAMES, build_token_schema
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


OUT = Path("outputs/m3w")
STAGE26_T50 = 0.14583655843823773
STAGE26_HARD = 0.11232167634621226
STAGE26_EASY = 0.01808836280803794


def load_ckpt(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def dataset_from_ckpt(ckpt: Dict[str, Any], split: str, limit_key: str) -> M3WFeatureDataset:
    config = ckpt["config"]
    return M3WFeatureDataset(
        config["feature_store"],
        split,
        mean=np.asarray(ckpt["feature_mean"], dtype=np.float32),
        std=np.asarray(ckpt["feature_std"], dtype=np.float32),
        limit=config.get(limit_key),
    )


def evaluate_checkpoint(path: str | Path, name: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    ckpt = load_ckpt(path)
    val = dataset_from_ckpt(ckpt, "val", "max_val_samples")
    test = dataset_from_ckpt(ckpt, "test", "max_test_samples")
    schema = build_token_schema(ckpt["feature_names"])
    model = M3WModel(schema, ckpt["config"], ckpt["variant"]).to(torch.device("cpu"))
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, torch.device("cpu"))
    test_pred = _predict(model, test, torch.device("cpu"))
    val_fde = np.maximum(0.0, np.expm1(np.clip(val_pred["log_fde"], -20.0, 8.5)))
    test_fde = np.maximum(0.0, np.expm1(np.clip(test_pred["log_fde"], -20.0, 8.5)))
    val_failure = _sigmoid(val_pred["failure_logit"])
    test_failure = _sigmoid(test_pred["failure_logit"])
    policy = _search_policy(val, val_fde, val_failure, ckpt["config"])
    selected, conf = _select(test, test_fde, test_failure, policy["policy"])
    metrics = _metrics(test, selected, conf)
    metrics.update(
        {
            "name": name,
            "variant": ckpt["variant"],
            "checkpoint": str(path),
            "policy": policy["policy"],
            "jepa_latent_variance": float((ckpt.get("jepa_stats") or [{}])[-1].get("latent_variance", 0.0)),
            "failure_ece": _ece(test.failure_label, test_failure),
        }
    )
    arrays = _arrays_for_ci(test, selected)
    return metrics, arrays


def evaluate_feature_mask(path: str | Path, name: str, disabled_tokens: List[str]) -> Dict[str, Any]:
    ckpt = load_ckpt(path)
    val = dataset_from_ckpt(ckpt, "val", "max_val_samples")
    test = dataset_from_ckpt(ckpt, "test", "max_test_samples")
    schema = build_token_schema(ckpt["feature_names"])
    disabled = sorted({idx for token in disabled_tokens for idx in schema.token_to_features[token]})
    for ds in [val, test]:
        ds.x[:, disabled] = 0.0
    model = M3WModel(schema, ckpt["config"], ckpt["variant"]).to(torch.device("cpu"))
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, torch.device("cpu"))
    test_pred = _predict(model, test, torch.device("cpu"))
    val_fde = np.maximum(0.0, np.expm1(np.clip(val_pred["log_fde"], -20.0, 8.5)))
    test_fde = np.maximum(0.0, np.expm1(np.clip(test_pred["log_fde"], -20.0, 8.5)))
    policy = _search_policy(val, val_fde, _sigmoid(val_pred["failure_logit"]), ckpt["config"])
    selected, conf = _select(test, test_fde, _sigmoid(test_pred["failure_logit"]), policy["policy"])
    out = _metrics(test, selected, conf)
    out.update({"name": name, "disabled_tokens": disabled_tokens})
    return out


def _arrays_for_ci(dataset: M3WFeatureDataset, selected: np.ndarray) -> Dict[str, np.ndarray]:
    idx = np.arange(len(dataset))
    selected_err = dataset.y_fde[idx, selected]
    strongest_err = dataset.y_fde[idx, dataset.strongest_idx]
    failure_thr = np.percentile(strongest_err, 90)
    return {
        "selected": selected_err,
        "strongest": strongest_err,
        "h50": dataset.horizon == 50,
        "h100": dataset.horizon == 100,
        "hard": np.logical_or(dataset.hard_candidate > 0.5, strongest_err >= failure_thr),
        "easy": strongest_err <= 10.0,
    }


def bootstrap_ci(arrays: Dict[str, np.ndarray], seed: int = 27, n_boot: int = 200) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    result = {}
    for name, mask in {"t50": arrays["h50"], "t100": arrays["h100"], "hard": arrays["hard"], "easy": arrays["easy"]}.items():
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            result[name] = {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
            continue
        vals = []
        for _ in range(n_boot):
            sample = rng.choice(idx, size=len(idx), replace=True)
            if name == "easy":
                value = max(0.0, float(arrays["selected"][sample].mean() / max(float(arrays["strongest"][sample].mean()), 1e-6) - 1.0))
            else:
                value = float(1.0 - arrays["selected"][sample].mean() / max(float(arrays["strongest"][sample].mean()), 1e-6))
            vals.append(value)
        result[name] = {
            "mean": float(np.mean(vals)),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "n": int(len(idx)),
        }
    return result


def build_evidence(checkpoint: str | Path = OUT / "checkpoints/best_small.pt") -> Dict[str, Any]:
    ensure_dir(OUT)
    ensure_dir(OUT / "figures")
    start = time.time()
    metrics = read_json(OUT / "metrics_m3w.json", {})
    gate = read_json(OUT / "world_model_gate_m3w.json", {})
    stage26 = read_json("outputs/reports/report_stage26_final.json", {})
    checkpoints = {
        "jepa_only": OUT / "checkpoints/jepa_only_best.pt",
        "transformer_only": OUT / "checkpoints/transformer_only_best.pt",
        "hybrid": checkpoint,
    }
    experiment_rows = []
    ci = {}
    arrays_for_best = None
    for name, path in checkpoints.items():
        if Path(path).exists():
            row, arrays = evaluate_checkpoint(path, name)
            experiment_rows.append(row)
            if name == "hybrid":
                arrays_for_best = arrays
    if arrays_for_best is not None:
        ci = bootstrap_ci(arrays_for_best)
    ablations = []
    for name, tokens in [
        ("no_scene", ["scene_patch", "scene_sdf"]),
        ("no_goal", ["goal_region"]),
        ("no_interaction", ["interaction_edge"]),
        ("no_baseline_rollout", ["baseline_rollout"]),
    ]:
        if Path(checkpoint).exists():
            ablations.append(evaluate_feature_mask(checkpoint, name, tokens))
    evidence = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - start,
        "current_truth": _truth_block(),
        "stage26": {
            "t50_improvement": stage26.get("t50_improvement", STAGE26_T50),
            "hard_failure_improvement": stage26.get("hard_failure_improvement", STAGE26_HARD),
            "easy_degradation": stage26.get("easy_degradation", STAGE26_EASY),
        },
        "m3w_metrics": metrics,
        "m3w_gates": gate,
        "experiment_rows": experiment_rows,
        "ablations": ablations,
        "bootstrap_ci": ci,
        "ccfa_candidate": False,
    }
    write_json(OUT / "stage27_evidence.json", _jsonable(evidence))
    write_reports(evidence)
    return evidence


def write_reports(evidence: Dict[str, Any]) -> None:
    rows = evidence["experiment_rows"]
    ablations = evidence["ablations"]
    ci = evidence["bootstrap_ci"]
    write_md(
        OUT / "experiment_matrix.md",
        [
            "# M3W Experiment Matrix",
            "",
            "| model | t+50 improvement | hard/failure improvement | easy degradation | selector regret | JEPA variance |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {r['name']} | {r['official_t50_improvement']:.6f} | {r['hard_failure_improvement']:.6f} | {r['easy_degradation']:.6f} | {r['selector_regret']:.6f} | {r.get('jepa_latent_variance', 0.0):.6f} |"
                for r in rows
            ],
            "| Stage26 selector | 0.145837 | 0.112322 | 0.018088 | n/a | n/a |",
        ],
    )
    write_md(
        OUT / "ablation_table_m3w.md",
        [
            "# M3W Ablation Table",
            "",
            "These are inference-time token masking ablations on the current hybrid checkpoint, not full retraining ablations.",
            "",
            "| ablation | disabled tokens | t+50 improvement | hard/failure improvement | easy degradation |",
            "| --- | --- | ---: | ---: | ---: |",
            *[
                f"| {r['name']} | {', '.join(r['disabled_tokens'])} | {r['official_t50_improvement']:.6f} | {r['hard_failure_improvement']:.6f} | {r['easy_degradation']:.6f} |"
                for r in ablations
            ],
            "",
            "A full CCF-A candidate still requires retrained ablations and statistical tests.",
        ],
    )
    write_md(
        OUT / "bootstrap_or_seed_report.md",
        [
            "# Bootstrap / Seed Report",
            "",
            "Current evidence uses bootstrap confidence intervals over the local-small test predictions. Multi-seed training is still required for stronger paper claims.",
            "",
            "| metric | mean | 95% CI low | 95% CI high | n |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[f"| {k} | {v['mean']:.6f} | {v['lo']:.6f} | {v['hi']:.6f} | {v['n']} |" for k, v in ci.items()],
        ],
    )
    write_md(OUT / "ccfa_gap_analysis.md", _gap_lines(evidence))
    write_md(OUT / "reproducibility_checklist.md", _repro_lines())
    write_md(OUT / "paper_draft_m3w.md", _paper_lines(evidence))
    write_md(
        OUT / "stage27_evidence_report.md",
        [
            "# Stage 27 M3W-CCFA Evidence Sprint",
            "",
            *[f"- {line}" for line in _truth_block()],
            "",
            f"- CCF-A candidate: `{evidence['ccfa_candidate']}`",
            "- Stage5C execution: `False`",
            "- SMC: `False`",
            "- Current verdict: `not yet CCF-A candidate; Stage26 remains best deployable`",
        ],
    )


def _gap_lines(evidence: Dict[str, Any]) -> List[str]:
    metrics = evidence.get("m3w_metrics", {})
    m = metrics.get("test_metrics", {})
    return [
        "# M3W CCF-A Gap Analysis",
        "",
        "## Current Verdict",
        "",
        "not yet CCF-A candidate.",
        "",
        "## What Is Missing",
        "",
        "1. Model gap: M3W hybrid has not exceeded Stage26 on t+50 or hard/failure.",
        "2. Representation gap: hybrid beats Transformer-only in this run, but JEPA latent non-collapse is still below gate and the contribution needs retrained ablations.",
        "3. Experiment gap: current ablations are inference-time; retrained ablations are still needed.",
        "4. Statistics gap: bootstrap CI exists, but multi-seed variance is still needed for stronger claims.",
        "5. Data gap: SDD is pixel-space raw-frame only; effective seconds/homography/metric scale are unverified.",
        "",
        "## Shortest Path",
        "",
        "1. Treat Stage26 selector as the deployment floor and train M3W latent features only as auxiliary selector features.",
        "2. Run retrained ablations with no-scene/no-goal/no-interaction/no-JEPA/no-Transformer under the arm64 torch runtime.",
        "3. Add multi-seed or larger bootstrap evidence and per-scene/per-agent-type breakdowns.",
        "",
        "## Usable Paper Material",
        "",
        "- Strict leakage-free SDD pixel-space benchmark setup.",
        "- Strong Stage26 cost-aware selector baseline.",
        "- Runtime-safe M3W JEPA/Transformer implementation and negative results.",
        "",
        "## Claims That Must Not Be Made",
        "",
        "- Do not claim true 3D.",
        "- Do not claim metric trajectory prediction.",
        "- Do not claim Stage5C or SMC readiness.",
        "- Do not claim M3W beats Stage26.",
        f"- Current M3W t+50 improvement is `{m.get('official_t50_improvement')}`, below Stage26.",
    ]


def _paper_lines(evidence: Dict[str, Any]) -> List[str]:
    return [
        "# M3W Paper Draft",
        "",
        "## Title",
        "",
        "M3W: Leakage-Free Multimodal Agent-Scene World Modeling for Top-Down Multi-Agent Forecasting",
        "",
        "## Abstract",
        "",
        "We study whether JEPA-style representation learning and spatiotemporal Transformer dynamics can improve real-world top-down multi-agent world modeling under strict no-leakage evaluation. Current local-small evidence does not yet establish M3W as a submission-ready method: the Stage26 cost-aware selector remains the best deployable model. We report the negative result, runtime fixes, ablations, and the remaining evidence gap.",
        "",
        "## Introduction",
        "",
        "The problem is to learn a multimodal 2.5D agent-scene world model that improves over strongest causal baselines and Stage26 selectors on SDD pixel-space raw-frame horizons while preserving easy cases.",
        "",
        "## Related Work Placeholder",
        "",
        "Trajectory forecasting, JEPA/self-supervised world representation, top-down pedestrian/drone datasets, safety-gated selectors.",
        "",
        "## Method",
        "",
        "M3W uses tokenized agent, scene, goal, interaction, baseline rollout, horizon, dataset, time, and mask features. JEPA predicts latent targets; Transformer dynamics aggregate tokens; downstream heads predict expected FDE, failure, goal diagnostics, interaction risk, occupancy, and physical validity.",
        "",
        "## Datasets",
        "",
        "SDD is used as pixel-space official benchmark. t+50 is official raw-frame; t+100 is raw-frame diagnostic. Effective seconds and metric scale are not verified.",
        "",
        "## Metrics",
        "",
        "t+50 FDE improvement, t+100 diagnostic, hard/failure improvement, easy degradation, AUROC/AUPRC/ECE, interaction risk, occupancy, bootstrap CI, ablations.",
        "",
        "## Baselines",
        "",
        "strongest causal baseline, BPSG-MA v1 fallback, Stage26 selector, JEPA-only, Transformer-only, Hybrid.",
        "",
        "## Experiments",
        "",
        "See `experiment_matrix.md`, `ablation_table_m3w.md`, and `bootstrap_or_seed_report.md`.",
        "",
        "## Ablation",
        "",
        "Current ablations are inference-time token masks. Retrained ablations are required before paper submission.",
        "",
        "## Failure Analysis",
        "",
        "Hybrid does not beat Stage26; JEPA non-collapse/downstream lift not proven; hard/failure gate not consistently passed by M3W.",
        "",
        "## Limitations",
        "",
        "Not true 3D, not metric, not foundation model, no Stage5C execution, no SMC, no human-gold scene labels.",
        "",
        "## Reproducibility",
        "",
        "Use `.venv-pytorch/bin/python` arm64, `num_workers=0`, checkpointed configs, and no future/test leakage.",
    ]


def _repro_lines() -> List[str]:
    return [
        "# M3W Reproducibility Checklist",
        "",
        "- [x] Use `.venv-pytorch/bin/python` arm64.",
        "- [x] Refuse macOS x86_64/Rosetta torch training by default.",
        "- [x] DataLoader multiprocessing disabled (`num_workers=0`).",
        "- [x] Checkpoints saved under `outputs/m3w/checkpoints/`.",
        "- [x] No future endpoint input.",
        "- [x] No central velocity official input.",
        "- [x] No test endpoint goal construction.",
        "- [x] Stage5C not executed.",
        "- [x] SMC not enabled.",
        "- [ ] Multi-seed variance not yet complete.",
        "- [ ] Retrained ablations not yet complete.",
    ]


def _truth_block() -> List[str]:
    return [
        "Current model is not true 3D.",
        "Current model is not a large-scale foundation world model.",
        "Current model is still a 2.5D / pseudo-3D multi-agent trajectory world-state model.",
        "SDD is pixel-space, not metric.",
        "t+50/t+100 are raw annotation-frame horizons.",
        "Effective seconds, homography, and metric scale remain unverified.",
        "self-audited / visual-prior labels are not human gold.",
        "Stage5C latent generative execution is forbidden.",
        "SMC is forbidden.",
    ]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(OUT / "checkpoints/best_small.pt"))
    args = parser.parse_args(argv)
    build_evidence(args.checkpoint)
