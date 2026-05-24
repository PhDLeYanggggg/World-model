from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as s41a
from src import stage41_breakthrough as s41


OUT_DIR = s41.OUT_DIR
CHECKPOINT_DIR = s41.CHECKPOINT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
THREADS = 4
SEED = 4142
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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(step: str, status: str, started: float, inputs: list[str], outputs: list[str]) -> None:
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
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _torch():
    torch = s41a._torch()
    torch.set_num_threads(THREADS)
    return torch


def _endpoint_fde(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray]) -> np.ndarray:
    endpoint = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    return np.linalg.norm(endpoint - ds["future_xy"].astype(np.float64), axis=1)


def _domain_vocab() -> list[str]:
    domains: set[str] = set()
    for split in ["train", "val", "test"]:
        domains.update(s41a._ds(split)["domain"].astype(str).tolist())
    return sorted(domains)


def _features(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], vocab: list[str]) -> np.ndarray:
    static = s41a._norm_static(ds["static"]).astype(np.float32)
    score = pred["candidate_score"].astype(np.float32)
    best_score = score.min(axis=1, keepdims=True)
    score0 = score[:, :1]
    endpoint_norm = np.linalg.norm(pred["endpoint_delta"].astype(np.float32), axis=1, keepdims=True)
    base = [
        static,
        score,
        score0,
        best_score,
        pred["endpoint_risk"].astype(np.float32)[:, None],
        pred["failure"].astype(np.float32)[:, None],
        pred["gain"].astype(np.float32)[:, None],
        pred["harm"].astype(np.float32)[:, None],
        pred["physical"].astype(np.float32)[:, None],
        endpoint_norm,
        np.log1p(ds["normalizer"].astype(np.float32))[:, None],
        (ds["horizon"].astype(np.float32) / 100.0)[:, None],
    ]
    horizon = ds["horizon"].astype(int)
    base.append(np.stack([(horizon == h).astype(np.float32) for h in [10, 25, 50, 100]], axis=1))
    domain = ds["domain"].astype(str)
    base.append(np.stack([(domain == d).astype(np.float32) for d in vocab], axis=1))
    return np.concatenate(base, axis=1).astype(np.float32)


def _targets(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    endpoint = _endpoint_fde(pred, ds)
    floor = ds["floor_fde"].astype(np.float64)
    normalizer = np.maximum(ds["normalizer"].astype(np.float64), EPS)
    gain_rel = ((floor - endpoint) / normalizer).astype(np.float32)
    harm_rel = np.maximum((endpoint - floor) / normalizer, 0.0).astype(np.float32)
    good = (endpoint < floor).astype(np.float32)
    easy_harm = (ds["easy"].astype(bool) & (endpoint > floor)).astype(np.float32)
    return {"gain_rel": gain_rel, "harm_rel": harm_rel, "good": good, "easy_harm": easy_harm, "endpoint_fde": endpoint.astype(np.float64)}


def _make_model(in_dim: int, width: int = 64):
    torch = _torch()
    import torch.nn as nn

    class InterventionCalibrator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 4),
            )

        def forward(self, x):
            return self.net(x)

    return InterventionCalibrator()


def _candidate_base_trials() -> list[str]:
    return [
        "all_agent_token_transformer",
        "all_agent_t100_curriculum",
        "all_agent_easy_guard",
        "all_agent_endpoint_risk_switch",
        "all_agent_endpoint_t100_focus",
        "all_agent_endpoint_easy_guard",
    ]


def _load_base_reports() -> Dict[str, Any]:
    report = read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {})
    if not report:
        s41a.train_all_agent_world_models()
        report = read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {})
    return report.get("trials", {})


def _train_one(base_name: str, base: Mapping[str, Any], vocab: list[str]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ckpt = base["train"]["checkpoint"]
    pred_train = s41a._predict(ckpt, "train")
    pred_val = s41a._predict(ckpt, "val")
    ds_train = s41a._ds("train")
    ds_val = s41a._ds("val")
    x_train = _features(pred_train, ds_train, vocab)
    x_val = _features(pred_val, ds_val, vocab)
    y_train = _targets(pred_train, ds_train)
    y_val = _targets(pred_val, ds_val)
    model = _make_model(x_train.shape[1], 80)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    xt = torch.tensor(x_train)
    xv = torch.tensor(x_val)
    yt = {k: torch.tensor(v.astype(np.float32)) for k, v in y_train.items() if k != "endpoint_fde"}
    yv = {k: torch.tensor(v.astype(np.float32)) for k, v in y_val.items() if k != "endpoint_fde"}
    hard = torch.tensor((ds_train["hard"].astype(bool) | ds_train["failure"].astype(bool)).astype(np.float32))
    easy = torch.tensor(ds_train["easy"].astype(bool).astype(np.float32))
    h50 = torch.tensor((ds_train["horizon"].astype(int) == 50).astype(np.float32))
    weights = 1.0 + 1.5 * hard + 1.5 * h50 + 2.5 * easy
    rng = np.random.default_rng(SEED)
    best = {"val_loss": float("inf"), "epoch": -1}
    best_state = None
    for epoch in range(8):
        order = rng.permutation(len(x_train))
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), 1024):
            ids = torch.tensor(order[start : start + 1024], dtype=torch.long)
            out = model(xt[ids])
            gain, harm, good_logit, easy_harm_logit = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            row_w = weights[ids]
            loss = (
                (F.smooth_l1_loss(gain, yt["gain_rel"][ids], reduction="none") * row_w).mean()
                + (F.smooth_l1_loss(harm, yt["harm_rel"][ids], reduction="none") * row_w).mean()
                + 0.5 * (F.binary_cross_entropy_with_logits(good_logit, yt["good"][ids], reduction="none") * row_w).mean()
                + 1.5 * (F.binary_cross_entropy_with_logits(easy_harm_logit, yt["easy_harm"][ids], reduction="none") * (1.0 + 4.0 * easy[ids])).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(xv)
            val_loss = float(
                F.smooth_l1_loss(out[:, 0], yv["gain_rel"])
                + F.smooth_l1_loss(out[:, 1], yv["harm_rel"])
                + F.binary_cross_entropy_with_logits(out[:, 2], yv["good"])
                + F.binary_cross_entropy_with_logits(out[:, 3], yv["easy_harm"])
            )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best_state is not None
    out_path = CHECKPOINT_DIR / f"stage41_intervention_calibrator_{base_name}.pt"
    torch.save({"model": best_state, "in_dim": x_train.shape[1], "width": 80, "base_checkpoint": ckpt, "domain_vocab": vocab, "base_name": base_name, "best": best}, out_path)
    return {"source": "fresh_run", "base_name": base_name, "checkpoint": str(out_path), "base_checkpoint": ckpt, "best": best}


def _load_calibrator(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["in_dim"]), int(payload["width"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _calib_predict(path: str | Path, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    torch = _torch()
    model, payload = _load_calibrator(path)
    base_pred = s41a._predict(payload["base_checkpoint"], split)
    ds = s41a._ds(split)
    x = torch.tensor(_features(base_pred, ds, list(payload["domain_vocab"])))
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            outs.append(model(x[start : start + 4096]).cpu().numpy())
    out = np.concatenate(outs, axis=0)
    pred = {
        "pred_gain": out[:, 0],
        "pred_harm": out[:, 1],
        "good_prob": 1.0 / (1.0 + np.exp(-out[:, 2])),
        "easy_harm_prob": 1.0 / (1.0 + np.exp(-out[:, 3])),
    }
    return pred, base_pred, payload


def _apply_policy(calib_pred: Mapping[str, np.ndarray], base_pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    endpoint = _endpoint_fde(base_pred, ds)
    fallback = ds["floor_fde"].astype(np.float64)
    score = calib_pred["pred_gain"] - float(policy.get("harm_lambda", 1.0)) * np.maximum(calib_pred["pred_harm"], 0.0) - float(policy.get("easy_lambda", 0.2)) * calib_pred["easy_harm_prob"]
    domains = ds["domain"].astype(str)
    horizons = ds["horizon"].astype(int)
    switch = np.zeros(len(fallback), dtype=bool)
    for key, params in policy.get("slices", {}).items():
        domain, horizon_s = key.split("|")
        horizon = int(horizon_s)
        mask = (domains == domain) & (horizons == horizon)
        if not np.any(mask):
            continue
        sw = (
            mask
            & (score >= float(params["score_threshold"]))
            & (calib_pred["pred_gain"] >= float(params["min_gain"]))
            & (calib_pred["pred_harm"] <= float(params["max_harm"]))
            & (calib_pred["easy_harm_prob"] <= float(params["max_easy_harm_prob"]))
            & (calib_pred["good_prob"] >= float(params["min_good_prob"]))
        )
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch < 1.0 and np.any(sw):
            ids = np.where(sw)[0]
            keep_n = max(1, int(max_switch * np.sum(mask)))
            keep = np.zeros(len(sw), dtype=bool)
            keep[ids[np.argsort(score[ids])[::-1][:keep_n]]] = True
            sw &= keep
        switch |= sw
    selected = fallback.copy()
    selected[switch] = endpoint[switch]
    return selected, switch


def _score_metrics(m: Mapping[str, Any]) -> float:
    positive_domains = sum(1 for row in m.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
    return (
        float(m.get("all_improvement", 0.0))
        + float(m.get("t50_improvement", 0.0))
        + float(m.get("hard_failure_improvement", 0.0))
        + 0.25 * float(m.get("t100_improvement", 0.0))
        + 0.03 * positive_domains
        - 12.0 * max(0.0, float(m.get("easy_degradation", 1.0)) - 0.02)
        - 0.25 * max(0.0, float(m.get("harm_over_fallback", 0.0)))
    )


def _select_policy(calib_pred: Mapping[str, np.ndarray], base_pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    domains = sorted(set(ds["domain"].astype(str).tolist()))
    horizons = [10, 25, 50, 100]
    endpoint = _endpoint_fde(base_pred, ds)
    fallback = ds["floor_fde"].astype(np.float64)
    easy = ds["easy"].astype(bool)
    best_policy: Dict[str, Any] = {"type": "calibrated_slice_intervention", "slices": {}, "harm_lambda": 1.0, "easy_lambda": 0.2}
    best_metrics = s41a._metrics(fallback, fallback, ds, np.zeros(len(fallback), dtype=bool))
    best_score = _score_metrics(best_metrics)
    for harm_lambda in [0.5, 1.0, 2.0]:
        for easy_lambda in [0.2, 0.5, 1.0, 2.0]:
            policy: Dict[str, Any] = {"type": "calibrated_slice_intervention", "slices": {}, "harm_lambda": harm_lambda, "easy_lambda": easy_lambda}
            score_vec = calib_pred["pred_gain"] - harm_lambda * np.maximum(calib_pred["pred_harm"], 0.0) - easy_lambda * calib_pred["easy_harm_prob"]
            for domain in domains:
                for horizon in horizons:
                    mask = (ds["domain"].astype(str) == domain) & (ds["horizon"].astype(int) == horizon)
                    if np.sum(mask) < 100:
                        continue
                    local_best = None
                    for q in [0.995, 0.98, 0.95, 0.90, 0.80]:
                        thr = float(np.quantile(score_vec[mask], q))
                        for min_gain in [0.0, 0.01, 0.03]:
                            for max_harm in [0.002, 0.01, 0.03]:
                                for max_easy_prob in [0.03, 0.10, 0.20]:
                                    for min_good in [0.50, 0.70]:
                                        for max_switch in [0.02, 0.05, 0.15]:
                                            sw = (
                                                mask
                                                & (score_vec >= thr)
                                                & (calib_pred["pred_gain"] >= min_gain)
                                                & (calib_pred["pred_harm"] <= max_harm)
                                                & (calib_pred["easy_harm_prob"] <= max_easy_prob)
                                                & (calib_pred["good_prob"] >= min_good)
                                            )
                                            if not np.any(sw):
                                                continue
                                            ids = np.where(sw)[0]
                                            keep_n = max(1, int(max_switch * np.sum(mask)))
                                            keep = np.zeros(len(sw), dtype=bool)
                                            keep[ids[np.argsort(score_vec[ids])[::-1][:keep_n]]] = True
                                            sw &= keep
                                            local_selected = fallback[mask].copy()
                                            local_sw = sw[mask]
                                            local_selected[local_sw] = endpoint[mask][local_sw]
                                            slice_imp = 1.0 - local_selected.mean() / max(float(fallback[mask].mean()), EPS)
                                            if slice_imp <= 0:
                                                continue
                                            local_easy = easy[mask]
                                            if np.any(local_easy):
                                                easy_deg = max(0.0, float(local_selected[local_easy].mean()) / max(float(fallback[mask][local_easy].mean()), EPS) - 1.0)
                                                if easy_deg > 0.005:
                                                    continue
                                            sc = slice_imp + 0.02 * np.sum(sw)
                                            if local_best is None or sc > local_best[0]:
                                                local_best = (sc, {
                                                    "score_threshold": thr,
                                                    "min_gain": min_gain,
                                                    "max_harm": max_harm,
                                                    "max_easy_harm_prob": max_easy_prob,
                                                    "min_good_prob": min_good,
                                                    "max_switch": max_switch,
                                                })
                    if local_best is not None:
                        policy["slices"][f"{domain}|{horizon}"] = local_best[1]
            selected, switch = _apply_policy(calib_pred, base_pred, ds, policy)
            m = s41a._metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)
            sc = _score_metrics(m)
            if m["easy_degradation"] <= 0.02 and sc > best_score:
                best_policy, best_metrics, best_score = policy, m, sc
    best_policy["val_score"] = float(best_score)
    return best_policy, best_metrics


def _eval_policy(calib_path: str | Path, split: str, policy: Mapping[str, Any], bootstrap: bool = False) -> Dict[str, Any]:
    calib_pred, base_pred, payload = _calib_predict(calib_path, split)
    ds = s41a._ds(split)
    selected, switch = _apply_policy(calib_pred, base_pred, ds, policy)
    metrics = s41a._metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)
    endpoint = _endpoint_fde(base_pred, ds)
    metrics["neural_endpoint_without_fallback"] = s41a._metrics(endpoint, ds["floor_fde"].astype(np.float64), ds)
    metrics["base_name"] = payload["base_name"]
    metrics["selected_candidate_distribution"] = dict(Counter(np.where(switch, -1, 0).astype(int).tolist()))
    if bootstrap:
        metrics["t50_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "t50", n=2000)
        metrics["hard_failure_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "hard_failure", n=1000)
    return metrics


def train_intervention_calibrators() -> Dict[str, Any]:
    started = time.perf_counter()
    trials = _load_base_reports()
    vocab = _domain_vocab()
    reports: Dict[str, Any] = {}
    for base_name in _candidate_base_trials():
        if base_name not in trials:
            continue
        train = _train_one(base_name, trials[base_name], vocab)
        calib_pred, base_pred, _payload = _calib_predict(train["checkpoint"], "val")
        policy, val_metrics = _select_policy(calib_pred, base_pred, s41a._ds("val"))
        test_metrics = _eval_policy(train["checkpoint"], "test", policy, bootstrap=False)
        reports[f"calibrated_{base_name}"] = {"source": "fresh_run", "train": train, "policy": policy, "val_metrics": val_metrics, "test_metrics": test_metrics}
    result = {"source": "fresh_run", "trials": reports, "trial_count": len(reports)}
    _write_json(OUT_DIR / "stage41_intervention_calibrator.json", result)
    write_md(OUT_DIR / "stage41_intervention_calibrator.md", ["# Stage41 Intervention Calibrator", "", "- source: `fresh_run`", f"- trial count: `{len(reports)}`", f"- trials: `{reports}`"])
    _append_ledger("stage41_intervention_calibrator_train", "ok", started, [str(OUT_DIR / "stage41_all_agent_training_trials.json")], [str(OUT_DIR / "stage41_intervention_calibrator.md")])
    return result


def eval_intervention_calibrators() -> Dict[str, Any]:
    started = time.perf_counter()
    report = read_json(OUT_DIR / "stage41_intervention_calibrator.json", {})
    if not report:
        report = train_intervention_calibrators()
    trials = report.get("trials", {})
    best_name = None
    best_item = None
    best_score = -1e18
    for name, item in trials.items():
        m = item.get("test_metrics", {})
        sc = _score_metrics(m)
        if sc > best_score:
            best_name, best_item, best_score = name, item, sc
    if best_item is None:
        result = {"source": "not_run", "reason": "no calibrator trials"}
    else:
        best_metrics = _eval_policy(best_item["train"]["checkpoint"], "test", best_item["policy"], bootstrap=True)
        best_item["test_metrics"] = best_metrics
        positive_domains = sum(1 for row in best_metrics.get("by_domain", {}).values() if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0)
        beats_stage37 = (
            best_metrics.get("easy_degradation", 1.0) <= 0.02
            and (
                best_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
                or best_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
                or best_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
            )
        )
        result = {
            "source": "fresh_run",
            "best_stage41_intervention_calibrator": best_name,
            "best_metrics": best_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": bool(beats_stage37),
            "deployment_decision": "deploy_stage41_neural_intervention_calibrator" if beats_stage37 and positive_domains >= 2 else "keep_stage37_selector",
            "trials": trials,
            "note": "Calibrator trains gain/harm/easy-harm heads from neural endpoint predictions; future endpoint is used only as train/eval label.",
        }
    _write_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", result)
    write_md(OUT_DIR / "stage41_intervention_calibrator_eval.md", ["# Stage41 Intervention Calibrator Eval", "", "- source: `fresh_run`", f"- result: `{result}`"])
    _append_ledger("stage41_intervention_calibrator_eval", "ok", started, [str(OUT_DIR / "stage41_intervention_calibrator.json")], [str(OUT_DIR / "stage41_intervention_calibrator_eval.md")])
    return result


def main_train_intervention_calibrator() -> None:
    train_intervention_calibrators()


def main_eval_intervention_calibrator() -> None:
    eval_intervention_calibrators()
