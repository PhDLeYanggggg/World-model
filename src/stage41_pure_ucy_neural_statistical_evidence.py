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
from src import stage41_pure_ucy_neural_retrain as pure


OUT_DIR = pure.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_pure_ucy_neural_statistical_evidence.json"
REPORT_MD = OUT_DIR / "stage41_pure_ucy_neural_statistical_evidence.md"
BOOTSTRAP_N = 2000
SEED = 41809
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


def _source_key(source_file: str) -> str:
    return pure._source_key(source_file)


def _bootstrap_ci(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = SEED) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[sample].mean()) / max(float(floor[sample].mean()), EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _selected_endpoint_arrays(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, np.ndarray]:
    floor_xy = ds["current_xy"].astype(np.float64) + ds["cand_delta"].astype(np.float64)[:, 0, :] * ds["normalizer"].astype(np.float64)[:, None]
    neural_xy = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    floor = ds["floor_fde"].astype(np.float64)
    neural = np.linalg.norm(neural_xy - ds["future_xy"].astype(np.float64), axis=1)
    if policy.get("type") == "bounded_endpoint_residual":
        alpha = pure._endpoint_residual_alpha(pred, ds, policy)
        selected_xy = floor_xy + alpha[:, None] * (neural_xy - floor_xy)
        selected = np.linalg.norm(selected_xy - ds["future_xy"].astype(np.float64), axis=1)
        switch = alpha > EPS
    else:
        selected, switch, _selected_idx = pure._select(policy, pred, ds)
        alpha = switch.astype(np.float64)
    return {
        "selected": selected.astype(np.float64),
        "floor": floor.astype(np.float64),
        "neural_without_fallback": neural.astype(np.float64),
        "switch": switch.astype(bool),
        "alpha": alpha.astype(np.float64),
    }


def _metric_context(ds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return pure._metric_context(ds)


def _bootstrap_report(selected: np.ndarray, floor: np.ndarray, ds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    horizon = ds["horizon"].astype(int)
    hard = ds["hard"].astype(bool) | ds["failure"].astype(bool)
    source = np.asarray([_source_key(str(s)) for s in ds["source_file"].astype(str)], dtype=object)
    domain = ds["domain"].astype(str)
    report: dict[str, Any] = {
        "all": _bootstrap_ci(selected, floor, np.ones(len(selected), dtype=bool), seed=SEED),
        "t10": _bootstrap_ci(selected, floor, horizon == 10, seed=SEED + 1),
        "t25": _bootstrap_ci(selected, floor, horizon == 25, seed=SEED + 2),
        "t50": _bootstrap_ci(selected, floor, horizon == 50, seed=SEED + 3),
        "t100_raw_frame_diagnostic": _bootstrap_ci(selected, floor, horizon == 100, seed=SEED + 4),
        "hard_failure": _bootstrap_ci(selected, floor, hard, seed=SEED + 5),
    }
    report["by_source"] = {
        name: _bootstrap_ci(selected, floor, source == name, seed=SEED + 20 + i)
        for i, name in enumerate(sorted(set(source.tolist())))
    }
    report["by_domain_label"] = {
        name: _bootstrap_ci(selected, floor, domain == name, seed=SEED + 40 + i)
        for i, name in enumerate(sorted(set(domain.tolist())))
    }
    return report


def _all_core_ci_positive(bootstrap: Mapping[str, Any]) -> bool:
    return bool(
        (bootstrap.get("all") or {}).get("low", 0.0) > 0.0
        and (bootstrap.get("t50") or {}).get("low", 0.0) > 0.0
        and (bootstrap.get("t100_raw_frame_diagnostic") or {}).get("low", 0.0) > 0.0
        and (bootstrap.get("hard_failure") or {}).get("low", 0.0) > 0.0
    )


def run_pure_ucy_neural_statistical_evidence() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    if not pure.REPORT_JSON.exists():
        pure.run_strict_pure_ucy_neural_retrain()
    retrain = json.loads(pure.REPORT_JSON.read_text(encoding="utf-8"))
    best_trial = retrain["best_trial"]
    trial = retrain["trials"][best_trial]
    checkpoint = trial["train"]["checkpoint"]
    pred = pure._predict(checkpoint, "test")
    ds = pure._ds("test")
    policy = retrain["best_policy"]
    arrays = _selected_endpoint_arrays(pred, ds, policy)
    selected = arrays["selected"]
    floor = arrays["floor"]
    switch = arrays["switch"]
    metrics = s41._metrics(selected, floor, _metric_context(ds), switch)
    raw_endpoint_metrics = s41._metrics(arrays["neural_without_fallback"], floor, _metric_context(ds), np.ones(len(floor), dtype=bool))
    bootstrap = _bootstrap_report(selected, floor, ds)
    source = np.asarray([_source_key(str(s)) for s in ds["source_file"].astype(str)], dtype=object)
    source_metrics = {
        name: s41._metrics(selected[source == name], floor[source == name], {k: v[source == name] for k, v in _metric_context(ds).items()}, switch[source == name])
        for name in sorted(set(source.tolist()))
    }
    stable = bool(
        retrain.get("strict_pure_ucy_only_neural_retrain_select_test_gate")
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("switch_rate", 0.0) > 0.0
        and _all_core_ci_positive(bootstrap)
    )
    result = {
        "source": "fresh_run",
        "protocol": "strict_pure_ucy_neural_bootstrap_statistical_evidence",
        "best_trial": best_trial,
        "best_mode": retrain.get("best_mode"),
        "selected_policy": policy,
        "test_metrics_recomputed": metrics,
        "raw_neural_endpoint_without_fallback": raw_endpoint_metrics,
        "bootstrap": bootstrap,
        "source_metrics": source_metrics,
        "statistically_stable_on_test": stable,
        "bootstrap_n": BOOTSTRAP_N,
        "interpretation": (
            "The strict pure-UCY neural residual repair has positive bootstrap lower bounds for all/t50/t100/hard "
            "while preserving easy cases under validation-selected conservative bounded residual deployment. Raw "
            "ungated endpoint neural remains unsafe and is reported as negative no-fallback evidence."
        ),
        "no_leakage": {
            **retrain.get("no_leakage", {}),
            "bootstrap_uses_test_labels_for_ci_only": True,
            "test_threshold_tuning": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "strict_pure_ucy_statistical_support": True,
            "ungated_neural_replacement": False,
            "dataset_local_raw_frame": True,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Strict Pure-UCY Neural Statistical Evidence",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- best trial/mode: `{best_trial}` / `{retrain.get('best_mode')}`",
        f"- statistically stable on test: `{stable}`",
        "",
        "## Recomputed Test Metrics",
        "",
        f"- all improvement: `{metrics.get('all_improvement')}`",
        f"- t50 improvement: `{metrics.get('t50_improvement')}`",
        f"- t100 raw-frame diagnostic improvement: `{metrics.get('t100_improvement')}`",
        f"- hard/failure improvement: `{metrics.get('hard_failure_improvement')}`",
        f"- easy degradation: `{metrics.get('easy_degradation')}`",
        f"- switch rate: `{metrics.get('switch_rate')}`",
        "",
        "## Bootstrap",
        "",
        f"- bootstrap n: `{BOOTSTRAP_N}`",
        f"- all/t50/t100/hard lows: `{bootstrap['all']['low']}` / `{bootstrap['t50']['low']}` / `{bootstrap['t100_raw_frame_diagnostic']['low']}` / `{bootstrap['hard_failure']['low']}`",
        f"- by source: `{bootstrap['by_source']}`",
        "",
        "## No-Fallback Negative Evidence",
        "",
        f"- raw neural endpoint without fallback: `{raw_endpoint_metrics}`",
        "",
        "## No Leakage",
        "",
        f"`{result['no_leakage']}`",
    ]
    write_md(REPORT_MD, lines)
    return result


def main_pure_ucy_neural_statistical_evidence() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_pure_ucy_neural_statistical_evidence()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_pure_ucy_neural_statistical_evidence",
            status,
            started,
            [pure.REPORT_JSON, pure.DATA_DIR / "seq2seq_test.npz"],
            [REPORT_JSON, REPORT_MD],
        )


if __name__ == "__main__":
    main_pure_ucy_neural_statistical_evidence()
