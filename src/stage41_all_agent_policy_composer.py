from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as aa
from src import stage41_all_agent_risk_repair as risk
from src import stage41_all_agent_t50_specialist as t50
from src import stage41_breakthrough as s41


OUT_DIR = s41.OUT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
RISK_REPORT = OUT_DIR / "stage41_all_agent_risk_repair.json"
T50_REPORT = OUT_DIR / "stage41_all_agent_t50_specialist.json"
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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str], outputs: Sequence[str]) -> None:
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
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage41 Breakthrough Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('command', '')}` | `{row.get('source', '')}` | `{row.get('status', '')}` | {float(row.get('wall_time_s', 0.0)):.3f} | `{str(row.get('input_hash', ''))[:12]}` | `{str(row.get('output_hash', ''))[:12]}` | `{row.get('git_commit', '')}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def _safe_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    risk_report = read_json(RISK_REPORT, {})
    t50_report = read_json(T50_REPORT, {})
    risk_ckpt = ""
    if risk_report:
        try:
            risk_ckpt = risk_report["candidates"][risk_report["best_trial"]]["checkpoint"]
        except Exception:
            risk_ckpt = ""
    if not risk_report or not risk_ckpt or not Path(str(risk_ckpt)).exists():
        risk.run_all_agent_risk_repair()
        risk_report = read_json(RISK_REPORT, {})
    t50_ckpt = ""
    if t50_report:
        try:
            t50_ckpt = t50_report["trials"][t50_report["best_trial"]]["train"]["checkpoint"]
        except Exception:
            t50_ckpt = ""
    if not t50_report or not t50_ckpt or not Path(str(t50_ckpt)).exists():
        t50.run_all_agent_t50_specialist()
        t50_report = read_json(T50_REPORT, {})
    return risk_report, t50_report


def _risk_arrays(split: str, risk_report: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    best = risk_report["candidates"][risk_report["best_trial"]]
    checkpoint = best["checkpoint"]
    policy = best["selected_policy"]
    pred = aa._predict(checkpoint, split)
    ds = aa._ds(split)
    selected, switch = risk._apply_group_policies(pred, ds, policy["policies"], policy["type"])
    return selected.astype(np.float64), switch.astype(bool)


def _t50_arrays(split: str, t50_report: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    best = t50_report["trials"][t50_report["best_trial"]]
    checkpoint = best["train"]["checkpoint"]
    policy_pack = best["policy"]
    pred = t50._predict(checkpoint, split)
    ds_t50 = t50._load_t50(split)
    domain = ds_t50["domain"].astype(str)
    selected_t50 = ds_t50["floor"].astype(np.float64).copy()
    switch_t50 = np.zeros(len(selected_t50), dtype=bool)
    if policy_pack["type"] == "global":
        groups = {"global": np.ones(len(domain), dtype=bool)}
    else:
        groups = {d: domain == d for d in sorted(set(domain.tolist()))}
    for name, mask in groups.items():
        policy = policy_pack["policies"].get(name, {"max_switch": 0.0})
        pred_g = {k: v[mask] for k, v in pred.items()}
        ds_g = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds_t50.items() if k != "raw_ds"}
        ds_g["raw_ds"] = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in ds_t50["raw_ds"].items()}
        sel, sw, _src = t50._select_t50(pred_g, ds_g, policy)
        selected_t50[mask] = sel
        switch_t50[mask] = sw
    full = aa._ds(split)
    h50 = full["horizon"].astype(int) == 50
    selected = full["floor_fde"].astype(np.float64).copy()
    switch = np.zeros(len(selected), dtype=bool)
    selected[h50] = selected_t50
    switch[h50] = switch_t50
    return selected, switch


def _compose_arrays(
    variant: str,
    floor: np.ndarray,
    horizon: np.ndarray,
    risk_selected: np.ndarray,
    risk_switch: np.ndarray,
    t50_selected: np.ndarray,
    t50_switch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h50 = horizon.astype(int) == 50
    if variant == "floor":
        return floor.copy(), np.zeros(len(floor), dtype=bool)
    if variant == "risk_only":
        return risk_selected.copy(), risk_switch.copy()
    if variant == "t50_only":
        selected = floor.copy()
        switch = np.zeros(len(floor), dtype=bool)
        selected[h50] = t50_selected[h50]
        switch[h50] = t50_switch[h50]
        return selected, switch
    if variant == "risk_non_t50_plus_t50":
        selected = risk_selected.copy()
        switch = risk_switch.copy()
        selected[h50] = floor[h50]
        switch[h50] = False
        selected[h50] = t50_selected[h50]
        switch[h50] = t50_switch[h50]
        return selected, switch
    if variant == "risk_all_t50_override":
        selected = risk_selected.copy()
        switch = risk_switch.copy()
        override = h50 & t50_switch
        selected[override] = t50_selected[override]
        switch[override] = True
        return selected, switch
    raise ValueError(variant)


VARIANTS = ["floor", "risk_only", "t50_only", "risk_non_t50_plus_t50", "risk_all_t50_override"]


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.5 * float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("hard_failure_improvement", 0.0))
        + 0.5 * float(metrics.get("t100_improvement", 0.0))
        - 30.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 0.2 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
    )


def _eval_variant(
    variant: str,
    split: str,
    risk_report: Mapping[str, Any],
    t50_report: Mapping[str, Any],
    cached_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Dict[str, Any]:
    ds = aa._ds(split)
    floor = ds["floor_fde"].astype(np.float64)
    if cached_arrays is None:
        risk_selected, risk_switch = _risk_arrays(split, risk_report)
        t50_selected, t50_switch = _t50_arrays(split, t50_report)
    else:
        risk_selected, risk_switch, t50_selected, t50_switch = cached_arrays
    selected, switch = _compose_arrays(variant, floor, ds["horizon"], risk_selected, risk_switch, t50_selected, t50_switch)
    metrics = aa._metrics(selected, floor, ds, switch)
    metrics["variant"] = variant
    metrics["selected_candidate_distribution"] = {
        "floor_rows": int(np.sum(~switch)),
        "switched_rows": int(np.sum(switch)),
        "t50_switched_rows": int(np.sum(switch & (ds["horizon"].astype(int) == 50))),
        "non_t50_switched_rows": int(np.sum(switch & (ds["horizon"].astype(int) != 50))),
    }
    return metrics


def run_all_agent_policy_composer() -> Dict[str, Any]:
    risk_report, t50_report = _safe_reports()
    val_cached = (*_risk_arrays("val", risk_report), *_t50_arrays("val", t50_report))
    test_cached = (*_risk_arrays("test", risk_report), *_t50_arrays("test", t50_report))
    val_metrics: Dict[str, Any] = {}
    for variant in VARIANTS:
        metrics = _eval_variant(variant, "val", risk_report, t50_report, val_cached)
        metrics["score"] = _score(metrics)
        val_metrics[variant] = metrics
    best_variant = max(val_metrics, key=lambda k: float(val_metrics[k]["score"]))
    test_metrics = _eval_variant(best_variant, "test", risk_report, t50_report, test_cached)
    ds_test = aa._ds("test")
    selected_test, switch_test = _compose_arrays(
        best_variant,
        ds_test["floor_fde"].astype(np.float64),
        ds_test["horizon"],
        test_cached[0],
        test_cached[1],
        test_cached[2],
        test_cached[3],
    )
    test_metrics["t50_ci"] = s41._bootstrap_ci(
        selected_test,
        ds_test["floor_fde"].astype(np.float64),
        ds_test,
        "t50",
        n=2000,
    )
    test_metrics["hard_failure_ci"] = s41._bootstrap_ci(
        selected_test,
        ds_test["floor_fde"].astype(np.float64),
        ds_test,
        "hard_failure",
        n=1000,
    )
    positive_domains = sum(
        1
        for row in test_metrics.get("by_domain", {}).values()
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
    )
    beats = (
        test_metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            test_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            or test_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            or test_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
    )
    result = {
        "source": "fresh_run",
        "hypothesis": "Compose the t50 specialist with the all/t100 risk-cap repair under validation selection, so all-agent t50 and t100 do not sabotage each other.",
        "best_variant": best_variant,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "positive_external_domains": int(positive_domains),
        "neural_exceeds_stage37_by_gate_margin": bool(beats),
        "deployment_decision": "deploy_all_agent_composed_policy" if beats and positive_domains >= 2 else "diagnostic_keep_m3w_neural_v1_endpoint_candidate",
        "stage37_reference": s41.STAGE37_REFERENCE,
        "stage37_margin_rule": "deploy only if all/t50/hard beats Stage37 by >=2% absolute with easy degradation <=2% and >=2 positive domains",
        "component_reports": {
            "risk_repair": str(RISK_REPORT),
            "t50_specialist": str(T50_REPORT),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "composition_selection_split": "val",
            "test_used_once_for_final_eval": True,
        },
    }
    _write_json(OUT_DIR / "stage41_all_agent_policy_composer.json", result)
    write_md(
        OUT_DIR / "stage41_all_agent_policy_composer.md",
        [
            "# Stage41 All-Agent Policy Composer",
            "",
            "- source: `fresh_run`",
            f"- best variant: `{best_variant}`",
            f"- deployment: `{result['deployment_decision']}`",
            f"- metrics: `{test_metrics}`",
            "",
            "## Interpretation",
            "",
            "This composer uses validation only to choose between floor, risk-cap, t50-specialist, and combined policies. It does not tune on test. If it remains diagnostic, the remaining blocker is not isolated t50 or t100 but joint all-agent policy compatibility.",
        ],
    )
    return result


def main_all_agent_policy_composer() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_all_agent_policy_composer()
        status = "success"
    finally:
        _append_ledger(
            "stage41_all_agent_policy_composer",
            status,
            started,
            [str(RISK_REPORT), str(T50_REPORT)],
            [str(OUT_DIR / "stage41_all_agent_policy_composer.md")],
        )
