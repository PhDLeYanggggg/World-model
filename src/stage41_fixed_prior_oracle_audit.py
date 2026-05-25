from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_full_trajectory_world_state as ft
from src import stage41_shape_policy_composer as composer
from src import stage41_dynamic_shape_meta_policy as meta
from src import stage41_pairwise_shape_switch_policy as pairwise
from src import stage41_fixed_prior_source_switch_policy as fixed_prior


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_fixed_prior_oracle_audit.json"
REPORT_MD = OUT_DIR / "stage41_fixed_prior_oracle_audit.md"
SOURCES = meta.SOURCES


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


def _source_cost_matrix(pack: Mapping[str, Any]) -> np.ndarray:
    return np.column_stack([ft._trajectory_errors(meta._source_xy(pack, source), pack["labels"])[0] for source in SOURCES]).astype(np.float64)


def _oracle_chosen(pack: Mapping[str, Any]) -> np.ndarray:
    costs = _source_cost_matrix(pack)
    return np.argmin(np.nan_to_num(costs, nan=1.0e12, posinf=1.0e12, neginf=1.0e12), axis=1).astype(int)


def _xy_from_chosen(pack: Mapping[str, Any], chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = pack["bridge_xy"].copy()
    shape_switch = np.zeros(len(chosen), dtype=bool)
    for i, source in enumerate(SOURCES):
        mask = chosen == i
        if not np.any(mask):
            continue
        selected[mask] = meta._source_xy(pack, source)[mask]
        if source != "bridge":
            shape_switch[mask] = meta._source_switch(pack, source)[mask]
    return selected, shape_switch


def _delta(compact: Mapping[str, float], fixed: Mapping[str, float]) -> dict[str, float]:
    keys = ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure"]
    return {key: float(compact[key] - fixed[key]) for key in keys}


def _source_distribution(chosen: np.ndarray) -> dict[str, float]:
    return {source: float(np.mean(chosen == i)) if len(chosen) else 0.0 for i, source in enumerate(SOURCES)}


def _margin_summary(pack: Mapping[str, Any], fixed_chosen: np.ndarray, oracle_chosen: np.ndarray) -> dict[str, Any]:
    costs = _source_cost_matrix(pack)
    rows = np.arange(len(oracle_chosen))
    fixed_cost = costs[rows, fixed_chosen]
    oracle_cost = costs[rows, oracle_chosen]
    sorted_cost = np.sort(costs, axis=1)
    best_margin = sorted_cost[:, 1] - sorted_cost[:, 0]
    residual_gain = fixed_cost - oracle_cost
    horizon = pack["horizon"].astype(int)
    labels = pack["labels"]
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    by_horizon: dict[str, Any] = {}
    for h in [10, 25, 50, 100]:
        mask = horizon == h
        by_horizon[f"t{h}"] = {
            "rows": int(np.sum(mask)),
            "oracle_switch_rate": float(np.mean(oracle_chosen[mask] != fixed_chosen[mask])) if np.any(mask) else 0.0,
            "mean_residual_gain": float(np.mean(residual_gain[mask])) if np.any(mask) else 0.0,
            "positive_residual_rate": float(np.mean(residual_gain[mask] > 1.0e-9)) if np.any(mask) else 0.0,
            "median_best_margin": float(np.median(best_margin[mask])) if np.any(mask) else 0.0,
        }
    return {
        "rows": int(len(oracle_chosen)),
        "oracle_switch_rate": float(np.mean(oracle_chosen != fixed_chosen)) if len(oracle_chosen) else 0.0,
        "positive_residual_rate": float(np.mean(residual_gain > 1.0e-9)) if len(residual_gain) else 0.0,
        "mean_residual_gain": float(np.mean(residual_gain)) if len(residual_gain) else 0.0,
        "median_residual_gain": float(np.median(residual_gain)) if len(residual_gain) else 0.0,
        "median_best_margin": float(np.median(best_margin)) if len(best_margin) else 0.0,
        "hard_oracle_switch_rate": float(np.mean(oracle_chosen[hard] != fixed_chosen[hard])) if np.any(hard) else 0.0,
        "hard_mean_residual_gain": float(np.mean(residual_gain[hard])) if np.any(hard) else 0.0,
        "easy_oracle_switch_rate": float(np.mean(oracle_chosen[~hard] != fixed_chosen[~hard])) if np.any(~hard) else 0.0,
        "by_horizon": by_horizon,
        "oracle_source_distribution": _source_distribution(oracle_chosen),
        "fixed_source_distribution": _source_distribution(fixed_chosen),
    }


def _audit_split(pack: Mapping[str, Any], fixed_policy: Mapping[str, str]) -> dict[str, Any]:
    fixed_xy, fixed_switch = composer._compose_sources(pack, fixed_policy)
    fixed_eval = composer._eval_selected(pack, fixed_xy, fixed_switch)
    fixed_chosen = fixed_prior._fixed_chosen(pack, fixed_policy)
    fixed_compact = fixed_prior._compact(fixed_eval, fixed_chosen, fixed_chosen)
    oracle_chosen = _oracle_chosen(pack)
    oracle_xy, oracle_switch = _xy_from_chosen(pack, oracle_chosen)
    oracle_eval = composer._eval_selected(pack, oracle_xy, oracle_switch)
    oracle_compact = fixed_prior._compact(oracle_eval, oracle_chosen, fixed_chosen)
    return {
        "fixed_compact": fixed_compact,
        "oracle_compact": oracle_compact,
        "oracle_delta_vs_fixed": _delta(oracle_compact, fixed_compact),
        "oracle_margin": _margin_summary(pack, fixed_chosen, oracle_chosen),
    }


def _audit_domain(domain: str) -> dict[str, Any]:
    packs = pairwise._build_domain_packs(domain)
    fixed_selection = composer._select_composer_on_val(packs["val_pack"])
    fixed_policy = fixed_selection["selected"]["policy"]
    val_audit = _audit_split(packs["val_pack"], fixed_policy)
    test_audit = _audit_split(packs["test_pack"], fixed_policy)
    d = test_audit["oracle_delta_vs_fixed"]
    meaningful_headroom = bool(d["all"] > 0.001 or d["t50"] > 0.001 or d["t100"] > 0.001 or d["hard_failure"] > 0.001)
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": packs["rows"],
        "fixed_policy": fixed_policy,
        "fixed_selection": fixed_selection,
        "val_oracle_audit": val_audit,
        "test_oracle_audit": test_audit,
        "meaningful_residual_oracle_headroom": meaningful_headroom,
        "diagnostic_only": True,
        "caveat": "Oracle uses future waypoint labels only to measure residual source-switch headroom. It is not an inference model and is not deployable.",
    }


def run_fixed_prior_oracle_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _audit_domain(domain) for domain in domains}
    headroom_domains = [domain for domain, row in results.items() if row.get("meaningful_residual_oracle_headroom")]
    result = {
        "source": "fresh_run",
        "protocol": "fixed_composer_residual_source_oracle_audit",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "oracle_is_diagnostic_not_deployable": True,
        "headroom_domains": headroom_domains,
        "two_domain_residual_oracle_headroom": len(headroom_domains) >= 2,
        "domain_results": results,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "oracle_future_labels_diagnostic_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "oracle_diagnostic": True,
            "deployable_model": False,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Fixed-Composer Residual Source Oracle Audit",
        "",
        "- source: `fresh_run`",
        "- oracle is diagnostic, not deployable: `True`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- headroom domains: `{headroom_domains}`",
        f"- two-domain residual oracle headroom: `{result['two_domain_residual_oracle_headroom']}`",
        "",
        "| domain | fixed policy | oracle delta all/t50/t100/hard | oracle switch | positive residual | hard switch | source distribution |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        test = row["test_oracle_audit"]
        d = test["oracle_delta_vs_fixed"]
        m = test["oracle_margin"]
        policy = row["fixed_policy"]
        lines.append(
            f"| `{domain}` | `{policy['short']}/{policy['t50']}/{policy['t100']}` | "
            f"{d['all']:.6f}/{d['t50']:.6f}/{d['t100']:.6f}/{d['hard_failure']:.6f} | "
            f"{m['oracle_switch_rate']:.6f} | {m['positive_residual_rate']:.6f} | {m['hard_oracle_switch_rate']:.6f} | "
            f"`{m['oracle_source_distribution']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit asks whether any per-row oracle over bridge / old-shape / gain-gate can beat the validation-selected fixed composer.",
            "- Future waypoint labels are used only for diagnostic oracle costs; they are not model inputs, thresholds, goals, or deployment features.",
            "- If oracle headroom is small, more source-switch learners are unlikely to help. If oracle headroom is large but learned switches fail, the blocker is causal feature separability.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_fixed_prior_oracle_audit() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_fixed_prior_oracle_audit()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_fixed_prior_oracle_audit",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_fixed_prior_oracle_audit()
