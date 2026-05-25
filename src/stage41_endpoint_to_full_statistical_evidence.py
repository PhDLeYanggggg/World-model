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
from src import stage41_endpoint_to_full_trajectory_repair as bridge


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_endpoint_to_full_statistical_evidence.json"
REPORT_MD = OUT_DIR / "stage41_endpoint_to_full_statistical_evidence.md"
DOMAINS = ["ETH_UCY", "TrajNet"]
BOOTSTRAP_N = 2000
SEED = 41667
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


def _bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = np.zeros(BOOTSTRAP_N, dtype=np.float64)
    for i in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals[i] = 1.0 - float(selected[sample].mean()) / max(float(floor[sample].mean()), EPS)
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _bootstrap_bundle(ev: Mapping[str, Any], labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    multi = ev["multi"].astype(bool)
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t10": horizon == 10,
        "t25": horizon == 25,
        "t50": horizon == 50,
        "t100_raw_frame_diagnostic": horizon == 100,
        "hard_failure": hard,
        "multi_agent": multi,
        "multi_agent_t50": multi & (horizon == 50),
        "multi_agent_hard_failure": multi & hard,
    }
    out: dict[str, Any] = {"ade": {}, "fde": {}}
    for i, (name, mask) in enumerate(masks.items()):
        out["ade"][name] = _bootstrap(ev["selected_ade"], ev["floor_ade"], mask, SEED + 17 * i)
        out["fde"][name] = _bootstrap(ev["selected_fde"], ev["floor_fde"], mask, SEED + 101 + 17 * i)
    return out


def _evaluate_domain_with_arrays(domain: str) -> dict[str, Any]:
    train = bridge._domain_data("train", domain)
    val = bridge._domain_data("val", domain)
    test = bridge._domain_data("test", domain)
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}

    training = dl._train_endpoint(domain, train, val)
    pred_train = dl._predict_endpoint(training["checkpoint"], train)
    pred_val = dl._predict_endpoint(training["checkpoint"], val)
    pred_test = dl._predict_endpoint(training["checkpoint"], test)
    fde_train = dl._endpoint_fde(pred_train["delta"], train)
    fde_val = dl._endpoint_fde(pred_val["delta"], val)
    fde_test = dl._endpoint_fde(pred_test["delta"], test)
    gate = dl._train_gate(train, pred_train, fde_train)
    gate_val = dl._predict_gate(gate, val, pred_val, fde_val)
    labels_val = bridge._align_full_labels("val", val)
    selection = bridge._select_policy_on_val(val, labels_val, pred_val, gate_val)

    gate_test = dl._predict_gate(gate, test, pred_test, fde_test)
    selected_delta, switch = bridge._apply_endpoint_policy(
        test,
        pred_test,
        gate_test,
        selection["selected"]["policy"],
        {"all_horizons": None, "t50_only": {50}, "long_horizon": {50, 100}}[selection["selected"]["variant"]],
    )
    labels_test = bridge._align_full_labels("test", test)
    selected_delta, switch, guarded_off = bridge._guard(test, labels_test, selected_delta, switch, float(selection["selected"]["min_sep"]))
    ev = bridge._eval_world_state(test, labels_test, selected_delta, switch)
    bootstrap = _bootstrap_bundle(ev, labels_test)
    lows = {
        "ade_all": bootstrap["ade"]["all"]["low"],
        "ade_t50": bootstrap["ade"]["t50"]["low"],
        "ade_t100": bootstrap["ade"]["t100_raw_frame_diagnostic"]["low"],
        "ade_hard": bootstrap["ade"]["hard_failure"]["low"],
        "ade_multi": bootstrap["ade"]["multi_agent"]["low"],
        "fde_all": bootstrap["fde"]["all"]["low"],
        "fde_t50": bootstrap["fde"]["t50"]["low"],
    }
    statistical_gate = bool(
        ev["ade_metrics"].get("all_improvement", 0.0) > 0.0
        and ev["ade_metrics"].get("t50_improvement", 0.0) > 0.0
        and ev["ade_metrics"].get("hard_failure_improvement", 0.0) > 0.0
        and ev["ade_metrics"].get("easy_degradation", 1.0) <= 0.02
        and ev["fde_metrics"].get("all_improvement", 0.0) > 0.0
        and ev["fde_metrics"].get("t50_improvement", 0.0) > 0.0
        and ev["collision_delta_005"] <= 0.01
        and ev["smoothness_jagged_delta"] <= 0.01
        and lows["ade_all"] > 0.0
        and lows["ade_t50"] > 0.0
        and lows["ade_hard"] > 0.0
        and lows["ade_multi"] > 0.0
        and lows["fde_all"] > 0.0
        and lows["fde_t50"] > 0.0
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "t50_rows": {"train": int(np.sum(train["horizon"] == 50)), "val": int(np.sum(val["horizon"] == 50)), "test": int(np.sum(test["horizon"] == 50))},
        "t100_rows": {"train": int(np.sum(train["horizon"] == 100)), "val": int(np.sum(val["horizon"] == 100)), "test": int(np.sum(test["horizon"] == 100))},
        "training": training,
        "selection": selection,
        "test_guarded_off": guarded_off,
        "ade_metrics_vs_floor": ev["ade_metrics"],
        "fde_metrics_vs_floor": ev["fde_metrics"],
        "multi_agent_ade_metrics": ev["multi_ade_metrics"],
        "collision_delta_vs_floor_005": ev["collision_delta_005"],
        "smoothness_jagged_delta": ev["smoothness_jagged_delta"],
        "bootstrap": bootstrap,
        "bootstrap_lows": lows,
        "endpoint_to_full_statistical_gate": statistical_gate,
        "claim_boundary": "Endpoint neural dynamics are projected through a linear waypoint bridge and scored against reconstructed waypoint labels. This is not learned full-waypoint shape dynamics, not metric, not seconds-level, not Stage5C, and not SMC.",
    }


def run_endpoint_to_full_statistical_evidence() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    results = {domain: _evaluate_domain_with_arrays(domain) for domain in DOMAINS}
    positive = [domain for domain, row in results.items() if row.get("endpoint_to_full_statistical_gate")]
    result = {
        "source": "fresh_run",
        "protocol": "endpoint_neural_to_full_waypoint_bootstrap_statistical_evidence",
        "bootstrap_n": BOOTSTRAP_N,
        "domains": DOMAINS,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_statistical_gate": len(positive) >= 2,
        "domain_results": results,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
        },
        "claim_boundary": {
            "endpoint_neural_dynamics": True,
            "linear_waypoint_bridge": True,
            "learned_full_waypoint_shape": False,
            "ungated_full_row_neural_safety": False,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Endpoint-To-Full Statistical Evidence",
        "",
        "- source: `fresh_run`",
        f"- bootstrap_n: `{BOOTSTRAP_N}`",
        f"- positive domains: `{positive}`",
        f"- two-domain statistical gate: `{result['two_domain_statistical_gate']}`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        "",
        "| domain | all ADE | all low | t50 ADE | t50 low | t100 ADE | t100 low | hard ADE | hard low | multi low | FDE all low | FDE t50 low | easy | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{row.get('reason')}` |")
            continue
        m = row["ade_metrics_vs_floor"]
        lows = row["bootstrap_lows"]
        lines.append(
            f"| `{domain}` | {m.get('all_improvement', 0.0):.4f} | {lows['ade_all']:.4f} | "
            f"{m.get('t50_improvement', 0.0):.4f} | {lows['ade_t50']:.4f} | "
            f"{m.get('t100_improvement', 0.0):.4f} | {lows['ade_t100']:.4f} | "
            f"{m.get('hard_failure_improvement', 0.0):.4f} | {lows['ade_hard']:.4f} | "
            f"{lows['ade_multi']:.4f} | {lows['fde_all']:.4f} | {lows['fde_t50']:.4f} | "
            f"{m.get('easy_degradation', 0.0):.4f} | `{row.get('endpoint_to_full_statistical_gate')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This strengthens the endpoint-to-full bridge by adding per-domain bootstrap lower bounds for actual waypoint ADE/FDE.",
            "- It does not convert the claim into learned full-waypoint shape dynamics; the waypoint path remains a protected linear bridge from endpoint neural dynamics.",
            "- Raw ungated full-row neural safety is still not claimed.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_endpoint_to_full_statistical_evidence() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_endpoint_to_full_statistical_evidence()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_endpoint_to_full_statistical_evidence",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_endpoint_to_full_statistical_evidence()
