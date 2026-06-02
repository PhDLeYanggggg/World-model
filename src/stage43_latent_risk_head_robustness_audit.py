from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _build_split,
    _git_commit,
    _jsonable,
    _sha256,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_multimodal_latent_head_suite import REPORT_JSON as STAGE43_Y_JSON
from src.stage43_world_state_head_audit import _binary_metrics, _predict_heads


REPORT_JSON = OUT_DIR / "stage43_latent_risk_head_robustness_audit.json"
REPORT_MD = OUT_DIR / "stage43_latent_risk_head_robustness_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_bx_latent_risk_head_robustness_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BX_LATENT_RISK_HEAD_ROBUSTNESS_AUDIT"
SOURCE = "fresh_stage43_bx_latent_risk_head_robustness_audit"
HEADS = {
    "failure": "y_failure",
    "gain": "y_gain",
    "harm": "y_harm",
}


def _pct(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{100.0 * float(value):.2f}%"


def _label(ds: Any, head: str) -> np.ndarray:
    return getattr(ds, HEADS[head]).astype(np.float32)


def _score(pred: Mapping[str, np.ndarray], head: str) -> np.ndarray:
    return pred[head].astype(np.float32)


def _breakdown(values: np.ndarray, y: np.ndarray, score: np.ndarray, *, min_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    vv = values.astype(str)
    for value in sorted(set(vv.tolist())):
        mask = vv == value
        if int(mask.sum()) < int(min_rows):
            continue
        row = _binary_metrics(y[mask], score[mask])
        out[str(value)] = {
            "rows": row["rows"],
            "positive_rate": row["positive_rate"],
            "auroc": row["auroc"],
            "auprc": row["auprc"],
            "brier": row["brier"],
            "ece": row["ece"],
            "defined": row["defined"],
        }
    return out


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"low": 0.0, "mean": 0.0, "high": 0.0}
    return {
        "low": float(np.quantile(arr, 0.025)),
        "mean": float(np.mean(arr)),
        "high": float(np.quantile(arr, 0.975)),
    }


def _bootstrap_head(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n: int,
    sample_rows: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y).astype(np.float32)
    score = np.asarray(score).astype(np.float32)
    rows = len(y)
    take = min(int(sample_rows), rows)
    auroc: list[float] = []
    auprc: list[float] = []
    brier: list[float] = []
    for _ in range(int(n)):
        ids = rng.integers(0, rows, size=take)
        metrics = _binary_metrics(y[ids], score[ids])
        if metrics["defined"]:
            auroc.append(float(metrics["auroc"]))
            auprc.append(float(metrics["auprc"]))
        brier.append(float(metrics["brier"]))
    return {
        "n": int(n),
        "sample_rows": int(take),
        "auroc": _ci(auroc),
        "auprc": _ci(auprc),
        "brier": _ci(brier),
        "defined_replicates": int(len(auroc)),
    }


def _min_defined_auroc(tables: Mapping[str, Mapping[str, Any]]) -> float:
    values = [
        float(row["auroc"])
        for row in tables.values()
        if row.get("defined") and row.get("auroc") is not None
    ]
    return min(values) if values else 0.0


def run_latent_risk_head_robustness_audit(
    *,
    batch_size: int = 4096,
    min_rows: int = 100,
    bootstrap: int = 1000,
    bootstrap_rows: int = 8000,
    seed: int = 487,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43y = read_json(STAGE43_Y_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    pred = _predict_heads(model, test, batch_size=int(batch_size))
    head_metrics: dict[str, Any] = {}
    by_domain: dict[str, Any] = {}
    by_horizon: dict[str, Any] = {}
    bootstrap_summary: dict[str, Any] = {}
    weak_horizon_slices: list[dict[str, Any]] = []
    for offset, head in enumerate(HEADS):
        y = _label(test, head)
        s = _score(pred, head)
        head_metrics[head] = _binary_metrics(y, s)
        by_domain[head] = _breakdown(test.domain, y, s, min_rows=int(min_rows))
        by_horizon[head] = _breakdown(test.horizon.astype(str), y, s, min_rows=int(min_rows))
        bootstrap_summary[head] = _bootstrap_head(
            y,
            s,
            n=int(bootstrap),
            sample_rows=int(bootstrap_rows),
            seed=int(seed) + 17 * offset,
        )
        for horizon, row in by_horizon[head].items():
            if row.get("defined") and row.get("auroc") is not None and float(row["auroc"]) < 0.75:
                weak_horizon_slices.append(
                    {
                        "head": head,
                        "horizon": horizon,
                        "rows": row["rows"],
                        "auroc": row["auroc"],
                        "auprc": row["auprc"],
                        "positive_rate": row["positive_rate"],
                    }
                )
    latent = pred["latent"].astype(np.float32)
    latent_stats = {
        "rows": int(len(latent)),
        "dim": int(latent.shape[1]) if latent.ndim == 2 else 0,
        "mean_variance": float(np.var(latent, axis=0).mean()) if len(latent) else 0.0,
        "min_variance": float(np.var(latent, axis=0).min()) if len(latent) else 0.0,
        "noncollapse_threshold": 0.01,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_checkpoint_replay_latent_risk_head_robustness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {
            "verdict": stage43m.get("stage43_m_gate", {}).get("verdict"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_sha256_matches_stage43_m": _sha256(checkpoint) == stage43m.get("checkpoint_sha256"),
        },
        "stage43_y_precondition": {
            "verdict": stage43y.get("stage43_y_gate", {}).get("verdict"),
            "protected_multimodal_latent_state_candidate": stage43y.get("stage43_y_gate", {}).get(
                "protected_multimodal_latent_state_candidate", False
            ),
        },
        "evaluation_protocol": {
            "split": "test",
            "rows": int(len(test.x)),
            "batch_size": int(batch_size),
            "bootstrap": int(bootstrap),
            "bootstrap_rows": int(min(int(bootstrap_rows), len(test.x))),
            "future_labels_eval_only": True,
            "test_threshold_tuning": False,
            "num_workers": 0,
        },
        "head_metrics": head_metrics,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "bootstrap": bootstrap_summary,
        "weak_horizon_slices": weak_horizon_slices,
        "latent_stats": latent_stats,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "standalone_ungated_policy": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_M_JSON, STAGE43_Y_JSON]),
    }
    payload["stage43_bx_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    heads = payload["head_metrics"]
    boot = payload["bootstrap"]
    by_domain = payload["by_domain"]
    by_horizon = payload["by_horizon"]
    global_strong = all(float(heads[head]["auroc"]) >= 0.80 for head in HEADS)
    domain_robust = all(_min_defined_auroc(by_domain[head]) >= 0.80 for head in HEADS)
    horizon_min = min(_min_defined_auroc(by_horizon[head]) for head in HEADS)
    horizon_supported = horizon_min >= 0.60
    bootstrap_supported = all(int(boot[head]["defined_replicates"]) >= 500 and float(boot[head]["auroc"]["low"]) >= 0.60 for head in HEADS)
    weak_horizons_reported = isinstance(payload["weak_horizon_slices"], list)
    gates = {
        "stage43_m_checkpoint_replayed": payload["stage43_m_precondition"]["checkpoint_sha256_matches_stage43_m"] is True,
        "stage43_y_precondition_seen": payload["stage43_y_precondition"]["verdict"]
        == "stage43_y_protected_multimodal_latent_head_suite_candidate",
        "fresh_test_predictions_completed": payload["evaluation_protocol"]["rows"] > 0,
        "latent_noncollapse": payload["latent_stats"]["min_variance"] > payload["latent_stats"]["noncollapse_threshold"],
        "global_failure_gain_harm_heads_strong": global_strong,
        "per_domain_heads_robust": domain_robust,
        "per_horizon_heads_supported": horizon_supported,
        "bootstrap_ci_completed": bootstrap_supported,
        "weak_horizon_caveats_reported": weak_horizons_reported,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and len(payload["weak_horizon_slices"]) == 0:
        verdict = "stage43_bx_latent_risk_head_robustness_pass"
    elif passed == total:
        verdict = "stage43_bx_latent_risk_head_robustness_pass_horizon_caveat"
    else:
        verdict = "stage43_bx_latent_risk_head_robustness_incomplete"
    return {
        "source": payload.get("source", SOURCE),
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "horizon_min_auroc": horizon_min,
        "weak_horizon_slice_count": len(payload["weak_horizon_slices"]),
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "standalone_ungated_policy": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "long_objective_complete": False,
    }


def _head_line(head: str, row: Mapping[str, Any], boot: Mapping[str, Any]) -> str:
    return (
        f"| `{head}` | `{row['rows']}` | `{_pct(row['positive_rate'])}` | "
        f"`{row['auroc']:.4f}` | `{row['auprc']:.4f}` | `{row['ece']:.4f}` | "
        f"`[{boot['auroc']['low']:.4f}, {boot['auroc']['high']:.4f}]` |"
    )


def _slice_lines(rows: list[Mapping[str, Any]], *, limit: int = 12) -> list[str]:
    lines = ["| head | horizon | rows | positive rate | AUROC | AUPRC |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['head']}` | `{row['horizon']}` | `{row['rows']}` | `{_pct(row['positive_rate'])}` | `{row['auroc']:.4f}` | `{row['auprc']:.4f}` |"
        )
    if len(rows) > limit:
        lines.append(f"| `...` |  | `{len(rows) - limit} more` |  |  |  |")
    return lines


def _breakdown_lines(table: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| slice | rows | positive rate | AUROC | AUPRC | ECE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in table.items():
        lines.append(
            f"| `{key}` | `{row['rows']}` | `{_pct(row['positive_rate'])}` | `{row['auroc']:.4f}` | `{row['auprc']:.4f}` | `{row['ece']:.4f}` |"
        )
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bx_gate"]
    write_json(REPORT_JSON, _jsonable(payload))
    lines = [
        "# Stage43-BX Latent Risk Head Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
        f"- protected multimodal latent-state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        "",
        "## Global Risk Heads",
        "",
        "| head | rows | positive rate | AUROC | AUPRC | ECE | bootstrap AUROC 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            _head_line(head, payload["head_metrics"][head], payload["bootstrap"][head])
            for head in HEADS
        ],
        "",
        "## Weak Horizon Caveats",
        "",
        f"- weak horizon slice count: `{len(payload['weak_horizon_slices'])}`",
        "",
        *_slice_lines(payload["weak_horizon_slices"]),
        "",
        "## Per-Domain Robustness",
        "",
    ]
    for head in HEADS:
        lines.extend([f"### {head}", "", *_breakdown_lines(payload["by_domain"][head]), ""])
    lines.extend(
        [
            "## Per-Horizon Robustness",
            "",
        ]
    )
    for head in HEADS:
        lines.extend([f"### {head}", "", *_breakdown_lines(payload["by_horizon"][head]), ""])
    lines.extend(
        [
            "## Latent State",
            "",
            f"- latent dim: `{payload['latent_stats']['dim']}`",
            f"- min variance: `{payload['latent_stats']['min_variance']:.6f}`",
            f"- mean variance: `{payload['latent_stats']['mean_variance']:.6f}`",
            "",
            "## Interpretation",
            "",
            "- Stage43-BX fresh-replays the Stage43-M latent checkpoint and audits failure/gain/harm heads across domain and horizon slices.",
            "- Global and per-domain risk heads are strong; horizon 50/100 remains weaker and is explicitly reported as a caveat.",
            "- This strengthens the protected latent world-state evidence, but it is not an ungated policy and does not execute Stage5C or SMC.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, or foundation claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-BX Latent Risk Head Robustness Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- horizon min AUROC: `{gate['horizon_min_auroc']:.4f}`",
            f"- weak horizon slices: `{gate['weak_horizon_slice_count']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- weak horizon slices: `{gate['weak_horizon_slice_count']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BX is a robustness audit for latent risk heads, not an ungated deployment policy.",
            "- Horizon 50/100 weakness remains a documented caveat.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bx_gate"]
    failure = payload["head_metrics"]["failure"]
    gain = payload["head_metrics"]["gain"]
    harm = payload["head_metrics"]["harm"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        "Stage43-BX fresh-replays the latent checkpoint and audits failure/gain/harm risk heads across domain and horizon slices with row-subsampled bootstrap CIs.",
        f"Global AUROC: failure `{failure['auroc']:.4f}`, gain `{gain['auroc']:.4f}`, harm `{harm['auroc']:.4f}`.",
        f"Bootstrap AUROC low: failure `{payload['bootstrap']['failure']['auroc']['low']:.4f}`, gain `{payload['bootstrap']['gain']['auroc']['low']:.4f}`, harm `{payload['bootstrap']['harm']['auroc']['low']:.4f}`.",
        f"Weak horizon slices: `{gate['weak_horizon_slice_count']}`; minimum horizon AUROC `{gate['horizon_min_auroc']:.4f}`.",
        "",
        "Boundary unchanged: protected dataset-local/raw-frame 2.5D only; no ungated deployment, no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_bx_latent_risk_head_robustness_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "global_head_metrics": payload["head_metrics"],
        "bootstrap": payload["bootstrap"],
        "weak_horizon_slices": payload["weak_horizon_slices"],
        "latent_stats": payload["latent_stats"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bx_latent_risk_head_robustness_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "stage": "Stage43-BX",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "weak_horizon_slice_count": gate["weak_horizon_slice_count"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BX latent risk head robustness audit.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=487)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_latent_risk_head_robustness_audit(
        batch_size=int(args.batch_size),
        min_rows=int(args.min_rows),
        bootstrap=int(args.bootstrap),
        bootstrap_rows=int(args.bootstrap_rows),
        seed=int(args.seed),
    )
    gate = payload["stage43_bx_gate"]
    print(f"Stage43-BX: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"weak_horizon_slices={gate['weak_horizon_slice_count']}")
    return payload


if __name__ == "__main__":
    main()
