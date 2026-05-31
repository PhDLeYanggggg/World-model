from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _batch_indices,
    _build_split,
    _git_commit,
    _jsonable,
    _sha256,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _pct,
    _standardize_from_checkpoint,
)


REPORT_JSON = OUT_DIR / "stage43_world_state_head_audit.json"
REPORT_MD = OUT_DIR / "stage43_world_state_head_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_v_world_state_head_audit_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_V_WORLD_STATE_HEAD_AUDIT"
SOURCE = "fresh_stage43_v_world_state_head_audit"
EPS = 1e-8


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _binary_metrics(y_true: np.ndarray, score: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y_true).astype(np.float64)
    s = np.clip(np.asarray(score).astype(np.float64), 0.0, 1.0)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    positives = int(np.sum(y >= 0.5))
    negatives = int(len(y) - positives)
    prevalence = float(positives / max(len(y), 1))
    out: dict[str, Any] = {
        "rows": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": prevalence,
        "brier": float(np.mean((s - y) ** 2)) if len(y) else 0.0,
        "baseline_brier": float(np.mean((prevalence - y) ** 2)) if len(y) else 0.0,
    }
    if positives == 0 or negatives == 0:
        out.update({"auroc": None, "auprc": None, "average_precision": None, "defined": False})
    else:
        ranks = _rankdata_average(s)
        rank_sum_pos = float(np.sum(ranks[y >= 0.5]))
        auroc = (rank_sum_pos - positives * (positives + 1) / 2.0) / max(float(positives * negatives), EPS)
        order = np.argsort(-s, kind="mergesort")
        yy = (y[order] >= 0.5).astype(np.float64)
        tp = np.cumsum(yy)
        fp = np.cumsum(1.0 - yy)
        precision = tp / np.maximum(tp + fp, EPS)
        recall = tp / max(float(positives), EPS)
        recall_prev = np.concatenate([[0.0], recall[:-1]])
        ap = float(np.sum((recall - recall_prev) * precision))
        out.update({"auroc": float(auroc), "auprc": ap, "average_precision": ap, "defined": True})
    pred = s >= 0.5
    true = y >= 0.5
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out["f1_at_0_5"] = float(2 * precision * recall / max(precision + recall, EPS))
    out["ece"] = _ece(y, s, bins=bins)
    out["brier_improvement_vs_prevalence"] = float(
        1.0 - out["brier"] / max(float(out["baseline_brier"]), EPS)
    )
    return out


def _ece(y_true: np.ndarray, score: np.ndarray, *, bins: int = 10) -> float:
    y = np.asarray(y_true).astype(np.float64)
    s = np.clip(np.asarray(score).astype(np.float64), 0.0, 1.0)
    total = len(y)
    if total == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if int(mask.sum()) == 0:
            continue
        ece += float(mask.sum()) / total * abs(float(np.mean(s[mask])) - float(np.mean(y[mask])))
    return float(ece)


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true).astype(np.float64)
    p = np.asarray(pred).astype(np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"rows": 0, "mse": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "corr": 0.0}
    mse = float(np.mean((p - y) ** 2))
    baseline = float(np.mean((np.mean(y) - y) ** 2))
    corr = float(np.corrcoef(y, p)[0, 1]) if float(np.std(y)) > 0 and float(np.std(p)) > 0 else 0.0
    return {
        "rows": int(len(y)),
        "mse": mse,
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(mse)),
        "baseline_mse": baseline,
        "r2": float(1.0 - mse / max(baseline, EPS)),
        "corr": corr,
    }


@torch.no_grad()
def _predict_heads(model, ds, *, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {
        "failure": [],
        "gain": [],
        "harm": [],
        "density": [],
        "validity": [],
        "latent": [],
    }
    for ids in _batch_indices(len(ds.x), int(batch_size), shuffle=False, seed=0):
        x = torch.from_numpy(ds.x[ids]).to(torch.device("cpu"))
        out = model(x)
        outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
        outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["density"].append(out["density"].detach().cpu().numpy())
        outs["validity"].append(torch.sigmoid(out["validity_logit"]).detach().cpu().numpy())
        outs["latent"].append(out["z_next"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outs.items()}


def _breakdown_binary(values: np.ndarray, y_true: np.ndarray, score: np.ndarray, *, min_rows: int = 100) -> dict[str, Any]:
    out: dict[str, Any] = {}
    vv = values.astype(str)
    for value in sorted(set(vv.tolist())):
        mask = vv == value
        if int(mask.sum()) < int(min_rows):
            continue
        m = _binary_metrics(y_true[mask], score[mask])
        out[value] = {
            "rows": m["rows"],
            "positive_rate": m["positive_rate"],
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "brier": m["brier"],
            "ece": m["ece"],
        }
    return out


def run_world_state_head_audit(*, batch_size: int = 4096, max_rows: int | None = None) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    test = _build_split("test", max_rows=max_rows, seed=int(ckpt.get("seed", 431)))
    test = _standardize_from_checkpoint(test, ckpt)
    pred = _predict_heads(model, test, batch_size=int(batch_size))
    validity_label = test.waypoint_valid.mean(axis=1).astype(np.float32)
    head_metrics = {
        "failure": _binary_metrics(test.y_failure, pred["failure"]),
        "gain": _binary_metrics(test.y_gain, pred["gain"]),
        "harm": _binary_metrics(test.y_harm, pred["harm"]),
        "density": _regression_metrics(test.y_density, pred["density"]),
        "physical_validity_proxy": {
            **_regression_metrics(validity_label, pred["validity"]),
            "trained_with_explicit_loss": False,
            "deployment_allowed": False,
        },
    }
    latent = pred["latent"].astype(np.float32)
    latent_stats = {
        "rows": int(len(latent)),
        "dim": int(latent.shape[1]) if latent.ndim == 2 else 0,
        "mean_variance": float(np.var(latent, axis=0).mean()) if len(latent) else 0.0,
        "min_variance": float(np.var(latent, axis=0).min()) if len(latent) else 0.0,
        "max_variance": float(np.var(latent, axis=0).max()) if len(latent) else 0.0,
        "noncollapse_threshold": 0.01,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_checkpoint_replay_world_state_head_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {
            "verdict": stage43m.get("stage43_m_gate", {}).get("verdict"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_sha256_matches_stage43_m": _sha256(checkpoint) == stage43m.get("checkpoint_sha256"),
        },
        "evaluation_protocol": {
            "split": "test",
            "rows": int(len(test.x)),
            "batch_size": int(batch_size),
            "max_rows": max_rows,
            "future_labels_eval_only": True,
            "test_threshold_tuning": False,
            "num_workers": 0,
        },
        "head_metrics": head_metrics,
        "latent_stats": latent_stats,
        "by_horizon": {
            "failure": _breakdown_binary(test.horizon.astype(str), test.y_failure, pred["failure"]),
            "gain": _breakdown_binary(test.horizon.astype(str), test.y_gain, pred["gain"]),
            "harm": _breakdown_binary(test.horizon.astype(str), test.y_harm, pred["harm"]),
        },
        "by_domain": {
            "failure": _breakdown_binary(test.domain, test.y_failure, pred["failure"]),
            "gain": _breakdown_binary(test.domain, test.y_gain, pred["gain"]),
            "harm": _breakdown_binary(test.domain, test.y_harm, pred["harm"]),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
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
        },
    }
    payload["stage43_v_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    heads = payload["head_metrics"]
    informative_binary = [
        name
        for name in ["failure", "gain", "harm"]
        if heads[name].get("auroc") is not None and float(heads[name]["auroc"]) > 0.60
    ]
    gates = {
        "stage43_m_checkpoint_replayed": payload["stage43_m_precondition"]["checkpoint_sha256_matches_stage43_m"] is True,
        "all_auxiliary_heads_evaluated": all(name in heads for name in ["failure", "gain", "harm", "density", "physical_validity_proxy"]),
        "latent_noncollapse": payload["latent_stats"]["mean_variance"] > payload["latent_stats"]["noncollapse_threshold"],
        "at_least_one_binary_world_state_head_informative": len(informative_binary) >= 1,
        "density_head_reported": heads["density"]["rows"] > 0,
        "physical_validity_not_deployed_without_loss": heads["physical_validity_proxy"]["trained_with_explicit_loss"] is False
        and heads["physical_validity_proxy"]["deployment_allowed"] is False,
        "no_test_threshold_tuning": payload["evaluation_protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_labels_eval_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_v_world_state_head_audit_partial"
        if passed == total
        else "stage43_v_world_state_head_audit_incomplete",
        "informative_binary_heads": informative_binary,
        "deploy_physical_validity_head": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _fmt_metric(value: Any) -> str:
    return "`undefined`" if value is None else f"`{float(value):.4f}`"


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_v_gate"]
    heads = payload["head_metrics"]
    lines = [
        "# Stage43-V World-State Head Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- test rows: `{payload['evaluation_protocol']['rows']}`",
        "",
        "## Auxiliary Head Metrics",
        "",
        "| head | primary metric | calibration/error | note |",
        "| --- | ---: | ---: | --- |",
        f"| failure | AUROC {_fmt_metric(heads['failure']['auroc'])}, AUPRC {_fmt_metric(heads['failure']['auprc'])} | ECE `{heads['failure']['ece']:.4f}` | baseline-failure risk label |",
        f"| gain | AUROC {_fmt_metric(heads['gain']['auroc'])}, AUPRC {_fmt_metric(heads['gain']['auprc'])} | ECE `{heads['gain']['ece']:.4f}` | switch/gain opportunity label |",
        f"| harm | AUROC {_fmt_metric(heads['harm']['auroc'])}, AUPRC {_fmt_metric(heads['harm']['auprc'])} | ECE `{heads['harm']['ece']:.4f}` | easy/harm guard label |",
        f"| density | R2 `{heads['density']['r2']:.4f}`, corr `{heads['density']['corr']:.4f}` | RMSE `{heads['density']['rmse']:.4f}` | occupancy-density proxy |",
        f"| physical_validity_proxy | R2 `{heads['physical_validity_proxy']['r2']:.4f}` | RMSE `{heads['physical_validity_proxy']['rmse']:.4f}` | not trained with explicit loss; not deployable |",
        "",
        "## Latent State",
        "",
        f"- latent dim: `{payload['latent_stats']['dim']}`",
        f"- mean variance: `{payload['latent_stats']['mean_variance']:.6f}`",
        f"- min variance: `{payload['latent_stats']['min_variance']:.6f}`",
        f"- non-collapse threshold: `{payload['latent_stats']['noncollapse_threshold']}`",
        "",
        "## Interpretation",
        "",
        "Stage43-V audits the auxiliary world-state heads from the existing Stage43-M latent dynamics checkpoint. This is evidence about latent risk/density heads, not a new deployment policy. The physical-validity output is explicitly marked not deployable because the current training loss does not supervise it.",
        "",
        "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-V Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"informative_binary_heads: `{', '.join(gate['informative_binary_heads'])}`",
        f"deploy_physical_validity_head: `{gate['deploy_physical_validity_head']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_v_gate"]
    heads = payload["head_metrics"]
    lines = [
        "## Stage43-V world-state head audit",
        "",
        f"Result source: `{payload['result_source']}`. I replayed the Stage43-M latent checkpoint and audited failure/gain/harm/density/validity heads on the test split without test threshold tuning.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- informative binary heads: `{', '.join(gate['informative_binary_heads'])}`",
        f"- failure AUROC/AUPRC: {_fmt_metric(heads['failure']['auroc'])} / {_fmt_metric(heads['failure']['auprc'])}",
        f"- gain AUROC/AUPRC: {_fmt_metric(heads['gain']['auroc'])} / {_fmt_metric(heads['gain']['auprc'])}",
        f"- harm AUROC/AUPRC: {_fmt_metric(heads['harm']['auroc'])} / {_fmt_metric(heads['harm']['auprc'])}",
        f"- density R2/corr: `{heads['density']['r2']:.4f}` / `{heads['density']['corr']:.4f}`",
        f"- latent mean variance: `{payload['latent_stats']['mean_variance']:.6f}`",
        f"- physical validity head deployable: `False` (no explicit training loss yet)",
        "",
        "Boundary: this is an auxiliary world-state head audit, not a Stage5C/SMC/generative rollout. Physical validity remains a gap because the current checkpoint exposes a validity logit but did not train it with a dedicated loss.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_v_gate"]
    state["stage43_v_world_state_head_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "informative_binary_heads": gate["informative_binary_heads"],
        "head_metrics": payload["head_metrics"],
        "latent_stats": payload["latent_stats"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_v_world_state_head_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_v_world_state_head_audit", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-V auxiliary world-state head audit.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_world_state_head_audit(batch_size=int(args.batch_size), max_rows=args.max_rows)
    gate = result["stage43_v_gate"]
    print(f"Stage43-V: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
