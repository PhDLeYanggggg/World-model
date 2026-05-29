from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import (
    OUT_DIR,
    ProtectedLatentStateModel,
    _build_split,
    _err_from_delta,
    _jsonable,
    _metrics,
    _predict,
    _select_with_policy,
)


REPORT_JSON = OUT_DIR / "stage43_latent_state_robustness_audit.json"
REPORT_MD = OUT_DIR / "stage43_latent_state_robustness_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_d_latent_robustness_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_D_LATENT_STATE_ROBUSTNESS_AUDIT"
SOURCE = "fresh_stage43_d_latent_state_robustness_audit"
EPS = 1e-8


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _apply_checkpoint_standardization(ds, ckpt: Mapping[str, Any]):
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    return ds


def _improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _easy_degradation(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS) - 1.0))


def _bootstrap_metric(
    selected: np.ndarray,
    floor: np.ndarray,
    mask: np.ndarray,
    *,
    easy: bool = False,
    n: int = 2000,
    seed: int = 4307,
) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"rows": int(len(ids)), "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n), dtype=np.float64)
    for i in range(int(n)):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals[i] = _easy_degradation(selected, floor, np.isin(np.arange(len(selected)), sample)) if easy else _improvement(selected, floor, np.isin(np.arange(len(selected)), sample))
    return {
        "rows": int(len(ids)),
        "mean": _easy_degradation(selected, floor, mask) if easy else _improvement(selected, floor, mask),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "bootstrap_n": int(n),
    }


def _bootstrap_metric_fast(
    selected: np.ndarray,
    floor: np.ndarray,
    ids: np.ndarray,
    *,
    easy: bool = False,
    n: int = 2000,
    seed: int = 4307,
) -> dict[str, Any]:
    if len(ids) < 30:
        return {"rows": int(len(ids)), "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    sel = selected[ids]
    flr = floor[ids]
    vals = np.empty(int(n), dtype=np.float64)
    for i in range(int(n)):
        j = rng.integers(0, len(ids), size=len(ids))
        if easy:
            vals[i] = max(0.0, float(np.mean(sel[j])) / max(float(np.mean(flr[j])), EPS) - 1.0)
        else:
            vals[i] = 1.0 - float(np.mean(sel[j])) / max(float(np.mean(flr[j])), EPS)
    return {
        "rows": int(len(ids)),
        "mean": max(0.0, float(np.mean(sel)) / max(float(np.mean(flr)), EPS) - 1.0) if easy else 1.0 - float(np.mean(sel)) / max(float(np.mean(flr)), EPS),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "bootstrap_n": int(n),
    }


def _slice_table(ds, selected: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for domain in sorted(set(ds.domain.astype(str).tolist())):
        mask = ds.domain.astype(str) == domain
        rows[f"domain:{domain}"] = {**_metrics(ds, selected, switched), "mask_rows": int(mask.sum()), "improvement": _improvement(selected, ds.floor_err, mask)}
    for horizon in sorted(set(ds.horizon.astype(int).tolist())):
        mask = ds.horizon.astype(int) == horizon
        rows[f"horizon:{horizon}"] = {"mask_rows": int(mask.sum()), "improvement": _improvement(selected, ds.floor_err, mask), "switch_rate": float(np.mean(switched[mask])) if int(mask.sum()) else 0.0}
    for name, mask in {
        "hard_failure": ds.hard | ds.failure,
        "easy": ds.easy,
        "not_easy": ~ds.easy,
    }.items():
        rows[f"subset:{name}"] = {
            "mask_rows": int(mask.sum()),
            "improvement": _improvement(selected, ds.floor_err, mask),
            "easy_degradation": _easy_degradation(selected, ds.floor_err, mask) if name == "easy" else 0.0,
            "switch_rate": float(np.mean(switched[mask])) if int(mask.sum()) else 0.0,
        }
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    boot = payload["bootstrap"]
    metrics = payload["metrics"]
    gates = {
        "stage43_c_checkpoint_exists": Path(payload["checkpoint"]).exists(),
        "full_or_larger_test_eval_completed": metrics["rows"] >= 60000,
        "latent_noncollapse": payload["latent_variance"] > 0.01,
        "all_ci_low_positive": boot["all"]["ci_low"] > 0.0,
        "t50_ci_low_positive": boot["t50"]["ci_low"] > 0.0,
        "hard_failure_ci_low_positive": boot["hard_failure"]["ci_low"] > 0.0,
        "easy_ci_high_safe": boot["easy_degradation"]["ci_high"] <= 0.02,
        "domain_scope_recorded": payload["domain_scope"]["test_domains"] == ["UCY"] and payload["domain_scope"]["multi_domain_test"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_d_latent_state_robustness_ucy_pass" if passed == total else "stage43_d_latent_state_robustness_partial",
        "multi_domain_claim_allowed": False,
    }


def run_audit(*, bootstrap_n: int = 2000, max_test: int | None = None, batch_size: int = 2048) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43c = read_json(OUT_DIR / "stage43_protected_latent_eval.json", {})
    checkpoint = Path(stage43c.get("checkpoint", OUT_DIR / "checkpoints/stage43_protected_latent_small.pt"))
    ckpt = _load_checkpoint(checkpoint)
    test = _build_split("test", max_rows=max_test, seed=int(ckpt.get("seed", 431)))
    test = _apply_checkpoint_standardization(test, ckpt)
    model = ProtectedLatentStateModel(test.x.shape[1], hidden_dim=int(ckpt["hidden_dim"]), latent_dim=int(ckpt["latent_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    pred = _predict(model, test, torch.device("cpu"), batch_size)
    policy = stage43c["validation_selected_policy"]["policy"]
    selected, switched = _select_with_policy(test, pred, policy)
    candidate = _err_from_delta(test, pred["delta"])
    metrics = _metrics(test, selected, switched)
    candidate_metrics = _metrics(test, candidate, np.ones(len(test.x), dtype=bool))
    latent_var = float(np.var(pred["latent"], axis=0).mean()) if len(pred["latent"]) else 0.0
    all_ids = np.arange(len(selected))
    boot = {
        "all": _bootstrap_metric_fast(selected, test.floor_err, all_ids, n=bootstrap_n, seed=4307),
        "t50": _bootstrap_metric_fast(selected, test.floor_err, np.where(test.horizon == 50)[0], n=bootstrap_n, seed=4308),
        "t100_raw_frame_diagnostic": _bootstrap_metric_fast(selected, test.floor_err, np.where(test.horizon == 100)[0], n=bootstrap_n, seed=4309),
        "hard_failure": _bootstrap_metric_fast(selected, test.floor_err, np.where(test.hard | test.failure)[0], n=bootstrap_n, seed=4310),
        "easy_degradation": _bootstrap_metric_fast(selected, test.floor_err, np.where(test.easy)[0], easy=True, n=bootstrap_n, seed=4311),
    }
    domains = sorted(set(test.domain.astype(str).tolist()))
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_eval_from_stage43_c_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "stage43_c_verdict": stage43c.get("stage43_c_gate", {}).get("verdict", ""),
        "policy": policy,
        "metrics": metrics,
        "ungated_candidate_metrics": candidate_metrics,
        "bootstrap": boot,
        "slice_table": _slice_table(test, selected, switched),
        "latent_variance": latent_var,
        "domain_scope": {
            "test_domains": domains,
            "multi_domain_test": len(domains) >= 2,
            "limitation": "Current Stage43-D robustness audit evaluates the full held-out UCY test split only; it does not yet prove multi-external-domain robustness.",
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_d_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_d_gate"]
    metrics = payload["metrics"]
    boot = payload["bootstrap"]
    write_md(
        REPORT_MD,
        [
            "# Stage43-D Latent-State Robustness Audit",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- checkpoint: `{payload['checkpoint']}`",
            "- checkpoint committed: `False`",
            f"- latent variance: `{payload['latent_variance']:.6f}`",
            f"- domain scope: `{payload['domain_scope']}`",
            "",
            "## Full Held-Out Test Metrics vs Stage37/Stage42 Floor",
            "",
            f"- rows: `{metrics['rows']}`",
            f"- all improvement: `{metrics['all_improvement_vs_floor']:.6f}`",
            f"- t50 improvement: `{metrics['t50_improvement_vs_floor']:.6f}`",
            f"- t100 raw-frame diagnostic: `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`",
            f"- hard/failure improvement: `{metrics['hard_failure_improvement_vs_floor']:.6f}`",
            f"- easy degradation: `{metrics['easy_degradation_vs_floor']:.6f}`",
            f"- switch rate: `{metrics['switch_rate']:.6f}`",
            "",
            "## Bootstrap CI",
            "",
            "| metric | rows | mean | ci low | ci high |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                f"| {name} | {row['rows']} | {row['mean']:.6f} | {row['ci_low']:.6f} | {row['ci_high']:.6f} |"
                for name, row in boot.items()
            ],
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
            "",
            "This audit supports a UCY held-out dataset-local/raw-frame protected latent-state result only. It does not authorize a multi-domain, metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-D Latent Robustness Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- multi-domain claim allowed: `{gate['multi_domain_claim_allowed']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"stage": "Stage43-D", "source": payload["source"], "verdict": gate["verdict"], "gate": f"{gate['passed']} / {gate['total']}", "generated_at_utc": payload["generated_at_utc"]}), ensure_ascii=False) + "\n")


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_d_gate"]
    metrics = payload["metrics"]
    boot = payload["bootstrap"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"multi_domain_claim_allowed = `{gate['multi_domain_claim_allowed']}`",
        "",
        "Stage43-D re-evaluates the Stage43-C protected latent-state checkpoint on the full held-out UCY test split and adds bootstrap confidence intervals. This is a robustness audit, not a new threshold-tuning run and not a Stage5C/SMC execution.",
        "",
        f"Full UCY test metrics: all `{metrics['all_improvement_vs_floor']:.6f}`, t50 `{metrics['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{metrics['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{metrics['easy_degradation_vs_floor']:.6f}`, switch rate `{metrics['switch_rate']:.6f}`.",
        "",
        f"Bootstrap CI lows: all `{boot['all']['ci_low']:.6f}`, t50 `{boot['t50']['ci_low']:.6f}`, hard/failure `{boot['hard_failure']['ci_low']:.6f}`, easy CI high `{boot['easy_degradation']['ci_high']:.6f}`.",
        "",
        "Scope limitation: this proves UCY held-out dataset-local/raw-frame robustness only; multi-domain robustness remains a next gate.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_d_latent_state_robustness_audit"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "metrics": payload["metrics"],
        "bootstrap": payload["bootstrap"],
        "multi_domain_claim_allowed": gate["multi_domain_claim_allowed"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args(argv)
    max_test = None if int(args.max_test) <= 0 else int(args.max_test)
    return run_audit(bootstrap_n=int(args.bootstrap), max_test=max_test, batch_size=int(args.batch_size))


if __name__ == "__main__":
    result = main()
    gate = result["stage43_d_gate"]
    print(f"Stage43-D latent robustness audit: {gate['verdict']} ({gate['passed']}/{gate['total']})")
