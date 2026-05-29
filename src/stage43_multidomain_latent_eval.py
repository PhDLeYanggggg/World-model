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
from src.stage43_latent_state_robustness_audit import _bootstrap_metric_fast
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


REPORT_JSON = OUT_DIR / "stage43_multidomain_latent_eval.json"
REPORT_MD = OUT_DIR / "stage43_multidomain_latent_eval.md"
GATE_MD = OUT_DIR / "stage43_stage_e_multidomain_latent_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_E_MULTIDOMAIN_LATENT_EVAL"
SOURCE = "fresh_stage43_e_multidomain_latent_eval"
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


def _domain_metrics(ds, selected: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in sorted(set(ds.domain.astype(str).tolist())):
        mask = ds.domain.astype(str) == domain
        hard_failure = ds.hard | ds.failure
        out[domain] = {
            "rows": int(mask.sum()),
            "horizon_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(ds.horizon[mask].astype(int), return_counts=True))
            },
            "all_improvement": _improvement(selected, ds.floor_err, mask),
            "t50_improvement": _improvement(selected, ds.floor_err, mask & (ds.horizon == 50)),
            "t100_raw_frame_diagnostic": _improvement(selected, ds.floor_err, mask & (ds.horizon == 100)),
            "hard_failure_improvement": _improvement(selected, ds.floor_err, mask & hard_failure),
            "easy_degradation": _easy_degradation(selected, ds.floor_err, mask & ds.easy),
            "switch_rate": float(np.mean(switched[mask])) if int(mask.sum()) else 0.0,
            "role": "train_seen" if ds.split == "train" else "validation_seen" if ds.split == "val" else "heldout_test",
        }
    return out


def _eval_split(model, ckpt: Mapping[str, Any], split: str, policy: Mapping[str, float], batch_size: int) -> dict[str, Any]:
    ds = _build_split(split, max_rows=None, seed=int(ckpt.get("seed", 431)))
    ds = _apply_checkpoint_standardization(ds, ckpt)
    pred = _predict(model, ds, torch.device("cpu"), batch_size)
    selected, switched = _select_with_policy(ds, pred, policy)
    candidate = _err_from_delta(ds, pred["delta"])
    metrics = _metrics(ds, selected, switched)
    candidate_metrics = _metrics(ds, candidate, np.ones(len(ds.x), dtype=bool))
    return {
        "split": split,
        "role": "train_seen" if split == "train" else "validation_seen" if split == "val" else "heldout_test",
        "metrics": metrics,
        "ungated_candidate_metrics": candidate_metrics,
        "domain_metrics": _domain_metrics(ds, selected, switched),
        "domains": sorted(set(ds.domain.astype(str).tolist())),
        "rows": int(len(ds.x)),
        "latent_variance": float(np.var(pred["latent"], axis=0).mean()) if len(pred["latent"]) else 0.0,
    }


def _heldout_coverage(split_payloads: Mapping[str, Any]) -> dict[str, Any]:
    train_domains = set(split_payloads["train"]["domains"])
    val_domains = set(split_payloads["val"]["domains"])
    test_domains = set(split_payloads["test"]["domains"])
    all_domains = sorted(train_domains | val_domains | test_domains)
    missing_heldout = sorted((train_domains | val_domains) - test_domains)
    return {
        "all_domains_observed": all_domains,
        "train_domains": sorted(train_domains),
        "val_domains": sorted(val_domains),
        "test_domains": sorted(test_domains),
        "multi_domain_heldout_test": len(test_domains) >= 2,
        "missing_heldout_domains": missing_heldout,
        "current_final_test_scope": "UCY-only heldout",
        "required_next_split": "source-level or scene-level heldout split containing ETH_UCY, TrajNet, and UCY without test endpoint goal leakage",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    coverage = payload["heldout_coverage"]
    test = payload["splits"]["test"]["metrics"]
    gates = {
        "checkpoint_exists": Path(payload["checkpoint"]).exists(),
        "train_val_test_evaluated": set(payload["splits"].keys()) == {"train", "val", "test"},
        "all_observed_domains_reported": set(coverage["all_domains_observed"]) >= {"ETH_UCY", "TrajNet", "UCY"},
        "ucy_heldout_positive": test["all_improvement_vs_floor"] > 0.0
        and test["t50_improvement_vs_floor"] > 0.0
        and test["hard_failure_improvement_vs_floor"] > 0.0
        and test["easy_degradation_vs_floor"] <= 0.02,
        "seen_domain_diagnostics_completed": bool(payload["splits"]["train"]["domain_metrics"])
        and bool(payload["splits"]["val"]["domain_metrics"]),
        "multi_domain_heldout_blocker_recorded": coverage["multi_domain_heldout_test"] is False
        and bool(coverage["missing_heldout_domains"]),
        "no_multi_domain_overclaim": payload["claim_boundary"]["multi_domain_latent_claim"] is False,
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
        "verdict": "stage43_e_multidomain_latent_eval_blocker_mapped" if passed == total else "stage43_e_multidomain_latent_eval_partial",
        "multi_domain_latent_candidate": False,
    }


def run_eval(*, bootstrap: int = 500, batch_size: int = 4096) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43c = read_json(OUT_DIR / "stage43_protected_latent_eval.json", {})
    checkpoint = Path(stage43c.get("checkpoint", OUT_DIR / "checkpoints/stage43_protected_latent_small.pt"))
    ckpt = _load_checkpoint(checkpoint)
    model = ProtectedLatentStateModel(int(ckpt["input_dim"]), hidden_dim=int(ckpt["hidden_dim"]), latent_dim=int(ckpt["latent_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    policy = stage43c["validation_selected_policy"]["policy"]
    splits = {split: _eval_split(model, ckpt, split, policy, batch_size) for split in ["train", "val", "test"]}
    coverage = _heldout_coverage(splits)
    test_ds = _build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431)))
    test_ds = _apply_checkpoint_standardization(test_ds, ckpt)
    test_pred = _predict(model, test_ds, torch.device("cpu"), batch_size)
    test_selected, _ = _select_with_policy(test_ds, test_pred, policy)
    boot = {
        "test_all": _bootstrap_metric_fast(test_selected, test_ds.floor_err, np.arange(len(test_selected)), n=bootstrap, seed=43101),
        "test_t50": _bootstrap_metric_fast(test_selected, test_ds.floor_err, np.where(test_ds.horizon == 50)[0], n=bootstrap, seed=43102),
        "test_hard_failure": _bootstrap_metric_fast(test_selected, test_ds.floor_err, np.where(test_ds.hard | test_ds.failure)[0], n=bootstrap, seed=43103),
        "test_easy_degradation": _bootstrap_metric_fast(test_selected, test_ds.floor_err, np.where(test_ds.easy)[0], easy=True, n=bootstrap, seed=43104),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_eval_from_stage43_c_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "policy": policy,
        "splits": splits,
        "heldout_coverage": coverage,
        "bootstrap": boot,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "multi_domain_latent_claim": False,
            "dataset_local_raw_frame_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_e_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_e_gate"]
    coverage = payload["heldout_coverage"]
    test = payload["splits"]["test"]["metrics"]
    lines = [
        "# Stage43-E Multidomain Latent Evaluation",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- multi-domain latent candidate: `{gate['multi_domain_latent_candidate']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        "- checkpoint committed: `False`",
        "",
        "## Heldout Coverage",
        "",
        f"- train domains: `{coverage['train_domains']}`",
        f"- val domains: `{coverage['val_domains']}`",
        f"- test domains: `{coverage['test_domains']}`",
        f"- missing heldout domains: `{coverage['missing_heldout_domains']}`",
        f"- required next split: {coverage['required_next_split']}",
        "",
        "## Split Metrics",
        "",
        "| split | role | rows | domains | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {split} | `{row['role']}` | {row['rows']} | `{row['domains']}` | {row['metrics']['all_improvement_vs_floor']:.6f} | {row['metrics']['t50_improvement_vs_floor']:.6f} | {row['metrics']['t100_raw_frame_diagnostic_vs_floor']:.6f} | {row['metrics']['hard_failure_improvement_vs_floor']:.6f} | {row['metrics']['easy_degradation_vs_floor']:.6f} | {row['metrics']['switch_rate']:.6f} |"
            for split, row in payload["splits"].items()
        ],
        "",
        "## UCY Heldout Test",
        "",
        f"- all improvement: `{test['all_improvement_vs_floor']:.6f}`",
        f"- t50 improvement: `{test['t50_improvement_vs_floor']:.6f}`",
        f"- t100 raw-frame diagnostic: `{test['t100_raw_frame_diagnostic_vs_floor']:.6f}`",
        f"- hard/failure improvement: `{test['hard_failure_improvement_vs_floor']:.6f}`",
        f"- easy degradation: `{test['easy_degradation_vs_floor']:.6f}`",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "Conclusion: current Stage43 latent-state model has UCY heldout support and seen/validation domain diagnostics, but it is not yet a multi-domain latent world model candidate because ETH_UCY and TrajNet are not held-out test domains in this split.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-E Multidomain Latent Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- multi-domain latent candidate: `{gate['multi_domain_latent_candidate']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"stage": "Stage43-E", "source": payload["source"], "verdict": gate["verdict"], "gate": f"{gate['passed']} / {gate['total']}", "generated_at_utc": payload["generated_at_utc"]}), ensure_ascii=False) + "\n")


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_e_gate"]
    coverage = payload["heldout_coverage"]
    test = payload["splits"]["test"]["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"multi_domain_latent_candidate = `{gate['multi_domain_latent_candidate']}`",
        "",
        "Stage43-E evaluates the Stage43 protected latent-state checkpoint across the currently available train/val/test domains. It confirms UCY heldout support but refuses a multi-domain claim because ETH_UCY and TrajNet are not present as held-out test domains in the current Stage43 split.",
        "",
        f"UCY heldout: all `{test['all_improvement_vs_floor']:.6f}`, t50 `{test['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{test['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{test['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{test['easy_degradation_vs_floor']:.6f}`.",
        "",
        f"Missing heldout domains for a real multi-domain latent claim: `{coverage['missing_heldout_domains']}`. Next required step is a source-level or scene-level split containing ETH_UCY, TrajNet, and UCY as held-out domains without test endpoint goal leakage.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_e_multidomain_latent_eval"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "multi_domain_latent_candidate": gate["multi_domain_latent_candidate"],
        "heldout_coverage": payload["heldout_coverage"],
        "ucy_test_metrics": payload["splits"]["test"]["metrics"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    return run_eval(bootstrap=int(args.bootstrap), batch_size=int(args.batch_size))


if __name__ == "__main__":
    result = main()
    gate = result["stage43_e_gate"]
    print(f"Stage43-E multidomain latent eval: {gate['verdict']} ({gate['passed']}/{gate['total']})")
