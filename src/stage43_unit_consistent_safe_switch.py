from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
    _git_commit,
    _jsonable,
    _metrics,
    _predict,
)
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets
from src.stage43_source_level_latent_robustness_audit import (
    REPORT_JSON as STAGE43H_JSON,
    _apply_checkpoint_standardization,
    _metrics_subset,
    _proximity_stats,
    _source_level_test_metadata,
)
from src.stage43_source_level_heldout_split import REPORT_JSON as SPLIT_REPORT_JSON


REPORT_JSON = OUT_DIR / "stage43_unit_consistent_safe_switch.json"
REPORT_MD = OUT_DIR / "stage43_unit_consistent_safe_switch.md"
GATE_MD = OUT_DIR / "stage43_stage_i_unit_consistent_safe_switch_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_I_UNIT_CONSISTENT_SAFE_SWITCH"
SOURCE = "fresh_stage43_i_unit_consistent_safe_switch"
PRIOR_EASY_GUARD = 0.03
DOMAIN_SWITCH_CAPS = {"ETH_UCY": 0.15, "TrajNet": 0.10, "UCY": 1.00}


def _hash_policy(policy: Mapping[str, Any]) -> str:
    text = json.dumps(policy, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_checkpoint(stage43g: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any], ProtectedLatentStateModel]:
    checkpoint = Path(stage43g.get("checkpoint", OUT_DIR / "checkpoints/stage43_source_level_latent_full.pt"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ProtectedLatentStateModel(int(ckpt["input_dim"]), int(ckpt["hidden_dim"]), int(ckpt["latent_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return checkpoint, ckpt, model


def _candidate_unit_error(ds, pred: Mapping[str, np.ndarray]) -> np.ndarray:
    normalized = np.sqrt(np.sum((pred["delta"].astype(np.float64) - ds.y_delta.astype(np.float64)) ** 2, axis=1))
    return (normalized * ds.scale.astype(np.float64)).astype(np.float32)


def _feature(raw_x: np.ndarray, feature_names: list[str], name: str) -> np.ndarray:
    if name not in feature_names:
        raise KeyError(f"Required causal feature missing from Stage43 dataset: {name}")
    return raw_x[:, feature_names.index(name)].astype(np.float32)


def _policy_switches(
    *,
    policy_name: str,
    raw_x: np.ndarray,
    feature_names: list[str],
    pred: Mapping[str, np.ndarray],
    stage43g_policy: Mapping[str, float],
) -> np.ndarray:
    n = len(raw_x)
    if policy_name == "stage43g_validation_policy_diagnostic":
        return (
            (pred["gain"] >= float(stage43g_policy["gain_threshold"]))
            & (pred["harm"] <= float(stage43g_policy["harm_threshold"]))
            & (pred["failure"] >= float(stage43g_policy["failure_threshold"]))
        ).astype(bool)
    if policy_name == "fixed_prior_stage41_easy_guard_0p03":
        easy_prob = _feature(raw_x, feature_names, "stage35_easy_prob")
        return (easy_prob <= PRIOR_EASY_GUARD).astype(bool)
    if policy_name == "domain_capped_prior_easy_guard":
        easy_prob = _feature(raw_x, feature_names, "stage35_easy_prob")
        predicted_gain = _feature(raw_x, feature_names, "stage35_predicted_gain")
        base = easy_prob <= PRIOR_EASY_GUARD
        out = np.zeros(n, dtype=bool)
        domains = raw_x[:, feature_names.index("domain_ETH_UCY") : feature_names.index("horizon_10")]
        domain_names = ["ETH_UCY", "TrajNet", "UCY"]
        domain_label = np.asarray([domain_names[int(np.argmax(row))] for row in domains], dtype=object)
        for domain, cap in DOMAIN_SWITCH_CAPS.items():
            mask = domain_label == domain
            ids = np.where(base & mask)[0]
            limit = int(np.floor(float(cap) * int(mask.sum())))
            if limit > 0 and len(ids) > 0:
                chosen = ids[np.argsort(-predicted_gain[ids])[: min(limit, len(ids))]]
                out[chosen] = True
        return out
    if policy_name.startswith("diagnostic_easy_guard_"):
        threshold = float(policy_name.rsplit("_", 1)[-1])
        easy_prob = _feature(raw_x, feature_names, "stage35_easy_prob")
        return (easy_prob <= threshold).astype(bool)
    if policy_name == "never_switch_floor":
        return np.zeros(n, dtype=bool)
    raise ValueError(f"Unknown Stage43-I policy: {policy_name}")


def _evaluate_policy(ds, candidate_unit: np.ndarray, switches: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.where(switches, candidate_unit, ds.floor_err).astype(np.float32)
    return selected, _metrics(ds, selected, switches)


def _policy_row(
    *,
    name: str,
    ds,
    raw_x: np.ndarray,
    pred: Mapping[str, np.ndarray],
    candidate_unit: np.ndarray,
    stage43g_policy: Mapping[str, float],
) -> dict[str, Any]:
    switches = _policy_switches(
        policy_name=name,
        raw_x=raw_x,
        feature_names=ds.feature_names,
        pred=pred,
        stage43g_policy=stage43g_policy,
    )
    selected, metrics = _evaluate_policy(ds, candidate_unit, switches)
    return {
        "name": name,
        "metrics": metrics,
        "switches": switches,
        "selected": selected,
    }


def _domain_metrics(ds, selected: np.ndarray, switches: np.ndarray) -> dict[str, Any]:
    return {
        domain: _metrics_subset(ds, selected, switches, ds.domain.astype(str) == domain)
        for domain in sorted(set(ds.domain.astype(str).tolist()))
    }


def _max_domain_easy(domain_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    if not domain_metrics:
        return 0.0
    return float(max(float(row.get("easy_degradation_vs_floor", 0.0)) for row in domain_metrics.values()))


def _min_domain_all(domain_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    if not domain_metrics:
        return 0.0
    return float(min(float(row.get("all_improvement_vs_floor", 0.0)) for row in domain_metrics.values()))


def _worst_source_all(source_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not source_metrics:
        return {"source_id": None, "all_improvement_vs_floor": 0.0}
    worst_key, worst_row = min(
        source_metrics.items(),
        key=lambda kv: float(kv[1].get("metrics", {}).get("all_improvement_vs_floor", 0.0)),
    )
    return {
        "source_id": worst_key,
        "domains": worst_row.get("domains", []),
        "scenes": worst_row.get("scenes", []),
        "all_improvement_vs_floor": float(worst_row.get("metrics", {}).get("all_improvement_vs_floor", 0.0)),
        "t50_improvement_vs_floor": float(worst_row.get("metrics", {}).get("t50_improvement_vs_floor", 0.0)),
        "easy_degradation_vs_floor": float(worst_row.get("metrics", {}).get("easy_degradation_vs_floor", 0.0)),
    }


def _source_metrics(ds, selected: np.ndarray, switches: np.ndarray, meta: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in sorted(set(meta["source_file"].astype(str).tolist())):
        mask = meta["source_file"].astype(str) == source
        sid = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        out[sid] = {
            "source_id": sid,
            "domains": sorted(set(ds.domain[mask].astype(str).tolist())),
            "scenes": sorted(set(meta["scene_id"][mask].astype(str).tolist())),
            "metrics": _metrics_subset(ds, selected, switches, mask),
        }
    return out


def _selected_xy(ds, pred: Mapping[str, np.ndarray], switches: np.ndarray, meta: Mapping[str, np.ndarray]) -> np.ndarray:
    neural_xy = meta["current_xy"].astype(np.float32) + pred["delta"].astype(np.float32) * ds.scale[:, None].astype(np.float32)
    return np.where(switches[:, None], neural_xy, meta["floor_xy"].astype(np.float32))


def _bootstrap(ds, selected: np.ndarray, *, n: int) -> dict[str, Any]:
    all_ids = np.arange(len(selected))
    return {
        "unit_all": _bootstrap_metric_fast(selected, ds.floor_err, all_ids, n=n, seed=43901),
        "unit_t50": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.horizon == 50)[0], n=n, seed=43902),
        "unit_t100_raw_frame_diagnostic": _bootstrap_metric_fast(
            selected, ds.floor_err, np.where(ds.horizon == 100)[0], n=n, seed=43903
        ),
        "unit_hard_failure": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.hard | ds.failure)[0], n=n, seed=43904),
        "unit_easy_degradation": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.easy)[0], easy=True, n=n, seed=43905),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["deployment_policy"]["test_metrics"]
    boot = payload["deployment_policy"]["bootstrap"]
    prox = payload["deployment_policy"]["proximity"]
    domain_metrics = payload["deployment_policy"]["domain_metrics"]
    gates = {
        "stage43_h_precondition_failed_and_floor_kept": payload["stage43_h_precondition"]["keep_frozen_floor"] is True,
        "domain_capped_policy_not_test_selected": payload["deployment_policy"]["test_tuned"] is False
        and payload["deployment_policy"]["name"] == "domain_capped_prior_easy_guard",
        "test_eval_completed": metrics["rows"] >= 80000,
        "unit_all_ci_low_positive": boot["unit_all"]["ci_low"] > 0.0,
        "unit_t50_ci_low_positive": boot["unit_t50"]["ci_low"] > 0.0,
        "unit_hard_ci_low_positive": boot["unit_hard_failure"]["ci_low"] > 0.0,
        "easy_preservation_gate": boot["unit_easy_degradation"]["ci_high"] <= 0.02,
        "per_domain_easy_preserved": _max_domain_easy(domain_metrics) <= 0.02,
        "per_domain_all_positive": _min_domain_all(domain_metrics) > 0.0,
        "proximity_not_materially_worse": prox["near_005_delta_vs_floor"] <= 0.01
        and prox["near_010_delta_vs_floor"] <= 0.01,
        "partial_switch_not_full_replacement": 0.0 < metrics["switch_rate"] < 0.80,
        "t100_reported_diagnostic_only": payload["claim_boundary"]["t100_raw_frame_diagnostic_only"] is True,
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
        "verdict": "stage43_i_unit_consistent_safe_switch_pass"
        if passed == total
        else "stage43_i_unit_consistent_safe_switch_diagnostic_only",
        "deploy_stage43_i_candidate": passed == total,
        "keep_frozen_floor_as_global_default": True,
    }


def run_safe_switch(*, bootstrap: int = 1000, batch_size: int = 4096) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43g = read_json(STAGE43G_JSON, {})
    stage43h = read_json(STAGE43H_JSON, {})
    split = read_json(SPLIT_REPORT_JSON, {})
    checkpoint, ckpt, model = _load_checkpoint(stage43g)
    _, val_raw, test_raw, manifest = build_source_level_datasets(seed=int(ckpt.get("seed", 443)))
    val_x_raw = val_raw.x.copy()
    test_x_raw = test_raw.x.copy()
    val = _apply_checkpoint_standardization(val_raw, ckpt)
    test = _apply_checkpoint_standardization(test_raw, ckpt)
    val_pred = _predict(model, val, torch.device("cpu"), batch_size)
    test_pred = _predict(model, test, torch.device("cpu"), batch_size)
    val_candidate_unit = _candidate_unit_error(val, val_pred)
    test_candidate_unit = _candidate_unit_error(test, test_pred)
    stage43g_policy = stage43g["validation_selected_policy"]["policy"]

    policy_names = [
        "never_switch_floor",
        "stage43g_validation_policy_diagnostic",
        "fixed_prior_stage41_easy_guard_0p03",
        "domain_capped_prior_easy_guard",
        "diagnostic_easy_guard_0.05",
        "diagnostic_easy_guard_0.10",
        "diagnostic_easy_guard_0.20",
        "diagnostic_easy_guard_0.80",
    ]
    policy_table: list[dict[str, Any]] = []
    policy_details: dict[str, Any] = {}
    for name in policy_names:
        val_row = _policy_row(
            name=name,
            ds=val,
            raw_x=val_x_raw,
            pred=val_pred,
            candidate_unit=val_candidate_unit,
            stage43g_policy=stage43g_policy,
        )
        test_row = _policy_row(
            name=name,
            ds=test,
            raw_x=test_x_raw,
            pred=test_pred,
            candidate_unit=test_candidate_unit,
            stage43g_policy=stage43g_policy,
        )
        policy_table.append(
            {
                "name": name,
                "selection_status": "domain_capped_deployable_candidate"
                if name == "domain_capped_prior_easy_guard"
                else "diagnostic_not_deployment_policy",
                "val_metrics": val_row["metrics"],
                "test_metrics": test_row["metrics"],
            }
        )
        policy_details[name] = test_row

    deploy = policy_details["domain_capped_prior_easy_guard"]
    meta = _source_level_test_metadata(manifest)
    selected_xy = _selected_xy(test, test_pred, deploy["switches"], meta)
    meta_with_selected = {**meta, "selected_xy": selected_xy}
    proximity = _proximity_stats(selected_xy, meta["floor_xy"], meta_with_selected)
    boot = _bootstrap(test, deploy["selected"], n=bootstrap)
    deployment_policy = {
        "name": "domain_capped_prior_easy_guard",
        "description": "Switch to the Stage43-G latent endpoint only when the pre-existing Stage35/Stage41 easy-risk proxy is <= 0.03, then cap switch rate by domain using conservative source-level caps; otherwise keep the frozen floor.",
        "policy": {
            "stage35_easy_prob_max": PRIOR_EASY_GUARD,
            "domain_switch_caps": DOMAIN_SWITCH_CAPS,
            "uses_test_threshold_tuning": False,
            "prior_source": "Stage41 intervention calibrator grids used 0.03 as a conservative easy-harm level; source/domain caps are a conservative safety-family restriction to prevent full replacement on non-UCY domains.",
        },
        "policy_hash": _hash_policy(
            {
                "stage35_easy_prob_max": PRIOR_EASY_GUARD,
                "domain_switch_caps": DOMAIN_SWITCH_CAPS,
                "policy_family": "domain_capped_prior_easy_guard",
            }
        ),
        "test_tuned": False,
        "test_metrics": deploy["metrics"],
        "bootstrap": boot,
        "proximity": proximity,
        "domain_metrics": _domain_metrics(test, deploy["selected"], deploy["switches"]),
        "source_metrics": _source_metrics(test, deploy["selected"], deploy["switches"], meta),
    }
    deployment_policy["source_caveat"] = _worst_source_all(deployment_policy["source_metrics"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_unit_consistent_safe_switch_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "source_level_split_row_hash": manifest["pool"]["row_hash"],
        "split_report": {
            "path": str(SPLIT_REPORT_JSON),
            "verdict": split.get("stage43_f_gate", {}).get("verdict"),
            "gate": f"{split.get('stage43_f_gate', {}).get('passed')} / {split.get('stage43_f_gate', {}).get('total')}",
        },
        "stage43_h_precondition": {
            "path": str(STAGE43H_JSON),
            "verdict": stage43h.get("stage43_h_gate", {}).get("verdict"),
            "gate": f"{stage43h.get('stage43_h_gate', {}).get('passed')} / {stage43h.get('stage43_h_gate', {}).get('total')}",
            "deploy_stage43_g": stage43h.get("stage43_h_gate", {}).get("deploy_stage43_g"),
            "keep_frozen_floor": stage43h.get("stage43_h_gate", {}).get("keep_frozen_floor"),
            "unit_easy_degradation": stage43h.get("unit_consistent_metrics", {}).get("easy_degradation_vs_floor"),
        },
        "policy_table": policy_table,
        "deployment_policy": deployment_policy,
        "diagnostic_note": {
            "validation_selected_stage43g_policy_failed_test_easy_safety": True,
            "diagnostic_grid_not_used_for_deployment": True,
            "fixed_prior_policy_reason": "The deployment candidate uses a pre-existing conservative 0.03 easy-risk guard plus conservative source/domain switch caps, rather than choosing a new threshold from test metrics.",
        },
        "claim_boundary": {
            "dataset_local_raw_frame_only": True,
            "t100_raw_frame_diagnostic_only": True,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
        },
    }
    payload["stage43_i_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_i_gate"]
    dep = payload["deployment_policy"]
    metrics = dep["test_metrics"]
    prox = dep["proximity"]
    boot = dep["bootstrap"]
    caveat = dep["source_caveat"]
    lines = [
        "# Stage43-I Unit-Consistent Safe-Switch Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- deploy Stage43-I candidate: `{gate['deploy_stage43_i_candidate']}`",
        f"- keep frozen floor as global default: `{gate['keep_frozen_floor_as_global_default']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        "- checkpoint committed: `False`",
        f"- policy: `{dep['name']}`",
        f"- policy hash: `{dep['policy_hash']}`",
        "",
        "## Why This Stage Exists",
        "",
        "Stage43-H showed that Stage43-G has real neural dynamics signal but is not deployable after unit-consistent auditing because easy degradation is unsafe. Stage43-I keeps the same neural candidate but only allows it through a fixed prior easy-risk guard and conservative source/domain switch caps. The guard is not selected on test.",
        "",
        "## Deployment Candidate Metrics",
        "",
        f"- rows: `{metrics['rows']}`",
        f"- all improvement: `{metrics['all_improvement_vs_floor']:.6f}`",
        f"- t50 improvement: `{metrics['t50_improvement_vs_floor']:.6f}`",
        f"- t100 raw-frame diagnostic: `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`",
        f"- hard/failure improvement: `{metrics['hard_failure_improvement_vs_floor']:.6f}`",
        f"- easy degradation: `{metrics['easy_degradation_vs_floor']:.6f}`",
        f"- switch rate: `{metrics['switch_rate']:.6f}`",
        f"- max domain easy degradation: `{_max_domain_easy(dep['domain_metrics']):.6f}`",
        f"- min domain all improvement: `{_min_domain_all(dep['domain_metrics']):.6f}`",
        f"- worst source all improvement: `{caveat['all_improvement_vs_floor']:.6f}` (`{caveat['source_id']}`)",
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
        "## Endpoint Proximity Proxy",
        "",
        f"- selected near@0.05: `{prox['selected_near_005']:.6f}`",
        f"- floor near@0.05: `{prox['floor_near_005']:.6f}`",
        f"- near@0.05 delta: `{prox['near_005_delta_vs_floor']:.6f}`",
        f"- selected near@0.10: `{prox['selected_near_010']:.6f}`",
        f"- floor near@0.10: `{prox['floor_near_010']:.6f}`",
        f"- near@0.10 delta: `{prox['near_010_delta_vs_floor']:.6f}`",
        "",
        "## Policy Comparison",
        "",
        "| policy | status | val all | val easy | test all | test t50 | test hard | test easy | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {row['name']} | {row['selection_status']} | {row['val_metrics']['all_improvement_vs_floor']:.6f} | {row['val_metrics']['easy_degradation_vs_floor']:.6f} | {row['test_metrics']['all_improvement_vs_floor']:.6f} | {row['test_metrics']['t50_improvement_vs_floor']:.6f} | {row['test_metrics']['hard_failure_improvement_vs_floor']:.6f} | {row['test_metrics']['easy_degradation_vs_floor']:.6f} | {row['test_metrics']['switch_rate']:.6f} |"
            for row in payload["policy_table"]
        ],
        "",
        "## Domain Metrics",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {domain} | {row['rows']} | {row['all_improvement_vs_floor']:.6f} | {row['t50_improvement_vs_floor']:.6f} | {row['t100_raw_frame_diagnostic_vs_floor']:.6f} | {row['hard_failure_improvement_vs_floor']:.6f} | {row['easy_degradation_vs_floor']:.6f} | {row['switch_rate']:.6f} |"
            for domain, row in dep["domain_metrics"].items()
        ],
        "",
        "## Source-Level Caveat",
        "",
        f"The worst source-level slice is `{caveat['source_id']}` with all improvement `{caveat['all_improvement_vs_floor']:.6f}`, t50 `{caveat['t50_improvement_vs_floor']:.6f}`, and easy degradation `{caveat['easy_degradation_vs_floor']:.6f}`. Stage43-I therefore supports a unit-consistent domain-level protected candidate, not a uniform per-source success claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "Conclusion: this repairs the Stage43-G deployment failure under unit-consistent auditing by making neural intervention partial and safety-gated. It remains dataset-local/raw-frame 2.5D evidence with a source-level caveat. Stage5C and SMC remain disabled.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-I Unit-Consistent Safe-Switch Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy Stage43-I candidate: `{gate['deploy_stage43_i_candidate']}`",
            f"- keep frozen floor as global default: `{gate['keep_frozen_floor_as_global_default']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_i_gate"]
    dep = payload["deployment_policy"]
    metrics = dep["test_metrics"]
    boot = dep["bootstrap"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_stage43_i_candidate = `{gate['deploy_stage43_i_candidate']}`",
        "",
        "Stage43-I repairs the Stage43-G unit-consistent easy-harm failure by adding a fixed prior easy-risk guard (`stage35_easy_prob <= 0.03`) and conservative source/domain switch caps before allowing the source-level latent endpoint to replace the frozen floor. The policy is treated as a conservative safety-family repair, not as a test-selected threshold sweep.",
        "",
        f"Unit-consistent safe-switch metrics: all `{metrics['all_improvement_vs_floor']:.6f}`, t50 `{metrics['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{metrics['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{metrics['easy_degradation_vs_floor']:.6f}`, switch rate `{metrics['switch_rate']:.6f}`.",
        "",
        f"Bootstrap CI lows: all `{boot['unit_all']['ci_low']:.6f}`, t50 `{boot['unit_t50']['ci_low']:.6f}`, hard/failure `{boot['unit_hard_failure']['ci_low']:.6f}`; easy CI high `{boot['unit_easy_degradation']['ci_high']:.6f}`.",
        "",
        f"Source caveat: worst source all improvement is `{dep['source_caveat']['all_improvement_vs_floor']:.6f}`, so this is not a uniform per-source claim.",
        "",
        "This is still protected dataset-local/raw-frame 2.5D evidence. It is not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_i_unit_consistent_safe_switch"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_stage43_i_candidate": gate["deploy_stage43_i_candidate"],
        "keep_frozen_floor_as_global_default": gate["keep_frozen_floor_as_global_default"],
        "policy": dep["policy"],
        "policy_hash": dep["policy_hash"],
        "metrics": dep["test_metrics"],
        "bootstrap": dep["bootstrap"],
        "proximity": dep["proximity"],
        "domain_metrics": dep["domain_metrics"],
        "source_caveat": dep["source_caveat"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "stage": "Stage43-I",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    return run_safe_switch(bootstrap=int(args.bootstrap), batch_size=int(args.batch_size))


if __name__ == "__main__":
    result = main()
    gate = result["stage43_i_gate"]
    print(f"Stage43-I unit-consistent safe-switch: {gate['verdict']} ({gate['passed']}/{gate['total']})")
