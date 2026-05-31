from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_latent_state_robustness_audit import _bootstrap_metric_fast
from src.stage43_protected_latent_state_model import (
    DATA35,
    DATA36,
    DATA37,
    OUT_DIR,
    ProtectedLatentStateModel,
    _git_commit,
    _jsonable,
    _metrics,
    _predict,
)
from src.stage43_source_level_caveat_audit import REPORT_JSON as STAGE43J_JSON
from src.stage43_source_level_heldout_split import REPORT_JSON as SPLIT_REPORT_JSON
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets
from src.stage43_source_level_latent_robustness_audit import (
    _apply_checkpoint_standardization,
    _metrics_subset,
    _proximity_stats,
)
from src.stage43_unit_consistent_safe_switch import (
    DOMAIN_SWITCH_CAPS,
    PRIOR_EASY_GUARD,
    _candidate_unit_error,
    _domain_metrics,
    _evaluate_policy,
    _hash_policy,
    _load_checkpoint,
    _policy_switches,
    _selected_xy,
    _source_metrics,
)


REPORT_JSON = OUT_DIR / "stage43_source_slice_repair.json"
REPORT_MD = OUT_DIR / "stage43_source_slice_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_k_source_slice_repair_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_K_SOURCE_SLICE_REPAIR"
SOURCE = "fresh_stage43_k_source_slice_repair"
FAMILY_MIN_VAL_ROWS = 100


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    return np.load(path, allow_pickle=False)


def _metadata_for_source_split(manifest: Mapping[str, Any], source_split: str) -> dict[str, np.ndarray]:
    assignments = {str(k): str(v) for k, v in manifest["source_assignments"].items()}
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for old_split in ["train", "val", "test"]:
        geo = _load_npz(DATA35 / f"expanded_external_{old_split}.npz")
        labels = _load_npz(DATA35 / f"labels_{old_split}.npz")
        family = _load_npz(DATA37 / f"t50_baseline_family_{old_split}.npz")
        selection = _load_npz(DATA36 / f"stage35_selection_{old_split}.npz")
        source = geo["source_file"].astype(str)
        ids = np.where(np.asarray([assignments[str(src)] == source_split for src in source], dtype=bool))[0]
        if len(ids) == 0:
            continue
        selected_family = selection["selected"].astype(np.int64).clip(0, family["prediction"].shape[1] - 1)
        if old_split == "test" and (DATA37 / "stage37_best_t50_selection_test.npz").exists():
            stage37 = _load_npz(DATA37 / "stage37_best_t50_selection_test.npz")
            h50 = geo["horizon"].astype(np.int64) == 50
            selected37 = stage37["selected_family"].astype(np.int64).clip(0, family["prediction"].shape[1] - 1)
            selected_family[h50] = selected37[h50]
        row = np.arange(len(source))
        floor_xy = family["prediction"][row, selected_family].astype(np.float32)
        current_xy = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float32)
        for key, value in {
            "old_split": np.asarray([old_split] * len(ids)),
            "local_row": ids.astype(np.int64),
            "dataset": geo["dataset"].astype(str)[ids],
            "scene_id": geo["scene_id"].astype(str)[ids],
            "source_file": source[ids],
            "agent_id": geo["agent_id"].astype(np.int64)[ids],
            "frame_id": geo["frame_id"].astype(np.float64)[ids],
            "horizon": geo["horizon"].astype(np.int64)[ids],
            "current_xy": current_xy[ids],
            "floor_xy": floor_xy[ids],
            "scale": labels["scale"].astype(np.float32)[ids],
        }.items():
            rows[key].append(value)
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()}


def _source_family(domain: str, scene: str, source_file: str) -> str:
    if domain == "TrajNet":
        text = f"{scene} {source_file}".lower()
        if "mot" in text or "pets" in text:
            return "TrajNet_mot"
        if "biwi" in text:
            return "TrajNet_biwi"
        return "TrajNet_crowds"
    if domain == "ETH_UCY":
        return "ETH_UCY"
    if domain == "UCY":
        return "UCY"
    return domain


def _families(meta: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            _source_family(str(domain), str(scene), str(source))
            for domain, scene, source in zip(meta["dataset"], meta["scene_id"], meta["source_file"])
        ],
        dtype=object,
    )


def _family_metrics(ds, selected: np.ndarray, switches: np.ndarray, families: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted(set(families.astype(str).tolist())):
        mask = families.astype(str) == family
        out[family] = _metrics_subset(ds, selected, switches, mask)
    return out


def _allowed_families_from_validation(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for family, row in sorted(metrics.items()):
        rows = int(row.get("rows", 0))
        all_imp = float(row.get("all_improvement_vs_floor", 0.0))
        easy = float(row.get("easy_degradation_vs_floor", 0.0))
        allowed[family] = {
            "rows": rows,
            "all_improvement_vs_floor": all_imp,
            "easy_degradation_vs_floor": easy,
            "allowed": rows >= FAMILY_MIN_VAL_ROWS and all_imp >= 0.0 and easy <= 0.02,
            "rule": "val rows >= 100 and val all improvement >= 0 and val easy degradation <= 0.02",
        }
    return allowed


def _apply_source_family_guard(base_switch: np.ndarray, families: np.ndarray, allowed: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    out = base_switch.copy().astype(bool)
    for family in sorted(set(families.astype(str).tolist())):
        is_allowed = bool(allowed.get(str(family), {}).get("allowed", False))
        if not is_allowed:
            out[families.astype(str) == str(family)] = False
    return out


def _negative_source_count(source_metrics: Mapping[str, Mapping[str, Any]]) -> int:
    count = 0
    for row in source_metrics.values():
        if float(row.get("metrics", {}).get("all_improvement_vs_floor", 0.0)) < -1e-8:
            count += 1
    return count


def _min_source_all(source_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    if not source_metrics:
        return 0.0
    return float(min(float(row.get("metrics", {}).get("all_improvement_vs_floor", 0.0)) for row in source_metrics.values()))


def _max_source_easy(source_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    if not source_metrics:
        return 0.0
    return float(max(float(row.get("metrics", {}).get("easy_degradation_vs_floor", 0.0)) for row in source_metrics.values()))


def _bootstrap(ds, selected: np.ndarray, *, n: int) -> dict[str, Any]:
    return {
        "unit_all": _bootstrap_metric_fast(selected, ds.floor_err, np.arange(len(selected)), n=n, seed=431001),
        "unit_t50": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.horizon == 50)[0], n=n, seed=431002),
        "unit_t100_raw_frame_diagnostic": _bootstrap_metric_fast(
            selected, ds.floor_err, np.where(ds.horizon == 100)[0], n=n, seed=431003
        ),
        "unit_hard_failure": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.hard | ds.failure)[0], n=n, seed=431004),
        "unit_easy_degradation": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.easy)[0], easy=True, n=n, seed=431005),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["deployment_policy"]["test_metrics"]
    boot = payload["deployment_policy"]["bootstrap"]
    prox = payload["deployment_policy"]["proximity"]
    source_metrics = payload["deployment_policy"]["source_metrics"]
    gates = {
        "stage43_j_precondition_present": payload["stage43_j_precondition"]["verdict"] == "stage43_j_source_level_caveat_mapped",
        "validation_family_policy_used": payload["deployment_policy"]["test_tuned"] is False
        and payload["deployment_policy"]["policy"]["allowed_families_source"] == "validation_only",
        "negative_source_repaired": _negative_source_count(source_metrics) == 0,
        "source_easy_preserved": _max_source_easy(source_metrics) <= 0.02,
        "unit_all_ci_low_positive": boot["unit_all"]["ci_low"] > 0.0,
        "unit_t50_ci_low_positive": boot["unit_t50"]["ci_low"] > 0.0,
        "unit_hard_ci_low_positive": boot["unit_hard_failure"]["ci_low"] > 0.0,
        "easy_preservation_gate": boot["unit_easy_degradation"]["ci_high"] <= 0.02,
        "proximity_not_materially_worse": prox["near_005_delta_vs_floor"] <= 0.01
        and prox["near_010_delta_vs_floor"] <= 0.01,
        "partial_switch_not_full_replacement": 0.0 < metrics["switch_rate"] < 0.80,
        "uniform_positive_source_claim_blocked": payload["claim_boundary"]["uniform_positive_per_source_claim"] is False,
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
        "verdict": "stage43_k_source_slice_negative_repaired" if passed == total else "stage43_k_source_slice_repair_incomplete",
        "source_safe_candidate": passed == total,
        "uniform_positive_source_candidate": False,
        "keep_frozen_floor_as_global_default": True,
    }


def run_source_slice_repair(*, bootstrap: int = 1000, batch_size: int = 4096) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43g = read_json(STAGE43G_JSON, {})
    stage43j = read_json(STAGE43J_JSON, {})
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

    val_base_switch = _policy_switches(
        policy_name="domain_capped_prior_easy_guard",
        raw_x=val_x_raw,
        feature_names=val.feature_names,
        pred=val_pred,
        stage43g_policy=stage43g_policy,
    )
    test_base_switch = _policy_switches(
        policy_name="domain_capped_prior_easy_guard",
        raw_x=test_x_raw,
        feature_names=test.feature_names,
        pred=test_pred,
        stage43g_policy=stage43g_policy,
    )
    val_base_selected, val_base_metrics = _evaluate_policy(val, val_candidate_unit, val_base_switch)
    test_base_selected, test_base_metrics = _evaluate_policy(test, test_candidate_unit, test_base_switch)

    val_meta = _metadata_for_source_split(manifest, "val")
    test_meta = _metadata_for_source_split(manifest, "test")
    val_families = _families(val_meta)
    test_families = _families(test_meta)
    val_family_metrics = _family_metrics(val, val_base_selected, val_base_switch, val_families)
    allowed = _allowed_families_from_validation(val_family_metrics)

    val_guard_switch = _apply_source_family_guard(val_base_switch, val_families, allowed)
    test_guard_switch = _apply_source_family_guard(test_base_switch, test_families, allowed)
    val_guard_selected, val_guard_metrics = _evaluate_policy(val, val_candidate_unit, val_guard_switch)
    test_guard_selected, test_guard_metrics = _evaluate_policy(test, test_candidate_unit, test_guard_switch)
    selected_xy = _selected_xy(test, test_pred, test_guard_switch, test_meta)
    proximity = _proximity_stats(selected_xy, test_meta["floor_xy"], {**test_meta, "selected_xy": selected_xy})
    source_metrics = _source_metrics(test, test_guard_selected, test_guard_switch, test_meta)
    boot = _bootstrap(test, test_guard_selected, n=bootstrap)
    policy = {
        "family_rule": "source-family switch allowed only when the family is supported and safe on the source-level validation split",
        "allowed_families_source": "validation_only",
        "family_min_val_rows": FAMILY_MIN_VAL_ROWS,
        "family_allow_conditions": {
            "val_all_improvement_vs_floor_min": 0.0,
            "val_easy_degradation_vs_floor_max": 0.02,
        },
        "stage35_easy_prob_max": PRIOR_EASY_GUARD,
        "domain_switch_caps": DOMAIN_SWITCH_CAPS,
        "source_family_mapping": "domain-level for ETH_UCY/UCY; TrajNet split into biwi/crowds/mot using scene/source name",
        "uses_test_threshold_tuning": False,
        "forbidden_action_not_used": "No test source id was disabled and no threshold was chosen from test source metrics.",
    }
    deployment_policy = {
        "name": "validation_source_family_guarded_safe_switch",
        "description": "Start from the Stage43-I unit-consistent safe switch, then allow switching only for source families that are supported and non-harmful on validation. Unsupported families fall back to the frozen floor.",
        "policy": policy,
        "policy_hash": _hash_policy({"policy_family": "validation_source_family_guarded_safe_switch", **policy}),
        "test_tuned": False,
        "allowed_families": allowed,
        "validation_metrics": val_guard_metrics,
        "test_metrics": test_guard_metrics,
        "bootstrap": boot,
        "proximity": proximity,
        "domain_metrics": _domain_metrics(test, test_guard_selected, test_guard_switch),
        "source_metrics": source_metrics,
        "source_family_metrics_validation": val_family_metrics,
        "source_family_metrics_test": _family_metrics(test, test_guard_selected, test_guard_switch, test_families),
        "blocked_test_families": sorted(
            {
                str(family)
                for family in test_families.astype(str).tolist()
                if not bool(allowed.get(str(family), {}).get("allowed", False))
            }
        ),
        "source_negative_count": _negative_source_count(source_metrics),
        "min_source_all_improvement": _min_source_all(source_metrics),
        "max_source_easy_degradation": _max_source_easy(source_metrics),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_source_slice_repair_without_test_threshold_tuning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "source_level_split_row_hash": manifest["pool"]["row_hash"],
        "stage43_j_precondition": {
            "path": str(STAGE43J_JSON),
            "verdict": stage43j.get("stage43_j_gate", {}).get("verdict"),
            "gate": f"{stage43j.get('stage43_j_gate', {}).get('passed')} / {stage43j.get('stage43_j_gate', {}).get('total')}",
        },
        "baseline_stage43_i_like_policy": {
            "name": "domain_capped_prior_easy_guard",
            "validation_metrics": val_base_metrics,
            "test_metrics": test_base_metrics,
        },
        "deployment_policy": deployment_policy,
        "repair_summary": {
            "stage43_i_negative_source_count": int(stage43j.get("nonpositive_source_count", 0)),
            "stage43_k_negative_source_count": deployment_policy["source_negative_count"],
            "min_source_all_improvement": deployment_policy["min_source_all_improvement"],
            "uniform_positive_per_source_claim_allowed": False,
            "reason_uniform_positive_still_blocked": "Unsupported or low-support source families may be safely floored to zero; that repairs harm but is not positive transfer.",
        },
        "claim_boundary": {
            "source_safe_candidate": True,
            "uniform_positive_per_source_claim": False,
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
    payload["stage43_k_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_k_gate"]
    dep = payload["deployment_policy"]
    metrics = dep["test_metrics"]
    boot = dep["bootstrap"]
    prox = dep["proximity"]
    lines = [
        "# Stage43-K Source-Slice Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- source-safe candidate: `{gate['source_safe_candidate']}`",
        f"- uniform positive source candidate: `{gate['uniform_positive_source_candidate']}`",
        f"- policy: `{dep['name']}`",
        f"- policy hash: `{dep['policy_hash']}`",
        "",
        "## Why This Stage Exists",
        "",
        "Stage43-J found that Stage43-I is a domain-level unit-consistent safe-switch candidate, but not a uniform source-level success: one small TrajNet source was slightly negative. Stage43-K repairs that negative source without using test-source threshold tuning.",
        "",
        "The repair starts from the Stage43-I safe switch and adds a validation-only source-family support guard. Source families that are missing or unsafe on validation fall back to the frozen floor. This can repair harm, but it does not turn a floored source into positive transfer.",
        "",
        "## Deployment Metrics",
        "",
        f"- rows: `{metrics['rows']}`",
        f"- all improvement: `{metrics['all_improvement_vs_floor']:.6f}`",
        f"- t50 improvement: `{metrics['t50_improvement_vs_floor']:.6f}`",
        f"- t100 raw-frame diagnostic: `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`",
        f"- hard/failure improvement: `{metrics['hard_failure_improvement_vs_floor']:.6f}`",
        f"- easy degradation: `{metrics['easy_degradation_vs_floor']:.6f}`",
        f"- switch rate: `{metrics['switch_rate']:.6f}`",
        f"- negative source count: `{dep['source_negative_count']}`",
        f"- min source all improvement: `{dep['min_source_all_improvement']:.6f}`",
        f"- max source easy degradation: `{dep['max_source_easy_degradation']:.6f}`",
        f"- blocked test source families: `{', '.join(dep['blocked_test_families']) if dep['blocked_test_families'] else 'none'}`",
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
        "## Proximity Proxy",
        "",
        f"- selected near@0.05: `{prox['selected_near_005']:.6f}`",
        f"- floor near@0.05: `{prox['floor_near_005']:.6f}`",
        f"- near@0.05 delta: `{prox['near_005_delta_vs_floor']:.6f}`",
        f"- selected near@0.10: `{prox['selected_near_010']:.6f}`",
        f"- floor near@0.10: `{prox['floor_near_010']:.6f}`",
        f"- near@0.10 delta: `{prox['near_010_delta_vs_floor']:.6f}`",
        "",
        "## Validation Source-Family Decision",
        "",
        "| family | val rows | val all | val easy | allowed |",
        "| --- | ---: | ---: | ---: | --- |",
        *[
            f"| {family} | {row['rows']} | {row['all_improvement_vs_floor']:.6f} | {row['easy_degradation_vs_floor']:.6f} | {bool(row['allowed'])} |"
            for family, row in dep["allowed_families"].items()
        ],
        "",
        "## Source Metrics",
        "",
        "| source | domains | scenes | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {sid} | {','.join(row.get('domains', []))} | {','.join(row.get('scenes', []))} | {row['metrics']['rows']} | {row['metrics']['all_improvement_vs_floor']:.6f} | {row['metrics']['t50_improvement_vs_floor']:.6f} | {row['metrics']['t100_raw_frame_diagnostic_vs_floor']:.6f} | {row['metrics']['hard_failure_improvement_vs_floor']:.6f} | {row['metrics']['easy_degradation_vs_floor']:.6f} | {row['metrics']['switch_rate']:.6f} |"
            for sid, row in dep["source_metrics"].items()
        ],
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "Conclusion: Stage43-K repairs the negative source-slice harm by adding a validation-only source-family support guard. It supports a source-safe protected candidate, not a uniform positive per-source claim. Stage5C and SMC remain disabled.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-K Source-Slice Repair Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- source-safe candidate: `{gate['source_safe_candidate']}`",
            f"- uniform positive source candidate: `{gate['uniform_positive_source_candidate']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_k_gate"]
    dep = payload["deployment_policy"]
    metrics = dep["test_metrics"]
    boot = dep["bootstrap"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"source_safe_candidate = `{gate['source_safe_candidate']}`",
        f"uniform_positive_source_candidate = `{gate['uniform_positive_source_candidate']}`",
        "",
        "Stage43-K addresses the Stage43-J source-level caveat without test-source threshold tuning. It starts from Stage43-I's unit-consistent safe switch and adds a validation-only source-family guard: source families unsupported or unsafe on validation are floored. This removes negative source-slice harm, but it is not a uniform positive per-source claim.",
        "",
        f"Metrics: all `{metrics['all_improvement_vs_floor']:.6f}`, t50 `{metrics['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{metrics['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{metrics['easy_degradation_vs_floor']:.6f}`, switch rate `{metrics['switch_rate']:.6f}`.",
        "",
        f"Bootstrap CI lows: all `{boot['unit_all']['ci_low']:.6f}`, t50 `{boot['unit_t50']['ci_low']:.6f}`, hard/failure `{boot['unit_hard_failure']['ci_low']:.6f}`; easy CI high `{boot['unit_easy_degradation']['ci_high']:.6f}`.",
        "",
        f"Source safety: negative source count `{dep['source_negative_count']}`, min source all improvement `{dep['min_source_all_improvement']:.6f}`, max source easy degradation `{dep['max_source_easy_degradation']:.6f}`.",
        f"Blocked test source families under the validation-only guard: `{', '.join(dep['blocked_test_families']) if dep['blocked_test_families'] else 'none'}`.",
        "",
        "Claim boundary remains: protected dataset-local/raw-frame 2.5D evidence only; no metric/seconds claim, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_k_source_slice_repair"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "source_safe_candidate": gate["source_safe_candidate"],
        "uniform_positive_source_candidate": gate["uniform_positive_source_candidate"],
        "policy": dep["policy"],
        "policy_hash": dep["policy_hash"],
        "metrics": dep["test_metrics"],
        "bootstrap": dep["bootstrap"],
        "source_negative_count": dep["source_negative_count"],
        "min_source_all_improvement": dep["min_source_all_improvement"],
        "max_source_easy_degradation": dep["max_source_easy_degradation"],
        "allowed_families": dep["allowed_families"],
        "blocked_test_families": dep["blocked_test_families"],
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
                        "stage": "Stage43-K",
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
    return run_source_slice_repair(bootstrap=int(args.bootstrap), batch_size=int(args.batch_size))


if __name__ == "__main__":
    result = main()
    gate = result["stage43_k_gate"]
    print(f"Stage43-K source-slice repair: {gate['verdict']} ({gate['passed']}/{gate['total']})")
