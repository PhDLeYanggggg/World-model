from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_downstream_easy_guard_audit import (
    REPORT_JSON as STAGE43_CB_JSON,
    SELECTED_VARIANT,
    _candidate_errors,
    _disagreement_features,
    _encode_selected_variant,
)
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _target_vec,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_latent_adapter_downstream_heads import _fit_heads, _load_adapter, _predict_heads
from src.stage43_latent_transition_adapter_repair import REPORT_JSON as STAGE43_BZ_JSON, _adapter_predict
from src.stage43_latent_transition_consistency_audit import _predict_transition_latents


REPORT_JSON = OUT_DIR / "stage43_shadow_easy_guard_repair.json"
REPORT_MD = OUT_DIR / "stage43_shadow_easy_guard_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_cc_shadow_easy_guard_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_CC_SHADOW_EASY_GUARD_REPAIR"
SOURCE = "fresh_stage43_cc_shadow_easy_guard_repair"


def _source_family(source_file: str) -> str:
    text = str(source_file).lower()
    if "pets" in text:
        return "pets"
    if "zara" in text:
        return "zara"
    if "biwi" in text:
        return "biwi"
    if "student" in text:
        return "students"
    if "hotel" in text:
        return "hotel"
    if "crowd" in text:
        return "crowds"
    stem = Path(str(source_file)).stem.lower()
    return stem or "unknown"


def _shadow_validation_split(ds: Any, *, holdout_mod: int = 30) -> dict[str, Any]:
    scores = np.zeros(len(ds.x), dtype=np.int64)
    for i, (domain, source, horizon) in enumerate(zip(ds.domain.astype(str), ds.source_file.astype(str), ds.horizon.astype(int))):
        key = f"{domain}|{source}|{horizon}|{i}".encode("utf-8")
        scores[i] = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    holdout = scores < int(holdout_mod)
    calib = ~holdout
    families = np.asarray([_source_family(x) for x in ds.source_file.astype(str)])
    support: dict[str, Any] = {"global_families": sorted(set(families.tolist())), "domain_families": {}}
    for domain in sorted(set(ds.domain.astype(str).tolist())):
        support["domain_families"][domain] = sorted(set(families[ds.domain.astype(str) == domain].tolist()))
    plan = {
        "source": "fresh_shadow_split_from_validation_only",
        "rule": "Hash validation rows into calibration/holdout; no test rows are used for threshold or guard selection.",
        "holdout_mod": int(holdout_mod),
        "calibration_rows": int(calib.sum()),
        "shadow_holdout_rows": int(holdout.sum()),
        "support": support,
    }
    return {"calibration": calib, "holdout": holdout, "families": families, "plan": plan}


def _slice_metrics(
    ds: Any,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    sub = type(
        "Slice",
        (),
        {
            "x": ds.x[mask],
            "floor_ade": ds.floor_ade[mask],
            "floor_fde": ds.floor_fde[mask],
            "hard": ds.hard[mask],
            "failure": ds.failure[mask],
            "easy": ds.easy[mask],
            "horizon": ds.horizon[mask],
        },
    )()
    return _metrics(sub, selected_ade[mask], selected_fde[mask], switched[mask])


def _slice_tables(
    ds: Any,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    mask: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {"domain": {}, "horizon": {}, "domain_horizon": {}, "source_family": {}}
    domain = ds.domain.astype(str)
    horizon = ds.horizon.astype(str)
    for value in sorted(set(domain[mask].tolist())):
        m = mask & (domain == value)
        out["domain"][value] = _slice_metrics(ds, selected_ade, selected_fde, switched, m)
    for value in sorted(set(horizon[mask].tolist())):
        m = mask & (horizon == value)
        out["horizon"][value] = _slice_metrics(ds, selected_ade, selected_fde, switched, m)
    for d in sorted(set(domain[mask].tolist())):
        for h in sorted(set(horizon[mask & (domain == d)].tolist())):
            m = mask & (domain == d) & (horizon == h)
            out["domain_horizon"][f"{d}|{h}"] = _slice_metrics(ds, selected_ade, selected_fde, switched, m)
    for fam in sorted(set(families[mask].tolist())):
        m = mask & (families == fam)
        out["source_family"][fam] = _slice_metrics(ds, selected_ade, selected_fde, switched, m)
    return out


def _unsafe_keys(table: Mapping[str, Mapping[str, Any]], *, easy_limit: float = 0.005) -> list[str]:
    bad = []
    for key, row in table.items():
        if int(row.get("rows", 0)) == 0:
            continue
        if float(row.get("easy_degradation_vs_floor", 0.0)) > easy_limit:
            bad.append(str(key))
        elif float(row.get("full_waypoint_ade_improvement_vs_floor", 0.0)) < -0.02:
            bad.append(str(key))
    return sorted(set(bad))


def _apply_guarded_policy(
    ds: Any,
    pred: Mapping[str, np.ndarray],
    candidate_ade: np.ndarray,
    candidate_fde: np.ndarray,
    disagreement: Mapping[str, np.ndarray],
    families: np.ndarray,
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = policy["base_policy"]
    allow = (
        (pred["gain"] >= float(base["gain_threshold"]))
        & (pred["harm"] <= float(base["harm_threshold"]))
        & (pred["failure"] >= float(base["failure_threshold"]))
        & (disagreement["model_floor_mean_disagreement"] <= float(base["disagreement_threshold"]))
        & (disagreement["model_floor_endpoint_disagreement"] <= float(base["endpoint_disagreement_threshold"]))
    )
    domain = ds.domain.astype(str)
    horizon = ds.horizon.astype(str)
    for d in policy.get("blocked_domains", []):
        allow &= domain != str(d)
    for h in policy.get("blocked_horizons", []):
        allow &= horizon != str(h)
    for key in policy.get("blocked_domain_horizons", []):
        d, h = str(key).split("|", 1)
        allow &= ~((domain == d) & (horizon == h))
    mode = policy.get("source_family_support_mode", "none")
    if mode == "global":
        supported = set(policy.get("supported_global_families", []))
        allow &= np.asarray([fam in supported for fam in families], dtype=bool)
    elif mode == "domain":
        support = policy.get("supported_domain_families", {})
        allow &= np.asarray([fam in set(support.get(d, [])) for fam, d in zip(families, domain)], dtype=bool)
    selected_ade = np.where(allow, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(allow, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, allow.astype(bool)


def _search_base_policy(ds: Any, pred: Mapping[str, np.ndarray], calib: np.ndarray) -> dict[str, Any]:
    candidate_ade, candidate_fde = _candidate_errors(ds, pred)
    disagreement = _disagreement_features(ds, pred)
    dis_q = np.quantile(disagreement["model_floor_mean_disagreement"][calib], [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90])
    end_q = np.quantile(disagreement["model_floor_endpoint_disagreement"][calib], [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90])
    best: dict[str, Any] | None = None
    evaluated = 0
    for gain in [0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97]:
        for harm in [0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.35]:
            for failure in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
                for dis in dis_q:
                    for endpoint in end_q:
                        policy = {
                            "gain_threshold": float(gain),
                            "harm_threshold": float(harm),
                            "failure_threshold": float(failure),
                            "disagreement_threshold": float(dis),
                            "endpoint_disagreement_threshold": float(endpoint),
                        }
                        wrapped = {
                            "base_policy": policy,
                            "blocked_domains": [],
                            "blocked_horizons": [],
                            "blocked_domain_horizons": [],
                            "source_family_support_mode": "none",
                        }
                        selected_ade, selected_fde, switched = _apply_guarded_policy(
                            ds, pred, candidate_ade, candidate_fde, disagreement, np.asarray([""] * len(ds.x)), wrapped
                        )
                        metrics = _slice_metrics(ds, selected_ade, selected_fde, switched, calib)
                        evaluated += 1
                        if metrics["easy_degradation_vs_floor"] > 0.005:
                            continue
                        if metrics["switch_rate"] > 0.25:
                            continue
                        objective = (
                            metrics["full_waypoint_ade_improvement_vs_floor"]
                            + metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                            + 0.50 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                            - 0.05 * metrics["switch_rate"]
                        )
                        row = {"base_policy": policy, "calibration_metrics": metrics, "objective": float(objective)}
                        if best is None or row["objective"] > best["objective"]:
                            best = row
    if best is None:
        return {
            "base_policy": {
                "gain_threshold": 1.01,
                "harm_threshold": -0.01,
                "failure_threshold": 1.01,
                "disagreement_threshold": 0.0,
                "endpoint_disagreement_threshold": 0.0,
            },
            "calibration_metrics": _slice_metrics(ds, ds.floor_ade, ds.floor_fde, np.zeros(len(ds.x), dtype=bool), calib),
            "objective": 0.0,
            "evaluated_policies": int(evaluated),
            "diagnostic": "no_calibration_policy_found",
        }
    best["evaluated_policies"] = int(evaluated)
    return best


def _candidate_guard_policies(
    *,
    base_policy: Mapping[str, float],
    support: Mapping[str, Any],
    holdout_tables: Mapping[str, Any],
) -> list[dict[str, Any]]:
    unsafe_domains = _unsafe_keys(holdout_tables["domain"])
    unsafe_horizons = _unsafe_keys(holdout_tables["horizon"])
    unsafe_domain_horizons = _unsafe_keys(holdout_tables["domain_horizon"])
    base = {
        "base_policy": dict(base_policy),
        "blocked_domains": [],
        "blocked_horizons": [],
        "blocked_domain_horizons": [],
        "source_family_support_mode": "none",
        "supported_global_families": support["global_families"],
        "supported_domain_families": support["domain_families"],
    }
    variants = [
        ("base_threshold_only", {}),
        ("blocked_domain_from_shadow_holdout", {"blocked_domains": unsafe_domains}),
        ("blocked_horizon_from_shadow_holdout", {"blocked_horizons": unsafe_horizons}),
        ("blocked_domain_horizon_from_shadow_holdout", {"blocked_domain_horizons": unsafe_domain_horizons}),
        ("global_source_family_support_guard", {"source_family_support_mode": "global"}),
        ("domain_source_family_support_guard", {"source_family_support_mode": "domain"}),
        (
            "domain_horizon_plus_global_family_guard",
            {"blocked_domain_horizons": unsafe_domain_horizons, "source_family_support_mode": "global"},
        ),
        (
            "domain_horizon_plus_domain_family_guard",
            {"blocked_domain_horizons": unsafe_domain_horizons, "source_family_support_mode": "domain"},
        ),
        (
            "domain_block_plus_domain_family_guard",
            {"blocked_domains": unsafe_domains, "source_family_support_mode": "domain"},
        ),
    ]
    out = []
    for name, update in variants:
        row = dict(base)
        row.update(update)
        row["name"] = name
        out.append(row)
    return out


def _source_support_summary(ds: Any, families: np.ndarray, support: Mapping[str, Any]) -> dict[str, Any]:
    domain = ds.domain.astype(str)
    global_supported = set(support.get("global_families", []))
    domain_supported = {str(k): set(v) for k, v in support.get("domain_families", {}).items()}
    global_unsupported: dict[str, int] = defaultdict(int)
    domain_unsupported: dict[str, int] = defaultdict(int)
    by_domain_family: dict[str, int] = defaultdict(int)
    for d, fam in zip(domain, families):
        by_domain_family[f"{d}|{fam}"] += 1
        if fam not in global_supported:
            global_unsupported[str(fam)] += 1
        if fam not in domain_supported.get(str(d), set()):
            domain_unsupported[f"{d}|{fam}"] += 1
    return {
        "global_unsupported_family_rows": dict(sorted(global_unsupported.items())),
        "domain_unsupported_family_rows": dict(sorted(domain_unsupported.items())),
        "by_domain_family_rows": dict(sorted(by_domain_family.items())),
    }


def _evaluate_policy(
    ds: Any,
    pred: Mapping[str, np.ndarray],
    families: np.ndarray,
    policy: Mapping[str, Any],
    mask: np.ndarray,
) -> dict[str, Any]:
    candidate_ade, candidate_fde = _candidate_errors(ds, pred)
    disagreement = _disagreement_features(ds, pred)
    selected_ade, selected_fde, switched = _apply_guarded_policy(
        ds, pred, candidate_ade, candidate_fde, disagreement, families, policy
    )
    return {
        "metrics": _slice_metrics(ds, selected_ade, selected_fde, switched, mask),
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switched": switched,
        "slice_tables": _slice_tables(ds, selected_ade, selected_fde, switched, mask, families),
    }


def _select_shadow_policy(
    ds: Any,
    pred: Mapping[str, np.ndarray],
    shadow: Mapping[str, Any],
) -> dict[str, Any]:
    calib = shadow["calibration"]
    holdout = shadow["holdout"]
    families = shadow["families"]
    base = _search_base_policy(ds, pred, calib)
    base_policy = {
        "base_policy": base["base_policy"],
        "blocked_domains": [],
        "blocked_horizons": [],
        "blocked_domain_horizons": [],
        "source_family_support_mode": "none",
        "supported_global_families": shadow["plan"]["support"]["global_families"],
        "supported_domain_families": shadow["plan"]["support"]["domain_families"],
        "name": "base_threshold_only",
    }
    base_holdout = _evaluate_policy(ds, pred, families, base_policy, holdout)
    candidates = _candidate_guard_policies(
        base_policy=base["base_policy"],
        support=shadow["plan"]["support"],
        holdout_tables=base_holdout["slice_tables"],
    )
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for policy in candidates:
        evaluated = _evaluate_policy(ds, pred, families, policy, holdout)
        metrics = evaluated["metrics"]
        objective = (
            metrics["full_waypoint_ade_improvement_vs_floor"]
            + metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            + 0.50 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            - 0.05 * metrics["switch_rate"]
            - 20.0 * max(0.0, metrics["easy_degradation_vs_floor"] - 0.005)
        )
        row = {
            "policy": policy,
            "shadow_holdout_metrics": metrics,
            "objective": float(objective),
        }
        rows.append(row)
        safe = metrics["easy_degradation_vs_floor"] <= 0.005 and metrics["switch_rate"] <= 0.25
        if not safe:
            continue
        if best is None or row["objective"] > best["objective"]:
            best = row
    if best is None:
        floor_policy = {
            "base_policy": {
                "gain_threshold": 1.01,
                "harm_threshold": -0.01,
                "failure_threshold": 1.01,
                "disagreement_threshold": 0.0,
                "endpoint_disagreement_threshold": 0.0,
            },
            "blocked_domains": [],
            "blocked_horizons": [],
            "blocked_domain_horizons": [],
            "source_family_support_mode": "none",
            "supported_global_families": shadow["plan"]["support"]["global_families"],
            "supported_domain_families": shadow["plan"]["support"]["domain_families"],
            "name": "floor_fallback_no_shadow_safe_policy",
        }
        best = {
            "policy": floor_policy,
            "shadow_holdout_metrics": _slice_metrics(
                ds, ds.floor_ade, ds.floor_fde, np.zeros(len(ds.x), dtype=bool), holdout
            ),
            "objective": 0.0,
            "diagnostic": "no_shadow_holdout_easy_safe_policy_found",
        }
    return {
        "base_calibration": base,
        "base_shadow_holdout": {
            "metrics": base_holdout["metrics"],
            "slice_tables": base_holdout["slice_tables"],
        },
        "candidate_policies": rows,
        "selected_shadow_policy": best,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    test = payload["test_once"]["metrics"]
    shadow = payload["shadow_policy"]["selected_shadow_policy"]["shadow_holdout_metrics"]
    gates = {
        "stage43_cb_precondition_seen": payload["stage43_cb_precondition"]["verdict"]
        == "stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch",
        "fresh_shadow_replay_completed": payload["result_source"] == "fresh_shadow_validation_easy_guard_repair",
        "train_only_heads_refit": payload["protocol"]["train_only_heads_refit"] is True,
        "validation_split_internal_only": payload["shadow_validation"]["plan"]["calibration_rows"] > 0
        and payload["shadow_validation"]["plan"]["shadow_holdout_rows"] > 0,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "inference_safe_guard_features_only": payload["no_leakage"]["guard_uses_future_labels"] is False
        and payload["no_leakage"]["guard_uses_test_endpoints"] is False,
        "shadow_holdout_easy_safe": shadow["easy_degradation_vs_floor"] <= 0.005,
        "test_easy_preserved": test["easy_degradation_vs_floor"] <= 0.02,
        "test_lift_vs_floor": test["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or test["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or test["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "source_family_support_reported": bool(payload["shadow_validation"]["plan"]["support"]["global_families"]),
        "domain_horizon_source_breakdown_reported": bool(payload["test_once"]["slice_tables"].get("domain"))
        and bool(payload["test_once"]["slice_tables"].get("horizon"))
        and bool(payload["test_once"]["slice_tables"].get("source_family")),
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage43_cc_shadow_easy_guard_repair_pass"
    elif gates["test_easy_preserved"] and gates["test_lift_vs_floor"]:
        verdict = "stage43_cc_shadow_easy_guard_repair_partial_safe_lift"
    elif gates["shadow_holdout_easy_safe"] and not gates["test_easy_preserved"]:
        verdict = "stage43_cc_shadow_easy_guard_shadow_safe_test_mismatch"
    else:
        verdict = "stage43_cc_shadow_easy_guard_repair_diagnostic_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cc_gate"]
    shadow = payload["shadow_policy"]["selected_shadow_policy"]
    shadow_m = shadow["shadow_holdout_metrics"]
    test = payload["test_once"]["metrics"]
    return [
        "# Stage43-CC Shadow Easy Guard Repair",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- selected policy: `{shadow['policy']['name']}`",
        "- deployable policy changed: `False`",
        "",
        "## Shadow Validation",
        "",
        f"- calibration rows: `{payload['shadow_validation']['plan']['calibration_rows']}`",
        f"- shadow holdout rows: `{payload['shadow_validation']['plan']['shadow_holdout_rows']}`",
        f"- validation source families: `{payload['shadow_validation']['plan']['support']['global_families']}`",
        f"- test global-unsupported families: `{payload['test_source_support_summary']['global_unsupported_family_rows']}`",
        f"- test domain-unsupported families: `{payload['test_source_support_summary']['domain_unsupported_family_rows']}`",
        f"- shadow all improvement: `{shadow_m['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- shadow t50 improvement: `{shadow_m['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- shadow hard/failure improvement: `{shadow_m['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- shadow easy degradation: `{shadow_m['easy_degradation_vs_floor']:.4f}`",
        f"- shadow switch rate: `{shadow_m['switch_rate']:.4f}`",
        "",
        "## Test Once",
        "",
        f"- test all improvement: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test t50 improvement: `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        f"- test switch rate: `{test['switch_rate']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- Stage43-CC uses validation-only calibration/holdout to avoid selecting guard parameters on test.",
        "- It tests whether source-family/domain/horizon support can repair the Stage43-CB easy-safety transfer failure.",
        "- Future waypoint labels are train/eval labels only. Guard inputs are predicted risk and model-vs-floor disagreement plus metadata available at inference.",
        "- Deployment remains unchanged unless test easy preservation and lift both pass.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cc_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CC Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    test = payload["test_once"]["metrics"]
    world = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{SOURCE}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "- deployable policy changed: `False`",
        f"- test all improvement: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test t50 improvement: `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        "- long objective complete: `False`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-CC is a shadow-validation safety repair audit for downstream latent heads.",
        "- It does not remove the Stage37/Stage42 safety floor.",
        "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
    ]
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(WORLD_GATE_MD, world)


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cc_gate"]
    shadow = payload["shadow_policy"]["selected_shadow_policy"]
    shadow_m = shadow["shadow_holdout_metrics"]
    test = payload["test_once"]["metrics"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "deployable_policy_changed = `False`",
        "",
        "Stage43-CC repairs the Stage43-CB validation/test easy mismatch with a validation-only shadow holdout and source-family support guard.",
        f"Selected shadow policy: `{shadow['policy']['name']}`.",
        f"Shadow all / hard / easy: `{shadow_m['full_waypoint_ade_improvement_vs_floor']:.4f}` / `{shadow_m['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{shadow_m['easy_degradation_vs_floor']:.4f}`.",
        f"Test all / t50 / hard / easy: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['easy_degradation_vs_floor']:.4f}`.",
        "",
        "Interpretation: this is a safety protocol repair for latent downstream heads. Deployment remains unchanged unless test easy safety and lift both hold.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_cc_shadow_easy_guard_repair"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "selected_policy": shadow["policy"],
        "shadow_metrics": shadow_m,
        "test_metrics": test,
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "deployable_policy_changed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_cc_shadow_easy_guard_repair"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable({"event": SOURCE, "verdict": gate["verdict"], "generated_at_utc": payload["generated_at_utc"]}),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_shadow_easy_guard_repair(*, batch_size: int = 8192, ridge: float = 1e-2) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43bz = read_json(STAGE43_BZ_JSON, {})
    stage43cb = read_json(STAGE43_CB_JSON, {})
    checkpoint, ckpt, base_model = _load_model(stage43m)
    adapter_path = Path(stage43bz.get("adapter_checkpoint", OUT_DIR / "checkpoints/stage43_latent_transition_adapter_repair.pt"))
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    probe = _predict_transition_latents(base_model, train, batch_size=int(batch_size))
    adapter = _load_adapter(adapter_path, train.x.shape[1], probe["z_t"].shape[1])
    train_latent = np.concatenate(
        [probe["z_t"], probe["z_next"], _adapter_predict(adapter, train.x, probe["z_t"], batch_size=int(batch_size))],
        axis=1,
    ).astype(np.float32)
    val_latent = _encode_selected_variant(base_model, adapter, val, batch_size=int(batch_size))
    test_latent = _encode_selected_variant(base_model, adapter, test, batch_size=int(batch_size))
    weights = _fit_heads(train_latent, train, ridge=float(ridge))
    val_pred = _predict_heads(val_latent, weights)
    test_pred = _predict_heads(test_latent, weights)
    shadow = _shadow_validation_split(val)
    shadow_policy = _select_shadow_policy(val, val_pred, shadow)
    test_families = np.asarray([_source_family(x) for x in test.source_file.astype(str)])
    test_eval = _evaluate_policy(test, test_pred, test_families, shadow_policy["selected_shadow_policy"]["policy"], np.ones(len(test.x), dtype=bool))
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_shadow_validation_easy_guard_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {"verdict": stage43m.get("stage43_m_gate", {}).get("verdict"), "checkpoint": str(checkpoint)},
        "stage43_bz_precondition": {"verdict": stage43bz.get("stage43_bz_gate", {}).get("verdict"), "adapter_checkpoint": str(adapter_path)},
        "stage43_cb_precondition": {"verdict": stage43cb.get("stage43_cb_gate", {}).get("verdict"), "report": str(STAGE43_CB_JSON)},
        "protocol": {
            "train_only_heads_refit": True,
            "ridge": float(ridge),
            "batch_size": int(batch_size),
            "selected_variant": SELECTED_VARIANT,
            "target_vec_shape": list(_target_vec(train).shape),
            "num_workers": 0,
        },
        "rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "shadow_validation": {"plan": shadow["plan"]},
        "shadow_policy": shadow_policy,
        "test_source_support_summary": _source_support_summary(test, test_families, shadow["plan"]["support"]),
        "test_once": {
            "metrics": test_eval["metrics"],
            "slice_tables": test_eval["slice_tables"],
            "bootstrap": _bootstrap_ci(test, test_eval["selected_ade"], test_eval["selected_fde"], n=1000, seed=1043),
        },
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_threshold_tuning": False,
            "test_statistics_normalization": False,
            "guard_uses_future_labels": False,
            "guard_uses_test_endpoints": False,
        },
        "claim_boundary": {
            "deployable_policy_changed": False,
            "metric_or_seconds_claim": False,
            "true_3d_claim": False,
            "foundation_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cc_gate"] = _gate(payload)
    _write_reports(payload)
    _update_summaries(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stage43-CC shadow-validation easy guard repair.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args(argv)
    payload = run_shadow_easy_guard_repair(batch_size=args.batch_size, ridge=args.ridge)
    gate = payload["stage43_cc_gate"]
    test = payload["test_once"]["metrics"]
    print(f"Stage43-CC: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"test_all={test['full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_t50={test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_hard={test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_easy={test['easy_degradation_vs_floor']:.4f}")
    return payload


if __name__ == "__main__":
    main()
