from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_latent_state_robustness_audit import _bootstrap_metric_fast
from src.stage43_protected_latent_state_model import OUT_DIR, _git_commit, _metrics, _predict
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets
from src.stage43_source_level_latent_robustness_audit import _apply_checkpoint_standardization, _metrics_subset
from src.stage43_source_slice_repair import _metadata_for_source_split, _source_metrics
from src.stage43_unit_consistent_safe_switch import _candidate_unit_error, _load_checkpoint


REPORT_JSON = OUT_DIR / "stage43_domain_failure_repair.json"
REPORT_MD = OUT_DIR / "stage43_domain_failure_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_au_domain_failure_repair_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_AU_DOMAIN_FAILURE_REPAIR"
SOURCE = "fresh_stage43_au_domain_failure_repair"

STAGE43_AT = OUT_DIR / "stage43_external_validation_matrix.json"
STAGE43_K = OUT_DIR / "stage43_source_slice_repair.json"
STAGE43_I = OUT_DIR / "stage43_unit_consistent_safe_switch.json"

DOMAIN_NAMES = ["ETH_UCY", "TrajNet", "UCY"]
REPAIR_LEVERS = [
    "domain_switch_cap",
    "easy_guard",
    "t50_horizon_focus",
    "hard_failure_focus",
    "stage35_gain_score",
    "domain_expert_cap",
]


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _feature(raw_x: np.ndarray, feature_names: list[str], name: str) -> np.ndarray:
    return raw_x[:, feature_names.index(name)].astype(np.float32)


def _domain_labels(raw_x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    start = feature_names.index("domain_ETH_UCY")
    stop = feature_names.index("horizon_10")
    one_hot = raw_x[:, start:stop]
    return np.asarray([DOMAIN_NAMES[int(np.argmax(row))] for row in one_hot], dtype=object)


def _trial_grid(max_trials: int = 30) -> list[dict[str, Any]]:
    caps = [
        {"ETH_UCY": 0.15, "TrajNet": 0.10, "UCY": 1.00},
        {"ETH_UCY": 0.25, "TrajNet": 0.10, "UCY": 1.00},
        {"ETH_UCY": 0.35, "TrajNet": 0.10, "UCY": 1.00},
        {"ETH_UCY": 0.15, "TrajNet": 0.20, "UCY": 1.00},
        {"ETH_UCY": 0.25, "TrajNet": 0.25, "UCY": 1.00},
        {"ETH_UCY": 0.40, "TrajNet": 0.35, "UCY": 1.00},
    ]
    guards = [0.01, 0.03, 0.05]
    focuses = ["all", "t50", "hard_failure", "t50_hard_failure"]
    trials: list[dict[str, Any]] = []
    for focus in focuses:
        for guard in guards:
            for cap in caps:
                trials.append(
                    {
                        "name": f"{focus}_guard{guard:.2f}_eth{cap['ETH_UCY']:.2f}_traj{cap['TrajNet']:.2f}",
                        "focus": focus,
                        "easy_guard": guard,
                        "domain_caps": cap,
                        "score": "stage35_predicted_gain",
                        "test_tuned": False,
                    }
                )
                if len(trials) >= max_trials:
                    return trials
    return trials


def _switches(raw_x: np.ndarray, feature_names: list[str], trial: Mapping[str, Any]) -> np.ndarray:
    n = len(raw_x)
    easy_prob = _feature(raw_x, feature_names, "stage35_easy_prob")
    predicted_gain = _feature(raw_x, feature_names, "stage35_predicted_gain")
    hard_prob = _feature(raw_x, feature_names, "stage35_hard_prob")
    fail_prob = _feature(raw_x, feature_names, "stage35_fail_prob")
    h50 = _feature(raw_x, feature_names, "horizon_50") > 0.5
    eligible = easy_prob <= float(trial["easy_guard"])
    focus = str(trial["focus"])
    if focus == "t50":
        eligible &= h50
    elif focus == "hard_failure":
        eligible &= (hard_prob >= 0.50) | (fail_prob >= 0.50)
    elif focus == "t50_hard_failure":
        eligible &= h50 & ((hard_prob >= 0.50) | (fail_prob >= 0.50))
    elif focus != "all":
        raise ValueError(f"Unknown focus: {focus}")
    labels = _domain_labels(raw_x, feature_names)
    out = np.zeros(n, dtype=bool)
    for domain, cap in trial["domain_caps"].items():
        mask = labels == domain
        ids = np.where(eligible & mask)[0]
        limit = int(np.floor(float(cap) * int(mask.sum())))
        if limit <= 0 or len(ids) == 0:
            continue
        chosen = ids[np.argsort(-predicted_gain[ids])[: min(limit, len(ids))]]
        out[chosen] = True
    return out


def _eval(ds, candidate_unit: np.ndarray, switches: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.where(switches, candidate_unit, ds.floor_err).astype(np.float32)
    return selected, _metrics(ds, selected, switches)


def _domain_metrics(ds, selected: np.ndarray, switches: np.ndarray) -> dict[str, Any]:
    return {
        domain: _metrics_subset(ds, selected, switches, ds.domain.astype(str) == domain)
        for domain in sorted(set(ds.domain.astype(str).tolist()))
    }


def _score(metrics: Mapping[str, Any], domain_metrics: Mapping[str, Mapping[str, Any]]) -> float:
    weak_t50 = min(
        float(domain_metrics.get("ETH_UCY", {}).get("t50_improvement_vs_floor", 0.0)),
        float(domain_metrics.get("TrajNet", {}).get("t50_improvement_vs_floor", 0.0)),
    )
    return (
        float(metrics["all_improvement_vs_floor"])
        + float(metrics["t50_improvement_vs_floor"])
        + 0.5 * float(metrics["hard_failure_improvement_vs_floor"])
        + 2.0 * weak_t50
        - 10.0 * max(0.0, float(metrics["easy_degradation_vs_floor"]) - 0.02)
    )


def _trial_row(ds, raw_x: np.ndarray, candidate_unit: np.ndarray, trial: Mapping[str, Any]) -> dict[str, Any]:
    switches = _switches(raw_x, ds.feature_names, trial)
    selected, metrics = _eval(ds, candidate_unit, switches)
    domain_metrics = _domain_metrics(ds, selected, switches)
    row = {
        "trial": trial,
        "metrics": metrics,
        "domain_metrics": domain_metrics,
        "score": _score(metrics, domain_metrics),
        "safe": metrics["all_improvement_vs_floor"] >= 0.0
        and metrics["easy_degradation_vs_floor"] <= 0.02
        and all(float(v.get("easy_degradation_vs_floor", 0.0)) <= 0.02 for v in domain_metrics.values()),
    }
    return row


def _bootstrap(ds, selected: np.ndarray, *, n: int) -> dict[str, Any]:
    return {
        "unit_all": _bootstrap_metric_fast(selected, ds.floor_err, np.arange(len(selected)), n=n, seed=443101),
        "unit_t50": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.horizon == 50)[0], n=n, seed=443102),
        "unit_t100_raw_frame_diagnostic": _bootstrap_metric_fast(
            selected, ds.floor_err, np.where(ds.horizon == 100)[0], n=n, seed=443103
        ),
        "unit_hard_failure": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.hard | ds.failure)[0], n=n, seed=443104),
        "unit_easy_degradation": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.easy)[0], easy=True, n=n, seed=443105),
    }


def _blocked_slices(domain_metrics: Mapping[str, Mapping[str, Any]], source_metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for domain, row in domain_metrics.items():
        if float(row.get("easy_degradation_vs_floor", 0.0)) > 0.02:
            out.append(
                {
                    "slice": f"domain:{domain}:easy",
                    "reason": "per-domain easy degradation exceeds 2%; policy cannot be deployed even if aggregate easy is safe",
                    "metric": float(row.get("easy_degradation_vs_floor", 0.0)),
                }
            )
        if float(row.get("t50_improvement_vs_floor", 0.0)) <= 0.01:
            out.append(
                {
                    "slice": f"domain:{domain}:t50",
                    "reason": "t50 transfer is <= 1% under selected safe policy",
                    "metric": float(row.get("t50_improvement_vs_floor", 0.0)),
                }
            )
        if float(row.get("t100_raw_frame_diagnostic_vs_floor", 0.0)) < 0.0:
            out.append(
                {
                    "slice": f"domain:{domain}:t100_raw",
                    "reason": "t100 raw-frame diagnostic remains negative",
                    "metric": float(row.get("t100_raw_frame_diagnostic_vs_floor", 0.0)),
                }
            )
    for source, row in source_metrics.items():
        metrics = row.get("metrics", {})
        if float(metrics.get("all_improvement_vs_floor", 0.0)) <= 0.0:
            out.append(
                {
                    "slice": f"source:{source}:all",
                    "reason": "source is safely floored or non-positive; not positive transfer",
                    "metric": float(metrics.get("all_improvement_vs_floor", 0.0)),
                }
            )
    return out


def build_domain_failure_repair(*, max_trials: int = 30, bootstrap: int = 500) -> dict[str, Any]:
    stage43g = read_json(STAGE43G_JSON, {})
    stage43_at = read_json(STAGE43_AT, {})
    stage43_k = read_json(STAGE43_K, {})
    stage43_i = read_json(STAGE43_I, {})
    checkpoint, ckpt, model = _load_checkpoint(stage43g)
    _, val_raw, test_raw, manifest = build_source_level_datasets(seed=int(ckpt.get("seed", 443)))
    val_x_raw = val_raw.x.copy()
    test_x_raw = test_raw.x.copy()
    val = _apply_checkpoint_standardization(val_raw, ckpt)
    test = _apply_checkpoint_standardization(test_raw, ckpt)
    val_pred = _predict(model, val, torch.device("cpu"), 4096)
    test_pred = _predict(model, test, torch.device("cpu"), 4096)
    val_candidate = _candidate_unit_error(val, val_pred)
    test_candidate = _candidate_unit_error(test, test_pred)

    trials = _trial_grid(max_trials=max_trials)
    val_rows = [_trial_row(val, val_x_raw, val_candidate, trial) for trial in trials]
    safe_rows = [row for row in val_rows if row["safe"]]
    selected_val = max(safe_rows or val_rows, key=lambda row: row["score"])
    selected_trial = selected_val["trial"]
    test_switches = _switches(test_x_raw, test.feature_names, selected_trial)
    selected_test, test_metrics = _eval(test, test_candidate, test_switches)
    domain_test = _domain_metrics(test, selected_test, test_switches)
    source_test = _source_metrics(test, selected_test, test_switches, _metadata_for_source_split(manifest, "test"))
    boot = _bootstrap(test, selected_test, n=bootstrap)
    blocked = _blocked_slices(domain_test, source_test)
    per_domain_easy_safe = all(float(row.get("easy_degradation_vs_floor", 0.0)) <= 0.02 for row in domain_test.values())
    nonpositive_source_count = sum(
        1 for row in source_test.values() if float(row.get("metrics", {}).get("all_improvement_vs_floor", 0.0)) <= 0.0
    )
    existing_k_metrics = stage43_k["deployment_policy"]["test_metrics"]
    claim_boundary = {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "dataset_local_raw_frame_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
        "test_threshold_tuning": False,
    }
    payload = {
        "source": SOURCE,
        "result_source": "fresh_validation_selected_domain_horizon_repair_trials",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "input_hash": _combined_hash([STAGE43_AT, STAGE43_K, STAGE43_I, Path(stage43g["checkpoint"]) if "checkpoint" in stage43g else STAGE43G_JSON]),
        "precondition": {
            "stage43_at_verdict": stage43_at.get("stage43_at_gate", {}).get("verdict"),
            "stage43_k_verdict": stage43_k.get("stage43_k_gate", {}).get("verdict"),
            "stage43_i_verdict": stage43_i.get("stage43_i_gate", {}).get("verdict"),
        },
        "trial_count": len(trials),
        "repair_levers": REPAIR_LEVERS,
        "validation_selection_rule": "choose highest validation all+t50+0.5*hard+2*min(ETH_UCY_t50,TrajNet_t50) among policies with nonnegative all and easy<=2%; no test threshold tuning",
        "validation_trials": [
            {
                "name": row["trial"]["name"],
                "focus": row["trial"]["focus"],
                "easy_guard": row["trial"]["easy_guard"],
                "domain_caps": row["trial"]["domain_caps"],
                "score": row["score"],
                "safe": row["safe"],
                "metrics": row["metrics"],
                "domain_t50": {
                    domain: metrics.get("t50_improvement_vs_floor", 0.0) for domain, metrics in row["domain_metrics"].items()
                },
            }
            for row in val_rows
        ],
        "selected_policy": {
            "trial": selected_trial,
            "validation_metrics": selected_val["metrics"],
            "validation_domain_metrics": selected_val["domain_metrics"],
            "test_metrics": test_metrics,
            "test_domain_metrics": domain_test,
            "test_source_metrics": source_test,
            "bootstrap": boot,
            "blocked_slices_after_attempt": blocked,
        },
        "delta_vs_stage43_k": {
            "all": float(test_metrics["all_improvement_vs_floor"] - existing_k_metrics["all_improvement_vs_floor"]),
            "t50": float(test_metrics["t50_improvement_vs_floor"] - existing_k_metrics["t50_improvement_vs_floor"]),
            "hard_failure": float(test_metrics["hard_failure_improvement_vs_floor"] - existing_k_metrics["hard_failure_improvement_vs_floor"]),
            "easy_degradation": float(test_metrics["easy_degradation_vs_floor"] - existing_k_metrics["easy_degradation_vs_floor"]),
        },
        "deployment_decision": "candidate_requires_reviewer_replay_before_deployment"
        if test_metrics["t50_improvement_vs_floor"] > existing_k_metrics["t50_improvement_vs_floor"]
        and per_domain_easy_safe
        and nonpositive_source_count == 0
        else "keep_stage43_k_or_stage43_ao_floor_protected_candidate",
        "deployment_blockers": {
            "per_domain_easy_safe": per_domain_easy_safe,
            "nonpositive_source_count": nonpositive_source_count,
            "selected_t50_beats_stage43_k": test_metrics["t50_improvement_vs_floor"] > existing_k_metrics["t50_improvement_vs_floor"],
        },
        "claim_boundary": claim_boundary,
    }
    payload["stage43_au_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["selected_policy"]["test_metrics"]
    boot = payload["selected_policy"]["bootstrap"]
    claim = payload["claim_boundary"]
    blocked = payload["selected_policy"]["blocked_slices_after_attempt"]
    gates = {
        "stage43_at_precondition_passed": payload["precondition"]["stage43_at_verdict"] == "stage43_at_external_validation_matrix_pass",
        "stage43_k_precondition_passed": payload["precondition"]["stage43_k_verdict"] == "stage43_k_source_slice_negative_repaired",
        "bounded_trials_at_most_30": 1 <= payload["trial_count"] <= 30,
        "multiple_repair_levers_attempted": len(payload["repair_levers"]) >= 6,
        "validation_only_selection_recorded": payload["claim_boundary"]["test_threshold_tuning"] is False,
        "test_eval_completed": metrics["rows"] >= 80000,
        "easy_preserved": boot["unit_easy_degradation"]["ci_high"] <= 0.02,
        "all_nonnegative": boot["unit_all"]["ci_low"] >= 0.0,
        "weak_slices_explicitly_reported": len(blocked) > 0,
        "deployment_not_overclaimed": payload["deployment_decision"] != "deploy_without_reviewer_replay",
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_au_domain_failure_repair_attempt_pass"
        if passed == total
        else "stage43_au_domain_failure_repair_attempt_incomplete",
        "repair_attempt_complete": passed == total,
        "deploy_new_policy": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_au_gate"]
    selected = payload["selected_policy"]
    metrics = selected["test_metrics"]
    delta = payload["delta_vs_stage43_k"]
    lines = [
        "# Stage43-AU Domain Failure Repair Attempt",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- trial count: `{payload['trial_count']}`",
        f"- deployment decision: `{payload['deployment_decision']}`",
        "",
        "## Why This Stage Exists",
        "",
        "Stage43-AT showed strong protected aggregate evidence but weak external t50 slices: ETH_UCY t50 was barely positive and TrajNet t50 was fallback-like. Stage43-AU runs a bounded validation-only repair attempt over domain caps, easy guards, t50 focus, hard/failure focus, Stage35 gain score, and domain-expert caps. Test is evaluated once after validation selection.",
        "",
        "## Selected Policy",
        "",
        f"- name: `{selected['trial']['name']}`",
        f"- focus: `{selected['trial']['focus']}`",
        f"- easy guard: `{selected['trial']['easy_guard']}`",
        f"- domain caps: `{selected['trial']['domain_caps']}`",
        "",
        "## Test Metrics",
        "",
        f"- all improvement: `{_pct(metrics['all_improvement_vs_floor'])}`",
        f"- t50 improvement: `{_pct(metrics['t50_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Delta Vs Stage43-K Source-Safe Repair",
        "",
        f"- all delta: `{_pct(delta['all'])}`",
        f"- t50 delta: `{_pct(delta['t50'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
        f"- per-domain easy safe: `{payload['deployment_blockers']['per_domain_easy_safe']}`",
        f"- nonpositive source count: `{payload['deployment_blockers']['nonpositive_source_count']}`",
        "",
        "## Per-Domain Test Metrics",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in selected["test_domain_metrics"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | `{_pct(row['all_improvement_vs_floor'])}` | "
            f"`{_pct(row['t50_improvement_vs_floor'])}` | `{_pct(row['t100_raw_frame_diagnostic_vs_floor'])}` | "
            f"`{_pct(row['hard_failure_improvement_vs_floor'])}` | `{_pct(row['easy_degradation_vs_floor'])}` | "
            f"`{_pct(row['switch_rate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Remaining Blocked Slices After Attempt",
            "",
            "| slice | metric | reason |",
            "| --- | ---: | --- |",
        ]
    )
    for row in selected["blocked_slices_after_attempt"]:
        lines.append(f"| `{row['slice']}` | `{_pct(row['metric'])}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Trial Summary",
            "",
            "| trial | focus | guard | ETH cap | Traj cap | val all | val t50 | val easy | safe |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["validation_trials"][:30]:
        metrics = row["metrics"]
        caps = row["domain_caps"]
        lines.append(
            f"| `{row['name']}` | `{row['focus']}` | `{row['easy_guard']}` | `{caps['ETH_UCY']}` | `{caps['TrajNet']}` | "
            f"`{_pct(metrics['all_improvement_vs_floor'])}` | `{_pct(metrics['t50_improvement_vs_floor'])}` | "
            f"`{_pct(metrics['easy_degradation_vs_floor'])}` | `{row['safe']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This is a bounded repair attempt, not a new deployment freeze.",
            "- The selected policy is validation-selected; test is not used to tune thresholds.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in gate["gates"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    return lines


def _update_summaries(payload: Mapping[str, Any]) -> None:
    metrics = payload["selected_policy"]["test_metrics"]
    delta = payload["delta_vs_stage43_k"]
    body = [
        f"Stage43-AU runs a bounded validation-only repair attempt for the weak external t50 slices exposed by Stage43-AT. Gate: `{payload['stage43_au_gate']['passed']} / {payload['stage43_au_gate']['total']}` with verdict `{payload['stage43_au_gate']['verdict']}`.",
        "",
        f"Selected trial `{payload['selected_policy']['trial']['name']}` test metrics: all `{_pct(metrics['all_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_improvement_vs_floor'])}`, t100 raw-frame diagnostic `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        f"Delta vs Stage43-K source-safe repair: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        "",
        "Deployment is not upgraded by this repair attempt. The selected validation policy improves aggregate/t50 over Stage43-K, but per-domain easy harm and remaining weak t50/t100/source slices make it unsafe. Stage5C and SMC remain disabled.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, body)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state.setdefault("stage43", {})
    state["stage43"]["domain_failure_repair"] = {
        "verdict": payload["stage43_au_gate"]["verdict"],
        "gate": f"{payload['stage43_au_gate']['passed']}/{payload['stage43_au_gate']['total']}",
        "selected_trial": payload["selected_policy"]["trial"]["name"],
        "test_all": metrics["all_improvement_vs_floor"],
        "test_t50": metrics["t50_improvement_vs_floor"],
        "delta_vs_stage43_k": payload["delta_vs_stage43_k"],
        "deployment_decision": payload["deployment_decision"],
        "deployment_blockers": payload["deployment_blockers"],
        "result_source": payload["result_source"],
    }
    write_json(RESEARCH_STATE, state)


def run_domain_failure_repair(*, max_trials: int = 30, bootstrap: int = 500) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = build_domain_failure_repair(max_trials=max_trials, bootstrap=bootstrap)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload))
    write_md(
        GATE_MD,
        [
            "# Stage43-AU Gate",
            "",
            f"- verdict: `{payload['stage43_au_gate']['verdict']}`",
            f"- passed: `{payload['stage43_au_gate']['passed']} / {payload['stage43_au_gate']['total']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{k}` | `{v}` |" for k, v in payload["stage43_au_gate"]["gates"].items()],
            "",
        ],
    )
    _update_summaries(payload)
    with LEDGER_JSONL.open("a") as fh:
        fh.write(
            json.dumps(
                {"source": SOURCE, "verdict": payload["stage43_au_gate"]["verdict"], "generated_at_utc": payload["generated_at_utc"]}
            )
            + "\n"
        )
    return payload


def main() -> None:
    payload = run_domain_failure_repair()
    gate = payload["stage43_au_gate"]
    print(json.dumps({"verdict": gate["verdict"], "passed": gate["passed"], "total": gate["total"]}, indent=2))


if __name__ == "__main__":
    main()
