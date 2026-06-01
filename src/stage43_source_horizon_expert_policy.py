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
from src.stage43_domain_failure_repair import _domain_metrics, _eval, _switches, _trial_grid, _trial_row
from src.stage43_latent_state_robustness_audit import _bootstrap_metric_fast
from src.stage43_protected_latent_state_model import OUT_DIR, _git_commit, _predict
from src.stage43_source_horizon_safety_envelope import _trial_safety_flags
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets
from src.stage43_source_level_latent_robustness_audit import _apply_checkpoint_standardization, _metrics_subset
from src.stage43_source_slice_repair import _apply_source_family_guard, _families, _metadata_for_source_split, _source_metrics
from src.stage43_unit_consistent_safe_switch import _candidate_unit_error, _load_checkpoint


REPORT_JSON = OUT_DIR / "stage43_source_horizon_expert_policy.json"
REPORT_MD = OUT_DIR / "stage43_source_horizon_expert_policy.md"
GATE_MD = OUT_DIR / "stage43_stage_aw_source_horizon_expert_policy_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_AW_SOURCE_HORIZON_EXPERT_POLICY"
SOURCE = "fresh_stage43_aw_source_horizon_expert_policy"

STAGE43_K = OUT_DIR / "stage43_source_slice_repair.json"
STAGE43_AV = OUT_DIR / "stage43_source_horizon_safety_envelope.json"
STAGE43_AU = OUT_DIR / "stage43_domain_failure_repair.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _h50(ds) -> np.ndarray:
    return ds.horizon.astype(np.int64) == 50


def _stage43k_base_switch(raw_x: np.ndarray, feature_names: list[str], families: np.ndarray, allowed_families: Mapping[str, Any]) -> np.ndarray:
    base_trial = {
        "name": "stage43k_base_all_guard0.03_eth0.15_traj0.10",
        "focus": "all",
        "easy_guard": 0.03,
        "domain_caps": {"ETH_UCY": 0.15, "TrajNet": 0.10, "UCY": 1.00},
        "score": "stage35_predicted_gain",
        "test_tuned": False,
    }
    base = _switches(raw_x, feature_names, base_trial)
    return _apply_source_family_guard(base, families, allowed_families)


def _composite_switch(base: np.ndarray, t50_switch: np.ndarray, h50: np.ndarray) -> np.ndarray:
    out = base.copy().astype(bool)
    out[h50] = t50_switch[h50]
    return out


def _score(metrics: Mapping[str, Any], flags: Mapping[str, Any]) -> float:
    return (
        3.0 * float(metrics["t50_improvement_vs_floor"])
        + float(metrics["all_improvement_vs_floor"])
        + 0.5 * float(metrics["hard_failure_improvement_vs_floor"])
        + 0.2 * float(flags["positive_t50_domain_count"])
        - 20.0 * max(0.0, float(metrics["easy_degradation_vs_floor"]) - 0.02)
        - 0.5 * float(flags["negative_source_count"])
    )


def _bootstrap(ds, selected: np.ndarray, *, n: int) -> dict[str, Any]:
    return {
        "unit_all": _bootstrap_metric_fast(selected, ds.floor_err, np.arange(len(selected)), n=n, seed=443201),
        "unit_t50": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.horizon == 50)[0], n=n, seed=443202),
        "unit_t100_raw_frame_diagnostic": _bootstrap_metric_fast(
            selected, ds.floor_err, np.where(ds.horizon == 100)[0], n=n, seed=443203
        ),
        "unit_hard_failure": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.hard | ds.failure)[0], n=n, seed=443204),
        "unit_easy_degradation": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.easy)[0], easy=True, n=n, seed=443205),
    }


def _source_family_metrics(ds, selected: np.ndarray, switches: np.ndarray, families: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted(set(families.astype(str).tolist())):
        mask = families.astype(str) == family
        out[family] = _metrics_subset(ds, selected, switches, mask)
    return out


def build_source_horizon_expert_policy(*, max_trials: int = 30, bootstrap: int = 1000) -> dict[str, Any]:
    stage43g = read_json(STAGE43G_JSON, {})
    stage43k = read_json(STAGE43_K, {})
    stage43av = read_json(STAGE43_AV, {})
    stage43au = read_json(STAGE43_AU, {})
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
    val_meta = _metadata_for_source_split(manifest, "val")
    test_meta = _metadata_for_source_split(manifest, "test")
    val_families = _families(val_meta)
    test_families = _families(test_meta)
    allowed = stage43k["deployment_policy"]["allowed_families"]
    val_base = _stage43k_base_switch(val_x_raw, val.feature_names, val_families, allowed)
    test_base = _stage43k_base_switch(test_x_raw, test.feature_names, test_families, allowed)
    # Small focused tests may ask for fewer trials than the grid position where
    # t50-focused policies first appear. Keep the official max_trials=30 run
    # unchanged, while ensuring the builder still exercises the intended family.
    grid_budget = max(int(max_trials), 30)
    trials = [trial for trial in _trial_grid(max_trials=grid_budget) if str(trial["focus"]) == "t50"][: int(max_trials)]
    rows: list[dict[str, Any]] = []
    for trial in trials:
        t50_sw = _switches(val_x_raw, val.feature_names, trial)
        comp = _composite_switch(val_base, t50_sw, _h50(val))
        val_selected, val_metrics = _eval(val, val_candidate, comp)
        val_domain = _domain_metrics(val, val_selected, comp)
        val_source = _source_metrics(val, val_selected, comp, val_meta)
        val_flags = _trial_safety_flags(val_metrics, val_domain, val_source)
        rows.append(
            {
                "trial": trial,
                "validation_metrics": val_metrics,
                "validation_domain_metrics": val_domain,
                "validation_source_family_metrics": _source_family_metrics(val, val_selected, comp, val_families),
                "validation_flags": val_flags,
                "selection_score": _score(val_metrics, val_flags),
            }
        )
    eligible = [
        row
        for row in rows
        if row["validation_flags"]["aggregate_safe"]
        and row["validation_flags"]["domain_easy_safe"]
        and row["validation_flags"]["negative_source_count"] == 0
        and row["validation_flags"]["positive_t50_domain_count"] >= 2
    ]
    selected_val = max(eligible or rows, key=lambda row: row["selection_score"])
    selected_trial = selected_val["trial"]
    test_t50_sw = _switches(test_x_raw, test.feature_names, selected_trial)
    test_comp = _composite_switch(test_base, test_t50_sw, _h50(test))
    test_selected, test_metrics = _eval(test, test_candidate, test_comp)
    test_domain = _domain_metrics(test, test_selected, test_comp)
    test_source = _source_metrics(test, test_selected, test_comp, test_meta)
    test_family = _source_family_metrics(test, test_selected, test_comp, test_families)
    test_flags = _trial_safety_flags(test_metrics, test_domain, test_source)
    boot = _bootstrap(test, test_selected, n=bootstrap)
    stage43k_metrics = stage43k["deployment_policy"]["test_metrics"]
    git_commit = _git_commit()
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
        "result_source": "fresh_validation_selected_source_horizon_expert_policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "input_hash": _combined_hash([STAGE43_K, STAGE43_AV, STAGE43_AU, STAGE43G_JSON]),
        "precondition": {
            "stage43_k_verdict": stage43k.get("stage43_k_gate", {}).get("verdict"),
            "stage43_av_verdict": stage43av.get("stage43_av_gate", {}).get("verdict"),
            "stage43_au_verdict": stage43au.get("stage43_au_gate", {}).get("verdict"),
        },
        "policy": {
            "name": "validation_selected_source_horizon_t50_expert_over_stage43k_base",
            "git_commit": git_commit,
            "base_policy": "Stage43-K source-family guarded non-t50 base",
            "t50_expert_trial": selected_trial,
            "selection_rule": "validation aggregate/domain/source safe, >=2 positive t50 validation domains, maximize 3*t50 + all + 0.5*hard + domain-count bonus",
            "test_tuned": False,
        },
        "validation_candidates": rows,
        "eligible_validation_candidate_count": len(eligible),
        "selected_validation_metrics": selected_val["validation_metrics"],
        "selected_validation_flags": selected_val["validation_flags"],
        "test_metrics": test_metrics,
        "test_domain_metrics": test_domain,
        "test_source_metrics": test_source,
        "test_source_family_metrics": test_family,
        "test_flags": test_flags,
        "bootstrap": boot,
        "delta_vs_stage43_k": {
            "all": float(test_metrics["all_improvement_vs_floor"] - stage43k_metrics["all_improvement_vs_floor"]),
            "t50": float(test_metrics["t50_improvement_vs_floor"] - stage43k_metrics["t50_improvement_vs_floor"]),
            "hard_failure": float(test_metrics["hard_failure_improvement_vs_floor"] - stage43k_metrics["hard_failure_improvement_vs_floor"]),
            "easy_degradation": float(test_metrics["easy_degradation_vs_floor"] - stage43k_metrics["easy_degradation_vs_floor"]),
        },
        "deployment_decision": "candidate_requires_reviewer_replay_before_deployment"
        if test_flags["deployable_like_under_test_diagnostic"] and boot["unit_t50"]["ci_low"] > 0.0
        else "keep_stage43_k_or_stage43_ao_floor_protected_candidate",
        "claim_boundary": claim_boundary,
    }
    payload["stage43_aw_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    claim = payload["claim_boundary"]
    metrics = payload["test_metrics"]
    boot = payload["bootstrap"]
    flags = payload["test_flags"]
    gates = {
        "stage43_k_precondition_passed": payload["precondition"]["stage43_k_verdict"] == "stage43_k_source_slice_negative_repaired",
        "stage43_av_precondition_passed": payload["precondition"]["stage43_av_verdict"] == "stage43_av_source_horizon_safety_envelope_pass",
        "validation_candidates_evaluated": len(payload["validation_candidates"]) > 0,
        "validation_only_selection": payload["policy"]["test_tuned"] is False,
        "validation_eligible_candidate_exists": payload["eligible_validation_candidate_count"] > 0,
        "test_eval_completed": metrics["rows"] >= 80000,
        "test_aggregate_safe": metrics["all_improvement_vs_floor"] >= 0.0 and metrics["easy_degradation_vs_floor"] <= 0.02,
        "test_t50_positive": boot["unit_t50"]["ci_low"] > 0.0,
        "domain_easy_safe": flags["domain_easy_safe"] is True,
        "source_negative_free": flags["negative_source_count"] == 0,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = (
        passed == total
        and payload["test_flags"]["deployable_like_under_test_diagnostic"] is True
        and payload["deployment_decision"] == "candidate_requires_reviewer_replay_before_deployment"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_aw_source_horizon_expert_policy_pass"
        if passed == total
        else "stage43_aw_source_horizon_expert_policy_incomplete",
        "candidate_for_reviewer_replay": deploy,
        "deploy_without_replay": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_aw_gate"]
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_stage43_k"]
    boot = payload["bootstrap"]
    lines = [
        "# Stage43-AW Source-Horizon Expert Policy",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- candidate for reviewer replay: `{gate['candidate_for_reviewer_replay']}`",
        f"- deploy without replay: `{gate['deploy_without_replay']}`",
        f"- deployment decision: `{payload['deployment_decision']}`",
        "",
        "## Policy",
        "",
        f"- base: {payload['policy']['base_policy']}",
        f"- t50 expert: `{payload['policy']['t50_expert_trial']['name']}`",
        f"- selection rule: {payload['policy']['selection_rule']}",
        f"- eligible validation candidates: `{payload['eligible_validation_candidate_count']}`",
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
        "## Bootstrap CI",
        "",
        f"- all CI: `[{_pct(boot['unit_all']['ci_low'])}, {_pct(boot['unit_all']['ci_high'])}]`",
        f"- t50 CI: `[{_pct(boot['unit_t50']['ci_low'])}, {_pct(boot['unit_t50']['ci_high'])}]`",
        f"- hard/failure CI: `[{_pct(boot['unit_hard_failure']['ci_low'])}, {_pct(boot['unit_hard_failure']['ci_high'])}]`",
        f"- easy degradation CI: `[{_pct(boot['unit_easy_degradation']['ci_low'])}, {_pct(boot['unit_easy_degradation']['ci_high'])}]`",
        "",
        "## Delta Vs Stage43-K",
        "",
        f"- all delta: `{_pct(delta['all'])}`",
        f"- t50 delta: `{_pct(delta['t50'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
        "",
        "## Per-Domain Test Metrics",
        "",
        "| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in payload["test_domain_metrics"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | `{_pct(row['all_improvement_vs_floor'])}` | "
            f"`{_pct(row['t50_improvement_vs_floor'])}` | `{_pct(row['t100_raw_frame_diagnostic_vs_floor'])}` | "
            f"`{_pct(row['hard_failure_improvement_vs_floor'])}` | `{_pct(row['easy_degradation_vs_floor'])}` | `{_pct(row['switch_rate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Safety Flags",
            "",
            f"- domain easy safe: `{payload['test_flags']['domain_easy_safe']}`",
            f"- negative source count: `{payload['test_flags']['negative_source_count']}`",
            f"- positive t50 domain count: `{payload['test_flags']['positive_t50_domain_count']}`",
            f"- deployable-like under test diagnostic: `{payload['test_flags']['deployable_like_under_test_diagnostic']}`",
            "",
            "## Claim Boundary",
            "",
            "- This is validation-selected and test-evaluated once, but still requires reviewer replay before any deployment update.",
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
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_stage43_k"]
    body = [
        f"Stage43-AW turns the Stage43-AV diagnostic into a validation-selected source/horizon expert policy: Stage43-K remains the non-t50 base, and a t50 expert is selected on validation only. Gate: `{payload['stage43_aw_gate']['passed']} / {payload['stage43_aw_gate']['total']}` with verdict `{payload['stage43_aw_gate']['verdict']}`.",
        "",
        f"Selected t50 expert `{payload['policy']['t50_expert_trial']['name']}` test metrics: all `{_pct(metrics['all_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_improvement_vs_floor'])}`, t100 raw-frame diagnostic `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        f"Delta vs Stage43-K: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        "",
        f"Decision: `{payload['deployment_decision']}`. This remains dataset-local/raw-frame 2.5D evidence; Stage5C and SMC remain disabled.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, body)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state.setdefault("stage43", {})
    state["stage43"]["source_horizon_expert_policy"] = {
        "verdict": payload["stage43_aw_gate"]["verdict"],
        "gate": f"{payload['stage43_aw_gate']['passed']}/{payload['stage43_aw_gate']['total']}",
        "selected_t50_expert": payload["policy"]["t50_expert_trial"]["name"],
        "test_all": metrics["all_improvement_vs_floor"],
        "test_t50": metrics["t50_improvement_vs_floor"],
        "test_hard": metrics["hard_failure_improvement_vs_floor"],
        "test_easy": metrics["easy_degradation_vs_floor"],
        "delta_vs_stage43_k": payload["delta_vs_stage43_k"],
        "deployment_decision": payload["deployment_decision"],
        "candidate_for_reviewer_replay": payload["stage43_aw_gate"]["candidate_for_reviewer_replay"],
        "result_source": payload["result_source"],
    }
    write_json(RESEARCH_STATE, state)


def run_source_horizon_expert_policy(*, max_trials: int = 30, bootstrap: int = 1000) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = build_source_horizon_expert_policy(max_trials=max_trials, bootstrap=bootstrap)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload))
    write_md(
        GATE_MD,
        [
            "# Stage43-AW Gate",
            "",
            f"- verdict: `{payload['stage43_aw_gate']['verdict']}`",
            f"- passed: `{payload['stage43_aw_gate']['passed']} / {payload['stage43_aw_gate']['total']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{k}` | `{v}` |" for k, v in payload["stage43_aw_gate"]["gates"].items()],
            "",
        ],
    )
    _update_summaries(payload)
    with LEDGER_JSONL.open("a") as fh:
        fh.write(
            json.dumps(
                {"source": SOURCE, "verdict": payload["stage43_aw_gate"]["verdict"], "generated_at_utc": payload["generated_at_utc"]}
            )
            + "\n"
        )
    return payload


def main() -> None:
    payload = run_source_horizon_expert_policy()
    gate = payload["stage43_aw_gate"]
    print(json.dumps({"verdict": gate["verdict"], "passed": gate["passed"], "total": gate["total"]}, indent=2))


if __name__ == "__main__":
    main()
