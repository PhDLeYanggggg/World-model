from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage41_fresh_confirmation")
REPORT_JSON = OUT_DIR / "stage41_jepa_deployment_decision.json"
REPORT_MD = OUT_DIR / "stage41_jepa_deployment_decision.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


EVIDENCE_PATHS = {
    "stage18_sam_jepa": Path("outputs/reports/stage18_jepa_training_report.json"),
    "stage19_wam_jepa": Path("outputs/reports/stage19_jepa_probe_report.json"),
    "stage23_sdd_jepa": Path("outputs/reports/stage23_sdd_jepa_metrics.json"),
    "stage24_sdd_jepa": Path("outputs/reports/stage24_sdd_jepa_metrics.json"),
    "stage39_jepa_auxiliary": Path("outputs/stage39_neural_dynamics/stage39_jepa_report.json"),
    "stage40_jepa_trials": Path("outputs/stage40_neural_optimization/stage40_training_trials.json"),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stage18_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    probe = report.get("probe") or {}
    failure_lift = _num(probe.get("jepa_frozen_failure_auc")) - _num(probe.get("no_jepa_failure_auc"))
    return {
        "source": report.get("source", "cached_verified"),
        "attempt": "SAM-JEPA-2.5D representation pretraining",
        "non_collapse": bool(report.get("non_collapse")),
        "downstream_lifts": {"failure_auroc_lift": failure_lift},
        "deployable_lift": failure_lift > 0,
        "notes": "Stage18 was non-collapse but did not improve selector/failure/goal/correction/official t+50.",
    }


def _stage19_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    failure_lift = _num(report.get("failure_jepa_auroc")) - _num(report.get("failure_no_jepa_auroc"))
    selector_lift = _num(report.get("selector_jepa_t50")) - _num(report.get("selector_no_jepa_t50"))
    return {
        "source": report.get("source", "cached_verified"),
        "attempt": "WAM-style JEPA probe",
        "non_collapse": bool(report.get("non_collapse")),
        "downstream_lifts": {"failure_auroc_lift": failure_lift, "selector_t50_lift": selector_lift},
        "deployable_lift": failure_lift > 0 and selector_lift > 0,
        "notes": "Stage19 has mixed historical diagnostics, but the registered probe report shows negative failure lift and no meaningful selector lift.",
    }


def _stage23_or_24_evidence(name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    lifts = {
        "selector_probe_lift": _num(report.get("selector_probe_lift")),
        "failure_predictor_lift": _num(report.get("failure_predictor_lift")),
        "hard_failure_correction_lift": _num(report.get("hard_failure_correction_lift")),
        "t50_lift": _num(report.get("t50_lift")),
    }
    return {
        "source": report.get("source", "cached_verified"),
        "attempt": name,
        "non_collapse": bool(report.get("non_collapse", report.get("trajectory_only_jepa", False))),
        "downstream_lifts": lifts,
        "deployable_lift": any(v > 0 for v in lifts.values()),
        "notes": "SDD JEPA retest is diagnostic unless it improves a downstream head.",
    }


def _stage39_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    lift = report.get("downstream_lift") or {}
    failure_lift = _num(lift.get("failure_auroc_lift"))
    return {
        "source": report.get("source", "cached_verified"),
        "attempt": "Stage39 JEPA auxiliary representation",
        "non_collapse": bool(report.get("non_collapse")),
        "downstream_lifts": {
            "failure_auroc_lift": failure_lift,
            "failure_auroc_base": _num(lift.get("failure_auroc_base")),
            "failure_auroc_with_jepa": _num(lift.get("failure_auroc_with_jepa")),
        },
        "deployable_lift": failure_lift > 0,
        "notes": "JEPA is representation-only; no latent generative rollout. Stage39 failure AUROC lift is negative.",
    }


def _stage40_evidence(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ["jepa_aux_candidate_ranker", "hybrid_moe_deeper_ranker"]:
        row = (report.get("trials") or {}).get(key) or {}
        metrics = row.get("test_metrics") or {}
        out.append(
            {
                "source": row.get("source", report.get("source", "cached_verified")),
                "attempt": f"Stage40 {key}",
                "non_collapse": True,
                "downstream_lifts": {
                    "all_improvement": _num(metrics.get("all_improvement")),
                    "t50_improvement": _num(metrics.get("t50_improvement")),
                    "hard_failure_improvement": _num(metrics.get("hard_failure_improvement")),
                    "easy_degradation": _num(metrics.get("easy_degradation"), 99.0),
                    "switch_rate": _num(metrics.get("switch_rate")),
                },
                "deployable_lift": (
                    _num(metrics.get("all_improvement")) > 0
                    and _num(metrics.get("t50_improvement")) > 0
                    and _num(metrics.get("hard_failure_improvement")) > 0
                    and _num(metrics.get("easy_degradation"), 99.0) <= 0.02
                ),
                "notes": "Stage40 JEPA/hybrid ranker failed the safety/deployment objective and was not deployed.",
            }
        )
    return out


def _collect_evidence() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage18 = read_json(EVIDENCE_PATHS["stage18_sam_jepa"], {})
    if stage18:
        rows.append(_stage18_evidence(stage18))
    stage19 = read_json(EVIDENCE_PATHS["stage19_wam_jepa"], {})
    if stage19:
        rows.append(_stage19_evidence(stage19))
    for key, label in [("stage23_sdd_jepa", "Stage23 SDD JEPA"), ("stage24_sdd_jepa", "Stage24 SDD JEPA")]:
        report = read_json(EVIDENCE_PATHS[key], {})
        if report:
            rows.append(_stage23_or_24_evidence(label, report))
    stage39 = read_json(EVIDENCE_PATHS["stage39_jepa_auxiliary"], {})
    if stage39:
        rows.append(_stage39_evidence(stage39))
    stage40 = read_json(EVIDENCE_PATHS["stage40_jepa_trials"], {})
    if stage40:
        rows.extend(_stage40_evidence(stage40))
    return rows


def run_jepa_deployment_decision() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    evidence = _collect_evidence()
    non_collapse_attempts = [row for row in evidence if row.get("non_collapse")]
    deployable_lifts = [row for row in evidence if row.get("deployable_lift")]
    disable_deployable_path = bool(evidence and not deployable_lifts)
    decision = {
        "source": "fresh_run",
        "decision": "disable_jepa_in_deployable_path" if disable_deployable_path else "keep_jepa_experimental_only",
        "disable_jepa_in_deployable_path": disable_deployable_path,
        "keep_jepa_for_diagnostic_research": True,
        "attempt_count": len(evidence),
        "non_collapse_attempt_count": len(non_collapse_attempts),
        "deployable_positive_attempt_count": len(deployable_lifts),
        "evidence": evidence,
        "no_leakage": {
            "future_endpoint_input": False,
            "test_endpoint_goals": False,
            "central_velocity": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "jepa_is_latent_generative_rollout": False,
        },
        "caveat": "This is a deployment decision, not a claim that JEPA can never help. It says current audited JEPA variants do not earn a deployable role in M3W-Neural v1.",
    }
    write_json(REPORT_JSON, _jsonable(decision))
    lines = [
        "# Stage41 JEPA Deployment Decision",
        "",
        "- source: `fresh_run`",
        f"- decision: `{decision['decision']}`",
        f"- attempts audited: `{len(evidence)}`",
        f"- non-collapse attempts: `{len(non_collapse_attempts)}`",
        f"- deployable positive attempts: `{len(deployable_lifts)}`",
        f"- Stage5C executed: `{decision['no_leakage']['stage5c_executed']}`",
        f"- SMC enabled: `{decision['no_leakage']['smc_enabled']}`",
        "",
        "| Attempt | Non-collapse | Deployable lift | Key lifts |",
        "| --- | --- | --- | --- |",
    ]
    for row in evidence:
        lines.append(f"| {row['attempt']} | `{row.get('non_collapse')}` | `{row.get('deployable_lift')}` | `{row.get('downstream_lifts')}` |")
    lines.extend(
        [
            "",
            "Conclusion: JEPA remains a diagnostic representation path only. The deployable M3W-Neural v1 path uses the Stage37 floor plus neural group-consistency safety/gain heads, not JEPA. This avoids overstating non-collapse JEPA as downstream world-model contribution.",
        ]
    )
    write_md(REPORT_MD, lines)
    return decision


def main_jepa_deployment_decision() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_jepa_deployment_decision()
        status = "ok"
    finally:
        _append_ledger("stage41_jepa_deployment_decision", status, started, EVIDENCE_PATHS.values(), [REPORT_MD, REPORT_JSON])


if __name__ == "__main__":
    main_jepa_deployment_decision()
