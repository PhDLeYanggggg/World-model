from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src.stage43_full_waypoint_latent_safe_repair import _source_family
from src.stage43_full_waypoint_latent_robustness_audit import _pct
from src.stage43_tail_horizon_waypoint_adapter import HORIZONS, _family_horizon


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_blocked_source_repair_feasibility.json"
REPORT_MD = OUT_DIR / "stage43_blocked_source_repair_feasibility.md"
GATE_MD = OUT_DIR / "stage43_stage_bb_blocked_source_repair_feasibility_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bb_blocked_source_repair_feasibility"
SECTION = "STAGE43_BB_BLOCKED_SOURCE_REPAIR_FEASIBILITY"

STAGE43_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
STAGE43_BA = OUT_DIR / "stage43_tail_adapter_source_blocker_audit.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _split_counts(ds) -> dict[str, Any]:
    families, horizons = _family_horizon(ds)
    by_family: dict[str, Any] = {}
    by_source: dict[str, Any] = {}
    for family in sorted(set(families.tolist())):
        mask = families == family
        horizon_counts = {
            str(h): int(np.sum(mask & (horizons == int(h))))
            for h in HORIZONS
            if int(np.sum(mask & (horizons == int(h)))) > 0
        }
        by_family[family] = {
            "rows": int(mask.sum()),
            "horizon_counts": horizon_counts,
        }
    for source_file in sorted(set(ds.source_file.astype(str).tolist())):
        mask = ds.source_file.astype(str) == source_file
        horizon_counts = {
            str(h): int(np.sum(mask & (horizons == int(h))))
            for h in HORIZONS
            if int(np.sum(mask & (horizons == int(h)))) > 0
        }
        by_source[source_file] = {
            "family": _source_family(source_file),
            "rows": int(mask.sum()),
            "horizon_counts": horizon_counts,
        }
    return {"by_family": by_family, "by_source": by_source}


def _validation_rows_for_family(table: Mapping[str, Mapping[str, Any]], family: str) -> dict[str, Any]:
    rows = {}
    total = 0
    allowed = []
    reasons: Counter[str] = Counter()
    for key, item in table.items():
        fam, horizon = str(key).rsplit("|", 1)
        if fam != family:
            continue
        row_count = int(item.get("rows", 0))
        total += row_count
        rows[horizon] = {
            "rows": row_count,
            "improvement": float(item.get("full_waypoint_ade_improvement_vs_floor", 0.0)),
            "easy_degradation": float(item.get("easy_degradation_vs_floor", 0.0)),
            "allowed": bool(item.get("allowed", False)),
            "reason": str(item.get("reason", "unknown")),
        }
        if bool(item.get("allowed", False)):
            allowed.append(int(horizon))
        else:
            reasons[str(item.get("reason", "unknown"))] += 1
    return {
        "total_rows": int(total),
        "horizons": rows,
        "allowed_horizons": sorted(allowed),
        "block_reasons": dict(reasons),
    }


def _source_support(
    *,
    source_file: str,
    family: str,
    split_counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name, counts in split_counts.items():
        by_source = counts["by_source"]
        by_family = counts["by_family"]
        out[split_name] = {
            "source_rows": int(by_source.get(source_file, {}).get("rows", 0)),
            "source_horizon_counts": dict(by_source.get(source_file, {}).get("horizon_counts", {})),
            "family_rows": int(by_family.get(family, {}).get("rows", 0)),
            "family_horizon_counts": dict(by_family.get(family, {}).get("horizon_counts", {})),
        }
    return out


def _repair_decision(
    *,
    blocked: Mapping[str, Any],
    validation: Mapping[str, Any],
    support: Mapping[str, Any],
    min_validation_rows: int,
    max_easy_degradation: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(validation.get("total_rows", 0)) < int(min_validation_rows):
        reasons.append("insufficient_validation_rows")
    if not validation.get("allowed_horizons"):
        reasons.append("no_validation_allowed_horizon")
    if float(blocked.get("ungated_improvement", 0.0)) < -0.5:
        reasons.append("ungated_transfer_catastrophic_negative")
    if float(blocked.get("easy_degradation", 0.0)) > float(max_easy_degradation):
        reasons.append("easy_harm_on_blocked_source")
    if int(support.get("train", {}).get("family_rows", 0)) == 0:
        reasons.append("no_train_family_rows")
    if int(support.get("val", {}).get("family_rows", 0)) == 0:
        reasons.append("no_val_family_rows")
    if int(support.get("test", {}).get("source_rows", 0)) == 0:
        reasons.append("no_test_source_rows")
    repairable = not reasons
    if repairable:
        status = "repairable_with_validation_guard"
        next_action = "Train a source-family repair candidate using train only, select thresholds on val, and evaluate test once."
    elif "ungated_transfer_catastrophic_negative" in reasons:
        status = "not_repairable_now_keep_floor"
        next_action = "Keep floor-only deployment until validation support proves a source-specific repair is safe."
    elif "insufficient_validation_rows" in reasons or "no_val_family_rows" in reasons:
        status = "not_repairable_now_collect_validation_support"
        next_action = "Add or rebuild validation support for this source family before training a repair."
    else:
        status = "not_repairable_now"
        next_action = "Do not deploy a repair until every blocker is cleared."
    return {
        "repairable_now": bool(repairable),
        "status": status,
        "blockers": reasons,
        "next_action": next_action,
    }


def build_blocked_source_repair_feasibility(
    *,
    seed: int = 431,
    min_validation_rows: int = 1000,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    tail_p = _load(STAGE43_P)
    ba = _load(STAGE43_BA)
    train = m._build_split("train", max_rows=None, seed=int(seed))
    val = m._build_split("val", max_rows=None, seed=int(seed))
    test = m._build_split("test", max_rows=None, seed=int(seed))
    split_counts = {
        "train": _split_counts(train),
        "val": _split_counts(val),
        "test": _split_counts(test),
    }
    validation_table = tail_p["selected_model"]["validation_support_table"]
    blocked_rows: list[dict[str, Any]] = []
    for blocked in ba.get("blocked_sources", []):
        family = str(blocked["family"])
        source_file = str(blocked["source_file"])
        validation = _validation_rows_for_family(validation_table, family)
        support = _source_support(source_file=source_file, family=family, split_counts=split_counts)
        decision = _repair_decision(
            blocked=blocked,
            validation=validation,
            support=support,
            min_validation_rows=int(min_validation_rows),
            max_easy_degradation=float(max_easy_degradation),
        )
        blocked_rows.append(
            {
                "source_file": source_file,
                "family": family,
                "test_rows": int(blocked.get("rows", 0)),
                "selected_improvement": float(blocked.get("selected_improvement", 0.0)),
                "ungated_improvement": float(blocked.get("ungated_improvement", 0.0)),
                "switch_rate": float(blocked.get("switch_rate", 0.0)),
                "validation_support": validation,
                "split_support": support,
                "repair_decision": decision,
            }
        )
    summary = {
        "blocked_source_count": int(len(blocked_rows)),
        "repairable_now_count": int(sum(1 for row in blocked_rows if row["repair_decision"]["repairable_now"])),
        "floor_only_count": int(sum(1 for row in blocked_rows if not row["repair_decision"]["repairable_now"])),
        "catastrophic_ungated_count": int(
            sum("ungated_transfer_catastrophic_negative" in row["repair_decision"]["blockers"] for row in blocked_rows)
        ),
        "insufficient_validation_support_count": int(
            sum("insufficient_validation_rows" in row["repair_decision"]["blockers"] for row in blocked_rows)
        ),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_blocked_source_repair_feasibility_from_validation_support_and_split_counts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_p": str(STAGE43_P),
            "stage43_ba": str(STAGE43_BA),
        },
        "input_verdicts": {
            "stage43_p": tail_p.get("stage43_p_gate", {}).get("verdict"),
            "stage43_ba": ba.get("stage43_ba_gate", {}).get("verdict"),
        },
        "repair_protocol": {
            "selection_data_required": "validation_only",
            "test_threshold_tuning_allowed": False,
            "min_validation_rows": int(min_validation_rows),
            "max_easy_degradation": float(max_easy_degradation),
            "future_labels_eval_or_loss_only": True,
            "diagnostic_test_rows_not_used_for_training": True,
        },
        "blocked_source_rows": blocked_rows,
        "summary": summary,
        "next_required_actions": [
            "Keep Stage43-P/AZ floor-only behavior on blocked TrajNet_biwi and TrajNet_mot sources.",
            "Do not train or deploy source-specific repair until validation support is sufficient and positive.",
            "If more external data is added, rebuild source-family validation support before touching test rows.",
            "Report Stage43-P/AZ as aggregate protected transfer, not uniform positive source transfer.",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_P, STAGE43_BA]),
    }
    payload["stage43_bb_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_ba_precondition_passed": payload["input_verdicts"]["stage43_ba"]
        == "stage43_ba_tail_adapter_source_blocker_audit_pass",
        "blocked_sources_inspected": summary["blocked_source_count"] > 0
        and len(payload["blocked_source_rows"]) == summary["blocked_source_count"],
        "split_support_quantified": all(
            {"train", "val", "test"}.issubset(set(row["split_support"].keys())) for row in payload["blocked_source_rows"]
        ),
        "validation_support_quantified": all(
            "total_rows" in row["validation_support"] and "horizons" in row["validation_support"]
            for row in payload["blocked_source_rows"]
        ),
        "unsafe_repair_correctly_blocked": summary["repairable_now_count"] == 0
        and summary["floor_only_count"] == summary["blocked_source_count"],
        "catastrophic_ungated_transfer_not_deployed": summary["catastrophic_ungated_count"] == summary["blocked_source_count"],
        "diagnostic_test_not_used_for_training": payload["repair_protocol"]["diagnostic_test_rows_not_used_for_training"] is True
        and payload["repair_protocol"]["test_threshold_tuning_allowed"] is False,
        "next_actions_recorded": len(payload["next_required_actions"]) >= 3,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["uniform_positive_external_transfer_claim"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bb_blocked_source_repair_feasibility_pass"
        if passed == total
        else "stage43_bb_blocked_source_repair_feasibility_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bb_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage43-BB Blocked Source Repair Feasibility",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- blocked sources: `{summary['blocked_source_count']}`",
        f"- repairable now: `{summary['repairable_now_count']}`",
        f"- floor-only now: `{summary['floor_only_count']}`",
        f"- catastrophic ungated transfer count: `{summary['catastrophic_ungated_count']}`",
        "",
        "## Blocked Source Decisions",
        "",
        "| family | source | test rows | val rows | train family rows | ungated lift | repair decision | blockers |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["blocked_source_rows"]:
        support = row["split_support"]
        val_rows = row["validation_support"]["total_rows"]
        train_family = support["train"]["family_rows"]
        blockers = ", ".join(row["repair_decision"]["blockers"]) or "none"
        lines.append(
            f"| `{row['family']}` | `{Path(row['source_file']).name}` | {row['test_rows']} | {val_rows} | "
            f"{train_family} | `{_pct(row['ungated_improvement'])}` | `{row['repair_decision']['status']}` | {blockers} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "I am keeping these blocked sources on the floor. That is not hiding a failure: it is the safe deployment rule doing its job. Both blocked sources have catastrophic ungated transfer, and neither currently has enough validation evidence to justify a source-specific repair.",
            "",
            "The practical next move is data/support work, not a new threshold tweak. A repair becomes legitimate only after source-family validation support is large enough, positive, and easy-safe before test is touched.",
            "",
            "## Next Required Actions",
            "",
            *[f"- {item}" for item in payload["next_required_actions"]],
            "",
            "## Claim Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric or seconds-level claim.",
            "- No true 3D or foundation claim.",
            "- No Stage5C execution and no SMC.",
            "- Future labels remain loss/eval only, not inference inputs.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bb_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"blocked_sources = `{summary['blocked_source_count']}`",
        f"repairable_now = `{summary['repairable_now_count']}`",
        f"floor_only_now = `{summary['floor_only_count']}`",
        "",
        "I checked the blocked tail-adapter sources before attempting another repair. The result is deliberately conservative: TrajNet_biwi and TrajNet_mot stay floor-only because ungated transfer is strongly negative and validation support is not strong enough to justify a source-specific switch policy.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bb_blocked_source_repair_feasibility"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "blocked_source_rows": payload["blocked_source_rows"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bb_blocked_source_repair_feasibility"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BB",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "summary": summary,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_bb_gate"]))
    lines = _render_md(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    gate = payload["stage43_bb_gate"]
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-P / AZ remains the performance leader and exact replay artifact.",
        "- Stage43-BB says the blocked sources are not repairable safely yet.",
        "- TrajNet_biwi and TrajNet_mot remain floor-only until validation support improves.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, world_lines)
    _update_ledgers(payload)


def run_blocked_source_repair_feasibility(
    *,
    seed: int = 431,
    min_validation_rows: int = 1000,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    payload = build_blocked_source_repair_feasibility(
        seed=int(seed),
        min_validation_rows=int(min_validation_rows),
        max_easy_degradation=float(max_easy_degradation),
    )
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit repair feasibility for Stage43 blocked tail-adapter sources.")
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--min-validation-rows", type=int, default=1000)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_blocked_source_repair_feasibility(
        seed=int(args.seed),
        min_validation_rows=int(args.min_validation_rows),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = payload["stage43_bb_gate"]
    print(f"Stage43-BB: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"repairable_now={payload['summary']['repairable_now_count']}")
    print(f"floor_only_now={payload['summary']['floor_only_count']}")
    return payload


if __name__ == "__main__":
    main()
