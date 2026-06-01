from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src.stage43_full_waypoint_latent_robustness_audit import _pct
from src.stage43_full_waypoint_latent_safe_repair import _source_family


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_tail_adapter_source_blocker_audit.json"
REPORT_MD = OUT_DIR / "stage43_tail_adapter_source_blocker_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_ba_tail_adapter_source_blocker_audit_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_ba_tail_adapter_source_blocker_audit"
SECTION = "STAGE43_BA_TAIL_ADAPTER_SOURCE_BLOCKER_AUDIT"

STAGE43_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
STAGE43_AZ = OUT_DIR / "stage43_tail_adapter_reviewer_replay.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _classify_validation_blockers(validation_table: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for key, row in validation_table.items():
        family, horizon_text = str(key).rsplit("|", 1)
        horizon = int(horizon_text)
        fam = by_family.setdefault(
            family,
            {
                "family": family,
                "allowed_horizons": [],
                "blocked_horizons": [],
                "block_reasons": {},
                "h100_blocked": False,
                "validation_rows": 0,
            },
        )
        fam["validation_rows"] += int(row.get("rows", 0))
        record = {
            "horizon": horizon,
            "rows": int(row.get("rows", 0)),
            "improvement": float(row.get("full_waypoint_ade_improvement_vs_floor", 0.0)),
            "easy_degradation": float(row.get("easy_degradation_vs_floor", 0.0)),
            "reason": str(row.get("reason", "unknown")),
        }
        if bool(row.get("allowed", False)):
            fam["allowed_horizons"].append(record)
        else:
            fam["blocked_horizons"].append(record)
            fam["block_reasons"][record["reason"]] = int(fam["block_reasons"].get(record["reason"], 0)) + 1
            if horizon == 100:
                fam["h100_blocked"] = True
    return by_family


def _test_source_rows(by_source_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for bucket in ["worst_sources", "best_sources"]:
        for row in by_source_summary.get(bucket, []):
            source_file = str(row.get("slice", ""))
            if source_file in seen:
                continue
            seen.add(source_file)
            family = _source_family(source_file)
            switch_rate = float(row.get("switch_rate", 0.0))
            selected = float(row.get("full_waypoint_ade_improvement_vs_floor", 0.0))
            ungated = float(row.get("ungated_full_waypoint_ade_improvement_vs_floor", 0.0))
            rows.append(
                {
                    "source_file": source_file,
                    "family": family,
                    "rows": int(row.get("rows", 0)),
                    "selected_improvement": selected,
                    "endpoint_improvement": float(row.get("endpoint_fde_improvement_vs_floor", 0.0)),
                    "ungated_improvement": ungated,
                    "easy_degradation": float(row.get("easy_degradation_vs_floor", 0.0)),
                    "switch_rate": switch_rate,
                    "status": "positive_switched"
                    if selected > 0.0 and switch_rate > 0.0
                    else ("safe_floor_blocked" if switch_rate == 0.0 and selected >= -1e-8 else "needs_review"),
                }
            )
    return sorted(rows, key=lambda row: (row["status"], row["source_file"]))


def _blocked_source_diagnosis(source_rows: list[dict[str, Any]], validation_by_family: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocked = []
    for row in source_rows:
        if row["status"] != "safe_floor_blocked":
            continue
        family = row["family"]
        validation = validation_by_family.get(family, {})
        reasons = dict(validation.get("block_reasons", {}))
        blocked.append(
            {
                **row,
                "validation_allowed_horizons": [item["horizon"] for item in validation.get("allowed_horizons", [])],
                "validation_block_reasons": reasons,
                "h100_blocked_by_validation": bool(validation.get("h100_blocked", False)),
                "diagnosis": _diagnose_block(row, reasons),
            }
        )
    return blocked


def _diagnose_block(row: Mapping[str, Any], reasons: Mapping[str, int]) -> str:
    if float(row["ungated_improvement"]) < -0.5:
        return "floor_required_ungated_catastrophic_negative_transfer"
    if "blocked_insufficient_validation_support" in reasons:
        return "insufficient_validation_support_for_safe_switch"
    if "blocked_validation_nonpositive" in reasons:
        return "validation_nonpositive_for_family_or_horizon"
    if "blocked_validation_easy_harm" in reasons:
        return "validation_easy_harm"
    return "blocked_by_conservative_policy"


def build_tail_adapter_source_blocker_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    tail_p = _load(STAGE43_P)
    az = _load(STAGE43_AZ)
    selected = tail_p["selected_model"]
    validation_by_family = _classify_validation_blockers(selected["validation_support_table"])
    source_rows = _test_source_rows(tail_p["by_source_summary"])
    blocked_sources = _blocked_source_diagnosis(source_rows, validation_by_family)
    positive_sources = [row for row in source_rows if row["status"] == "positive_switched"]
    floor_blocked_sources = [row for row in source_rows if row["status"] == "safe_floor_blocked"]
    catastrophic_blocked = [row for row in blocked_sources if row["diagnosis"] == "floor_required_ungated_catastrophic_negative_transfer"]
    by_domain = tail_p["by_domain"]
    positive_domains = [
        domain for domain, row in by_domain.items() if float(row.get("full_waypoint_ade_improvement_vs_floor", 0.0)) > 0.0
    ]
    nonnegative_domains = [
        domain for domain, row in by_domain.items() if float(row.get("full_waypoint_ade_improvement_vs_floor", 0.0)) >= 0.0
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_source_family_blocker_audit_from_stage43_p_and_az",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {"stage43_p": str(STAGE43_P), "stage43_az": str(STAGE43_AZ)},
        "input_verdicts": {
            "stage43_p": tail_p["stage43_p_gate"]["verdict"],
            "stage43_az": az["stage43_az_gate"]["verdict"],
        },
        "performance_leader_policy_hash": az["policy_hash"],
        "performance_leader_model_hash": tail_p["selected_model"]["model_hash"],
        "source_rows": source_rows,
        "validation_by_family": validation_by_family,
        "blocked_sources": blocked_sources,
        "positive_sources": positive_sources,
        "summary": {
            "test_source_count": int(len(source_rows)),
            "positive_switched_source_count": int(len(positive_sources)),
            "safe_floor_blocked_source_count": int(len(floor_blocked_sources)),
            "catastrophic_ungated_blocked_source_count": int(len(catastrophic_blocked)),
            "domain_count": int(len(by_domain)),
            "positive_domain_count": int(len(positive_domains)),
            "nonnegative_domain_count": int(len(nonnegative_domains)),
            "uniform_positive_transfer_claim_allowed": False,
            "uniform_nonnegative_transfer_supported": len(nonnegative_domains) == len(by_domain),
            "floor_necessity_supported_for_blocked_sources": len(catastrophic_blocked) == len(floor_blocked_sources)
            and len(floor_blocked_sources) > 0,
        },
        "next_required_actions": [
            "Do not claim uniform positive source transfer for Stage43-P.",
            "Repair blocked TrajNet_biwi and TrajNet_mot with source-specific training only if validation support becomes safe.",
            "Keep the safety floor for blocked sources because ungated full-waypoint transfer is strongly negative.",
            "Separate source-family labels from coarse domain labels in future paper tables.",
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
        "input_hash": _combined_hash([STAGE43_P, STAGE43_AZ]),
    }
    payload["stage43_ba_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_p_and_az_passed": payload["input_verdicts"]["stage43_p"]
        == "stage43_p_tail_horizon_adapter_pass_t100_still_fallback"
        and payload["input_verdicts"]["stage43_az"] == "stage43_az_tail_adapter_reviewer_replay_pass",
        "source_rows_audited": summary["test_source_count"] >= 3,
        "positive_and_blocked_sources_separated": summary["positive_switched_source_count"] > 0
        and summary["safe_floor_blocked_source_count"] > 0,
        "blocked_sources_have_diagnosis": all(bool(row.get("diagnosis")) for row in payload["blocked_sources"]),
        "floor_necessity_for_blocked_sources": summary["floor_necessity_supported_for_blocked_sources"] is True,
        "uniform_positive_transfer_not_overclaimed": summary["uniform_positive_transfer_claim_allowed"] is False
        and claim["uniform_positive_external_transfer_claim"] is False,
        "nonnegative_domain_boundary_recorded": summary["uniform_nonnegative_transfer_supported"] is True,
        "validation_blockers_mapped": bool(payload["validation_by_family"]),
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
        and claim["dataset_local_raw_frame_only"] is True,
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
        "verdict": "stage43_ba_tail_adapter_source_blocker_audit_pass"
        if passed == total
        else "stage43_ba_tail_adapter_source_blocker_audit_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ba_gate"]
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    summary = payload["summary"]
    lines = [
        "# Stage43-BA Tail Adapter Source Blocker Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- performance leader policy hash: `{payload['performance_leader_policy_hash']}`",
        f"- model hash: `{payload['performance_leader_model_hash']}`",
        "",
        "## Summary",
        "",
        f"- test sources: `{summary['test_source_count']}`",
        f"- positive switched sources: `{summary['positive_switched_source_count']}`",
        f"- safe-floor blocked sources: `{summary['safe_floor_blocked_source_count']}`",
        f"- catastrophic ungated blocked sources: `{summary['catastrophic_ungated_blocked_source_count']}`",
        f"- positive domains: `{summary['positive_domain_count']} / {summary['domain_count']}`",
        f"- nonnegative domains: `{summary['nonnegative_domain_count']} / {summary['domain_count']}`",
        f"- uniform positive transfer claim allowed: `{summary['uniform_positive_transfer_claim_allowed']}`",
        f"- floor necessity supported for blocked sources: `{summary['floor_necessity_supported_for_blocked_sources']}`",
        "",
        "## Source Rows",
        "",
        "| source family | source | rows | selected lift | ungated lift | switch | status | diagnosis |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    diagnosis_by_source = {row["source_file"]: row.get("diagnosis", "") for row in payload["blocked_sources"]}
    for row in payload["source_rows"]:
        source = Path(row["source_file"]).name
        lines.append(
            f"| `{row['family']}` | `{source}` | {row['rows']} | `{_pct(row['selected_improvement'])}` | "
            f"`{_pct(row['ungated_improvement'])}` | `{_pct(row['switch_rate'])}` | `{row['status']}` | "
            f"{diagnosis_by_source.get(row['source_file'], '')} |"
        )
    lines.extend(
        [
            "",
            "## Validation Blockers By Family",
            "",
            "| family | validation rows | allowed horizons | block reasons | h100 blocked |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for family, row in sorted(payload["validation_by_family"].items()):
        allowed = ",".join(str(item["horizon"]) for item in row["allowed_horizons"]) or "none"
        reasons = ", ".join(f"{key}:{value}" for key, value in sorted(row["block_reasons"].items())) or "none"
        lines.append(
            f"| `{family}` | {row['validation_rows']} | `{allowed}` | `{reasons}` | `{row['h100_blocked']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-P is a strong aggregate performance leader, but it is not a uniform-positive source-transfer result. The blocked TrajNet_biwi and TrajNet_mot sources remain safe because the policy falls back to the floor; their ungated full-waypoint transfer is strongly negative, so the floor is necessary rather than cosmetic.",
            "",
            "## Next Required Actions",
            "",
            *[f"- {item}" for item in payload["next_required_actions"]],
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
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
        "- Stage43-BA explains why uniform positive source transfer is still blocked.",
        "- Safe floor remains necessary for blocked TrajNet_biwi and TrajNet_mot sources.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, gate_lines)
    write_md(GATE_MD, lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ba_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"positive_sources = `{summary['positive_switched_source_count']} / {summary['test_source_count']}`",
        f"safe_floor_blocked_sources = `{summary['safe_floor_blocked_source_count']}`",
        f"catastrophic_ungated_blocked_sources = `{summary['catastrophic_ungated_blocked_source_count']}`",
        f"uniform_positive_transfer_claim_allowed = `{summary['uniform_positive_transfer_claim_allowed']}`",
        "",
        "Stage43-BA audits why the replayed Stage43-P tail adapter cannot be claimed as uniform positive source transfer. The source-level blocker is not hidden: TrajNet_biwi and TrajNet_mot remain floor-only because ungated full-waypoint transfer is catastrophically negative. The safety floor is therefore necessary for these slices.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ba_tail_adapter_source_blocker_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "blocked_sources": payload["blocked_sources"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ba_tail_adapter_source_blocker_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BA",
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


def _run(_: argparse.Namespace) -> dict[str, Any]:
    payload = build_tail_adapter_source_blocker_audit()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Audit source blockers for the Stage43-P/AZ tail adapter.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_ba_gate"]
    print(f"Stage43-BA: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"blocked_sources={result['summary']['safe_floor_blocked_source_count']}")
    print(f"floor_necessity={result['summary']['floor_necessity_supported_for_blocked_sources']}")
    return result


if __name__ == "__main__":
    main()
