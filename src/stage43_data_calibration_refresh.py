from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage42_data_calibration as s42
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "data_calibration_stage43.json"
REPORT_MD = OUT_DIR / "data_calibration_stage43.md"
GATE_MD = OUT_DIR / "stage43_stage_as_data_calibration_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE42_DATA = Path("outputs/stage42_long_research/data_calibration_stage42.json")
STAGE42_TIME = Path("outputs/stage42_long_research/source_time_geometry_calibration_stage42.json")
STAGE23_SDD_TIME = Path("outputs/reports/stage23_sdd_time_geometry_audit.json")
STAGE30_SDD_TIME = Path("outputs/stage30_m3w_verified/time_geometry_raw_audit.json")

SECTION = "STAGE43_AS_DATA_CALIBRATION_REFRESH"
SOURCE = "fresh_stage43_as_data_calibration_refresh"


def _fresh_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec in s42.DATASET_SPECS:
        copy = dict(spec)
        if copy["id"] == "aerialmpt":
            raw = list(copy["raw_candidates"])
            for extra in ["data/aerialmpt", "data/aerialmpt/DLR_AerialMPT_Dataset.zip"]:
                if extra not in raw:
                    raw.append(extra)
            copy["raw_candidates"] = raw
        specs.append(copy)
    return specs


def _source_specific_calibration(time_payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(time_payload.get("source_records", []))
    supported = [row for row in records if row.get("source_specific_metric_time_evidence")]
    by_domain: dict[str, int] = {}
    for row in supported:
        by_domain[str(row.get("domain", "unknown"))] = by_domain.get(str(row.get("domain", "unknown")), 0) + 1
    return {
        "source": "cached_verified_from_stage42_bn",
        "supported_source_count": len(supported),
        "supported_source_ids": [str(row.get("source_id", "")) for row in supported],
        "supported_by_domain": by_domain,
        "global_metric_claim_allowed": bool(time_payload.get("summary", {}).get("global_metric_claim_allowed", False)),
        "global_seconds_claim_allowed": bool(time_payload.get("summary", {}).get("global_seconds_claim_allowed", False)),
        "m3w_official_metric_seconds_claim_allowed": bool(
            time_payload.get("summary", {}).get("m3w_official_metric_seconds_claim_allowed", False)
        ),
    }


def _sdd_status(stage23: Mapping[str, Any], stage30: Mapping[str, Any]) -> dict[str, Any]:
    conclusion = (
        stage30.get("allowed_conclusion")
        or stage30.get("conclusion")
        or stage23.get("conclusion")
        or "pixel-space only, effective seconds unknown"
    )
    return {
        "source": "cached_verified",
        "coordinate_unit": "pixel",
        "metric_claim_allowed": False,
        "seconds_claim_allowed": False,
        "effective_seconds_status": "unknown_or_not_globally_verified",
        "conclusion": conclusion,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    calibration = payload["source_specific_calibration"]
    datasets = {row["dataset_id"]: row for row in payload["datasets"]}
    gates = {
        "all_required_sources_audited": {"sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"}.issubset(
            set(datasets)
        ),
        "fresh_local_path_audit_ran": all(row.get("source") == "fresh_run" for row in payload["datasets"]),
        "sdd_pixel_raw_frame_guard": payload["sdd_status"]["coordinate_unit"] == "pixel"
        and payload["sdd_status"]["metric_claim_allowed"] is False
        and payload["sdd_status"]["seconds_claim_allowed"] is False,
        "external_domains_available": {"opentraj", "eth_ucy", "trajnet", "ucy"}.issubset(
            set(summary["external_domains_ready_from_existing_state"])
        ),
        "source_specific_calibration_recorded": calibration["supported_source_count"] >= 6,
        "global_metric_seconds_blocked": summary["global_metric_claim_allowed"] is False
        and summary["global_seconds_claim_allowed"] is False
        and calibration["global_metric_claim_allowed"] is False
        and calibration["global_seconds_claim_allowed"] is False,
        "tgsim_diagnostic_only": datasets["tgsim"]["data_role"] == "diagnostic_only"
        and datasets["tgsim"]["metric_claim_allowed"] is True,
        "aerialmpt_audited_no_metric_claim": "aerialmpt" in datasets
        and datasets["aerialmpt"]["metric_claim_allowed"] is False
        and datasets["aerialmpt"]["seconds_claim_allowed"] is False,
        "no_training_or_download": payload["training_run"] is False and payload["auto_download_executed"] is False,
        "stage5c_and_smc_false": payload["stage5c_executed"] is False and payload["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_as_data_calibration_refresh_pass" if passed == total else "stage43_as_data_calibration_refresh_incomplete",
        "data_calibration_ready": passed == total,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _build_payload(_: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    registry = s42._load_registry()
    known = s42._read_known_metrics()
    datasets = [s42._audit_dataset(spec, registry, known) for spec in _fresh_specs()]
    metric_ready = [d["dataset_id"] for d in datasets if d["stage42_readiness"]["ready_for_metric_claim"]]
    seconds_ready = [d["dataset_id"] for d in datasets if d["stage42_readiness"]["ready_for_seconds_claim"]]
    external_ready = [
        d["dataset_id"]
        for d in datasets
        if d["dataset_id"] in {"eth_ucy", "trajnet", "ucy", "opentraj"}
        and d["stage42_readiness"]["can_train_or_eval_from_existing_local_state"]
    ]
    time_payload = read_json(STAGE42_TIME, {})
    stage23_sdd = read_json(STAGE23_SDD_TIME, {})
    stage30_sdd = read_json(STAGE30_SDD_TIME, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_local_path_audit_plus_cached_verified_stage42_time_geometry",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "current_facts": [
            "M3W remains protected dataset-local/raw-frame 2.5D multi-agent world-state evidence.",
            "SDD remains pixel-space with effective seconds unknown unless source-specific audits prove otherwise.",
            "ETH/UCY source-specific calibration candidates exist, but global metric/seconds claims remain blocked.",
            "TGSIM is traffic diagnostic only and cannot be counted as pedestrian top-down world-model success.",
            "AerialMPT is audited as local candidate/diagnostic until source terms and geometry are verified.",
            "No Stage5C execution and no SMC.",
        ],
        "datasets": datasets,
        "summary": {
            "datasets_audited": len(datasets),
            "raw_paths_found": sum(1 for d in datasets if d["raw_path_found"]),
            "converted_paths_found": sum(1 for d in datasets if d["converted_path_found"]),
            "external_domains_ready_from_existing_state": external_ready,
            "metric_claim_ready_datasets": metric_ready,
            "seconds_claim_ready_datasets": seconds_ready,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage43_external_validation_ready": len(external_ready) >= 2,
            "stage43_full_waypoint_prereq_ready": bool(
                read_json("outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json", {})
            )
            and len(external_ready) >= 2,
        },
        "source_specific_calibration": _source_specific_calibration(time_payload),
        "sdd_status": _sdd_status(stage23_sdd, stage30_sdd),
        "training_run": False,
        "auto_download_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "input_hash": _combined_hash([STAGE42_DATA, STAGE42_TIME, STAGE23_SDD_TIME, STAGE30_SDD_TIME]),
    }
    payload["stage43_as_gate"] = _gate(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_as_gate"]
    summary = payload["summary"]
    calibration = payload["source_specific_calibration"]
    lines = [
        "# Stage43-AS Data Calibration Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- data calibration ready: `{gate['data_calibration_ready']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- datasets audited: `{summary['datasets_audited']}`",
        f"- raw paths found: `{summary['raw_paths_found']}`",
        f"- converted paths found: `{summary['converted_paths_found']}`",
        f"- external domains ready from existing state: `{', '.join(summary['external_domains_ready_from_existing_state'])}`",
        f"- source-specific calibration candidates: `{', '.join(calibration['supported_source_ids'])}`",
        f"- global metric claim allowed: `{summary['global_metric_claim_allowed']}`",
        f"- global seconds claim allowed: `{summary['global_seconds_claim_allowed']}`",
        "",
        "## Dataset Table",
        "",
        "| dataset | raw | converted | coordinate | calibration | metric | seconds | role |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["datasets"]:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['raw_path_found']}` | `{row['converted_path_found']}` | {row['known_coordinate_unit']} | {row['calibration_state']} | `{row['metric_claim_allowed']}` | `{row['seconds_claim_allowed']}` | {row['data_role']} |"
        )
    lines.extend(
        [
            "",
            "## Source-Specific Calibration",
            "",
            f"- supported source count: `{calibration['supported_source_count']}`",
            f"- supported by domain: `{calibration['supported_by_domain']}`",
            "- Interpretation: ETH/UCY source-specific timing/coordinate evidence can support restricted future audits, but it does not upgrade the global M3W claim.",
            "",
            "## SDD Status",
            "",
            f"- coordinate unit: `{payload['sdd_status']['coordinate_unit']}`",
            f"- metric claim allowed: `{payload['sdd_status']['metric_claim_allowed']}`",
            f"- seconds claim allowed: `{payload['sdd_status']['seconds_claim_allowed']}`",
            f"- conclusion: {payload['sdd_status']['conclusion']}",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
            "",
            "## Decision",
            "",
            "Stage43 can continue using existing SDD/external data under raw-frame/dataset-local language. Metric/seconds-level claims remain blocked globally; restricted ETH/UCY calibrated-subset work must be explicitly source-specific and separately gated.",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_as_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"external_domains_ready = `{', '.join(summary['external_domains_ready_from_existing_state'])}`",
        f"global_metric_claim_allowed = `{summary['global_metric_claim_allowed']}`",
        f"global_seconds_claim_allowed = `{summary['global_seconds_claim_allowed']}`",
        "",
        "Stage43-AS refreshes the data/calibration state by rerunning local path audits for SDD, OpenTraj, ETH/UCY, TrajNet, UCY, TGSIM, and AerialMPT, then reconciling with Stage42-BN source time/geometry evidence. The result keeps global M3W in raw-frame/dataset-local 2.5D language while preserving source-specific ETH/UCY calibration candidates for separately gated future work.",
        "",
        "No training, no auto-download, no Stage5C, no SMC, and no metric/seconds/true-3D/foundation claim.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_as_data_calibration_refresh"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "data_calibration_ready": gate["data_calibration_ready"],
        "summary": payload["summary"],
        "source_specific_calibration": payload["source_specific_calibration"],
        "sdd_status": payload["sdd_status"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_as_data_calibration_refresh"
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
                        "stage": "Stage43-AS",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "global_metric_claim_allowed": summary["global_metric_claim_allowed"],
                        "global_seconds_claim_allowed": summary["global_seconds_claim_allowed"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _build_payload(args)
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Refresh Stage43 data calibration and source geometry guard.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_as_gate"]
    print(f"Stage43-AS: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
