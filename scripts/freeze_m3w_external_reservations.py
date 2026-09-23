"""Freeze source-wide holdouts without reading model outputs or future losses."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_source_reservations import (
    KIND, REGISTRY, SourceReservations, metadata_identities,
)
from src.evaluation.m3w_intake_admission import validate_admission


BASE = "outputs/publication_readiness_2026_09/"
INPUTS = {
    "permission": BASE + "delegated_research_authorization_20260923.json",
    "constraints": BASE + "dronecrowd_grouping_v2/exclusion_constraints.json",
    "drone_cache": BASE + "dronecrowd_recordings_v1/analysis.json",
    "dut_screen": BASE + "calibration_support_v1/dut_screen.json",
    "observation_disposition": BASE + "dronecrowd_observation_disposition_20260923.md",
}
EXPECTED = {
    "constraints": "12c8479daf35215d47bde4111aaf74c56849947b1c0d8c43476d9c5778353976",
    "drone_cache": "1bb9946fa1bb62c720b40acb5aa7348dc4e8bf852a44c0d35b8765f36e98e9be",
}


def frozen_json(path, content, verify):
    payload = json.dumps(content, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Frozen output changed: {path}")
    elif verify:
        raise ValueError(f"Missing frozen output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(payload)


def build_registry(root):
    bindings = {relative: sha256(root / relative) for relative in INPUTS.values()}
    for key, expected in EXPECTED.items():
        if bindings[INPUTS[key]] != expected:
            raise ValueError(f"Changed audited reservation input: {key}")
    permission = json.loads((root / INPUTS["permission"]).read_text())
    if not permission.get("evidence_based_observation_protocol_and_data_role_decisions_delegated"):
        raise ValueError("Protocol decision not delegated")
    constraints = json.loads((root / INPUTS["constraints"]).read_text())
    drone = json.loads((root / INPUTS["drone_cache"]).read_text())
    screen = json.loads((root / INPUTS["dut_screen"]).read_text())
    for key in ("source_manifest", "conversion_report", "quality_audit"):
        reference = screen[key]
        if sha256(root / reference["path"]) != reference["sha256"]:
            raise ValueError(f"Changed DUT {key}")
        bindings[reference["path"]] = reference["sha256"]
    components = constraints["conservative_components"]
    sequence_ids = [row["sequence_id"] for row in drone["recordings"]]
    flattened = [sequence for group in components for sequence in group]
    if sorted(flattened) != sorted(sequence_ids) or len(set(sequence_ids)) != 112:
        raise ValueError("Grouping does not cover all 112 source recordings once")
    group_ids = {sequence: f"exclusion_{i:03d}" for i, group in enumerate(components) for sequence in group}
    records = {}
    arrays_verified = 0
    bytes_verified = 0

    def add(name, dataset, directory, metadata_hash, role, group, quality):
        nonlocal arrays_verified, bytes_verified
        metadata_path = root / directory / "metadata.json"
        if sha256(metadata_path) != metadata_hash:
            raise ValueError(f"Changed metadata: {name}")
        metadata = json.loads(metadata_path.read_text())
        for filename, receipt in metadata["artifacts"].items():
            path = root / directory / filename
            if path.parent != root / directory or sha256(path) != receipt["sha256"]:
                raise ValueError(f"Changed recording payload: {name}/{filename}")
            arrays_verified += 1
            bytes_verified += path.stat().st_size
        records[name] = {
            "dataset": dataset, "cache_path": directory, "metadata_sha256": metadata_hash,
            "source_identities": sorted(metadata_identities(metadata)),
            "reserved_role": role, "predictive_admission": False,
            "exclusion_group": group, "quality_disposition": quality,
        }

    for row in drone["recordings"]:
        sequence = row["sequence_id"]
        add("dronecrowd_" + sequence, "DroneCrowd",
            "data/stage_cvpr2027_experiments/dronecrowd_recordings_v1/" + sequence,
            row["metadata_sha256"], "confirmation", group_ids[sequence],
            "offline_XML_audited_not_independent_site_certified")
    if len(screen["recordings"]) != 28:
        raise ValueError("Expected complete 28-recording DUT inventory")
    for name, entry in screen["recordings"].items():
        role = "excluded" if entry["quality_disposition"] == "quarantine" else "calibration"
        add(name, "DUT", entry["cache_path"], entry["metadata_sha256"], role,
            entry["physical_scene"], entry["quality_disposition"])
    excluded = [name for name, item in records.items() if item["reserved_role"] == "excluded"]
    if excluded != ["dut_intersection_04"]:
        raise ValueError("Unexpected quality quarantine set")
    registry = {
        "kind": KIND, "status": "frozen_reservations_not_predictive_admission",
        "decision_date": "2026-09-23", "decision_authority": INPUTS["permission"],
        "independence_certified": False, "bindings": bindings, "records": records,
        "sources": {
            "DroneCrowd": {
                "reserved_role": "confirmation", "partition_scope": "whole_source",
                "recordings": sorted(name for name, item in records.items() if item["dataset"] == "DroneCrowd"),
                "independent_physical_sites": None, "exclusion_groups_are_independent_sites": False,
                "official_release_split_used_for_predictive_roles": False,
                "predictive_outcomes_reviewed_for_assignment": False,
                "remaining_admission_requirements": ["source_use_and_historical_exposure_review",
                    "frozen_predictor_policy_and_producer_chain", "independent_source_overlap_disposition",
                    "one_shot_evaluation_claim_and_appropriate_uncertainty_unit"],
            },
            "DUT": {
                "reserved_role": "calibration", "partition_scope": "whole_source_except_quarantine",
                "recordings": sorted(name for name, item in records.items() if item["dataset"] == "DUT"),
                "author_documented_locations": 2, "independent_physical_sites": None,
                "predictive_outcomes_reviewed_for_assignment": False,
                "remaining_admission_requirements": ["source_use_conditions_and_historical_exposure_review",
                    "frozen_policy_family_and_complete_producer_exclusion",
                    "explicit_small_support_risk_disposition_no_formal_two_percent_claim"],
            },
        },
        "task_unchanged": {"history_steps": 8, "prediction_steps": 12,
            "native_annotation_stride_not_seconds": True, "raw_t50_supplemental": True,
            "easy_degradation_limit_percent": 2.0},
        "uncertainty": "Do not count clips, exclusion groups or windows as independent physical sites",
        "stage5c_execution": False, "smc_enabled": False,
    }
    return registry, arrays_verified, bytes_verified


def audit_guards(root, registry):
    reservations = SourceReservations(root)
    wrong_role_refusals, matching_but_not_admitted = 0, 0
    for name, row in registry["records"].items():
        metadata = json.loads((root / row["cache_path"] / "metadata.json").read_text())
        for role in ("fit", "development", "calibration", "confirmation"):
            if role != row["reserved_role"]:
                try:
                    reservations.validate(name, row, metadata, role)
                except ValueError as exc:
                    if "forbids this data role" not in str(exc):
                        raise
                    wrong_role_refusals += 1
                else:
                    raise AssertionError("Whole-source cross-role reuse was admitted")
            else:
                reservations.validate(name, row, metadata, role)
                try:
                    validate_admission(root, name, row, metadata, role)
                except ValueError as exc:
                    if "source intake screen required" not in str(exc):
                        raise
                    matching_but_not_admitted += 1
                else:
                    raise AssertionError("Reservation silently granted predictive admission")
    return wrong_role_refusals, matching_but_not_admitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verification-receipt", default="verification_execution.json")
    args = parser.parse_args()
    if (Path(args.verification_receipt).name != args.verification_receipt
            or not args.verification_receipt.startswith("verification_")
            or not args.verification_receipt.endswith(".json")):
        parser.error("Verification receipt must be a verification_*.json filename")
    started = time.monotonic()
    registry, arrays, size = build_registry(ROOT)
    frozen_json(ROOT / REGISTRY, registry, args.verify)
    refused, reserved = audit_guards(ROOT, registry)
    output = ROOT / BASE / "external_role_reservations_v1"
    analysis = {
        "scope": "whole_source_reservation_and_admission_guard_not_predictive_experiment",
        "registry_sha256": sha256(ROOT / REGISTRY), "recordings": len(registry["records"]),
        "dronecrowd_confirmation_reserved": 112, "dut_calibration_reserved": 27, "dut_quarantined": 1,
        "arrays_hash_verified": arrays, "array_bytes_hash_verified": size,
        "cross_role_attempts_refused": refused, "matching_roles_still_require_admission": reserved,
        "new_training": "not_run", "forecast_evaluation": "not_run",
        "independent_calibration": "not_run", "independent_confirmation": "not_run",
        "independence_certified": False, "raw_arrays_loaded": False,
        "stage5c_execution": False, "smc_enabled": False,
    }
    frozen_json(output / "analysis.json", analysis, args.verify)
    execution = {"result_source": "cached_verified" if args.verify else "fresh_run",
        "analysis_sha256": sha256(output / "analysis.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/freeze_m3w_external_reservations.py", "src/evaluation/m3w_source_reservations.py",
            "src/evaluation/m3w_intake_admission.py")},
        "wall_seconds": time.monotonic() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat()}
    destination = output / (args.verification_receipt if args.verify else "execution.json")
    if destination.exists():
        raise FileExistsError("Preserve existing execution receipt; use a new explicit verification receipt")
    with destination.open("x") as stream:
        stream.write(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**analysis, **execution}), flush=True)


if __name__ == "__main__":
    main()
