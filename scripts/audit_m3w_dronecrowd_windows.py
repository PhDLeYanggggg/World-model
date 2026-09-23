"""Read authorized annotations for window support; no model data export or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_m3w_dronecrowd_annotations import require_archive_identity, require_authorization
from scripts.audit_m3w_dronecrowd_metadata import plain_path, verify_local
from src.evaluation.m3w_dronecrowd_archive import archive_inventory
from src.evaluation.m3w_dronecrowd_intake import sequence_ids, sha
from src.evaluation.m3w_dronecrowd_windows import PROBES, audit_support, parse_recording, verify_prefix_invariance


REFERENCE_SHA256 = "b680ca560c2dc73f2db70a64c12ec1d23b0b62d96deecb6cc999d8628732407a"
IMPLEMENTATION = (
    "src/evaluation/m3w_dronecrowd_windows.py", "scripts/audit_m3w_dronecrowd_windows.py",
    "tests/test_m3w_dronecrowd_windows.py",
)


def verify_reference(reference_path):
    data = reference_path.read_bytes()
    if sha(data) != REFERENCE_SHA256:
        raise ValueError("Frozen annotation audit identity changed")
    reference = json.loads(data)
    if (reference["structural_failures"] or reference["admitted_recordings"] != 0
            or reference["scientific_roles_assigned"]):
        raise ValueError("Unexpected prior audit scope or failed recordings")
    for name, digest in reference["implementation_sha256"].items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise ValueError("Prior audit implementation changed; review before reuse")
    return reference


def aggregate_support(recordings):
    aggregate = {}
    for split in ("train", "test", "all"):
        selected = [r for r in recordings if split == "all" or r["release_split"] == split]
        probes = {}
        for spec in PROBES:
            histogram = Counter()
            totals = Counter()
            for record in selected:
                for key, value in record["probes"][spec.name].items():
                    if key == "target_count_histogram":
                        histogram.update(value)
                    else:
                        totals[key] += value
            probes[spec.name] = dict(totals)
            probes[spec.name]["target_count_histogram"] = dict(sorted(histogram.items(), key=lambda p: int(p[0])))
        aggregate[split] = {"recordings": len(selected), "probes": probes}
    return aggregate


def audit_windows(archive_path, reference, train_ids, test_ids, progress=None):
    if set(train_ids) & set(test_ids):
        raise ValueError("Release IDs overlap")
    all_ids = set(train_ids) | set(test_ids)
    reference_records = {r["sequence_id"]: r for r in reference["sequences"]}
    if set(reference_records) != all_ids:
        raise ValueError("Release and reference recording coverage differ")
    members = {v["name"]: v["sha256"] for v in reference["members"]}
    recordings = []
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive_inventory(archive)
        names = {e.filename for e in entries if not e.is_dir()}
        if names != set(members):
            raise ValueError("Archive and reference member coverage differ")
        for sequence in sorted(all_ids):
            name = f"annotations/{sequence}.xml"
            data = archive.read(name)
            if sha(data) != members[name]:
                raise ValueError("XML differs from reference member hash")
            record = parse_recording(data, sequence)
            probes = {spec.name: audit_support(record, spec) for spec in PROBES}
            previous = reference_records[sequence]["structural_windows"]
            for spec in PROBES:
                old_key = spec.name.replace("raw_t", "pred")
                if probes[spec.name]["target_agent_windows"] != previous[old_key]:
                    raise ValueError("Independent multi-agent count differs from prior per-agent run count")
            recordings.append({
                "sequence_id": sequence, "release_split": "train" if sequence in train_ids else "test",
                "source_xml_sha256": record.source_xml_sha256, "probes": probes,
                "prefix_invariance": verify_prefix_invariance(record),
            })
            if progress:
                progress(sequence, len(recordings))
    return {
        "schema_version": 1, "result_source": "fresh_run",
        "scope": "annotation_structural_multiagent_support_and_exported_prefix_separation",
        "archive_sha256": reference["archive_sha256"], "reference_audit_sha256": REFERENCE_SHA256,
        "source_format": "xml_only_audit_reference_not_final_scientific_choice",
        "window_probes": [asdict(spec) for spec in PROBES],
        "probe_choice_is_not_forecasting_protocol_approval": True,
        "aggregate": aggregate_support(recordings), "recordings": recordings,
        "prefix_invariance": {
            "query_prefixes_checked": sum(r["prefix_invariance"]["query_prefixes_checked"] for r in recordings),
            "transformations_per_prefix": 2, "input_mismatches": 0,
            "source_time_causality_proved": False,
        },
        "past_input_agent_inventory": "currently_visible_agents_regardless_of_future_validity",
        "past_incomplete_history": "zero_masked_not_imputed_or_discarded_by_future",
        "target_availability": "loss_and_evaluation_only_full_contiguous_raw_span",
        "neighbor_features_from_target_filtered_inventory": False,
        "bounds_policy": "flag_and_count_sensitivity_no_clipping_or_adopted_quality_exclusion",
        "temporal_nonoverlap_is_not_independent_scene_count": True,
        "observation_mode": "offline_annotated_source_time_provenance_unresolved",
        "coordinate_unit": "image_pixel", "metric_status": "unverified",
        "effective_seconds": "unknown", "physical_sites": None,
        "model_feature_rows_exported": 0, "admitted_recordings": 0,
        "scientific_roles_assigned": False, "new_images_or_videos_downloaded": False,
        "forecast_evaluation": "not_run", "training": "not_run",
        "calibration": "not_run", "confirmation": "not_run",
        "stage5c_executed": False, "smc_enabled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/publication_readiness_2026_09/dronecrowd_window_separation_v1")
    args = parser.parse_args()
    output = plain_path(args.output_dir)
    if not output.is_relative_to(ROOT / "outputs"):
        raise SystemExit("Reports must remain under outputs")
    archive_path = plain_path(ROOT / "external_data/DroneCrowd_annotations/annotations.zip")
    if subprocess.run(["git", "check-ignore", "--quiet", str(archive_path)], cwd=ROOT).returncode:
        raise SystemExit("Annotation source must remain Git-ignored")
    prior = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_annotations_v1"
    reference = verify_reference(prior / "analysis.json")
    permission = json.loads((prior / "authorization.json").read_text())
    require_authorization(permission)
    require_archive_identity(archive_path, permission)
    if reference["archive_sha256"] != permission["archive_sha256"]:
        raise SystemExit("Reference and authorized archive differ")
    metadata_path = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_metadata_v1/source_manifest.json"
    if sha(metadata_path.read_bytes()) != reference["metadata_manifest_sha256"]:
        raise SystemExit("Prior metadata manifest changed")
    metadata = verify_local(ROOT / "external_data/DroneCrowd_release_metadata", json.loads(metadata_path.read_text()))
    start = time.monotonic()
    def progress(sequence, count):
        if count % 16 == 0:
            print(json.dumps({"audited_recordings": count, "last_sequence": sequence,
                              "elapsed_seconds": round(time.monotonic()-start, 1)}), flush=True)
    result = audit_windows(archive_path, reference, sequence_ids(metadata["trainlist.txt"]),
                           sequence_ids(metadata["testlist.txt"]), progress)
    result["authorization_sha256"] = sha((prior / "authorization.json").read_bytes())
    result["metadata_manifest_sha256"] = sha(metadata_path.read_bytes())
    result["implementation_sha256"] = {n: sha((ROOT / n).read_bytes()) for n in IMPLEMENTATION}
    # JSON's round-trip canonicalizes tuple probe offsets before equality checks.
    result = json.loads(json.dumps(result, allow_nan=False))
    if args.verify:
        if result != json.loads((output / "analysis.json").read_text()):
            raise SystemExit("Frozen window audit differs; never overwrite")
        status = "cached_verified"
    else:
        if output.exists():
            raise SystemExit("Output exists; use --verify or a new version")
        output.mkdir(parents=True)
        with (output / "analysis.json").open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "seconds": round(time.monotonic()-start, 3),
                   "analysis_sha256": sha((output / "analysis.json").read_bytes()),
                   "implementation_sha256": result["implementation_sha256"],
                   "runtime": sys.version, "raw_archive_extracted": False,
                   "new_downloads": False, "training_or_prediction": False}
        with (output / "execution.json").open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
        status = "fresh_run"
    print(json.dumps({"status": status, "aggregate": result["aggregate"],
                      "prefix_invariance": result["prefix_invariance"],
                      "analysis_sha256": sha((output / "analysis.json").read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
