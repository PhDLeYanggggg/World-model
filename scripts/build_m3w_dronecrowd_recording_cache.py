"""Convert all audited XML recordings to lossless mmap caches, without fitting."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_dronecrowd_windows import verify_reference
from scripts.audit_m3w_dronecrowd_metadata import plain_path
from src.data_unification.m3w_dronecrowd_cache import (
    ARRAY_NAMES, DroneCrowdSceneWindows, digest_file, write_recording_cache,
)
from src.evaluation.m3w_dronecrowd_intake import sha
from src.evaluation.m3w_dronecrowd_windows import PROBES, compare_inputs, parse_recording, past_inputs


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    source = ROOT / "external_data/DroneCrowd_annotations/annotations.zip"
    reference_path = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_annotations_v1/analysis.json"
    permission_path = ROOT / "outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json"
    permission = json.loads(permission_path.read_text())
    if not permission.get("technical_and_no_leakage_audits_delegated"):
        raise ValueError("Missing delegated source audit authorization")
    reference = verify_reference(reference_path)
    if digest_file(source) != reference["archive_sha256"]:
        raise ValueError("Annotation archive changed")
    cache = plain_path(ROOT / "data/stage_cvpr2027_experiments/dronecrowd_recordings_v1")
    report = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_recordings_v1"
    if subprocess.run(["git", "check-ignore", "--quiet", str(cache / "recording.npy")], cwd=ROOT).returncode:
        raise ValueError("Derived cache must be Git ignored")
    cache.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    lock = (cache / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another cache conversion is active")
    provenance = {
        "archive_sha256": digest_file(source), "annotation_audit_sha256": digest_file(reference_path),
        "authorization_sha256": digest_file(permission_path),
        "code_sha256": {name: digest_file(ROOT / name) for name in (
            "src/data_unification/m3w_dronecrowd_cache.py", "src/evaluation/m3w_dronecrowd_windows.py",
            "src/evaluation/m3w_dronecrowd_intake.py", "scripts/build_m3w_dronecrowd_recording_cache.py")},
    }
    identity = cache / "identity.json"
    if identity.exists():
        if not args.resume:
            raise SystemExit("Existing cache; use --resume to verify rather than overwrite")
        if json.loads(identity.read_text()) != provenance:
            raise ValueError("Converter/source identity changed")
    else:
        atomic_json(identity, provenance)
    member_hashes = {item["name"]: item["sha256"] for item in reference["members"]}
    rows, fresh, reused, prefix_checks = [], 0, 0, 0
    xml_seconds, cache_open_seconds, input_seconds = 0., 0., 0.
    with zipfile.ZipFile(source) as archive:
        for item in reference["sequences"]:
            sequence = item["sequence_id"]
            data = archive.read(f"annotations/{sequence}.xml")
            if sha(data) != member_hashes[f"annotations/{sequence}.xml"]:
                raise ValueError("XML reference mismatch")
            tick = time.monotonic()
            record = parse_recording(data, sequence)
            xml_seconds += time.monotonic()-tick
            path = cache / sequence
            if path.exists():
                reused += 1
            else:
                # A crash can leave a partial directory; fail explicitly instead
                # of treating it as valid or deleting unverified local material.
                write_recording_cache(path, record, provenance)
                fresh += 1
            tick = time.monotonic()
            reader = DroneCrowdSceneWindows(path, PROBES[0], expected_xml_sha256=record.source_xml_sha256,
                                           expected_provenance=provenance)
            cache_open_seconds += time.monotonic()-tick
            for name in ARRAY_NAMES:
                if not np.array_equal(getattr(reader.record, name), getattr(record, name)):
                    raise ValueError("Lossless source-array comparison failed")
            for query in (7, 149, 298):
                tick = time.monotonic()
                actual = reader.get_inputs(query)
                input_seconds += time.monotonic()-tick
                if not compare_inputs(actual, past_inputs(record, query, PROBES[0])):
                    raise ValueError("Cached input differs from source")
                # Remove every future row and every future-only agent. Reusing
                # the same API verifies the mmap cache does not change causality.
                keep = record.valid[:, :query+1].any(axis=1)
                prefix = replace(record, agent_ids=record.agent_ids[keep],
                                 positions=record.positions[keep, :query+1],
                                 valid=record.valid[keep, :query+1], in_bounds=record.in_bounds[keep, :query+1])
                if not compare_inputs(actual, past_inputs(prefix, query, PROBES[0])):
                    raise ValueError("Cached scene inventory depends on future rows")
                prefix_checks += 1
            rows.append({"sequence_id": sequence, "source_xml_sha256": record.source_xml_sha256,
                         "metadata_sha256": digest_file(path / "metadata.json"),
                         "agents": len(record.agent_ids), "visible_rows": int(record.valid.sum()),
                         "main_query_frames": len(reader),
                         "array_bytes": sum(a["bytes"] for a in reader.metadata["artifacts"].values())})
            if len(rows)%16 == 0 or len(rows)==len(reference["sequences"]):
                progress = {"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(),
                            "recordings_verified": len(rows), "total": len(reference["sequences"]),
                            "seconds": round(time.monotonic()-started, 3)}
                atomic_json(cache / "heartbeat.json", progress)
                print(json.dumps(progress), flush=True)
    result = {
        "scope": "lossless_causal_recording_cache_not_forecast_results", "provenance": provenance,
        "schema": reader.schema(), "recordings": rows, "recordings_count": len(rows),
        "agent_tracks": sum(x["agents"] for x in rows), "visible_rows": sum(x["visible_rows"] for x in rows),
        "main_query_frames": sum(x["main_query_frames"] for x in rows),
        "array_bytes": sum(x["array_bytes"] for x in rows),
        "full_array_equal_source": True, "prefix_checks": prefix_checks, "prefix_mismatches": 0,
        "scientific_roles_assigned": False, "independent_sites_verified": False,
        "training": "not_run", "forecast_evaluation": "not_run", "episodes_materialized": False,
        "stage5c_executed": False, "smc_enabled": False,
    }
    target = report / "analysis.json"
    result_source = "fresh_run"
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError("Verified cache reconstruction differs from previous analysis")
        result_source = "cached_verified"
    else:
        atomic_json(target, result)
    execution = {"result_source": result_source, "fresh_recordings": fresh, "cached_verified_recordings": reused,
                 "xml_parse_seconds": xml_seconds, "hash_verified_cache_open_seconds": cache_open_seconds,
                 "past_input_queries": prefix_checks, "past_input_seconds": input_seconds,
                 "wall_seconds": time.monotonic()-started, "analysis_sha256": digest_file(target),
                 "completed_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(report / ("verification_execution.json" if target.exists() and result_source=="cached_verified"
                          else "execution.json"), execution)
    print(json.dumps(execution), flush=True)


if __name__ == "__main__":
    main()
