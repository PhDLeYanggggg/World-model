"""Audit an explicitly authorized, locally downloaded DroneCrowd ZIP, without extraction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_dronecrowd_metadata import plain_path, verify_local
from src.evaluation.m3w_dronecrowd_archive import ARCHIVE_FILE_ID, audit_archive
from src.evaluation.m3w_dronecrowd_intake import sequence_ids, sha


def require_authorization(value):
    if (value.get("route") != "independent_external_scenes_first"
            or value.get("annotation_download_and_audit") is not True
            or value.get("archive_file_id") != ARCHIVE_FILE_ID
            or value.get("scientific_roles_assigned") is not False
            or value.get("execute_downloaded_content") is not False
            or not value.get("user_message")):
        raise ValueError("Explicit scoped authorization receipt required; no role assignment")


def require_archive_identity(path, authorization):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (digest != authorization.get("archive_sha256")
            or path.stat().st_size != authorization.get("downloaded_bytes")):
        raise ValueError("Archive differs from the authorized acquisition snapshot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "external_data/DroneCrowd_annotations/annotations.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/publication_readiness_2026_09/dronecrowd_annotations_v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    archive, output = plain_path(args.archive), plain_path(args.output_dir)
    if not archive.is_relative_to(ROOT / "external_data") or not output.is_relative_to(ROOT / "outputs"):
        raise SystemExit("Source must remain under ignored external_data; reports under outputs")
    if subprocess.run(["git", "check-ignore", "--quiet", str(archive)], cwd=ROOT).returncode:
        raise SystemExit("Archive must be excluded from Git")
    authorization = output / "authorization.json"
    permission = json.loads(authorization.read_text())
    require_authorization(permission)
    require_archive_identity(archive, permission)
    metadata_output = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_metadata_v1"
    metadata = verify_local(ROOT / "external_data/DroneCrowd_release_metadata",
                            json.loads((metadata_output / "source_manifest.json").read_text()))
    start = time.monotonic()
    def progress(sequence, passed, failed):
        if int(sequence) % 16 == 0 or sequence == "00112":
            print(json.dumps({"last_sequence": sequence, "audited": passed,
                              "quarantined": failed, "elapsed_seconds": round(time.monotonic()-start, 1)}), flush=True)
    result = audit_archive(archive, sequence_ids(metadata["trainlist.txt"]),
                           sequence_ids(metadata["testlist.txt"]), progress)
    result["authorization_sha256"] = sha(authorization.read_bytes())
    result["metadata_manifest_sha256"] = sha((metadata_output / "source_manifest.json").read_bytes())
    result["implementation_sha256"] = {name: sha((ROOT / name).read_bytes()) for name in (
        "src/evaluation/m3w_dronecrowd_archive.py", "scripts/audit_m3w_dronecrowd_annotations.py",
        "src/evaluation/m3w_dronecrowd_intake.py", "tests/test_m3w_dronecrowd_archive.py")}
    result_path = output / "analysis.json"
    if args.verify:
        if result != json.loads(result_path.read_text()):
            raise SystemExit("Frozen raw audit differs; do not overwrite")
        status = "cached_verified"
    else:
        with result_path.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "seconds": round(time.monotonic()-start, 3), "analysis_sha256": sha(result_path.read_bytes()),
                   "raw_archive_extracted": False, "third_party_code_executed": False}
        with (output / "execution.json").open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
        status = "fresh_run"
    print(json.dumps({"status": status, "aggregate": result["aggregate"],
                      "structural_failures": result["structural_failures"],
                      "analysis_sha256": sha(result_path.read_bytes()), "admitted_recordings": 0}, indent=2))


if __name__ == "__main__":
    main()
