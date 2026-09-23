import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.m3w_intake_admission import validate_admission
from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_source_reservations import (
    KIND, REGISTRY, SourceReservations, check_source_reservation, metadata_identities,
)
from test_m3w_experiment_contract import fixture_contract, approve


def write_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item))
    return sha256(path)


def fixture(root, *, name="reserved", dataset="DroneCrowd", role="confirmation"):
    metadata = {"dataset": dataset, "source_conditions_review": "pending",
                "source_xml_sha256": "1"*64,
                "artifacts": {"positions.npy": {"sha256": "2"*64}, "valid.npy": {"sha256": "3"*64}}}
    digest = write_json(root / name / "metadata.json", metadata)
    record = {"dataset": dataset, "cache_path": name, "metadata_sha256": digest,
              "source_identities": sorted(metadata_identities(metadata)),
              "reserved_role": role, "predictive_admission": False}
    evidence = write_json(root / "decision.json", {"synthetic_fixture_only": True})
    registry = {"kind": KIND, "status": "frozen_reservations_not_predictive_admission",
                "independence_certified": False, "bindings": {"decision.json": evidence},
                "records": {name: record}, "sources": {dataset: {"recordings": [name], "reserved_role": role}}}
    write_json(root / REGISTRY, registry)
    return name, record, metadata


@pytest.mark.parametrize("role", ["fit", "development", "calibration"])
def test_no_training_selection_or_calibration_on_confirmation_source(tmp_path, role):
    name, record, metadata = fixture(tmp_path)
    with pytest.raises(ValueError, match="forbids this data role"):
        validate_admission(tmp_path, name, record, metadata, role)


def test_matching_reservation_is_not_admission_or_legacy_bypass(tmp_path):
    name, record, metadata = fixture(tmp_path)
    metadata.pop("source_conditions_review")
    # Use the same reviewed metadata to exercise the legacy path, not hash drift.
    record["metadata_sha256"] = write_json(tmp_path / name / "metadata.json", metadata)
    path = tmp_path / REGISTRY
    registry = json.loads(path.read_text())
    registry["records"][name] = record
    write_json(path, registry)
    assert check_source_reservation(tmp_path, name, record, metadata, "confirmation")
    with pytest.raises(ValueError, match="source intake screen required"):
        validate_admission(tmp_path, name, record, metadata, "confirmation")


@pytest.mark.parametrize("identity", ["source", "geometry"])
def test_renamed_copied_source_stays_reserved(tmp_path, identity):
    name, record, metadata = fixture(tmp_path)
    renamed = {"cache_path": "renamed", "metadata_sha256": "0"*64}
    fake_metadata = {"dataset": "renamed_unknown"}
    if identity == "source":
        fake_metadata["source_xml_sha256"] = metadata["source_xml_sha256"]
    else:
        fake_metadata["artifacts"] = {"positions.npy": metadata["artifacts"]["positions.npy"]}
    with pytest.raises(ValueError, match="forbids this data role"):
        validate_admission(tmp_path, "renamed", renamed, fake_metadata, "fit")
    with pytest.raises(ValueError, match="exact reviewed recording identity"):
        validate_admission(tmp_path, "renamed", renamed, fake_metadata, "confirmation")


def test_identical_valid_masks_are_not_duplicate_source_evidence(tmp_path):
    _, _, metadata = fixture(tmp_path)
    other = {"cache_path": "unrelated", "metadata_sha256": "0"*64}
    other_metadata = {"dataset": "synthetic_other", "artifacts": {"valid.npy": metadata["artifacts"]["valid.npy"]}}
    assert validate_admission(tmp_path, "unrelated", other, other_metadata, "fit") == {}


@pytest.mark.parametrize("role", ["fit", "development", "calibration", "confirmation"])
def test_quarantine_cannot_be_reassigned(tmp_path, role):
    name, record, metadata = fixture(tmp_path, dataset="DUT", role="excluded")
    with pytest.raises(ValueError, match="forbids this data role"):
        validate_admission(tmp_path, name, record, metadata, role)
    assert validate_admission(tmp_path, name, record, metadata, "excluded") == {}


@pytest.mark.parametrize("change", ["evidence", "metadata", "cross_role", "independence", "admission", "missing_record", "escape"])
def test_drift_and_false_declarations_fail_closed(tmp_path, change):
    name, record, metadata = fixture(tmp_path)
    path = tmp_path / REGISTRY
    document = json.loads(path.read_text())
    if change == "evidence":
        (tmp_path / "decision.json").write_text("changed")
    elif change == "metadata":
        (tmp_path / name / "metadata.json").write_text("changed")
    elif change == "cross_role":
        document["sources"]["DroneCrowd"]["reserved_role"] = "calibration"
    elif change == "independence":
        document["independence_certified"] = True
    elif change == "admission":
        document["records"][name]["predictive_admission"] = True
    elif change == "missing_record":
        document["sources"]["DroneCrowd"]["recordings"].append("missing")
    else:
        document["records"][name]["cache_path"] = "../escaped"
    write_json(path, document)
    with pytest.raises(ValueError):
        SourceReservations(tmp_path).validate(name, record, metadata, "confirmation")


@pytest.mark.parametrize("dataset", ["DUT", "DroneCrowd", "dronecrowd"])
def test_missing_registry_never_grants_protected_source_legacy_admission(tmp_path, dataset):
    with pytest.raises(ValueError, match="requires a frozen reservation"):
        validate_admission(tmp_path, "a", {"cache_path": "a"}, {"dataset": dataset}, "fit")


def test_missing_inventory_entry_cannot_add_unreviewed_protected_recording(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="absent from frozen reservation"):
        validate_admission(tmp_path, "new", {"cache_path": "new"}, {"dataset": "DUT"}, "calibration")


def test_cache_schema_cannot_hide_behind_removed_dataset_field(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="absent from frozen reservation"):
        validate_admission(tmp_path, "new", {"cache_path": "new"},
                           {"schema": "m3w_dronecrowd_recording_cache_v1"}, "fit")


def test_unrelated_portable_run_does_not_require_reserved_raw_data(tmp_path):
    name, _, _ = fixture(tmp_path)
    (tmp_path / name / "metadata.json").unlink()
    assert validate_admission(tmp_path, "synthetic", {"cache_path": "synthetic"},
                              {"dataset": "synthetic_other"}, "fit") == {}


def test_caller_cannot_strip_source_identity_from_reviewed_metadata(tmp_path):
    name, record, metadata = fixture(tmp_path)
    metadata.pop("source_xml_sha256")
    with pytest.raises(ValueError, match="metadata disagrees"):
        check_source_reservation(tmp_path, name, record, metadata, "confirmation")


def test_production_cli_refuses_renamed_reserved_geometry_before_torch(tmp_path):
    protocol = fixture_contract(tmp_path)
    fixture(tmp_path)
    path = tmp_path / "a/metadata.json"
    metadata = json.loads(path.read_text())
    registry_path = tmp_path / REGISTRY
    registry = json.loads(registry_path.read_text())
    # The existing synthetic fixture's real geometry is designated as held out.
    registry["records"]["reserved"]["source_identities"].append(metadata["artifacts"]["points.npy"]["sha256"])
    write_json(registry_path, registry)
    approve(protocol)
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    output = tmp_path / "must_not_train"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts/train_m3w_causal_forecaster.py"),
        "--protocol", str(protocol_path), "--workspace-root", str(tmp_path), "--output-dir", str(output),
        "--fit-recordings", "a", "--baseline", "constant_velocity_causal_fd", "--seed", "1"],
        cwd=repo, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    message = json.loads(result.stdout)
    assert not message["torch_training_started"]
    assert "forbids this data role" in message["reason"]
    assert not output.exists()
