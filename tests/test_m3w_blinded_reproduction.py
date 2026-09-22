import json
import subprocess
import sys
import zipfile

import pytest

from scripts.build_m3w_blinded_reproduction import (
    ROOT, archive_bytes, check_equivalence, clean_run, extract, make_files, scan_files,
)
from scripts.build_m3w_evidence_manuscript_v2 import build, load_sources
from src.evaluation.m3w_portable_evidence import package_inputs, reconstruct


@pytest.fixture(scope="module")
def docs():
    return load_sources(ROOT)


@pytest.fixture(scope="module")
def files(docs):
    return make_files(docs)[0]


def materialize(tmp_path, files):
    for name, data in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_all_numeric_fields_match_frozen_paper(docs):
    result = reconstruct(extract(docs))
    assert check_equivalence(result, build(docs)) == 1062
    assert len(result["rows"]) == 8
    assert len(result["site_metrics"]) == 32
    assert len(result["site_seed_metrics"]) == 96
    assert len(result["contrasts"]) == 14
    assert len(result["joint_support"]) == 3
    assert not result["submission_ready"] and not result["new_training"]


def test_excerpt_is_allowlisted_not_whole_source_metadata(docs, files):
    evidence = json.loads(files["evidence.json"])
    assert "identity" not in evidence
    assert "source_bindings" not in files["evidence.json"].decode()
    assert "checkpoint" not in files["evidence.json"].decode()
    assert not evidence["independent_confirmation"]
    scan_files(files)


@pytest.mark.parametrize("key", ["independent_confirmation", "risk_calibrated", "deployment", "new_training"])
def test_cannot_relabel_results_as_completion(docs, key):
    evidence = extract(docs)
    evidence[key] = True
    with pytest.raises(ValueError, match="completion claim"):
        reconstruct(evidence)


def test_missing_seed_is_not_silently_averaged(docs):
    evidence = extract(docs)
    del evidence["policies"][0]["summary"]["seeds"]["43"]
    with pytest.raises(ValueError, match="seed population"):
        reconstruct(evidence)


def test_modified_scene_arithmetic_is_refused(docs):
    evidence = extract(docs)
    evidence["policies"][0]["summary"]["ADE"]["by_scene"]["coupa"]["model_error"] += 1
    with pytest.raises(ValueError, match="scene gain"):
        reconstruct(evidence)


def test_nan_is_refused(docs):
    evidence = extract(docs)
    evidence["policies"][0]["summary"]["ADE"]["by_scene"]["coupa"]["model_error"] = float("nan")
    with pytest.raises(ValueError, match="scene gain"):
        reconstruct(evidence)


def test_joint_support_is_not_predictive_success(docs):
    evidence = extract(docs)
    evidence["joint_support"][0]["predictive_lift_status"] = "pass"
    with pytest.raises(ValueError, match="predictive evidence"):
        reconstruct(evidence)


def test_archive_bytes_are_deterministic(files):
    assert archive_bytes(files) == archive_bytes(dict(reversed(list(files.items()))))


@pytest.mark.parametrize("name,data", [
    ("../secret.json", b"{}"), ("/secret.json", b"{}"), ("weights.pt", b"{}"),
    ("metadata.json", b'{"path":"/Users/researcher/data"}'),
    ("metadata.json", b'{"repo":"https://github.com/PhDLeYanggggg/World-model"}'),
])
def test_unsafe_or_identifying_payload_is_refused(name, data):
    with pytest.raises(ValueError):
        scan_files({name: data})


def test_missing_file_is_refused(tmp_path, files):
    materialize(tmp_path, files)
    (tmp_path / "evidence.json").unlink()
    with pytest.raises(FileNotFoundError):
        package_inputs(tmp_path)


def test_changed_payload_is_refused(tmp_path, files):
    materialize(tmp_path, files)
    (tmp_path / "evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed package file"):
        package_inputs(tmp_path)


def test_symlink_evidence_is_refused(tmp_path, files):
    materialize(tmp_path, files)
    source = tmp_path / "evidence.json"
    source.rename(tmp_path / "moved.json")
    source.symlink_to(tmp_path / "moved.json")
    with pytest.raises(ValueError, match="Symlink"):
        package_inputs(tmp_path)


def test_clean_extraction_reconstructs_every_output(tmp_path, files):
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(archive_bytes(files))
    receipt = clean_run(archive)
    assert receipt["exit_code"] == 0
    assert len(receipt["output_hashes"]) == 7
    assert receipt["isolated_interpreter"] and receipt["network_and_subprocess_guard"]
    assert not receipt["receipt"][0]["new_training"]
    assert not receipt["receipt"][0]["submission_ready"]


@pytest.mark.parametrize("injection", [
    'open("/etc/hosts").read()\n',
    'import socket; socket.socket()\n',
    'import subprocess; subprocess.run(["true"])\n',
])
def test_clean_execution_rejects_external_access(tmp_path, files, injection):
    changed = dict(files)
    changed["reproduce.py"] = injection.encode()+changed["reproduce.py"]
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(archive_bytes(changed))
    with pytest.raises(RuntimeError, match="forbidden|outside extracted"):
        clean_run(archive)


def test_repeat_does_not_overwrite_previous_outputs(tmp_path, files):
    materialize(tmp_path, files)
    command = [sys.executable, "-I", str(tmp_path / "reproduce.py"), "--verify",
               "--output", str(tmp_path / "result")]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    before = (tmp_path / "result/results.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "Refusing overwrite" in second.stderr
    assert (tmp_path / "result/results.json").read_bytes() == before


def test_archive_members_have_no_personal_timestamps(tmp_path, files):
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(archive_bytes(files))
    with zipfile.ZipFile(archive) as content:
        assert len(content.namelist()) == 13
        assert all(x.date_time == (1980, 1, 1, 0, 0, 0) for x in content.infolist())
        assert all(x.create_system == 3 for x in content.infolist())
