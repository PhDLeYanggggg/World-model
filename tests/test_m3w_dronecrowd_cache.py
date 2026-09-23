from dataclasses import replace
import json

import numpy as np
import pytest

from src.data_unification.m3w_dronecrowd_cache import DroneCrowdSceneWindows, write_recording_cache
from src.evaluation.m3w_dronecrowd_windows import Recording, WindowSpec, compare_inputs, past_inputs


SPEC = WindowSpec("obs8_pred12", 8, 1, tuple(range(1, 13)))
PROVENANCE = {"archive_sha256": "fixture", "converter_sha256": "fixture-code"}


def record():
    positions = np.zeros((3, 30, 2), dtype=np.float64)
    positions[:, :, 0] = np.arange(30)
    positions[:, :, 1] = np.arange(3)[:, None]
    valid = np.ones((3, 30), bool)
    valid[1, 10:] = False
    valid[2, :7] = False
    return Recording("00001", "xml-fixture", np.arange(3, dtype=np.int64), positions, valid, valid.copy())


def open_cache(path, item=None, spec=SPEC):
    write_recording_cache(path, item or record(), PROVENANCE)
    return DroneCrowdSceneWindows(path, spec, expected_xml_sha256="xml-fixture", expected_provenance=PROVENANCE)


def test_lossless_mmap_and_separate_labels(tmp_path):
    reader = open_cache(tmp_path / "cache")
    assert isinstance(reader.record.positions, np.memmap)
    assert not reader.record.positions.flags.writeable
    assert reader.metadata["data_role"] == "unassigned_quarantine"
    inputs = reader.get_inputs(7)
    assert compare_inputs(inputs, past_inputs(record(), 7, SPEC))
    assert inputs["agent_ids"].tolist() == [0, 1, 2]
    assert reader.get_labels(inputs)["complete_target"].tolist() == [True, False, False]
    assert not any("future" in key or "target" in key for key in inputs)


def test_future_changes_do_not_change_inputs_or_neighbor_inventory(tmp_path):
    item = record()
    pos, valid, bounds = item.positions.copy(), item.valid.copy(), item.in_bounds.copy()
    pos[:, 8:] += 1000
    valid[:, 8:] = False
    bounds[:, 8:] = False
    original = open_cache(tmp_path / "original", item)
    changed = open_cache(tmp_path / "changed", replace(item, positions=pos, valid=valid, in_bounds=bounds))
    assert compare_inputs(original.get_inputs(7), changed.get_inputs(7))
    np.testing.assert_array_equal(original.query_frames, changed.query_frames)
    assert not changed.get_labels(changed.get_inputs(7))["future_mask"].any()


def test_inference_tail_works_without_future_labels(tmp_path):
    reader = open_cache(tmp_path / "cache")
    assert len(reader) == 11
    inputs = reader.get_inputs(29)
    assert inputs["query_frame"] == 29
    with pytest.raises(ValueError, match="future"):
        reader.get_labels(inputs)


def test_stride_does_not_restore_missing_velocity(tmp_path):
    item = record()
    item.valid[0, 3] = False
    reader = open_cache(tmp_path / "cache", item, WindowSpec("stride2", 8, 2, (2,)))
    inputs = reader.get_inputs(14)
    assert inputs["history_mask"][0].all()
    assert not inputs["history_complete"][0]
    assert not inputs["velocity_mask"][0, 2]


def test_existing_cache_cannot_be_overwritten(tmp_path):
    path = tmp_path / "cache"
    open_cache(path)
    with pytest.raises(FileExistsError):
        write_recording_cache(path, record(), PROVENANCE)


def test_schema_survives_exact_json_round_trip(tmp_path):
    reader = open_cache(tmp_path / "cache")
    assert json.loads(json.dumps(reader.schema())) == reader.schema()


def test_incomplete_write_does_not_publish_recording(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated storage interruption")
    monkeypatch.setattr(np, "save", fail)
    path = tmp_path / "cache"
    with pytest.raises(OSError):
        write_recording_cache(path, record(), PROVENANCE)
    assert not path.exists()


@pytest.mark.parametrize("change", ["array", "source", "role", "provenance"])
def test_tampering_refused(tmp_path, change):
    path = tmp_path / "cache"
    open_cache(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if change == "array":
        with (path / "positions.npy").open("ab") as stream:
            stream.write(b"changed")
    elif change == "source":
        metadata["source_xml_sha256"] = "changed"
    elif change == "role":
        metadata["data_role"] = "confirmation"
    else:
        metadata["provenance"]["converter_sha256"] = "changed"
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        DroneCrowdSceneWindows(path, SPEC, expected_xml_sha256="xml-fixture", expected_provenance=PROVENANCE)
