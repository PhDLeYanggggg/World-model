"""Lossless, scene-query DroneCrowd cache; source admission remains separate.

The input API uses the exported annotation prefix, not sensor-time detections.
No model, role assignment, goal construction or future-complete agent filter is
introduced here. Dense recording arrays are mapped; episodes are not stored.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from src.evaluation.m3w_dronecrowd_windows import (
    Recording, WindowSpec, past_inputs, target_labels,
)


ARRAY_NAMES = ("agent_ids", "positions", "valid", "in_bounds")
SCHEMA = "m3w_dronecrowd_recording_cache_v1"


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_recording_cache(directory, record, provenance):
    """Create only; an existing cache must be verified, never overwritten."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Publish a recording only after every payload and its manifest are complete.
    # A killed writer leaves a separate pending directory, never a valid cache.
    pending = Path(tempfile.mkdtemp(prefix=f".{directory.name}.pending-", dir=directory.parent))
    artifacts = {}
    for name in ARRAY_NAMES:
        path = pending / (name + ".npy")
        np.save(path, getattr(record, name), allow_pickle=False)
        artifacts[path.name] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
    metadata = {
        "schema": SCHEMA, "sequence_id": record.sequence_id,
        "source_xml_sha256": record.source_xml_sha256,
        "coordinate_unit": "image_pixel", "metric_status": "unverified",
        "frame_unit": "raw_annotation_frame", "effective_seconds": None,
        "observation_mode": "offline_annotated_source_time_provenance_unresolved",
        "geometry": "XML_visible_nonoccluded_head_box_center",
        "data_role": "unassigned_quarantine", "physical_site_id": None,
        "source_admitted_for_training": False, "provenance": provenance,
        "agents": len(record.agent_ids), "frames": record.valid.shape[1],
        "visible_rows": int(record.valid.sum()), "artifacts": artifacts,
        "future_endpoint_input": False, "central_velocity": False,
        "test_endpoint_goals": False, "episodes_materialized": False,
    }
    (pending / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    pending.rename(directory)
    return metadata


class DroneCrowdSceneWindows:
    """Memory-mapped scene queries with explicitly separate input/label access."""

    def __init__(self, directory, spec, *, expected_xml_sha256, expected_provenance):
        spec.validate()
        self.directory, self.spec = Path(directory), spec
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if (self.metadata.get("schema") != SCHEMA
                or self.metadata.get("source_xml_sha256") != expected_xml_sha256
                or self.metadata.get("provenance") != expected_provenance
                or self.metadata.get("data_role") != "unassigned_quarantine"
                or self.metadata.get("source_admitted_for_training") is not False):
            raise ValueError("Cache schema, source, provenance or quarantine identity changed")
        if set(self.metadata["artifacts"]) != {name + ".npy" for name in ARRAY_NAMES}:
            raise ValueError("Unexpected recording arrays")
        arrays = {}
        for name in ARRAY_NAMES:
            path = self.directory / (name + ".npy")
            receipt = self.metadata["artifacts"][path.name]
            if digest_file(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
                raise ValueError("Recording array checksum mismatch")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        n, frames = self.metadata["agents"], self.metadata["frames"]
        if (arrays["agent_ids"].shape != (n,) or arrays["positions"].shape != (n, frames, 2)
                or arrays["valid"].shape != (n, frames) or arrays["in_bounds"].shape != (n, frames)
                or arrays["valid"].dtype != bool or arrays["in_bounds"].dtype != bool
                or arrays["positions"].dtype != np.float64
                or arrays["agent_ids"].dtype != np.int64
                or np.any(np.diff(arrays["agent_ids"]) <= 0)):
            raise ValueError("Invalid array shape, dtype or agent identity")
        self.record = Recording(self.metadata["sequence_id"], expected_xml_sha256, **arrays)
        # This is a clip-boundary schedule, not a future-visibility selection.
        self.query_frames = np.arange(spec.history_span, max(spec.history_span, frames-max(spec.future_offsets)))

    def __len__(self):
        return len(self.query_frames)

    def get_inputs(self, query_frame):
        return past_inputs(self.record, query_frame, self.spec)

    def get_labels(self, inputs):
        return target_labels(self.record, inputs, self.spec)

    def schema(self):
        window = asdict(self.spec)
        window["future_offsets"] = list(window["future_offsets"])
        return {"cache_schema": SCHEMA, "window": window,
                "agent_inventory": "all_visible_at_query_including_incomplete_history",
                "history_gaps": "masked_not_interpolated",
                "velocity": "backward_fd_only_valid_if_raw_interval_complete",
                "future_availability_only_in_labels": True,
                "image_features": "not_in_this_cache",
                "source_admitted_for_training": False}
