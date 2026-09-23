"""Annotation-only window audit; not a training loader or source-admission grant.

Inputs depend on exported past rows only. Unrecorded VATIC control-point
dependencies remain unknown, so this does not establish sensor-time causality.
"""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from src.evaluation.m3w_dronecrowd_intake import inspect_xml


@dataclass(frozen=True)
class WindowSpec:
    name: str
    history: int
    observation_stride: int
    future_offsets: tuple[int, ...]

    def validate(self):
        if (not isinstance(self.name, str) or not self.name
                or type(self.history) is not int or self.history < 1
                or type(self.observation_stride) is not int or self.observation_stride < 1
                or not self.future_offsets
                or any(type(i) is not int or i <= 0 for i in self.future_offsets)
                or tuple(sorted(set(self.future_offsets))) != self.future_offsets):
            raise ValueError("Invalid history/future window specification")

    @property
    def history_span(self):
        return (self.history - 1) * self.observation_stride


PROBES = (
    WindowSpec("obs8_pred12_stride1", 8, 1, tuple(range(1, 13))),
    WindowSpec("obs8_pred12_stride12", 8, 12, tuple(range(12, 145, 12))),
    *(WindowSpec(f"obs8_raw_t{h}_stride1", 8, 1, (h,)) for h in (10, 25, 50, 100)),
)


@dataclass(frozen=True)
class Recording:
    sequence_id: str
    source_xml_sha256: str
    agent_ids: np.ndarray
    positions: np.ndarray
    valid: np.ndarray
    in_bounds: np.ndarray


def parse_recording(data, sequence_id):
    """Use XML alone as an audit reference, never silently prefer MAT/TXT."""
    try:
        screen = inspect_xml(data, sequence_id)
    except ET.ParseError as error:
        raise ValueError("Malformed XML; other geometry formats are not accepted") from error
    root = ET.fromstring(data.decode("utf-8-sig"))
    if (root.tag != "annotations" or any(t.tag != "track" for t in root)
            or len(list(root)) != screen["tracks"]
            or screen["labels"] != {"human": screen["tracks"]}):
        raise ValueError("Expected reviewed direct human track XML structure")
    tracks = sorted(root, key=lambda t: int(t.get("id")))
    nframes = screen["observed_frame_range"][1] + 1
    positions = np.zeros((len(tracks), nframes, 2), dtype=np.float64)
    valid = np.zeros((len(tracks), nframes), dtype=bool)
    in_bounds = np.zeros_like(valid)
    for index, track in enumerate(tracks):
        for box in track:
            if box.tag != "box":
                raise ValueError("Unexpected non-box track child")
            frame = int(box.get("frame"))
            if box.get("outside") == box.get("occluded") == "0":
                x0, y0, x1, y1 = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
                positions[index, frame] = ((x0 + x1) / 2, (y0 + y1) / 2)
                valid[index, frame] = True
                in_bounds[index, frame] = x0 >= 0 and y0 >= 0 and x1 <= 1920 and y1 <= 1080
    return Recording(sequence_id, screen["source_xml_sha256"],
                     np.array([int(t.get("id")) for t in tracks], dtype=np.int64),
                     positions, valid, in_bounds)


def past_inputs(record, query_frame, spec):
    spec.validate()
    if (type(query_frame) is not int or query_frame < spec.history_span
            or query_frame >= record.valid.shape[1]):
        raise ValueError("Requested history outside available raw frame range")
    frames = np.arange(query_frame - spec.history_span, query_frame + 1, spec.observation_stride)
    # Current visibility defines the agent inventory. Future availability never does.
    indices = np.flatnonzero(record.valid[:, query_frame])
    mask = record.valid[np.ix_(indices, frames)].copy()
    positions = record.positions[indices][:, frames].copy()
    positions[~mask] = 0
    velocities = np.zeros_like(positions)
    velocity_mask = np.zeros_like(mask)
    for i in range(1, len(frames)):
        complete = record.valid[indices, frames[i-1]:frames[i]+1].all(axis=1)
        velocity_mask[:, i] = complete
        velocities[complete, i] = (positions[complete, i] - positions[complete, i-1]) / spec.observation_stride
    return {
        "sequence_id": record.sequence_id, "query_frame": query_frame,
        "observation_frames": frames, "agent_ids": record.agent_ids[indices].copy(),
        "positions": positions, "history_mask": mask,
        "history_complete": record.valid[indices, frames[0]:query_frame+1].all(axis=1),
        "in_bounds_mask": record.in_bounds[indices][:, frames].copy() & mask,
        "velocities": velocities, "velocity_mask": velocity_mask,
        "agent_type": "human", "coordinate_unit": "image_pixel",
        "velocity_unit": "image_pixel_per_raw_annotation_frame",
        "observation_mode": "offline_annotated_source_time_provenance_unresolved",
    }


def compare_inputs(left, right):
    if set(left) != set(right):
        return False
    return all(np.array_equal(left[k], right[k]) if isinstance(left[k], np.ndarray)
               else left[k] == right[k] for k in left)


def target_labels(record, inputs, spec):
    """A separate label path, including target availability, never fed to inputs."""
    spec.validate()
    query = inputs["query_frame"]
    if query + max(spec.future_offsets) >= record.valid.shape[1]:
        raise ValueError("Requested future outside available raw frame range")
    if not compare_inputs(inputs, past_inputs(record, query, spec)):
        raise ValueError("Inputs do not match the recording prefix and history specification")
    indices = np.searchsorted(record.agent_ids, inputs["agent_ids"])
    frames = query + np.array(spec.future_offsets, dtype=np.int64)
    mask = record.valid[np.ix_(indices, frames)].copy()
    positions = record.positions[indices][:, frames].copy()
    positions[~mask] = 0
    complete = record.valid[indices, query-spec.history_span:frames[-1]+1].all(axis=1)
    return {"future_frames": frames, "future_positions": positions,
            "future_mask": mask, "complete_target": complete}


def _interval_valid(valid, starts, ends):
    missing = np.c_[np.zeros(len(valid), dtype=np.int64), np.cumsum(~valid, axis=1)]
    return missing[:, ends + 1] == missing[:, starts]


def audit_support(record, spec):
    """Count shared queries, not independent samples. No prediction errors read."""
    spec.validate()
    queries = np.arange(spec.history_span, record.valid.shape[1] - max(spec.future_offsets))
    current = record.valid[:, queries]
    history = _interval_valid(record.valid, queries-spec.history_span, queries)
    targets = _interval_valid(record.valid, queries-spec.history_span, queries+max(spec.future_offsets))
    bounded_targets = targets & _interval_valid(record.in_bounds, queries-spec.history_span,
                                                queries+max(spec.future_offsets))
    counts = targets.sum(axis=0)
    history_counts = history.sum(axis=0)
    has_targets = counts > 0
    losing = (history & ~targets).sum(axis=0)
    hist_values, hist_counts = np.unique(counts, return_counts=True)
    nonoverlap, previous_end = 0, -1
    for query in queries[has_targets]:
        if query - spec.history_span > previous_end:
            nonoverlap += 1
            previous_end = int(query + max(spec.future_offsets))
    return {
        "query_frames_considered": len(queries),
        "queries_with_any_current_agent": int(current.any(axis=0).sum()),
        "queries_with_targets": int(has_targets.sum()),
        "queries_with_at_least_2_targets": int((counts >= 2).sum()),
        "queries_with_at_least_5_targets": int((counts >= 5).sum()),
        "queries_with_at_least_10_targets": int((counts >= 10).sum()),
        "target_agent_windows": int(counts.sum()),
        "target_agent_windows_all_boxes_in_bounds": int(bounded_targets.sum()),
        "past_visible_agent_instances": int(current.sum()),
        "history_complete_agent_instances": int(history_counts.sum()),
        "history_complete_but_future_incomplete_instances": int(losing.sum()),
        "history_complete_but_future_incomplete_instances_on_queries_with_targets": int(losing[has_targets].sum()),
        "queries_losing_history_complete_neighbors_if_target_filtered": int(((losing > 0) & has_targets).sum()),
        "nonoverlapping_raw_intervals_with_targets": nonoverlap,
        "target_count_histogram": {str(int(k)): int(v) for k, v in zip(hist_values, hist_counts)},
    }


def verify_prefix_invariance(record):
    """Fresh metamorphic checks on exported rows, not original annotator controls."""
    spec = PROBES[0]
    queries = sorted({spec.history_span, (record.valid.shape[1]-1)//2, record.valid.shape[1]-2})
    checked, changed_labels, surviving_neighbors = 0, 0, 0
    for query in queries:
        if query < spec.history_span:
            continue
        original = past_inputs(record, int(query), spec)
        keep = record.valid[:, :query+1].any(axis=1)
        truncated = Recording(record.sequence_id, "not_an_input", record.agent_ids[keep],
                              record.positions[keep, :query+1].copy(), record.valid[keep, :query+1].copy(),
                              record.in_bounds[keep, :query+1].copy())
        if not compare_inputs(original, past_inputs(truncated, int(query), spec)):
            raise AssertionError("Future truncation changed exported past-only inputs")
        changed_positions = record.positions.copy()
        changed_valid, changed_bounds = record.valid.copy(), record.in_bounds.copy()
        changed_positions[:, query+1:] += 12345
        changed_valid[:, query+1:] = False
        changed_bounds[:, query+1:] = False
        changed = Recording(record.sequence_id, "not_an_input", record.agent_ids,
                            changed_positions, changed_valid, changed_bounds)
        if not compare_inputs(original, past_inputs(changed, int(query), spec)):
            raise AssertionError("Future perturbation changed exported past-only inputs")
        if query + max(spec.future_offsets) < record.valid.shape[1]:
            before = target_labels(record, original, spec)
            after = target_labels(changed, original, spec)
            changed_labels += int(not np.array_equal(before["future_mask"], after["future_mask"]))
            surviving_neighbors += int(before["complete_target"].sum())
        checked += 1
    return {"query_prefixes_checked": checked, "transformations_per_prefix": 2,
            "input_mismatches": 0, "prefixes_with_changed_label_availability": changed_labels,
            "complete_targets_hidden_in_perturbation": surviving_neighbors,
            "source_time_causality_proved": False}
