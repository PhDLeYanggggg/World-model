"""Recording-centric lazy windows, rebuilt from raw positions without legacy teachers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from src.evaluation.m3w_recording_lineage import read_track, sha256


INDEX_DTYPE = np.dtype([
    ("history_start", "<i8"), ("current_row", "<i8"), ("future_end", "<i8"),
    ("protocol", "<i2"), ("horizon_raw", "<i8"), ("future_steps", "<i4"),
])
PROTOCOL_STEPS = 0
PROTOCOL_RAW = 1
BASELINES = ("constant_position", "constant_velocity_causal_fd", "damped_velocity_005",
             "damped_velocity_010", "damped_velocity_020", "constant_acceleration_causal", "constant_turn_rate")
FEATURE_NAMES = (
    "last_speed_raw_per_frame", "history_path_length", "normalization_scale",
    "last_acceleration_norm", "heading_change", "history_curvature",
    "current_neighbor_count", "nearest_neighbor_distance_over_scale",
    "density_within_history_scale", "minimum_time_to_closest_approach_over_horizon", "maximum_closing_speed_over_scale",
    "stationary_history", "raw_frame_horizon", "observed_frame_step",
)


def validate_catalog(config: Mapping, root: Path, audit: Mapping) -> list[dict]:
    source_root = root / config["source_root"]
    expected = {row["source"]: row["raw_sha256"] for row in audit["sources"]}
    owners = {}
    hashes = {}
    seen_ids = set()
    records = []
    for recording in config["recordings"]:
        name = recording["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]+", name):
            raise ValueError("Recording ids must be safe directory names")
        if name in seen_ids:
            raise ValueError(f"Duplicate recording id: {name}")
        seen_ids.add(name)
        files = []
        for alias in [recording["canonical"], *recording["aliases"]]:
            path = source_root / alias
            relative = str(path.relative_to(root))
            if relative in owners:
                raise ValueError(f"Source assigned twice: {relative}")
            digest = sha256(path)
            if expected.get(relative) != digest:
                raise ValueError(f"Unreviewed or changed source: {relative}")
            if digest in hashes and hashes[digest] != name:
                raise ValueError("Byte-identical recordings assigned to different groups")
            owners[relative] = name
            hashes[digest] = name
            files.append({"path": relative, "sha256": digest})
        if recording["alias_evidence"] == "byte_identical" and len({f["sha256"] for f in files}) != 1:
            raise ValueError(f"Byte equality no longer holds for {name}")
        records.append({**recording, "files": files})
    if set(owners) != set(expected):
        raise ValueError("Catalog must account for every audited source, including quarantined aliases")
    return records


def validate_split_groups(catalog: list[dict], assignments: Mapping[str, str], *, physical_scene: bool = True) -> None:
    active = {r["id"] for r in catalog if r["enabled"]}
    if set(assignments) != active:
        raise ValueError("Every active recording must have exactly one split assignment")
    groups: dict[str, set[str]] = {}
    for record in catalog:
        if not record["enabled"]:
            continue
        key = record["physical_scene"] if physical_scene else record["id"]
        groups.setdefault(key, set()).add(assignments[record["id"]])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("A recording/physical scene crosses split boundaries")


def clean_points(points: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4 or not len(arr):
        raise ValueError("Expected nonempty [frame, agent, x, y] positions")
    if not np.isfinite(arr).all() or not np.equal(arr[:, :2], np.round(arr[:, :2])).all():
        raise ValueError("Nonfinite positions or nonintegral frame/agent identity")
    unique = np.unique(arr, axis=0)
    order = np.lexsort((unique[:, 0], unique[:, 1]))
    unique = unique[order]
    same_key = (np.diff(unique[:, 1]) == 0) & (np.diff(unique[:, 0]) == 0)
    if same_key.any():
        raise ValueError("Conflicting positions for one frame/agent; quarantine source")
    return unique, len(arr) - len(unique)


def track_boundaries(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(np.diff(points[:, 1]) != 0) + 1]
    ends = np.r_[starts[1:], len(points)]
    return starts, ends


def build_window_index(points: np.ndarray, history_steps: int = 8, future_steps: int = 12,
                       raw_horizons: tuple[int, ...] = (10, 25, 50, 100)) -> np.ndarray:
    if history_steps < 3 or future_steps < 1 or any(h <= 0 for h in raw_horizons):
        raise ValueError("At least three past states and positive future horizons are required")
    starts, ends = track_boundaries(points)
    rows = []
    for start, end in zip(starts, ends):
        frames = points[start:end, 0].astype(np.int64)
        for i in range(history_steps - 1, len(frames) - 1):
            step = int(frames[i] - frames[i - 1])
            if step <= 0 or np.any(np.diff(frames[i - history_steps + 1:i + 1]) != step):
                continue
            requests = [(PROTOCOL_STEPS, future_steps, step * future_steps)]
            requests += [(PROTOCOL_RAW, h // step, h) for h in raw_horizons if h % step == 0]
            for protocol, n_future, horizon in requests:
                j = i + n_future
                if j >= len(frames) or np.any(np.diff(frames[i:j + 1]) != step):
                    continue
                rows.append((start + i - history_steps + 1, start + i, start + j,
                             protocol, horizon, n_future))
    return np.asarray(rows, dtype=INDEX_DTYPE)


def causal_baselines(history_xy: np.ndarray, history_frames: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    p = history_xy[-1]
    dt = np.diff(history_frames)
    if len(dt) < 2 or np.any(dt <= 0) or np.any(offsets <= 0):
        raise ValueError("Causal finite differences require increasing past timestamps")
    velocity = np.diff(history_xy, axis=0) / dt[:, None]
    v = velocity[-1]
    acceleration = (v - velocity[-2]) / (0.5 * (dt[-1] + dt[-2]))
    t = offsets[:, None]
    predictions = [np.broadcast_to(p, (len(offsets), 2)).copy(), p + t * v]
    for decay in (0.05, 0.10, 0.20):
        rate = decay / dt[-1]
        predictions.append(p + (-np.expm1(-rate * t) / rate) * v)
    predictions.append(p + t * v + 0.5 * t * t * acceleration)
    a0, a1 = np.arctan2(velocity[-2:, 1], velocity[-2:, 0])
    turn = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    omega = turn / (0.5 * (dt[-1] + dt[-2]))
    if abs(omega) < 1e-8 or np.linalg.norm(velocity[-2]) < 1e-8:
        predictions.append(p + t * v)
    else:
        phase = a1 + omega * offsets
        displacement = np.column_stack([np.sin(phase) - np.sin(a1), np.cos(a1) - np.cos(phase)])
        predictions.append(p + np.linalg.norm(v) / omega * displacement)
    return np.asarray(predictions)


class RecordingWindows:
    """Inputs and labels have separate APIs; neither contains inherited selector outputs."""

    def __init__(self, directory: Path, max_neighbors: int = 8):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if max_neighbors < 1:
            raise ValueError("At least one neighbor slot is required")
        for name in ("points.npy", "index.npy", "frame_order.npy"):
            if sha256(self.directory / name) != self.metadata["artifacts"][name]["sha256"]:
                raise ValueError(f"Recording cache identity mismatch: {name}")
        self.points = np.load(self.directory / "points.npy", mmap_mode="r", allow_pickle=False)
        self.index = np.load(self.directory / "index.npy", mmap_mode="r", allow_pickle=False)
        self.frame_order = np.load(self.directory / "frame_order.npy", mmap_mode="r", allow_pickle=False)
        self.frame_values = self.points[self.frame_order, 0]
        self.starts, self.ends = track_boundaries(self.points)
        self.max_neighbors = max_neighbors

    def __len__(self) -> int:
        return len(self.index)

    def _track_start(self, row: int) -> int:
        return int(self.starts[np.searchsorted(self.starts, row, side="right") - 1])

    def get_inputs(self, item: int) -> dict[str, np.ndarray]:
        row = self.index[item]
        return self._inputs_for_row(row)

    def _inputs_for_row(self, row) -> dict[str, np.ndarray]:
        start, current = int(row["history_start"]), int(row["current_row"])
        history = self.points[start:current + 1]
        frame, agent, px, py = history[-1]
        xy = history[:, 2:4]
        dt = np.diff(history[:, 0])
        velocity = np.diff(xy, axis=0) / dt[:, None]
        v = velocity[-1]
        heading = np.arctan2(v[1], v[0]) if np.linalg.norm(v) > 1e-8 else 0.0
        c, s = np.cos(heading), np.sin(heading)
        rotate = np.array([[c, -s], [s, c]])
        path_length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        horizon = int(row["horizon_raw"])
        scale = max(float(path_length), float(np.linalg.norm(v) * horizon), 1e-3)
        center = np.array([px, py])
        offsets = np.arange(1, int(row["future_steps"]) + 1, dtype=np.float64) * dt[-1]
        rollouts = causal_baselines(xy, history[:, 0], offsets)
        lo, hi = np.searchsorted(self.frame_values, frame, side="left"), np.searchsorted(self.frame_values, frame, side="right")
        visible = self.frame_order[lo:hi]
        visible = visible[self.points[visible, 1] != agent]
        distance = np.linalg.norm(self.points[visible, 2:4] - center, axis=1)
        ordered = np.lexsort((self.points[visible, 1], distance))
        neighbors = visible[ordered[:self.max_neighbors]]
        k = len(history)
        neighbor_xy = np.zeros((self.max_neighbors, k, 2), dtype=np.float64)
        neighbor_mask = np.zeros((self.max_neighbors, k), dtype=bool)
        neighbor_times = np.zeros((self.max_neighbors, k), dtype=np.float64)
        all_ttc, all_closing = [], []
        for out_id, nr in enumerate(neighbors):
            begin = max(self._track_start(int(nr)), int(nr) - k + 1)
            past = self.points[begin:int(nr) + 1]
            n = len(past)
            assert past[:, 0].max() <= frame
            neighbor_xy[out_id, -n:] = (past[:, 2:4] - center) @ rotate / scale
            neighbor_times[out_id, -n:] = past[:, 0] - frame
            neighbor_mask[out_id, -n:] = True
            nv = (past[-1, 2:4] - past[-2, 2:4]) / (past[-1, 0] - past[-2, 0]) if n >= 2 else np.zeros(2)
            rel = past[-1, 2:4] - center
            rel_v = nv - v
            vv = float(rel_v @ rel_v)
            dot = float(rel @ rel_v)
            ttc = -dot / vv if vv > 1e-12 and dot < 0 else 10.0 * horizon
            all_ttc.append(min(ttc / horizon, 10.0))
            all_closing.append(max(0.0, -dot / max(float(np.linalg.norm(rel)), 1e-8)) / scale)
        headings = np.arctan2(velocity[:, 1], velocity[:, 0])
        angles = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
        acceleration = (velocity[-1] - velocity[-2]) / (0.5 * (dt[-1] + dt[-2]))
        features = np.array([
            np.linalg.norm(v), path_length, scale, np.linalg.norm(acceleration), angles[-1],
            np.abs(angles).sum() / max(path_length, 1e-3), len(visible),
            distance.min() / scale if len(distance) else 100.0,
            (distance <= scale).sum(), min(all_ttc, default=10.0), max(all_closing, default=0.0),
            float(path_length < 1e-6), horizon, dt[-1],
        ], dtype=np.float32)
        return {
            "history_xy": ((xy - center) @ rotate / scale).astype(np.float32),
            "history_frame_offsets": (history[:, 0] - frame).astype(np.float32),
            "history_velocity": (velocity @ rotate / scale).astype(np.float32),
            "history_mask": np.ones(k, dtype=bool),
            "neighbor_xy": neighbor_xy.astype(np.float32), "neighbor_mask": neighbor_mask,
            "neighbor_frame_offsets": neighbor_times.astype(np.float32),
            "baseline_rollouts": ((rollouts - center) @ rotate / scale).astype(np.float32),
            "prediction_frame_offsets": offsets.astype(np.float32), "causal_features": features,
        }

    def get_scene_inputs(self, frame_id: int, horizon_raw: int, history_steps: int = 8) -> dict:
        """Select targets by observable past support, never future track completeness."""
        if horizon_raw <= 0 or history_steps < 3:
            raise ValueError("Positive raw horizon and at least three past observations required")
        lo, hi = np.searchsorted(self.frame_values, frame_id, side="left"), np.searchsorted(self.frame_values, frame_id, side="right")
        agents, excluded = [], []
        for current in self.frame_order[lo:hi]:
            current = int(current)
            agent_id = int(self.points[current, 1])
            begin = current - history_steps + 1
            if begin < self._track_start(current):
                excluded.append({"agent_id": agent_id, "reason": "insufficient_past"})
                continue
            deltas = np.diff(self.points[begin:current + 1, 0])
            step = int(deltas[-1])
            if step <= 0 or np.any(deltas != step) or horizon_raw % step:
                excluded.append({"agent_id": agent_id, "reason": "past_grid_incompatible_with_request"})
                continue
            inputs = self._inputs_for_row({"history_start": begin, "current_row": current,
                                           "horizon_raw": horizon_raw, "future_steps": horizon_raw // step})
            agents.append({"agent_id": agent_id, "inputs": inputs})
        return {"recording_id": self.metadata["id"], "physical_scene": self.metadata["physical_scene"],
                "frame_id": int(frame_id), "horizon_raw": int(horizon_raw), "agents": agents,
                "excluded_past_support": excluded, "data_role": "diagnostic_only"}

    def get_scene_labels(self, scene_inputs: Mapping) -> list[dict]:
        """Missing future observations are loss masks, not an inference-time agent filter."""
        if scene_inputs["recording_id"] != self.metadata["id"]:
            raise ValueError("Scene labels requested from a different recording")
        ids = self.points[self.starts, 1].astype(np.int64)
        labels = []
        for agent in scene_inputs["agents"]:
            position = int(np.searchsorted(ids, agent["agent_id"]))
            if position >= len(ids) or ids[position] != agent["agent_id"]:
                raise ValueError("Unknown scene agent")
            track = self.points[self.starts[position]:self.ends[position]]
            frames = scene_inputs["frame_id"] + agent["inputs"]["prediction_frame_offsets"].astype(np.int64)
            if np.any(frames <= scene_inputs["frame_id"]):
                raise ValueError("Loss targets must follow the current frame")
            target_xy = np.zeros((len(frames), 2), dtype=np.float64)
            mask = np.zeros(len(frames), dtype=bool)
            for k, frame in enumerate(frames):
                i = int(np.searchsorted(track[:, 0], frame))
                if i < len(track) and track[i, 0] == frame and np.isfinite(track[i, 2:4]).all():
                    target_xy[k] = track[i, 2:4]
                    mask[k] = True
            labels.append({"agent_id": agent["agent_id"], "future_frame_ids": frames,
                           "future_xy_dataset_local": target_xy, "future_label_mask": mask})
        return labels

    def get_labels(self, item: int) -> dict[str, np.ndarray]:
        row = self.index[item]
        cur, end = int(row["current_row"]), int(row["future_end"])
        return {"future_xy_dataset_local": self.points[cur + 1:end + 1, 2:4].copy(),
                "future_frame_ids": self.points[cur + 1:end + 1, 0].copy()}

    def identity(self, item: int) -> dict:
        row = self.index[item]
        current = self.points[int(row["current_row"])]
        return {"recording_id": self.metadata["id"], "physical_scene": self.metadata["physical_scene"],
                "agent_id": int(current[1]), "frame_id": int(current[0]),
                "protocol": "observation_steps" if row["protocol"] == PROTOCOL_STEPS else "raw_exact",
                "horizon_raw": int(row["horizon_raw"]), "data_role": "diagnostic_only"}


def write_recording(directory: Path, points: np.ndarray, recording: Mapping,
                    history_steps: int = 8, future_steps: int = 12) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    points, duplicate_rows = clean_points(points)
    index = build_window_index(points, history_steps, future_steps)
    order = np.lexsort((points[:, 1], points[:, 0]))
    for name, data in (("points", points), ("index", index), ("frame_order", order)):
        np.save(directory / f"{name}.npy", data, allow_pickle=False)
    starts, ends = track_boundaries(points)
    lengths = ends - starts
    raw_counts = {str(h): int(((index["protocol"] == PROTOCOL_RAW) & (index["horizon_raw"] == h)).sum()) for h in (10, 25, 50, 100)}
    steps = index[index["protocol"] == PROTOCOL_STEPS]
    metadata = {
        **recording, "points": len(points), "agents": len(starts), "exact_duplicate_point_rows_removed": duplicate_rows,
        "history_steps": history_steps, "future_observation_steps": future_steps,
        "observation_step_windows": len(steps), "raw_exact_windows": raw_counts,
        "observation_step_actual_raw_horizons": {str(int(v)): int((steps["horizon_raw"] == v).sum()) for v in np.unique(steps["horizon_raw"])},
        "track_length_quantiles": np.quantile(lengths, [0, .25, .5, .75, 1]).tolist(),
        "schema": {"causal_features": FEATURE_NAMES, "baseline_names": BASELINES,
                   "future_inputs": False, "legacy_teacher_inputs": False, "goals_used": False,
                   "central_velocity_used": False, "neighbor_cutoff": "current frame or earlier"},
        "coordinate_claim": "dataset_local_unverified", "time_claim": "raw_frames_or_steps_not_seconds",
        "role": "diagnostic_only", "official_benchmark": False,
        "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in directory.glob("*.npy")},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_catalog(root: Path, config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    audit_path = root / config["source_identity_audit"]
    audit = json.loads(audit_path.read_text())
    catalog = validate_catalog(config, root, audit)
    outputs = []
    for recording in catalog:
        if not recording["enabled"]:
            continue
        raw_path = root / config["source_root"] / recording["canonical"]
        points, skipped = read_track(raw_path)
        if skipped:
            raise ValueError(f"Canonical source has missing labels or parse errors: {raw_path}")
        result = write_recording(output_dir / recording["id"], points, recording)
        outputs.append(result)
    manifest = {
        "result_source": "fresh_run", "scope": "raw_position_rebuild_no_training",
        "config_sha256": sha256(config_path), "identity_audit_sha256": sha256(audit_path),
        "catalog": catalog, "recordings": outputs, "cache_directory": str(output_dir),
        "canonical_recordings_used": len(outputs),
        "physical_scene_groups": sorted({r["physical_scene"] for r in outputs}),
        "legacy_npz_or_teacher_read": False, "official_split_selected": False,
        "independent_confirmation_holdout": False, "supervised_training_run": False,
        "submission_ready": False, "stage5c_executed": False, "smc_enabled": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
