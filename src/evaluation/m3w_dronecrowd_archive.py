"""Read-only release audit. Structural windows are not admitted model samples."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import stat
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.io import loadmat

from src.evaluation.m3w_dronecrowd_intake import inspect_xml, sha


ARCHIVE_FILE_ID = "1NeUK0AqgACG1iPiu4rjz3J68Pnsaj5LN"
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024**3
WINDOWS = ((8, 12, 1), (8, 12, 12), (8, 10, 1), (8, 25, 1),
           (8, 50, 1), (8, 100, 1))


def archive_inventory(archive):
    entries, seen = archive.infolist(), set()
    if not 1 <= len(entries) <= 1024:
        raise ValueError("Archive member count outside bounded audit")
    total = 0
    for entry in entries:
        name, path = entry.filename, PurePosixPath(entry.filename)
        if (name in seen or path.is_absolute() or ".." in path.parts
                or "\\" in name or str(path) != name.rstrip("/")):
            raise ValueError("Duplicate or unsafe archive path")
        seen.add(name)
        kind = stat.S_IFMT(entry.external_attr >> 16)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or entry.flag_bits & 1:
            raise ValueError("Special/link/encrypted archive member refused")
        if entry.is_dir():
            if name != "annotations/" or entry.file_size:
                raise ValueError("Unexpected archive directory")
            continue
        if not re.fullmatch(r"annotations/[0-9]{5}(?:\.(?:xml|mat)|_clean\.txt)", name):
            raise ValueError("Unexpected non-annotation archive member")
        if not 1 <= int(path.name[:5]) <= 112:
            raise ValueError("Unreviewed release sequence")
        if not 0 < entry.file_size <= MAX_MEMBER_BYTES or entry.compress_size <= 0:
            raise ValueError("Invalid or oversized member")
        if entry.file_size / entry.compress_size > 1000:
            raise ValueError("Excessive compression ratio")
        total += entry.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Archive exceeds decompression budget")
    return entries


def contiguous_lengths(frames):
    frames = np.asarray(frames, dtype=np.int64)
    if not len(frames):
        return []
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Frame sequence must be strictly increasing")
    boundaries = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, len(frames)]
    return np.diff(boundaries).tolist()


def window_counts(lengths):
    # A sample requires a completely visible raw-frame run, even at stride 12.
    return {f"obs{k}_pred{h}_stride{s}": sum(max(0, n - (k + h - 1) * s)
                                            for n in lengths)
            for k, h, s in WINDOWS}


def ordered_rows(rows):
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError("Expected finite six-column annotation rows")
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def inspect_recording(data, sequence_id):
    result = inspect_xml(data, sequence_id)
    root = ET.fromstring(data.decode("utf-8-sig"))
    visible_rows, track_sizes, run_sizes = [], [], []
    frame_population = np.zeros(300, dtype=np.int64)
    all_frame_steps, track_attributes, box_values = Counter(), Counter(), defaultdict(Counter)
    fingerprints = []
    outside_image_boxes, visible_outside_image_boxes = 0, 0
    for track in root.findall("track"):
        track_attributes.update(track.attrib.keys())
        agent, frames, track_rows = int(track.get("id")), [], []
        for box in track:
            frame = int(box.get("frame"))
            coords = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            for key in ("keyframe", "generated", "interpolated", "outside", "occluded"):
                if key in box.attrib:
                    box_values[key][box.get(key)] += 1
            beyond_image = coords[0] < 0 or coords[1] < 0 or coords[2] > 1920 or coords[3] > 1080
            if beyond_image:
                outside_image_boxes += 1
            if box.get("outside") == box.get("occluded") == "0":
                visible_outside_image_boxes += beyond_image
                frames.append(frame)
                track_rows.append([frame, agent, *coords])
                frame_population[frame] += 1
        all_frames = [int(b.get("frame")) for b in track]
        all_frame_steps.update(np.diff(all_frames).tolist())
        track_sizes.append(len(frames))
        run_sizes.extend(contiguous_lengths(frames))
        visible_rows.extend(track_rows)
        if len(track_rows) >= 20:
            canonical = np.asarray(track_rows, dtype="<f8")[:, [0, 2, 3, 4, 5]]
            fingerprints.append((sha(canonical.tobytes()), len(track_rows), agent))
    rows = np.asarray(visible_rows, dtype=np.float64).reshape(-1, 6)
    result.update({
        "root_tag": root.tag, "root_attributes": dict(root.attrib),
        "track_attribute_counts": dict(sorted(track_attributes.items())),
        "box_flag_values": {k: dict(sorted(v.items())) for k, v in sorted(box_values.items())},
        "raw_frame_step_counts": {str(k): v for k, v in sorted(all_frame_steps.items())},
        "boxes_outside_nominal_1920x1080": outside_image_boxes,
        "visible_boxes_outside_nominal_1920x1080": visible_outside_image_boxes,
        "structural_windows": window_counts(run_sizes),
        "history_only_windows": {str(k): sum(max(0, n - k + 1) for n in run_sizes)
                                 for k in (8, 16, 32, 64)},
        "visible_track_length_quantiles": np.quantile(track_sizes, [0, .25, .5, .75, 1]).tolist(),
        "visible_frame_agent_quantiles": np.quantile(frame_population, [0, .25, .5, .75, 1]).tolist(),
        "tracks_without_visible_rows": track_sizes.count(0),
    })
    return result, rows, fingerprints


def compare_mat(data, xml_rows):
    arrays = loadmat(io.BytesIO(data), verify_compressed_data_integrity=True)
    if "anno" not in arrays:
        raise ValueError("MAT lacks anno")
    rows = np.asarray(arrays["anno"])
    if rows.dtype.kind not in "uif":
        raise ValueError("MAT anno must be numeric data, not objects")
    expected, actual = ordered_rows(xml_rows), ordered_rows(rows)
    shape_matches = actual.shape == expected.shape
    return {"rows": len(actual), "shape_matches_visible_xml": shape_matches,
            "exact_rows_match_visible_xml": bool(shape_matches and np.array_equal(actual, expected)),
            "mismatched_scalar_values": int(np.count_nonzero(actual != expected)) if shape_matches else None,
            "identity_columns_match": bool(shape_matches and np.array_equal(actual[:, :2], expected[:, :2])),
            "mismatched_rows": int(np.any(actual != expected, axis=1).sum()) if shape_matches else None,
            "mismatched_values_by_column": np.count_nonzero(actual != expected, axis=0).tolist() if shape_matches else None,
            "max_absolute_coordinate_difference": float(np.max(np.abs(actual[:, 2:] - expected[:, 2:]), initial=0)) if shape_matches else None,
            "variables": sorted(k for k in arrays if not k.startswith("__")),
            "anno_dtype": str(rows.dtype), "label": np.asarray(arrays.get("label", [])).tolist()}


def compare_clean_text(data, xml_rows):
    rows = np.loadtxt(io.BytesIO(data), delimiter=",", ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 10 or not np.isfinite(rows).all():
        raise ValueError("Clean text is not finite ten-column data")
    # Check the hypothesis against every value; do not silently use it as a converter.
    candidate = rows[:, :6].copy()
    candidate[:, 0] -= 1
    candidate[:, 4:6] += candidate[:, 2:4]
    expected, actual = ordered_rows(xml_rows), ordered_rows(candidate)
    shape_matches = actual.shape == expected.shape
    match = shape_matches and np.array_equal(actual, expected)
    return {"rows": len(rows), "frame_plus_one_xywh_hypothesis_exact_match": bool(match),
            "identity_columns_match": bool(shape_matches and np.array_equal(actual[:, :2], expected[:, :2])),
            "mismatched_rows": int(np.any(actual != expected, axis=1).sum()) if shape_matches else None,
            "mismatched_values_by_column": np.count_nonzero(actual != expected, axis=0).tolist() if shape_matches else None,
            "max_absolute_coordinate_difference": float(np.max(np.abs(actual[:, 2:] - expected[:, 2:]), initial=0)) if shape_matches else None,
            "tail_column_unique_values": [np.unique(rows[:, i]).tolist() for i in range(6, 10)]}


def audit_archive(path, train_ids, test_ids, progress=None):
    path = Path(path)
    with path.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if set(train_ids) & set(test_ids):
        raise ValueError("Release split overlap")
    expected_ids = set(train_ids) | set(test_ids)
    sequences, members, failures, duplicate_tracks = [], [], [], defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        entries = archive_inventory(archive)
        names = {e.filename for e in entries}
        xml_ids = {PurePosixPath(n).stem for n in names if n.endswith(".xml")}
        if xml_ids != expected_ids:
            raise ValueError("XML coverage disagrees with frozen release lists")
        if {PurePosixPath(n).stem for n in names if n.endswith(".mat")} != expected_ids:
            raise ValueError("MAT coverage disagrees with frozen release lists")
        for sequence_id in sorted(xml_ids):
            contents = {}
            for suffix in (".xml", ".mat", "_clean.txt"):
                name = f"annotations/{sequence_id}{suffix}"
                if name not in names:
                    continue
                raw = archive.read(name)  # zipfile validates CRC; nothing is extracted or executed.
                entry = archive.getinfo(name)
                if len(raw) != entry.file_size:
                    raise ValueError("Decompressed size mismatch")
                members.append({"name": name, "bytes": len(raw), "compressed_bytes": entry.compress_size,
                                "sha256": sha(raw), "crc32": f"{entry.CRC:08x}"})
                contents[suffix] = raw
            try:
                result, rows, fingerprints = inspect_recording(contents[".xml"], sequence_id)
                result["release_split"] = "train" if sequence_id in train_ids else "test"
                result["mat_check"] = compare_mat(contents[".mat"], rows)
                if "_clean.txt" in contents:
                    result["clean_text_check"] = compare_clean_text(contents["_clean.txt"], rows)
                for digest, count, agent in fingerprints:
                    duplicate_tracks[digest].append({"sequence": sequence_id, "agent_id": agent,
                                                     "visible_rows": count})
                sequences.append(result)
            except (ValueError, KeyError, ET.ParseError) as error:
                failures.append({"sequence_id": sequence_id, "error": str(error)})
            if progress:
                progress(sequence_id, len(sequences), len(failures))
    aggregate = {}
    for split in ("train", "test", "all"):
        selected = [r for r in sequences if split == "all" or r["release_split"] == split]
        aggregate[split] = {
            "sequences": len(selected),
            **{k: sum(r[k] for r in selected) for k in (
                "tracks", "raw_boxes", "converter_retained_boxes", "outside_boxes", "occluded_boxes",
                "all_box_discontinuous_edges", "retained_box_discontinuous_edges",
                "boxes_outside_nominal_1920x1080", "visible_boxes_outside_nominal_1920x1080",
                "tracks_without_visible_rows")},
            "structural_windows": {k: sum(r["structural_windows"][k] for r in selected)
                                   for k in window_counts([])},
            "history_only_windows": {str(k): sum(r["history_only_windows"][str(k)] for r in selected)
                                     for k in (8, 16, 32, 64)},
        }
    repeated = [v for v in duplicate_tracks.values() if len({x["sequence"] for x in v}) > 1]
    identical_files = defaultdict(list)
    for member in members:
        if member["name"].endswith(".xml"):
            identical_files[member["sha256"]].append(member["name"])
    return {
        "schema_version": 1, "result_source": "fresh_run",
        "scope": "authorized_annotation_only_structural_audit_no_predictive_readout",
        "archive_sha256": archive_hash, "archive_bytes": path.stat().st_size,
        "member_count": len(entries), "uncompressed_bytes": sum(e.file_size for e in entries),
        "members": members, "aggregate": aggregate, "sequences": sequences,
        "structural_failures": failures,
        "clean_text_matches_official_test_ids": {n.split("/")[1][:5] for n in names if n.endswith("_clean.txt")} == set(test_ids),
        "identical_xml_groups": [v for v in identical_files.values() if len(v) > 1],
        "exact_cross_sequence_track_groups_min20_rows": repeated,
        "duplicate_screen_does_not_establish_physical_site_independence": True,
        "site_map": None, "camera_motion": "not_run_no_images_or_videos_authorized",
        "annotation_source_time_provenance": "unresolved_not_inferred_from_dense_or_linear_motion",
        "coordinate_unit": "image_pixel", "metric_status": "unverified_no_homography_or_scale",
        "effective_seconds": "unknown_no_verified_video_to_annotation_time_mapping",
        "labels": "released_offline_annotations_not_sensor_causal_observations",
        "admitted_recordings": 0, "scientific_roles_assigned": False,
        "model_feature_rows_exported": 0, "training": "not_run", "forecast_evaluation": "not_run",
        "calibration": "not_run", "independent_confirmation": "not_run",
        "third_party_code_executed": False, "stage5c_executed": False, "smc_enabled": False,
    }
