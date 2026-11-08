"""Verify one ACME pixel-trajectory sample per site and audit explicit clocks."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha1, sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_external_causal_sources import timing_support


REVISION = "a0f117c11a1aee25413da95cf8612f5f8209c171"
TREE_ROOT = "BEV/BEV_annotations/trajectory-transform/trajectories"
SITE_FILES = {"bonn": 23, "cmu": 5, "gmu": 21, "keio": 8, "nus": 39, "short-term": 19, "umich": 12}
REPORT = ROOT / "outputs/publication_readiness_2026_09/acme_prefix_intake_v1"


def inventory(tree):
    by_site = defaultdict(list)
    seen = set()
    for item in tree:
        if item["type"] != "file":
            continue
        path = PurePosixPath(item["path"]).relative_to(TREE_ROOT)
        if path == PurePosixPath("cmu/intrinsics-cmu.txt"):
            continue
        site = path.parts[0]
        if (site not in SITE_FILES or path.suffix != ".toml" or
                ".." in path.parts or path in seen or item["size"] <= 0):
            raise ValueError(f"Unexpected trajectory tree entry: {path}")
        seen.add(path)
        by_site[site].append(item)
    if {site: len(files) for site, files in by_site.items()} != SITE_FILES:
        raise ValueError("Unexpected release inventory or truncated tree response")
    return {site: sorted(files, key=lambda row: row["path"]) for site, files in sorted(by_site.items())}


def verify_blob(data, item):
    digest = sha256(data).hexdigest()
    if len(data) != item["size"]:
        raise ValueError("ACME sample byte count mismatch")
    if "lfs" in item:
        if digest != item["lfs"]["oid"]:
            raise ValueError("ACME sample LFS hash mismatch")
    elif sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest() != item["oid"]:
        raise ValueError("ACME sample Git blob hash mismatch")
    return digest


def track_clocks(record):
    points = record.get("trajectories")
    if not isinstance(points, list) or not points:
        return "no_positions", (), None, 0
    n = len(points)
    if any(not isinstance(p, list) or len(p) != 2 or
           any(type(x) not in (int, float) or not math.isfinite(x) for x in p)
           for p in points):
        return "invalid_positions", (), None, n
    start = record.get("traj_start")
    ordinal = range(start, start+n) if type(start) is int and start >= 0 else None
    frames = record.get("frame_number")
    if not isinstance(frames, list) or not frames:
        return "missing_explicit_clock", (), ordinal, n
    if len(frames) != n:
        return "clock_length_mismatch", (), ordinal, n
    if (any(type(f) is not int or f < 0 for f in frames) or
            any(a >= b for a, b in zip(frames, frames[1:]))):
        return "invalid_explicit_clock", (), ordinal, n
    if ordinal is None or start != frames[0]:
        return "start_clock_disagreement", (), ordinal, n
    return "verified_explicit_clock", tuple(frames), ordinal, n


def support_summary(tracks):
    if not tracks:
        return None
    extent = [min(map(min, tracks)), max(map(max, tracks))]
    values = {}
    for stride in (1, 4):
        counts = timing_support(tracks, extent, stride)
        counts.pop("disjoint_first_ticks")
        values[str(stride)] = counts
    return {"frame_extent": extent, "grids": values}


def audit_tables(raw):
    track_counts = Counter()
    row_counts = Counter()
    explicit, ordinal = [], []
    verified_gapped_tracks = 0
    ordinal_moved_observations = 0
    for key, record in raw.items():
        if not re.fullmatch(r"human\d+", key):
            continue
        reason, frames, inferred, rows = track_clocks(record)
        track_counts[reason] += 1
        row_counts[reason] += rows
        if frames:
            explicit.append(set(frames))
            if any(b-a > 1 for a, b in zip(frames, frames[1:])):
                verified_gapped_tracks += 1
            ordinal_moved_observations += sum(a != b for a, b in zip(frames, inferred))
        if inferred is not None:
            ordinal.append(set(inferred))
    if not track_counts:
        raise ValueError("No ACME human track records")
    return {
        "tracks_by_clock_status": dict(sorted(track_counts.items())),
        "rows_by_clock_status": dict(sorted(row_counts.items())),
        "verified_gapped_tracks": verified_gapped_tracks,
        "ordinal_moved_observations_among_verified_tracks": ordinal_moved_observations,
        "strict": support_summary(explicit),
        "ordinal_diagnostic_not_validated": support_summary(ordinal),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "external_data/ACME_source_audit")
    parser.add_argument("--output", type=Path, default=REPORT / "analysis.json")
    args = parser.parse_args()
    metadata = (args.directory / "trajectory_tree.json").read_bytes()
    sites = inventory(json.loads(metadata))
    results = {}
    for site, entries in sites.items():
        entry = entries[0]
        data = (args.directory / f"{site}.toml").read_bytes()
        digest = verify_blob(data, entry)
        counts = audit_tables(tomllib.loads(data.decode("utf-8")))
        results[site] = dict(
            inventory_files=len(entries), inventory_bytes=sum(e["size"] for e in entries),
            sample=entry["path"], sample_bytes=entry["size"], sample_sha256=digest,
            publisher_blob=entry.get("lfs", entry)["oid"], **counts,
        )
        print(f"{site}: {sum(counts['rows_by_clock_status'].values()):,} position rows", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        result_source="fresh_run", dataset="acmesocnavdataset/ACME-dataset",
        revision=REVISION, inventory_sha256=sha256(metadata).hexdigest(),
        selection="lexicographically_first_pixel_trajectory_file_per_site",
        published_annotation_fps=10, timebase_verified_against_video=False,
        coordinate_input="publisher_pretransform_pixel_annotations",
        live_detection_prefix_verified=False, sites=results,
        forecast_evaluation="not_run", formal_roles_assigned=False,
        dronecrowd_confirmation_access=False,
    ), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
