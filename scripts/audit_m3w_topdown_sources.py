"""Count source-level observation and label support without scoring forecasts."""

import argparse
from collections import defaultdict
from contextlib import closing
from hashlib import file_digest
from itertools import groupby
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_unification.m3w_topdown_source_support import grid_indices, support


def identity(path):
    with path.open("rb") as stream:
        return {"archive": path.name, "bytes": path.stat().st_size,
                "sha256": file_digest(stream, "sha256").hexdigest()}


def mot_tracks(lines, frame_count):
    tracks = defaultdict(list)
    for line in lines:
        cols = line.split()
        if len(cols) != 9 or cols[6:] != ["1", "-1", "-1"]:
            raise ValueError("Unexpected AerialMPT MOT row")
        frame, agent = map(int, cols[:2])
        if not 1 <= frame <= frame_count or agent < 0:
            raise ValueError("Invalid MOT frame or agent identifier")
        tracks[agent].append(frame - 1)
    ordered = [sorted(a) for a in tracks.values()]
    if any(len(a) != len(set(a)) for a in ordered):
        raise ValueError("Duplicate MOT track/frame row")
    return ordered


def totals(counts):
    sums = defaultdict(int)
    for count in counts:
        for key, value in count.items():
            sums[key] += value
    return dict(sums)


def audit_aerial(path):
    clips = {}
    frames_by_group_role = defaultdict(lambda: defaultdict(set))
    with ZipFile(path) as archive:
        for name in archive.namelist():
            m = re.fullmatch(r"(train|test)/([^/]+)/\2_gts\.txt", name)
            if m is None:
                continue
            role, clip = m.groups()
            frames = archive.read(f"{role}/{clip}/image_list.txt").decode().split()
            if clip in clips or not frames or len(frames) != len(set(frames)):
                raise ValueError(f"Duplicate clip or frame identifier: {clip}")
            group = next((prefix for prefix in ("bauma", "pasing") if clip.startswith(prefix)), clip)
            frames_by_group_role[group][role].update(frames)
            tracks = mot_tracks(archive.read(name).decode().splitlines(), len(frames))
            clips[clip] = {"publisher_role": role, "provisional_group": group, "frames": len(frames),
                           "observations": sum(map(len, tracks)), "tracks": len(tracks),
                           "stride_1": support(tracks, len(frames), 1)}
    if len(clips) != 14:
        raise ValueError(f"Expected 14 AerialMPT clips, found {len(clips)}")
    overlap = {group: len(roles["train"] & roles["test"]) for group, roles in
               frames_by_group_role.items() if set(roles) == {"train", "test"}}
    return {"source": "DLR AerialMPT", "asset": identity(path), "clips": clips,
            "provisional_groups": len(frames_by_group_role), "published_role_group_overlap": overlap,
            "frames": sum(c["frames"] for c in clips.values()),
            "observations": sum(c["observations"] for c in clips.values()),
            "tracks": sum(c["tracks"] for c in clips.values()),
            "stride_1": totals(c["stride_1"] for c in clips.values())}


def opendd_record(conn, table):
    start, stop, rows = conn.execute(f'SELECT MIN(TIMESTAMP), MAX(TIMESTAMP), COUNT(*) FROM "{table}"').fetchone()
    ends, residual = grid_indices([start, stop])
    frame_count = int(ends[1] - ends[0] + 1)
    ids = defaultdict(list)
    counts = defaultdict(int)
    query = (f'SELECT OBJID, TIMESTAMP, CLASS FROM "{table}" '
             "WHERE CLASS IN ('Pedestrian', 'Bicycle') ORDER BY OBJID, TIMESTAMP")
    for _, group in groupby(conn.execute(query), key=lambda row: row[0]):
        track = list(group)
        classes = {r[2] for r in track}
        if len(classes) != 1:
            raise ValueError(f"Object category changes within {table}")
        category = classes.pop()
        indices, error = grid_indices([r[1] for r in track])
        residual = max(residual, error)
        ids[category].append(indices - ends[0])
        counts[category] += len(indices)
    return {"timeline_grid_positions": frame_count, "all_observations": rows,
            "max_checked_grid_residual": residual, "classes": {
                cls: {"tracks": len(ids[cls]), "observations": counts[cls], **{
                    f"stride_{stride}": support(ids[cls], frame_count, stride) for stride in (1, 12)}}
                for cls in ("Pedestrian", "Bicycle")}}


def audit_opendd(path, site):
    records = {}
    issues = {}
    with ZipFile(path) as archive, TemporaryDirectory(dir=path.parent) as tmp:
        names = [n for n in archive.namelist() if re.fullmatch(
            rf"trajectories_{site}(?:_v\d+)?\.sqlite", Path(n).name)]
        if len(names) != 1:
            raise ValueError(f"Expected one trajectory DB for {site}, got {names}")
        db = Path(tmp) / "tracks.sqlite"
        with archive.open(names[0]) as src, db.open("wb") as dest:
            shutil.copyfileobj(src, dest, 1024 * 1024)
        with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro&immutable=1", uri=True)) as conn:
            conn.execute("PRAGMA trusted_schema=OFF")
            conn.execute("PRAGMA query_only=ON")
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            tables = [t for t in tables if re.fullmatch(rf"{site}_\d+", t)]
            if not tables:
                raise ValueError(f"No recording tables in {site}")
            for table in tables:
                try:
                    records[table] = opendd_record(conn, table)
                except ValueError as e:
                    issues[table] = str(e)
    by_class = {}
    for cls in ("Pedestrian", "Bicycle"):
        values = [r["classes"][cls] for r in records.values()]
        by_class[cls] = {"tracks": sum(v["tracks"] for v in values),
                         "observations": sum(v["observations"] for v in values),
                         "recordings_with_tracks": sum(v["tracks"] > 0 for v in values), **{
            f"stride_{stride}": {**totals(v[f"stride_{stride}"] for v in values),
                                 "recordings_with_multiagent_complete": sum(
                                     v[f"stride_{stride}"]["multiagent_complete_queries"] > 0 for v in values)}
            for stride in (1, 12)}}
    return {"site": site, "asset": identity(path), "recordings": len(tables), "issues": issues,
            "verified_recordings": len(records), "classes": by_class,
            "all_observations_verified_recordings": sum(r["all_observations"] for r in records.values()),
            "max_checked_grid_residual": max((r["max_checked_grid_residual"] for r in records.values()), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aerial", type=Path)
    parser.add_argument("--opendd-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.aerial) == bool(args.opendd_dir):
        parser.error("Pass exactly one source")
    result = {"scope": "structure_only", "observation_steps": 8, "prediction_steps": 12,
              "source_roles_assigned": False, "predictions_scored": False,
              "dronecrowd_confirmation_accessed": False}
    if args.aerial:
        result["aerial"] = audit_aerial(args.aerial)
    else:
        sites = []
        for i in range(1, 8):
            print(f"Reading OpenDD site {i}/7", flush=True)
            sites.append(audit_opendd(args.opendd_dir / f"opendd_v3-rdb{i}.zip", f"rdb{i}"))
        result["opendd"] = {"candidate_export_grid_seconds": 1001 / 30000, "sites": sites}
    payload = json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != payload:
            raise SystemExit(f"Audit does not match frozen output: {args.output}")
        print(f"Exact replay matched: {args.output}")
    else:
        pending = args.output.with_suffix(".pending")
        pending.write_text(payload)
        pending.replace(args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
