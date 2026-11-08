"""Structural support from released SQLite trajectories; no forecast outcomes."""

from collections import Counter
from contextlib import closing
import hashlib
from itertools import groupby
import re
import sqlite3

import numpy as np


CLOCKS = (1001/30000, 1/30, 1/25, 1/24, 1001/24000, 1/20, 1/10)
STRIDES = (1, 12)
HORIZONS = (10, 25, 50, 100)
TOLERANCE = .0001


def ticks(tracks):
    """Recognize a source-time lattice without filling missing ticks."""
    time = np.unique(np.concatenate(tracks))
    for step in CLOCKS:
        frame = np.rint(time/step).astype(np.int64)
        residual = float(np.abs(time-frame*step).max())
        if residual <= TOLERANCE and len(frame) == len(np.unique(frame)):
            return step, residual
    return None, None


def support(tick, stride, k=8):
    """Anchor on observed history; audit future availability separately."""
    def exists(x):
        i = np.searchsorted(tick, x)
        return tick[i.clip(max=len(tick)-1)] == x

    past = tick[exists(tick[:, None]-stride*np.arange(1, k)).all(1)]
    future_count = exists(past[:, None]+stride*np.arange(1, 13)).sum(1)
    endpoint = exists(past[:, None]+np.asarray(HORIZONS))
    full = future_count == 12
    return dict(past_eligible=len(past), complete_future12=int(full.sum()),
        partial_future12=int(((future_count > 0) & ~full).sum()),
        no_future12=int((future_count == 0).sum()),
        **{f"raw_endpoint_{h}": int(endpoint[:, i].sum()) for i, h in enumerate(HORIZONS)}), past, past[full]


def tables(db, site, file_id):
    table = "TRAJECTORIES" if site == "crosswalk" else f"TRAJECTORIES_{int(file_id):04d}"
    expected = {"TRACKS": {"TRACK_ID", "TYPE"}, table: {"TRACK_ID", "TIME", "X", "Y"}}
    names = {k for k, in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
    for name, fields in expected.items():
        cols = {col for col, in db.execute("SELECT name FROM pragma_table_info(?)", (name,))}
        if name not in names or not fields <= cols:
            raise ValueError(f"Unknown {site} schema: {name}")
    return table


def audit_db(path, site, member):
    pattern = r"crosswalk_(\d+)\.db" if site == "crosswalk" else r"intsc_data_(\d+)\.db"
    matched = re.fullmatch(pattern, member)
    if not matched:
        raise ValueError("Expected published recording-local filename")
    file_id = matched[1]
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA query_only=ON")
        table = tables(db, site, file_id)
        class_rows = db.execute("SELECT TRACK_ID, TYPE FROM TRACKS").fetchall()
        cls = dict(class_rows)
        if len(cls) != len(class_rows) or any(not isinstance(k, int) for k in cls):
            raise ValueError("Track metadata contains ambiguous IDs")
        it = db.execute(f'SELECT TRACK_ID, TIME, X, Y FROM "{table}" ORDER BY TRACK_ID,TIME')
        tracks = []
        for ident, group in groupby(it, key=lambda r:r[0]):
            if not isinstance(ident, int):
                raise ValueError("Invalid track identifier")
            values = np.asarray([r[1:] for r in group], dtype=np.float64)
            t = values[:, 0]
            if not np.isfinite(values).all() or (t < 0).any() or (np.diff(t) <= 0).any():
                raise ValueError(f"Invalid timestamps or positions: {member}")
            tracks.append((cls.get(ident, "unknown") or "unknown", t))
    if not tracks:
        raise ValueError(f"Empty trajectory table: {member}")
    stride, jitter = ticks([t for _, t in tracks])
    classes = Counter(c for c, _ in tracks)
    lengths = np.asarray([len(t) for _, t in tracks])
    result = dict(site=site, file_id=file_id, member=member, classes=dict(classes),
        rows=int(lengths.sum()), track_ids=len(tracks), metadata_ids=len(cls),
        pedestrian_tracks=classes["Pedestrian"], pedestrian_rows=sum(len(t) for c,t in tracks if c=="Pedestrian"),
        track_length_p10_p50_p90=np.quantile(lengths, [.1,.5,.9]).tolist(),
        lattice_step_source_seconds=stride, max_timestamp_lattice_residual_seconds=jitter)
    if stride is None:
        return dict(**result, support_status="no_validated_uniform_timestamp_lattice", availability=None)
    out, gap_tracks = {}, Counter()
    for s in STRIDES:
        totals = {c: Counter() for c in ("all", "pedestrian")}
        past_frames = {c: Counter() for c in totals}
        complete_frames = {c: Counter() for c in totals}
        for category, t in tracks:
            frame = np.rint(t/stride).astype(np.int64)
            if s == 1 and (np.diff(frame) > 1).any():
                gap_tracks[category] += 1
            counts, past, complete = support(frame, s)
            for c in ("all", "pedestrian") if category == "Pedestrian" else ("all",):
                totals[c].update(counts)
                past_frames[c].update(map(int,past))
                complete_frames[c].update(map(int,complete))
        out[str(s)] = {c: dict(totals[c], query_frames=len(past_frames[c]),
                               queries_with_two_past_agents=sum(v>=2 for v in past_frames[c].values()),
                               queries_with_two_complete_agents=sum(v>=2 for v in complete_frames[c].values()))
                       for c in totals}
    return dict(**result, support_status="lattice_verified", availability=out,
                tracks_with_native_gaps=sum(gap_tracks.values()), pedestrian_tracks_with_gaps=gap_tracks["Pedestrian"])


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
