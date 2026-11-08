"""Measure released pedestrian timing support, without forecasting or assigning roles."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import file_digest
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/m3w_external_causal_sources_v1.json"
REPORT = ROOT / "outputs/publication_readiness_2026_09/independent_source_qualification_v1"


def read_tracks(stream, sind):
    reader = csv.DictReader(stream)
    required = {"track_id", "frame_id", "timestamp_ms", "agent_type"} if sind else {"id", "time"}
    if not required.issubset(reader.fieldnames or ()) or (
        not sind and set(reader.fieldnames) != required
    ):
        raise ValueError("Unexpected timing schema")
    tracks = defaultdict(set)
    max_residual_seconds = 0.0
    for row in reader:
        if sind:
            if row["agent_type"] != "pedestrian":
                raise ValueError("Unexpected agent type in SinD pedestrian sample")
            key, tick = row["track_id"], int(row["frame_id"])
            time = float(row["timestamp_ms"]) / 1000
            residual = abs(time - tick / 9.99)
        else:
            key, time = row["id"], float(row["time"])
            if not math.isfinite(time):
                raise ValueError("Nonfinite time")
            tick = round(10 * time)
            residual = abs(time - tick / 10)
        if not key or tick < 0 or not math.isfinite(residual) or residual > 1e-6:
            raise ValueError("Missing ID or off-grid timestamp")
        if tick in tracks[key]:
            raise ValueError("Duplicate track/frame; support is ambiguous")
        tracks[key].add(tick)
        max_residual_seconds = max(max_residual_seconds, residual)
    if not tracks:
        raise ValueError("No pedestrian tracks")
    return tracks, max_residual_seconds


def timing_support(tracks, extent, stride):
    """Select disjoint query intervals from past support, then score label presence."""
    start, stop = extent
    history_offsets = tuple(i * stride for i in range(1, 8))
    future_offsets = tuple(i * stride for i in range(8, 20))
    eligible, complete, partial = Counter(), Counter(), Counter()
    for ticks in tracks:
        for first in ticks:
            if first < start or first + future_offsets[-1] > stop:
                continue
            if not all(first + offset in ticks for offset in history_offsets):
                continue
            eligible[first] += 1
            n = sum(first + offset in ticks for offset in future_offsets)
            if n == 12:
                complete[first] += 1
            elif n:
                partial[first] += 1

    def labels(queries):
        n = sum(eligible[t] for t in queries)
        c = sum(complete[t] for t in queries)
        p = sum(partial[t] for t in queries)
        return dict(queries=len(queries), targets=n, complete12=c, partial12=p,
                    no_future=n-c-p, complete_fraction=c/n if n else None,
                    queries_with_two_complete=sum(complete[t] >= 2 for t in queries))

    multiagent = sorted(t for t, n in eligible.items() if n >= 2)
    selected = []
    next_start = start
    for t in multiagent:
        if t >= next_start:
            selected.append(t)
            next_start = t + future_offsets[-1] + 1
    return dict(all_history_queries=labels(sorted(eligible)),
                all_multiagent_history_queries=labels(multiagent),
                disjoint_past_selected=labels(selected),
                disjoint_first_ticks=selected)


def audit(path, spec):
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if digest != spec["sha256"] or path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Unexpected snapshot for {path.name}")
    sind = spec["release"] == "SinD public sample"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        tracks, residual = read_tracks(stream, sind)
    count = sum(map(len, tracks.values()))
    extent = [min(map(min, tracks.values())), max(map(max, tracks.values()))]
    if count != spec["rows"] or (
        not sind and (len(tracks) != spec["tracks"] or extent != spec["tick_extent"])
    ):
        raise ValueError(f"Population mismatch for {path.name}")
    sizes = sorted(map(len, tracks.values()))
    gaps = Counter()
    for ticks in tracks.values():
        ordered = sorted(ticks)
        gaps.update(b-a for a, b in zip(ordered, ordered[1:]))
    rate = 9.99 if sind else 10
    return {**spec, "track_ids": len(tracks), "tick_extent": extent,
            "track_length_rows": {f"p{p}": sizes[(len(sizes)-1)*p//100] for p in (0, 25, 50, 75, 90, 100)},
            "adjacent_track_tick_gaps": dict(sorted(gaps.items())),
            "published_grid_hz": rate, "max_clock_residual_seconds": residual,
            "strict_causal_positions_verified": False,
            "grids": {str(s): {"sample_period_seconds": s/rate,
                               **timing_support(tracks.values(), extent, s)} for s in (1, 4)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "external_data/ExternalTrajectorySourceAudit")
    parser.add_argument("--output", type=Path, default=REPORT / "analysis.json")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    results = []
    for spec in config["sources"]:
        results.append(audit(args.directory / spec["file"], spec))
        print(f"Audited {spec['name']}: {results[-1]['rows']:,} rows", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        result_source="fresh_run",
        scope="released_pedestrian_ids_and_timing_only",
        qualification_only=True,
        sind_commit=config["sind_commit"],
        sources=results,
        forecasts="not_run",
        formal_roles_assigned=False,
        dronecrowd_confirmation_access=False,
    ), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
