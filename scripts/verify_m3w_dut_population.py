"""Independent raw-CSV coverage recount for the frozen descriptive DUT readout."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "outputs/publication_readiness_2026_09/dut_frozen_readout_v1"
COUNT_KEYS = ("queries", "targets", "visible", "unknown_cv_context", "complete", "known_steps")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recount(rows):
    """Count frame membership without imported cache/index/inference helpers."""
    tracks, visible = defaultdict(set), defaultdict(set)
    for frame, agent in rows:
        if frame in tracks[agent]:
            raise ValueError("Duplicate raw agent/frame")
        tracks[agent].add(frame)
        visible[frame].add(agent)
    if not visible:
        raise ValueError("Empty recording")
    queries = sorted(f for f in visible if f >= min(visible) + 7)
    result = dict.fromkeys(COUNT_KEYS, 0)
    result.update(raw_rows=sum(map(len, tracks.values())), agents=len(tracks),
                  partial_targets=0, zero_future_targets=0,
                  complete_pedestrian_targets=0, complete_vehicle_targets=0)
    for frame in queries:
        result["queries"] += 1
        result["visible"] += len(visible[frame])
        for agent in visible[frame]:
            frames = tracks[agent]
            result["unknown_cv_context"] += int(frame - 1 not in frames)
            # Future availability never decides inference target membership.
            if not all(frame - j in frames for j in range(8)):
                continue
            result["targets"] += 1
            known = sum(frame + j in frames for j in range(1, 13))
            result["known_steps"] += known
            result["complete"] += int(known == 12)
            result["partial_targets"] += int(0 < known < 12)
            result["zero_future_targets"] += int(known == 0)
            if known == 12:
                result["complete_vehicle_targets" if agent % 2 else "complete_pedestrian_targets"] += 1
    return result


def compare_counts(expected, receipt):
    if receipt["next_query"] != expected["queries"] or receipt["queries"] != expected["queries"]:
        raise ValueError("Incomplete query coverage")
    if not receipt["stats"]:
        raise ValueError("No model views")
    for view, counts in receipt["stats"].items():
        for key in COUNT_KEYS:
            if counts[key] != expected[key]:
                raise ValueError(f"Coverage mismatch: {view}/{key}")


def audit(source_root, *, check_results=False):
    manifest_path = PUBLIC / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_records = {}
    for name, record in manifest["records"].items():
        metadata_path = ROOT / record["cache_path"] / "metadata.json"
        if digest(metadata_path) != record["metadata_sha256"]:
            raise ValueError("Metadata binding changed")
        metadata = json.loads(metadata_path.read_text())
        rows = []
        for entry in metadata["files"]:
            path = (source_root / entry["path"]).resolve()
            if not path.is_relative_to(source_root.resolve()) or digest(path) != entry["sha256"]:
                raise ValueError("Raw source binding changed")
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["label"] not in ("ped", "veh"):
                        raise ValueError("Unknown raw agent label")
                    rows.append((int(row["frame"]), 2 * int(row["id"]) + int(row["label"] == "veh")))
        expected_records[name] = dict(site=record["physical_scene"], **recount(rows))
    checks = []
    if check_results:
        analysis = json.loads((PUBLIC / "analysis.json").read_text())
        if analysis["manifest_sha256"] != digest(manifest_path):
            raise ValueError("Readout manifest changed")
        if {r["recording"] for r in analysis["records"]} != set(expected_records):
            raise ValueError("Readout recording population mismatch")
        for entry in analysis["records"]:
            path = ROOT / entry["path"]
            if digest(path) != entry["sha256"]:
                raise ValueError("Completed receipt changed")
            receipt = json.loads(path.read_text())
            if receipt["manifest_sha256"] != digest(manifest_path) or receipt["recording"] != entry["recording"]:
                raise ValueError("Completed receipt identity mismatch")
            if set(receipt["stats"]) != set(analysis["views"]):
                raise ValueError("Missing fixed model view")
            compare_counts(expected_records[entry["recording"]], receipt)
            checks.append(entry["recording"])
    keys = set(next(iter(expected_records.values()))) - {"site"}
    return dict(result_source="fresh_run_independent_raw_csv_recount",
        manifest_sha256=digest(manifest_path), code_sha256=digest(Path(__file__)),
        totals={k: sum(r[k] for r in expected_records.values()) for k in sorted(keys)},
        recording_count=len(expected_records), physical_sites=len({r["site"] for r in expected_records.values()}),
        records=expected_records, result_coverage_verified=check_results,
        completed_recordings_checked=checks, prediction_values_read=False,
        inference_rerun=False, scope="counts_only_not_accuracy_or_independence_certificate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "external_data/DUT_author_raw")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    result = audit(args.source_root, check_results=args.check_results)
    target = PUBLIC / ("population_verification.json" if args.check_results else "population_recount.json")
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
