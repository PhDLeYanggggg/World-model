"""Separate annotation, point matching and identity limits of the camera roster."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_m3w_acme_camera_roster as roster


def support_counts(past, future, ticks_by_label, matched_labels):
    observed = {label for label, ticks in ticks_by_label.items() if past <= ticks}
    complete = {label for label in observed if future <= ticks_by_label[label]}
    if not matched_labels <= observed:
        raise ValueError("Matched reference must appear in every observed tick")
    counts = dict(reference_past=len(observed), reference_complete=len(complete),
                  framewise_past=len(matched_labels), framewise_complete=len(complete & matched_labels))
    return {**counts, **{k + "_pair": v >= 2 for k, v in counts.items()}}


def ceiling_window(hybrid, frames, spec, labels, ticks, start):
    frames = hybrid.old.association_frames(frames, start, spec["factor"])
    past = set(range(start, start + roster.OBS * roster.STRIDE, roster.STRIDE))
    future = set(range(start + roster.OBS * roster.STRIDE,
                       start + (roster.OBS + roster.PRED) * roster.STRIDE, roster.STRIDE))
    matches = []
    for t, frame in zip(sorted(past), frames, strict=True):
        targets = [(label, path[t]) for label, path in labels.items() if t in path]
        rows = hybrid.np.asarray(frame["rows"]).reshape(-1, 7)
        matches.append({label for _, label in hybrid.old.gmu.associate_labels(rows, targets)})
    return support_counts(past, future, ticks, set.intersection(*matches))


def main():
    from scripts import run_m3w_acme_hybrid_detector as hybrid

    reg = roster.PUBLIC / "registration.json"
    report = roster.PUBLIC / "analysis.json"
    fixed = json.loads(reg.read_text())
    key = hybrid.sha(reg)
    measured = json.loads(report.read_text())
    if fixed != roster.registration(hybrid) or measured["registration_sha256"] != key:
        raise ValueError("Roster readout does not match the frozen inputs")
    prior_key, prior_sites, *_ = hybrid.register()
    sites = {}
    for site, spec in fixed["sites"].items():
        labels = hybrid.old.label_tables(spec)
        ticks = {label: set(path) for label, path in labels.items()}
        records = []
        shared = set(spec["starts"]) & set(prior_sites[site]["starts"])
        for start, scored in zip(spec["starts"], measured["sites"][site]["camera_clock_grid"]["per_window"],
                                 strict=True):
            path = roster.PRIVATE / f"{site}_{start:05d}.json"
            frames = roster.load_window(hybrid, path, key, start, spec)
            if start in shared and frames != hybrid.read_window(site, start, prior_key):
                raise ValueError("The frozen detector differs at a shared start")
            row = ceiling_window(hybrid, frames, spec, labels, ticks, start)
            if (scored["start"] != start or scored["stable"] > row["framewise_past"]
                    or scored["complete"] > row["framewise_complete"]):
                raise ValueError("Identity-stable results cannot exceed fixed framewise matches")
            records.append(dict(start=start, **row))
        sites[site] = dict(windows=len(records), identical_shared_detector_frames=len(shared) * roster.OBS,
                           **roster.aggregate(records), per_window=records)
    hybrid.immutable(roster.PUBLIC / "support_ceiling.json", dict(
        result_source="fresh_run", frame_cache_source="cached_verified",
        code_sha256=hybrid.sha(Path(__file__)), parent_analysis_sha256=hybrid.sha(report),
        meaning="fixed one-to-one framewise matches ignoring online identity; not an optimized rematch",
        future_geometry_read_for_scoring=False, forecast_accuracy="not_run", sites=sites))
    print(json.dumps({k: {n: v for n, v in s.items() if n != "per_window"} for k, s in sites.items()}))


if __name__ == "__main__":
    main()
