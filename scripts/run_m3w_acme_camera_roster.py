"""Measure full online roster support on an annotation-independent camera grid."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "outputs/publication_readiness_2026_09/acme_camera_roster_v1"
PRIVATE = ROOT / "data/stage_cvpr2027_experiments/acme_camera_roster_v1"
OBS, PRED, STRIDE, QUERIES = 8, 12, 4, 12


def scheduled_starts(frame_count, factor):
    last_tick = (frame_count - 1) // factor
    max_slot = last_tick // (STRIDE * (OBS + PRED)) - 1
    if max_slot < QUERIES - 1:
        raise ValueError("Insufficient disjoint camera intervals")
    return [i * max_slot // (QUERIES - 1) * STRIDE * (OBS + PRED) for i in range(QUERIES)]


def roster_counts(online_by_frame, matches_by_frame, label_ticks, start):
    if len(online_by_frame) != OBS or len(matches_by_frame) != OBS:
        raise ValueError("Roster requires exactly eight observations")
    for ids, matches in zip(online_by_frame, matches_by_frame, strict=True):
        if (not {i for i, _ in matches} <= ids or len({i for i, _ in matches}) != len(matches)
                or len({j for _, j in matches}) != len(matches)):
            raise ValueError("Invalid one-to-one reference match")
    online = set.intersection(*online_by_frame)
    stable = set.intersection(*matches_by_frame)
    future = set(range(start + OBS * STRIDE, start + (OBS + PRED) * STRIDE, STRIDE))
    stable_labels = {label for _, label in stable}
    complete = sum(future <= label_ticks[label] for label in stable_labels)
    partial = sum(bool(future & label_ticks[label]) for label in stable_labels) - complete
    counts = dict(online=len(online), stable=len(stable), complete=complete, partial=partial,
                  absent=len(stable) - complete - partial)
    return dict(**counts, unmatched=len(online) - len(stable),
                multiagent=len(online) >= 2, complete_pair=complete >= 2,
                whole_multiagent_roster=len(online) >= 2 and complete == len(online))


def digest(frames):
    return hashlib.sha256(json.dumps(frames, sort_keys=True, allow_nan=False).encode()).hexdigest()


def registration(hybrid):
    old = hybrid.old
    base_key, previous, *_ = hybrid.register()
    selected = {site: {**spec, "starts": scheduled_starts(spec["frame_count"], spec["factor"])}
                for site, spec in previous.items()}
    data = dict(runner_sha256=hybrid.sha(Path(__file__)), backend_registration_sha256=base_key,
                backend_sha256=hybrid.sha(Path(hybrid.__file__)), sites=selected,
                observations=OBS, predictions=PRED, annotation_stride=STRIDE, phase=0,
                schedule="equally spaced disjoint camera-clock slots; never use labels or detections",
                label_use="one-to-one association and future presence only, after all prefixes are cached",
                scope="exposed cameras, development only; no new split or deployment test")
    if old.gmu.STRIDE != STRIDE:
        raise ValueError("Registered stride mismatch")
    return data


def load_window(hybrid, path, key, start, spec):
    row = json.loads(path.read_text())
    frames = row["frames"]
    if row["registration_sha256"] != key or row["frames_sha256"] != digest(frames):
        raise ValueError("Camera cache binding or digest mismatch")
    hybrid.old.association_frames(frames, start, spec["factor"])
    if any(not 0 <= x["pts_error_ms"] <= .1 for x in frames):
        raise ValueError("Cached frame clock mismatch")
    return frames


def produce(hybrid, key, sites):
    torch, cv2 = hybrid.torch, hybrid.cv2
    if not torch.backends.mps.is_available():
        raise RuntimeError("The pinned CPU/MPS detector is unavailable")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    model = hybrid.cpu.detector()
    model.backbone = hybrid.Backbone(model.backbone)
    model.roi_heads.box_head = hybrid.BoxHead(model.roi_heads.box_head)
    began = time.monotonic()
    for site, spec in sites.items():
        source = hybrid.old.LOCAL / (spec["stem"] + "." + spec["ext"])
        if hybrid.sha(source) != spec["video_sha256"]:
            raise ValueError("Pinned video changed")
        cap = cv2.VideoCapture(str(source))
        try:
            properties = (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                          cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
            if not cap.isOpened() or tuple(cap.get(p) for p in properties) != (
                    spec["fps"], spec["frame_count"], 1920, 1080):
                raise ValueError("Pinned video metadata changed")
            for index, start in enumerate(spec["starts"], 1):
                path = PRIVATE / f"{site}_{start:05d}.json"
                fresh = not path.exists()
                if fresh:
                    frames = hybrid.old.camera_window(cap, model, start, spec)
                    hybrid.immutable(path, dict(registration_sha256=key, frames=frames,
                                                frames_sha256=digest(frames)))
                else:
                    load_window(hybrid, path, key, start, spec)
                beat = dict(pid=os.getpid(), site=site, windows=index, total=len(spec["starts"]),
                            fresh=fresh, elapsed_seconds=round(time.monotonic() - began, 2))
                pending = PRIVATE / "heartbeat.pending"
                pending.write_text(json.dumps(beat) + "\n")
                pending.replace(PRIVATE / "heartbeat.json")
                print(beat, flush=True)
        finally:
            cap.release()


def readout(hybrid, frames, spec, labels, label_ticks, start):
    old = hybrid.old
    frames = old.reassociate(old.association_frames(frames, start, spec["factor"]))
    online, matches = [], []
    for step, frame in enumerate(frames):
        tick = start + step * STRIDE
        rows = hybrid.np.asarray(frame["rows"]).reshape(-1, 7)
        targets = [(label, path[tick]) for label, path in labels.items() if tick in path]
        online.append(set(rows[:, 0].astype(int)))
        matches.append(set(old.gmu.associate_labels(rows, targets)))
    return roster_counts(online, matches, label_ticks, start)


def aggregate(rows):
    return {name: sum(row[name] for row in rows) for name in rows[0] if name != "start"}


def evaluate(hybrid, key, sites):
    prior_key, prior_sites, _, _, mobile_key = hybrid.register()
    # Verify that all camera-only prefixes exist before loading evaluation references.
    caches = {site: [load_window(hybrid, PRIVATE / f"{site}_{start:05d}.json", key, start, spec)
                     for start in spec["starts"]] for site, spec in sites.items()}
    result = {}
    for site, spec in sites.items():
        labels = hybrid.old.label_tables(spec)
        ticks = {label: set(path) for label, path in labels.items()}
        prior = prior_sites[site]
        cached = []
        for start in prior["starts"]:
            frames = hybrid.read_window(site, start, prior_key)
            hybrid.cpu.verify_frames(frames, hybrid.cpu.read_reference(site, start, mobile_key))
            cached.append(frames)
        views = (
            ("camera_clock_grid", spec["starts"], caches[site]),
            ("reference_enriched", prior["starts"], cached),
        )
        result[site] = {}
        for name, starts, windows in views:
            rows = [dict(start=start, **readout(hybrid, frames, spec, labels, ticks, start))
                    for start, frames in zip(starts, windows, strict=True)]
            result[site][name] = dict(result_source="fresh_run", prefix_cache_source=(
                "fresh_run" if name == "camera_clock_grid" else "cached_verified"),
                windows=len(rows), **aggregate(rows), per_window=rows)
    report = dict(registration_sha256=key, sites=result, forecast_accuracy="not_run",
                  independent_confirmation=False, deployment_risk="not_run",
                  stage5c=False, smc=False)
    hybrid.immutable(PUBLIC / "analysis.json", report)
    print(json.dumps({site: {mode: {k: v for k, v in group.items() if k != "per_window"}
                            for mode, group in groups.items()} for site, groups in result.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "produce", "evaluate"))
    args = p.parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts import run_m3w_acme_hybrid_detector as hybrid

    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg = PUBLIC / "registration.json"
        data = registration(hybrid)
        if args.phase == "register":
            hybrid.immutable(reg, data)
            return
        if json.loads(reg.read_text()) != data:
            raise ValueError("Frozen camera registration changed")
        fn = produce if args.phase == "produce" else evaluate
        fn(hybrid, hybrid.sha(reg), data["sites"])


if __name__ == "__main__":
    main()
