"""Predeclared camera-input support and motion comparison on an exposed GMU clip."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import tomllib

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use the native arm64 .venv-pytorch interpreter")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from scripts.audit_m3w_external_causal_sources import timing_support
from scripts.audit_m3w_acme_prefix_sources import track_clocks
from scripts.run_m3w_acme_camera_repair import associate_labels, feet
from scripts.run_m3w_waterloo_raw_prefix import (
    WEIGHTS, detect, detector, file_digest_array, immutable, sha,
)
from src.world_model.m3w_streaming_boxes import StreamingTracks


LOCAL = ROOT/"external_data/ACME_source_audit"
VIDEO = LOCAL/"gmu_engr_elev_1.avi"
LABELS = LOCAL/"gmu.toml"
VIDEO_SHA = "eda774eaf3def7579ea9e68d97a8ecf55a818d8a5ce757ea2af50f1b714e7763"
LABEL_SHA = "b4f3e0c2cf2667bc50d4dde1d5cedee9b711aeddc3bcaf55ed0e7cc770a6c634"
PRIVATE = ROOT/"data/stage_cvpr2027_experiments/acme_gmu_camera_v1"
PUBLIC = ROOT/"outputs/publication_readiness_2026_09/acme_gmu_camera_v1"
STRIDE, CAMERA_FACTOR, QUERIES = 4, 3, 12
MIN_HISTORIES, MIN_QUERIES = 8, 3


def label_tables():
    result = {}
    for key, item in tomllib.loads(LABELS.read_text()).items():
        _, frames, _, _ = track_clocks(item)
        if frames:
            result[key] = dict(zip(frames, item["trajectories"], strict=True))
    return result


def sampled_starts(tracks):
    times = [set(a) for a in tracks.values()]
    extent = min(map(min, times)), max(map(max, times))
    grid = timing_support(times, extent, STRIDE)["disjoint_first_ticks"]
    if len(grid) < QUERIES:
        raise ValueError("Insufficient structural query coverage")
    return [grid[i] for i in np.linspace(0, len(grid)-1, QUERIES, dtype=int)]


def rollout(history):
    h = np.asarray(history, dtype=float)
    if h.shape != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Expected eight finite past positions")
    cv = h[-1]-h[-2]
    ols = np.array([-1.5, -.5, .5, 1.5]) @ (h[-4:]-h[-1])/5
    dt = np.arange(1., 13.)[:, None]
    return np.stack((h[-1]+cv*dt, h[-1]+ols*dt))


def query(cap, model, start):
    first = CAMERA_FACTOR*start
    last = first+7*STRIDE*CAMERA_FACTOR
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tracker = StreamingTracks()
    frames = []
    for i in range(first, last+1):
        ok, image = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != i+1:
            raise OSError(f"Cannot decode publisher camera frame {i}")
        if (i-first) % (STRIDE*CAMERA_FACTOR):
            continue
        if abs(cap.get(cv2.CAP_PROP_POS_MSEC)-1000*i/30) > .1:
            raise ValueError("Observed camera timestamp does not match declared grid")
        d = detect(model, image)
        rows, _ = tracker.step(len(frames), d[d[:, -1] == 1])
        frames.append(dict(camera_frame=i, frame_sha256=file_digest_array(image), rows=rows.tolist()))
    return frames


def associate_query(frames, labels, start):
    histories, matches = {}, []
    past_support = set(labels)
    raw, points, counts = 0, 0, []
    for k, f in enumerate(frames):
        t = start+k*STRIDE
        if f["camera_frame"] != t*CAMERA_FACTOR:
            raise ValueError("Camera-to-label frame mismatch")
        rows = np.asarray(f["rows"]).reshape(-1, 7)
        histories = {int(r[0]): histories.get(int(r[0]), [])+[p]
                     for r, p in zip(rows, feet(rows[:, 1:5]), strict=True)}
        targets = [(key, v[t]) for key, v in labels.items() if t in v]
        past_support.intersection_update(key for key, _ in targets)
        raw += len(rows)
        points += len(targets)
        pairs = set(associate_labels(rows, targets))
        matches.append(pairs)
        counts.append(len(pairs))
    stable = set.intersection(*matches)
    complete_histories = {key: xy for key, xy in histories.items() if len(xy) == 8}
    return dict(start=start, detector_points=raw, reference_points=points,
                matched_points=sum(counts), reference_histories=len(past_support),
                online_histories=len(complete_histories), stable_matches=len(stable)), [
                    (lab, rollout(complete_histories[online])) for online, lab in sorted(stable)
                ]


def evaluate(starts, binding):
    labels = label_tables()
    support = []
    banks = []
    for s in starts:
        saved = json.loads((PRIVATE/f"window_{s:05d}.json").read_text())
        if saved["registration"] != binding:
            raise ValueError("Unregistered prefix cache")
        counts, bank = associate_query(saved["frames"], labels, s)
        support.append(counts)
        banks.append(bank)
    n = sum(s["stable_matches"] for s in support)
    q = sum(s["stable_matches"] > 0 for s in support)
    gate = n >= MIN_HISTORIES and q >= MIN_QUERIES
    result = dict(support=support, stable_histories=n, matched_queries=q,
                  support_gate=gate, scoring="not_run", future_labels_scored=False)
    if gate:
        scores = []
        complete = 0
        for start, bank in zip(starts, banks, strict=True):
            window = []
            for lab, paths in bank:
                # This is the only lookup of future positions used in scoring.
                future = [labels[lab].get(start+(8+i)*STRIDE) for i in range(12)]
                if any(p is None for p in future):
                    continue
                dist = np.linalg.norm(paths-np.asarray(future), axis=-1)
                window.append(np.column_stack((dist.mean(-1), dist[:, -1])))
            complete += len(window)
            if window:
                scores.append(np.mean(window, axis=0))
        result.update(scoring="observed_complete_cases_diagnostic" if scores else "no_complete_labels",
                      future_labels_scored=True, complete_futures=complete, missing_futures=n-complete,
                      windows_with_scores=len(scores),
                      query_equal_mean_ade_fde_px=dict(zip(("CV", "OLS4"), np.mean(scores, axis=0).tolist()))
                      if scores else None)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "produce", "evaluate"))
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    if sha(VIDEO) != VIDEO_SHA or sha(LABELS) != LABEL_SHA:
        raise ValueError("Source snapshot mismatch")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    register = PUBLIC/"registration.json"
    if args.phase == "register":
        immutable(register, dict(
            scope="one_exposed_GMU_camera_engineering", video_sha256=VIDEO_SHA, label_sha256=LABEL_SHA,
            source_clock_hz=30, annotation_hz=10, camera_factor=CAMERA_FACTOR, camera_phase=0,
            starts=sampled_starts(label_tables()), stride=STRIDE, observed=8, forecast=12,
            required_stable_histories=MIN_HISTORIES, required_matched_queries=MIN_QUERIES,
            code_sha256=sha(Path(__file__)), weights_sha256=sha(WEIGHTS),
            choice="even_quantiles_of_nonoverlapping_past_supported_reference_queries",
            method="CV_and_anchored_OLS4_frozen_before_outcome_lookup",
            official_or_confirmation=False,
        ))
        return
    spec = json.loads(register.read_text())
    if spec["code_sha256"] != sha(Path(__file__)) or spec["weights_sha256"] != sha(WEIGHTS):
        raise ValueError("Registered implementation or detector changed")
    binding = sha(register)
    starts = spec["starts"]
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "evaluate":
            result = evaluate(starts, binding)
            immutable(PUBLIC/"analysis.json", {**result, "registration_sha256": binding,
                                              "independent_confirmation": False,
                                              "risk_or_neural_evaluation": "not_run"})
            print(json.dumps(result), flush=True)
            return
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(1)
        model = detector()
        cap = cv2.VideoCapture(str(VIDEO))
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) != 30:
            raise ValueError("Expected pinned 30 FPS video")
        try:
            for k, s in enumerate(starts, 1):
                out = PRIVATE/f"window_{s:05d}.json"
                if out.exists() and args.resume:
                    if json.loads(out.read_text())["registration"] != binding:
                        raise ValueError("Resume cache mismatch")
                else:
                    immutable(out, dict(registration=binding, frames=query(cap, model, s)))
                beat = dict(pid=os.getpid(), windows=k, total=len(starts), last_start=s)
                (PRIVATE/"heartbeat.json").write_text(json.dumps(beat)+"\n")
                print(beat, flush=True)
        finally:
            cap.release()


if __name__ == "__main__":
    main()
