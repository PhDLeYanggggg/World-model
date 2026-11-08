"""Transfer the frozen GMU camera pipeline to two unused ACME cameras."""

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

from scripts.audit_m3w_acme_prefix_sources import track_clocks
from scripts import run_m3w_acme_gmu_camera as gmu
from scripts.run_m3w_acme_gmu_reassociate import all_frame_label_support, reassociate
from scripts.run_m3w_waterloo_raw_prefix import WEIGHTS, detect, detector, file_digest_array, immutable, sha
from src.world_model.m3w_streaming_boxes import StreamingTracks


LOCAL = ROOT/"external_data/ACME_source_audit"
PUBLIC = ROOT/"outputs/publication_readiness_2026_09/acme_held_camera_probe_v1"
PRIVATE = ROOT/"data/stage_cvpr2027_experiments/acme_held_camera_probe_v1"
SOURCES = {
    "cmu": dict(stem="cmu_327_1", ext="mp4", fps=10, frame_count=23349, factor=1,
                label_sha256="1866a542763a7a65017244de13396040b287fc7aa4068f9230eb25f2ff9a71b9",
                video_sha256="9624facb881b15f68ba6aaae448e9809283831f26c57efc457d96c169b95bb67",
                video_member="BEV/BEV_videos/cmu/327-1.mp4",
                label_member="BEV/BEV_annotations/trajectory-transform/trajectories/cmu/FINAL_327-1.toml"),
    "miraikan": dict(stem="miraikan_elder", ext="MP4", fps=60, frame_count=23370, factor=6,
                     label_sha256="68644c037f35e544b572fe21f9b1f9696473ae0bc22677ab023a90d7daa6a6dd",
                     video_sha256="3f03e8e2f8a29ab0bac63914aa29d013771fecb00deea2d619b364095aa86235",
                     video_member="BEV/BEV_videos/miraikan/03-09-2025-2-3F_Elder_Corridor_0_0.MP4",
                     label_member="BEV/BEV_annotations/trajectory-transform/trajectories/short-term/completed_2-3F_elder_corridor_0_0.toml"),
}


def label_tables(spec):
    p = LOCAL/(spec["stem"]+".toml")
    if sha(p) != spec["label_sha256"]:
        raise ValueError("Label snapshot mismatch")
    tracks = {}
    for key, row in tomllib.loads(p.read_text()).items():
        _, clock, _, _ = track_clocks(row)
        if clock:
            tracks[key] = dict(zip(clock, row["trajectories"], strict=True))
    return tracks


def association_frames(frames, start, factor):
    """Check physical frame indices before invoking the shared GMU scorer."""
    if len(frames) != 8:
        raise ValueError("Exactly eight camera frames required")
    out = []
    for k, frame in enumerate(frames):
        tick = start+k*gmu.STRIDE
        if frame["camera_frame"] != tick*factor:
            raise ValueError("Mapped camera frame mismatch")
        out.append({**frame, "camera_frame": tick*gmu.CAMERA_FACTOR})
    return out


def bind():
    parent = gmu.PUBLIC/"registration.json"
    transport = gmu.PUBLIC/"reassociation_registration.json"
    p, t = [json.loads(path.read_text()) for path in (parent, transport)]
    weights = sha(WEIGHTS)
    if weights != p["weights_sha256"] or sha(Path(gmu.__file__)) != p["code_sha256"]:
        raise ValueError("GMU detector or baseline parent changed")
    if sha(ROOT/"scripts/run_m3w_acme_gmu_reassociate.py") != t["code_sha256"]:
        raise ValueError("GMU association code changed")
    return dict(runner_sha256=sha(Path(__file__)), detector_weights_sha256=weights,
                gmu_registration_sha256=sha(parent), gmu_association_registration_sha256=sha(transport),
                dependencies={n:sha(ROOT/n) for n in (
                    "scripts/run_m3w_waterloo_raw_prefix.py", "scripts/run_m3w_acme_camera_repair.py",
                    "src/world_model/m3w_streaming_boxes.py")})


def camera_window(cap, model, start, spec):
    first = start*spec["factor"]
    stride = gmu.STRIDE*spec["factor"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    online, frames = StreamingTracks(), []
    for i in range(first, first+7*stride+1):
        ok, image = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != i+1:
            raise OSError(f"Decode mismatch at native frame {i}")
        if (i-first) % stride:
            continue
        residual = abs(cap.get(cv2.CAP_PROP_POS_MSEC)-1000*i/spec["fps"])
        if residual > .1:
            raise ValueError(f"Presentation timestamp mismatch at native frame {i}: {residual:.4f} ms")
        d = detect(model, image)
        rows, _ = online.step(len(frames), d[d[:, -1] == 1])
        frames.append(dict(camera_frame=i, frame_sha256=file_digest_array(image),
                           pts_error_ms=residual, rows=rows.tolist()))
    return frames


def produce(site, spec, registration, resume):
    video = LOCAL/(spec["stem"]+"."+spec["ext"])
    if sha(video) != spec["video_sha256"]:
        raise ValueError("Video snapshot mismatch")
    cap = cv2.VideoCapture(str(video))
    try:
        expected = (spec["fps"], spec["frame_count"], 1920, 1080)
        actual = tuple(cap.get(k) for k in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                                             cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or actual != expected:
            raise ValueError(f"Camera clock or raster mismatch: {actual} vs {expected}")
        model = detector()
        for k, start in enumerate(spec["starts"], 1):
            target = PRIVATE/f"{site}_{start:05d}.json"
            if target.exists() and resume:
                entry = json.loads(target.read_text())
                if entry["registration_sha256"] != registration:
                    raise ValueError("Cannot resume another registration")
                association_frames(entry["frames"], start, spec["factor"])
            else:
                immutable(target, dict(registration_sha256=registration,
                                       frames=camera_window(cap, model, start, spec)))
            beat = dict(pid=os.getpid(), site=site, windows=k, total=len(spec["starts"]), last_start=start)
            (PRIVATE/"heartbeat.json").write_text(json.dumps(beat)+"\n")
            print(beat, flush=True)
    finally:
        cap.release()


def score(banks, starts, labels):
    observed, n = [], 0
    for start, bank in zip(starts, banks, strict=True):
        per_window = []
        for ident, path in bank:
            ys = [labels[ident].get(start+(8+k)*gmu.STRIDE) for k in range(12)]
            if any(y is None for y in ys):
                continue
            d = np.linalg.norm(path-np.asarray(ys), axis=-1)
            per_window.append(np.column_stack((d.mean(-1), d[:, -1])))
        n += len(per_window)
        if per_window:
            observed.append(np.mean(per_window, axis=0))
    return dict(complete_futures=n, scored_windows=len(observed),
                query_equal_ade_fde_px=dict(zip(("CV", "OLS4"), np.mean(observed, axis=0).tolist()))
                if observed else None)


def evaluate(site, spec, registration):
    labels = label_tables(spec)
    windows, banks, errors = [], [], []
    for start in spec["starts"]:
        data = json.loads((PRIVATE/f"{site}_{start:05d}.json").read_text())
        if data["registration_sha256"] != registration:
            raise ValueError("Unregistered sensor cache")
        frames = association_frames(data["frames"], start, spec["factor"])
        errors.extend(r["pts_error_ms"] for r in frames)
        original, _ = gmu.associate_query(frames, labels, start)
        transported, bank = gmu.associate_query(reassociate(frames), labels, start)
        banks.append(bank)
        windows.append(dict(**original, transport_stable=transported["stable_matches"],
                            transport_online=transported["online_histories"],
                            matches_ignoring_ids=all_frame_label_support(frames, labels, start)))
    n = sum(w["transport_stable"] for w in windows)
    q = sum(w["transport_stable"] > 0 for w in windows)
    gate = n >= gmu.MIN_HISTORIES and q >= gmu.MIN_QUERIES
    return dict(windows=windows, original_stable_histories=sum(w["stable_matches"] for w in windows),
                transported_stable_histories=n, matched_windows=q, support_gate=gate,
                max_pts_error_ms=max(errors),
                future_scoring=score(banks, spec["starts"], labels) if gate else "not_run")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "produce", "evaluate"))
    cli.add_argument("--site", choices=SOURCES)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    reg = PUBLIC/"registration.json"
    binding = bind()
    if args.phase == "register":
        sites = {k: dict(**v, starts=gmu.sampled_starts(label_tables(v))) for k,v in SOURCES.items()}
        immutable(reg, dict(bindings=binding, sites=sites, observations=8, predictions=12,
                            annotation_stride=gmu.STRIDE, camera_phase=0,
                            diagnostic_support_gate=[gmu.MIN_HISTORIES, gmu.MIN_QUERIES],
                            source_selection="smallest published video with matched label file per chosen institution",
                            scope="frozen camera transfer diagnostic not independent policy confirmation"))
        return
    fixed = json.loads(reg.read_text())
    if fixed["bindings"] != binding:
        raise ValueError("Registered runner or model changed")
    key = sha(reg)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "produce":
            for site in ([args.site] if args.site else SOURCES):
                produce(site, fixed["sites"][site], key, args.resume)
        else:
            rows = {site:evaluate(site, fixed["sites"][site], key) for site in SOURCES}
            immutable(PUBLIC/"analysis.json", dict(registration_sha256=key, sites=rows,
                                                   independent_confirmation=False,
                                                   deployment_risk_evaluation="not_run",
                                                   neural_forecaster_evaluation="not_run"))
            print(json.dumps({k: {a:b for a,b in v.items() if a != "windows"} for k,v in rows.items()}))


if __name__ == "__main__":
    main()
