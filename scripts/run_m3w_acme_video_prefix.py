"""Run a camera-only ACME streaming probe; never load annotations."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use the native arm64 .venv-pytorch interpreter")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_waterloo_raw_prefix import (
    WEIGHTS, detect, detector, file_digest_array, immutable, sha,
)
from src.world_model.m3w_streaming_boxes import StreamingTracks
import cv2
import numpy as np
import torch
import torchvision


VIDEO = ROOT / "external_data/ACME_source_audit/umich_GX010036_BEV1_DOW.MP4"
VIDEO_SHA = "dc4bc94cc46c49a2247c39cb9e9b4f6c274cb4421ad8e7a8c3b6824eb6976ddd"
VIDEO_URL = (
    "https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset/resolve/"
    "a0f117c11a1aee25413da95cf8612f5f8209c171/BEV/BEV_videos/umich/GX010036_BEV1_DOW.MP4"
)
PRIVATE = ROOT / "data/stage_cvpr2027_experiments/acme_video_prefix_v1"
PUBLIC = ROOT / "outputs/publication_readiness_2026_09/acme_prefix_intake_v1"
FRAMES, COLD = 32, 16


def video_clock(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 5 or fps > 240:
        raise ValueError("Unexpected container frame rate")
    stride = round(0.4*fps)
    return dict(container_fps=fps, stride_raw_frames=stride,
                stride_period_seconds=stride/fps,
                container_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def run_prefix(count, binding, resume, cold):
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise OSError("Cannot open ACME publisher video")
    rows = []
    state = StreamingTracks()
    network = None
    try:
        clock = video_clock(capture)
        stride = clock["stride_raw_frames"]
        for i in range((count-1)*stride+1):
            good, frame = capture.read()
            if not good:
                raise OSError(f"Unexpected EOF at frame {i}")
            if i % stride:
                continue
            token = dict(raw_frame=i, frame_sha256=file_digest_array(frame), registration=binding)
            cache = PRIVATE / f"frame_{i:05d}.npz"
            if cache.exists() and not cold:
                if not resume:
                    raise FileExistsError("Existing prefix cache: pass --resume")
                with np.load(cache, allow_pickle=False) as stored:
                    det = stored["detections"]
                    if (json.loads(str(stored["binding"])) != token or
                            str(stored["digest"]) != file_digest_array(det)):
                        raise ValueError("Video prefix cache mismatch")
            else:
                if network is None:
                    network = detector()
                all_detections = detect(network, frame)
                det = all_detections[all_detections[:, -1] == 1]
                if not cold:
                    pending = cache.with_suffix(".pending.npz")
                    np.savez(pending, detections=det, binding=json.dumps(token), digest=file_digest_array(det))
                    pending.replace(cache)
            tracked, pred = state.step(i//stride, det)
            forecasts = np.asarray([np.stack((cv, ols)) for _, cv, ols in pred])
            digests = [file_digest_array(x) for x in (det, tracked, forecasts)]
            rows.append(dict(**token, time_ms=capture.get(cv2.CAP_PROP_POS_MSEC),
                             causal_output_sha256=hashlib.sha256("".join(digests).encode()).hexdigest(),
                             detections=len(det), forecasts=len(pred)))
            if (i//stride) % 4 == 0 or len(rows) == count:
                beat = dict(pid=os.getpid(), done=len(rows), total=count, cold=cold)
                (PRIVATE / "heartbeat.json").write_text(json.dumps(beat)+"\n")
                print(json.dumps(beat), flush=True)
    finally:
        capture.release()
    return dict(video=clock, frames=rows, detector_rows=sum(r["detections"] for r in rows),
                forecast_queries=sum(r["forecasts"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "verify"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if VIDEO.stat().st_size != 489731567 or sha(VIDEO) != VIDEO_SHA:
            raise ValueError("Video does not match the pinned publisher object")
        weight_sha = sha(WEIGHTS)
        if not weight_sha.startswith("907ea3f9"):
            raise ValueError("Unexpected pretrained detector weights")
        registration = PUBLIC / "video_probe_registration.json"
        immutable(registration, dict(
            video_url=VIDEO_URL, video_sha256=VIDEO_SHA, video_bytes=489731567,
            detector_weights_sha256=weight_sha, frames=FRAMES, cold_frames=COLD,
            selection="first_video_frame_then_approximate_0.4s_stride",
            role="camera_prefix_engineering", person_class=1, cpu_threads=4, workers=0,
            code_sha256={name:sha(ROOT / name) for name in (
                "scripts/run_m3w_acme_video_prefix.py", "scripts/run_m3w_waterloo_raw_prefix.py",
                "src/world_model/m3w_streaming_boxes.py",
            )},
            software=dict(torch=torch.__version__, torchvision=torchvision.__version__, opencv=cv2.__version__),
            annotations_read=False,
        ))
        cold = args.phase == "verify"
        result = run_prefix(COLD if cold else FRAMES, sha(registration), args.resume, cold)
        report = PUBLIC / "video_probe.json"
        if cold:
            saved = json.loads(report.read_text())
            if result["video"] != saved["video"] or result["frames"] != saved["frames"][:COLD]:
                raise ValueError("Cold prefix disagrees with the longer run")
            immutable(PUBLIC / "video_probe_verification.json", dict(
                exact=True, frames=COLD, detector_rows=result["detector_rows"],
                forecast_queries=result["forecast_queries"], report_sha256=sha(report),
            ))
        else:
            immutable(report, dict(
                **result, annotations_read=False, forecast_accuracy="not_run",
                annotation_video_alignment_verified=False, independent_confirmation=False,
            ))


if __name__ == "__main__":
    main()
