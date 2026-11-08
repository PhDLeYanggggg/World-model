"""Change only the frozen detector on the two previously exposed ACME cameras."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use the native arm64 .venv-pytorch interpreter")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_acme_held_camera_probe as prior

cv2, np, torch = prior.cv2, prior.np, prior.torch
sha, immutable, gmu = prior.sha, prior.immutable, prior.gmu
PUBLIC = prior.PUBLIC.parent / "acme_detector_repair_v1"
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
WEIGHTS = ROOT / "external_data/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
URL = "https://download.pytorch.org/models/" + WEIGHTS.name
WEIGHT_HASH = "dd69338a"


def register():
    parent = prior.PUBLIC / "registration.json"
    old = prior.PUBLIC / "analysis.json"
    r, a = (json.loads(p.read_text()) for p in (parent, old))
    if (a["registration_sha256"] != sha(parent) or r["bindings"] != prior.bind()
            or a["independent_confirmation"]):
        raise ValueError("Require frozen previous camera diagnostic")
    digest = sha(WEIGHTS)
    if not digest.startswith(WEIGHT_HASH):
        raise ValueError("ResNet50 weight checksum mismatch")
    f = PUBLIC / "registration.json"
    immutable(f, dict(runner_sha256=sha(Path(__file__)), parent_registration=sha(parent),
        parent_analysis=sha(old), weights_url=URL, weights_sha256=digest,
        detector="Torchvision Faster R-CNN ResNet50 FPN v2 COCO_V1",
        change="only detector architecture and weights; same tiling, detector thresholds, tracker, clock and matching",
        original_2pct=True, forecast_policy_evaluated=False, independent_confirmation=False,
        sites=r["sites"], support_gate=r["diagnostic_support_gate"]))
    return sha(f), r["sites"], a["sites"], sha(parent)


def detector():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    net = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None).eval()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.transform.min_size = (512,)
    net.transform.max_size = 768
    return net


def verify_frames(frames, reference):
    old = [(r["camera_frame"], r["frame_sha256"], r["pts_error_ms"]) for r in reference]
    new = [(r["camera_frame"], r["frame_sha256"], r["pts_error_ms"]) for r in frames]
    if len(old) != 8 or old != new:
        raise ValueError("Detector comparison did not decode identical frames")


def read_reference(site, start, key):
    path = prior.PRIVATE / f"{site}_{start:05d}.json"
    row = json.loads(path.read_text())
    if row["registration_sha256"] != key:
        raise ValueError("Frozen detector input receipt changed")
    return row["frames"]


def produce(key, sites, previous_key, resume):
    model = detector()
    begin = time.monotonic()
    for site, spec in sites.items():
        p = prior.LOCAL / (spec["stem"]+"."+spec["ext"])
        if sha(p) != spec["video_sha256"]:
            raise ValueError("Pinned video changed")
        cap = cv2.VideoCapture(str(p))
        try:
            expected = (spec["fps"], spec["frame_count"], 1920, 1080)
            if not cap.isOpened() or tuple(cap.get(k) for k in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)) != expected:
                raise ValueError("Camera metadata changed")
            for i, start in enumerate(spec["starts"], 1):
                f = PRIVATE / f"{site}_{start:05d}.json"
                ref = read_reference(site, start, previous_key)
                if f.exists() and resume:
                    x = json.loads(f.read_text())
                    if x["registration_sha256"] != key:
                        raise ValueError("Cannot resume a different detector")
                    verify_frames(x["frames"], ref)
                else:
                    frames = prior.camera_window(cap, model, start, spec)
                    verify_frames(frames, ref)
                    immutable(f, dict(registration_sha256=key, frames=frames))
                event = dict(pid=os.getpid(), site=site, windows=i, total=len(spec["starts"]),
                             seconds=round(time.monotonic()-begin, 3))
                (PRIVATE/"heartbeat.json").write_text(json.dumps(event)+"\n")
                print(event, flush=True)
        finally:
            cap.release()


def evaluate(key, sites, old, previous_key):
    out = {}
    for site, spec in sites.items():
        labels = prior.label_tables(spec)
        windows, banks = [], []
        for start in spec["starts"]:
            x = json.loads((PRIVATE/f"{site}_{start:05d}.json").read_text())
            if x["registration_sha256"] != key:
                raise ValueError("Unregistered detector cache")
            verify_frames(x["frames"], read_reference(site, start, previous_key))
            frames = prior.association_frames(x["frames"], start, spec["factor"])
            matches = prior.all_frame_label_support(frames, labels, start)
            result, bank = gmu.associate_query(prior.reassociate(frames), labels, start)
            windows.append(dict(stable=result["stable_matches"], online=result["online_histories"], label_support=matches))
            banks.append(bank)
        n = sum(w["stable"] for w in windows)
        nw = sum(w["stable"] > 0 for w in windows)
        ok = n >= gmu.MIN_HISTORIES and nw >= gmu.MIN_QUERIES
        out[site] = dict(stable=n, matched_windows=nw, support_gate=ok, windows=windows,
            prior_stable=old[site]["transported_stable_histories"], prior_windows=old[site]["matched_windows"],
            future_scoring=prior.score(banks, spec["starts"], labels) if ok else "not_run")
    immutable(PUBLIC/"analysis.json", dict(registration_sha256=key, sites=out, original_2pct=True,
        independent_confirmation=False, deployment_risk_evaluation="not_run", neural_forecaster_evaluation="not_run"))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "windows"} for k, v in out.items()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "produce", "evaluate"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    key, sites, old, previous_key = register()
    if a.phase == "register":
        return
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.phase == "produce":
            produce(key, sites, previous_key, a.resume)
        else:
            evaluate(key, sites, old, previous_key)


if __name__ == "__main__":
    main()
