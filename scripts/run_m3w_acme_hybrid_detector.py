"""Run the registered ResNet50 comparison with MPS convolutions and CPU proposals."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_acme_detector_repair as cpu

old = cpu.prior
torch, cv2, np = cpu.torch, cpu.cv2, cpu.np
sha, immutable = cpu.sha, cpu.immutable
PUBLIC = cpu.PUBLIC.parent / "acme_detector_hybrid_v1"
PRIVATE = cpu.PRIVATE.parent / PUBLIC.name


class Backbone(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.to("mps")
        self.out_channels = net.out_channels

    def forward(self, x):
        return {k: v.cpu() for k, v in self.net(x.to("mps")).items()}


class BoxHead(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.to("mps")

    def forward(self, x):
        return self.net(x.to("mps")).cpu()


def register():
    cpu_key, sites, previous, mobile_key = cpu.register()
    f = PUBLIC/"registration.json"
    immutable(f, dict(runner_sha256=sha(Path(__file__)), cpu_registration_sha256=cpu_key,
        mobile_registration_sha256=mobile_key, clock_and_frames_unchanged=True,
        runtime="MPS backbone/FPN and ROI box head; CPU preprocessing, proposal and box selection and tracking",
        threads=4, workers=0, separate_backend_caches=True, no_online_label_input=True,
        overlap_check="all preexisting CPU windows; compare frames, ordered rows, and support without mixing backends",
        original_2pct=True, deployment_risk_evaluation="not_run", independent_confirmation=False))
    return sha(f), sites, previous, cpu_key, mobile_key


def read_window(site, start, key, directory=PRIVATE):
    row = json.loads((directory/f"{site}_{start:05d}.json").read_text())
    if row["registration_sha256"] != key:
        raise ValueError("Cached detector uses a different registration")
    return row["frames"]


def run(key, sites, mobile_key):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; preserve the registered CPU resume separately")
    model = cpu.detector()
    model.backbone = Backbone(model.backbone)
    model.roi_heads.box_head = BoxHead(model.roi_heads.box_head)
    began = time.monotonic()
    for site, spec in sites.items():
        path = old.LOCAL/(spec["stem"]+"."+spec["ext"])
        if sha(path) != spec["video_sha256"]:
            raise ValueError("Pinned video changed")
        cap = cv2.VideoCapture(str(path))
        try:
            props = (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
            if not cap.isOpened() or tuple(cap.get(k) for k in props) != (spec["fps"], spec["frame_count"], 1920, 1080):
                raise ValueError("Pinned camera metadata changed")
            for i, start in enumerate(spec["starts"], 1):
                f = PRIVATE/f"{site}_{start:05d}.json"
                mobile = cpu.read_reference(site, start, mobile_key)
                fresh = not f.exists()
                if fresh:
                    frames = old.camera_window(cap, model, start, spec)
                    cpu.verify_frames(frames, mobile)
                    immutable(f, dict(registration_sha256=key, frames=frames))
                else:
                    cpu.verify_frames(read_window(site, start, key), mobile)
                event = dict(pid=os.getpid(), site=site, windows=i, total=len(spec["starts"]), fresh=fresh,
                             seconds=round(time.monotonic()-began, 3))
                pending = PRIVATE/"heartbeat.pending"
                pending.write_text(json.dumps(event)+"\n")
                pending.replace(PRIVATE/"heartbeat.json")
                print(event, flush=True)
        finally:
            cap.release()


def rows_compare(a, b):
    cpu.verify_frames(a, b)
    max_box, max_confidence = 0., 0.
    mismatches = 0
    for af, bf in zip(a, b, strict=True):
        x, y = [np.asarray(f["rows"]).reshape(-1, 7) for f in (af, bf)]
        if x.shape != y.shape or not np.array_equal(x[:, [0, 6]], y[:, [0, 6]]):
            mismatches += 1
        elif len(x):
            max_box = max(max_box, float(np.abs(x[:, 1:5]-y[:, 1:5]).max()))
            max_confidence = max(max_confidence, float(np.abs(x[:, 5]-y[:, 5]).max()))
    return dict(frames=len(a), count_or_identity_mismatches=mismatches, max_box_difference_px=max_box,
                max_confidence_difference=max_confidence)


def summarize_window(frames, spec, labels, start):
    frames = old.association_frames(frames, start, spec["factor"])
    result, bank = old.gmu.associate_query(old.reassociate(frames), labels, start)
    return dict(stable=result["stable_matches"], online=result["online_histories"],
                label_support=old.all_frame_label_support(frames, labels, start)), bank


def evaluate(key, sites, previous, cpu_key, mobile_key):
    result = {}
    for site, spec in sites.items():
        labels = old.label_tables(spec)
        windows, forecasts, overlaps = [], [], []
        for start in spec["starts"]:
            frames = read_window(site, start, key)
            cpu.verify_frames(frames, cpu.read_reference(site, start, mobile_key))
            support, bank = summarize_window(frames, spec, labels, start)
            windows.append(support)
            forecasts.append(bank)
            if (cpu.PRIVATE/f"{site}_{start:05d}.json").is_file():
                ref = read_window(site, start, cpu_key, cpu.PRIVATE)
                par = rows_compare(frames, ref)
                ref_support, _ = summarize_window(ref, spec, labels, start)
                overlaps.append(dict(**par, support_identical=support == ref_support))
        n = sum(v["stable"] for v in windows)
        q = sum(v["stable"] > 0 for v in windows)
        ok = n >= old.gmu.MIN_HISTORIES and q >= old.gmu.MIN_QUERIES
        result[site] = dict(stable=n, windows_with_stable=q, support_gate=ok, windows=windows, cpu_overlap=overlaps,
            prior_stable=previous[site]["transported_stable_histories"], prior_windows=previous[site]["matched_windows"],
            future_scoring=old.score(forecasts, spec["starts"], labels) if ok else "not_run")
    out = dict(registration_sha256=key, sites=result, evaluated_frames=8*sum(len(v["starts"]) for v in sites.values()),
        no_training=True, original_2pct=True, independent_confirmation=False, deployment_risk_evaluation="not_run",
        neural_forecaster_evaluation="not_run")
    immutable(PUBLIC/"analysis.json", out)
    print(json.dumps({k: {a: b for a, b in v.items() if a not in ("windows", "cpu_overlap")}
                      for k, v in result.items()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("produce", "evaluate"))
    args = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    key, sites, previous, cpu_key, mobile_key = register()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "produce":
            run(key, sites, mobile_key)
        else:
            evaluate(key, sites, previous, cpu_key, mobile_key)


if __name__ == "__main__":
    main()
