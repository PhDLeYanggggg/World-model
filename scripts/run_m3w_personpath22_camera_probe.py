"""Verify released video timing and online roster coverage on one training clip."""

import argparse
from collections import defaultdict
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile
from zlib import crc32

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_personpath22_sources as intake
from scripts.run_m3w_acme_camera_roster import roster_counts
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap

LOCAL = intake.LOCAL
PRIVATE = ROOT / "data/stage_cvpr2027_experiments/personpath22_camera_probe_v1"
DECODER = ROOT / "data/stage_cvpr2027_experiments/media_decode_runtime"


def schedule(fps, length):
    return [[round(k * fps / 5) for k in range(t, t + 40, 2)]
            for t in range(0, round((length - 1) * 5 / fps) - 37, 40)]


def choose(audit):
    return min(uid for uid, row in audit["training_support"].items()
               if row["clock_ok"] and row["in_video_zip"] and schedule(row["fps"], row["frames"]))


def get_video(uid):
    name, *spec = intake.VIDEO
    path = LOCAL / uid
    with intake.PinnedArchive((f"{intake.BASE}/{name}", *spec), 144 << 20) as archive:
        archive.block_size = 4 << 20
        with ZipFile(archive) as z:
            info = z.getinfo(uid)
            if not path.is_file():
                intake.write_atomic(path, z.read(info))
            body = path.read_bytes()
            if len(body) != info.file_size or crc32(body) != info.CRC:
                raise ValueError("Video member differs from the pinned catalog")
    return path


def decoder():
    sys.path.insert(0, str(DECODER))
    import av
    return av


def timestamps(path, fps, length):
    with decoder().open(str(path)) as media:
        stream = media.streams.video[0]
        stream.codec_context.thread_count = 4
        decoded_fps = float(stream.average_rate)
        size = [stream.width, stream.height]
        times = [float(frame.pts * frame.time_base) for frame in media.decode(video=0)]
    if (abs(decoded_fps - fps) > 1e-6 or len(times) != length
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Decoded video metadata differs from annotation clock")
    max_residual_ms = max(abs(t - times[0] - i / fps) for i, t in enumerate(times)) * 1000
    if max_residual_ms > 1:
        raise ValueError(f"Video PTS cannot be bound to a constant cadence: {max_residual_ms:.3f} ms")
    return size, max_residual_ms


def produce(h, path, reg, key):
    cv2, torch = h.cv2, h.torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    model = h.cpu.detector()
    backend = "cpu"
    if torch.backends.mps.is_available():
        model.backbone = h.Backbone(model.backbone)
        model.roi_heads.box_head = h.BoxHead(model.roi_heads.box_head)
        backend = "cpu_mps"
    began = time.monotonic()
    with decoder().open(str(path)) as media:
        media.streams.video[0].codec_context.thread_count = 4
        decoded = enumerate(media.decode(video=0))
        for i, schedule_ in enumerate(reg["native_frame_schedule"], 1):
            out = PRIVATE / f"window_{i}.json"
            if out.is_file():
                load_window(i, schedule_, key, backend)
            else:
                past = schedule_[:8]
                frames, tracker = [], StreamingTracks()
                for wanted in past:
                    f, image = next(decoded)
                    while f < wanted:
                        f, image = next(decoded)
                    if f != wanted:
                        raise OSError("Camera frame index changed")
                    detected = h.old.detect(model, image.to_ndarray(format="bgr24"))
                    rows, _ = tracker.step(len(frames), detected[detected[:, -1] == 1])
                    frames.append(dict(camera_frame=f, rows=rows.tolist()))
                h.immutable(out, dict(registration=key, backend=backend, frames=frames))
            beat = dict(pid=os.getpid(), windows=i, total=len(reg["native_frame_schedule"]),
                        backend=backend, seconds=round(time.monotonic() - began, 3))
            intake.write_atomic(PRIVATE / "heartbeat.json", json.dumps(beat).encode())
            print(beat, flush=True)


def load_window(i, frames, key, backend=None):
    data = json.loads((PRIVATE / f"window_{i}.json").read_text())
    if (data["registration"] != key or (backend is not None and data["backend"] != backend)
            or [f["camera_frame"] for f in data["frames"]] != frames[:8]):
        raise ValueError("Camera window or backend changed")
    return data


def coverage(frames, future, by_frame, id_ticks):
    online, match = [], []
    for frame in frames:
        rows = np.asarray(frame["rows"]).reshape(-1, 7)
        targets = by_frame.get(frame["camera_frame"], [])
        boxes = np.asarray([b for _, b in targets]).reshape(-1, 4)
        ious = overlap(rows[:, 1:5], boxes)
        ious[ious < .5] = 0
        ix, jx = linear_sum_assignment(-ious)
        match.append({(int(rows[i, 0]), targets[j][0]) for i, j in zip(ix, jx, strict=True) if ious[i, j] > 0})
        online.append(set(rows[:, 0].astype(int)))
    translated = {label: {32 + k * 4 for k, f in enumerate(future) if f in ticks}
                  for label, ticks in id_ticks.items()}
    return roster_counts(online, match, translated, 0)


def evaluate(h, reg, key):
    caches = [load_window(i, q, key) for i, q in enumerate(reg["native_frame_schedule"], 1)]
    label_zip = intake.read_small(intake.ANNOTATIONS)
    if intake.sha(label_zip) != reg["annotation_archive_sha256"]:
        raise ValueError("Annotation ZIP changed")
    with ZipFile(io.BytesIO(label_zip)) as z:
        data = json.loads(z.read(f"anno_amodal_2022/{reg['uid']}.json"))
    by_frame, id_ticks = defaultdict(list), defaultdict(set)
    for row in data["entities"]:
        f, ident = row["blob"]["frame_idx"], row["id"]
        if any(row["labels"].get(x, 0) for x in ("reflection", "person in vehicle", "person on open vehicle")):
            continue
        x, y, w, height = row["bb"]
        if w > 0 and height > 0:
            by_frame[f].append((ident, (x, y, x + w, y + height)))
            id_ticks[ident].add(f)
    windows = [coverage(h.old.reassociate(c["frames"]), q[8:], by_frame, id_ticks)
               for c, q in zip(caches, reg["native_frame_schedule"], strict=True)]
    result = dict(registration_sha256=key, windows=windows, scope="one training camera diagnostic",
                  totals={k: sum(row[k] for row in windows) for k in windows[0]},
                  forecasts_scored=False, official_test_read=False, independent_confirmation=False)
    h.immutable(intake.OUT / "camera_probe.json", result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "produce", "evaluate"))
    args = p.parse_args()
    from scripts import run_m3w_acme_hybrid_detector as h

    for d in (LOCAL, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    audit_path = intake.OUT / "analysis.json"
    audit = json.loads(audit_path.read_text())
    uid = choose(audit)
    row = audit["training_support"][uid]
    reg_path = intake.OUT / "camera_registration_v2.json"
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bindings = dict(audit_sha256=h.sha(audit_path), runner_sha256=h.sha(Path(__file__)),
                        detector_registration=h.register()[0], uid=uid,
                        native_frame_schedule=schedule(row["fps"], row["frames"]),
                        annotation_archive_sha256=audit["annotation_archive_sha256"],
                        decoder_version=decoder().__version__)
        if args.phase == "prepare":
            path = get_video(uid)
            size, pts = timestamps(path, row["fps"], row["frames"])
            h.immutable(reg_path, dict(**bindings, video_sha256=h.sha(path), image_size=size,
                                       max_pts_residual_ms=pts, label_use="evaluation only"))
            print(json.dumps(dict(uid=uid, frames=row["frames"], fps=row["fps"], image_size=size, pts_ms=pts)))
        else:
            reg = json.loads(reg_path.read_text())
            if any(reg[k] != v for k, v in bindings.items()):
                raise ValueError("Frozen camera registration changed")
            if args.phase == "produce":
                path = LOCAL / uid
                if h.sha(path) != reg["video_sha256"]:
                    raise ValueError("Frozen camera video changed")
                produce(h, path, reg, h.sha(reg_path))
            else:
                evaluate(h, reg, h.sha(reg_path))


if __name__ == "__main__":
    main()
