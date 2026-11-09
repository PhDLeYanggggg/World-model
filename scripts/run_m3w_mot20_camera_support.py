"""Audit a new MOT20 camera with a fixed, label-independent 8-to-12 schedule."""

import argparse
import configparser
from email.parser import BytesHeaderParser
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from zipfile import ZipFile
from zlib import crc32

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_acme_camera_roster import roster_counts
from src.data_unification.m3w_public_zip_range import RangeArchive
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap


NAME = "mot20_camera_support_v1"
PUBLIC = ROOT / "outputs/publication_readiness_2026_09" / NAME
PRIVATE = ROOT / "data/stage_cvpr2027_experiments" / NAME
LOCAL = ROOT / "external_data" / "MOT20_camera_support"
URL = "https://motchallenge.net/data/MOT20.zip"
ARCHIVE_LENGTH, ARCHIVE_VERSION = 5028926248, '"12bbf5328-59fa45dc4c7c8"'
SEQUENCE = "MOT20-02"
PREFIX = "MOT20/train/" + SEQUENCE
FPS, STRIDE, OBS, PRED, LENGTH = 25, 10, 8, 12, 2782
RASTER = (1920, 1080)
DISTRACTORS = {2, 7, 8, 12, 6}


def starts():
    return list(range(1, LENGTH - (OBS + PRED - 1) * STRIDE + 1, (OBS + PRED) * STRIDE))


def frame_id(start, step):
    return start + step * STRIDE


class MOTRangeArchive(RangeArchive):
    def _fetch(self, first, last):
        expected = last - first + 1
        if self.total + expected > self.byte_limit:
            raise ValueError("MOT20 byte budget exceeded")
        raw = subprocess.check_output([
            "curl", "--fail", "--silent", "--show-error", "--include", "--suppress-connect-headers",
            "--max-time", "20", "--max-filesize", str(expected), "--range", f"{first}-{last}",
            "--header", "Accept-Encoding: identity", "--proto", "=https", self.url,
        ])
        head, separator, data = raw.partition(b"\r\n\r\n")
        status, _, rest = head.partition(b"\r\n")
        headers = BytesHeaderParser().parsebytes(rest)
        if (not separator or status.split()[1:2] != [b"206"] or len(data) != expected
                or headers.get("ETag") != ARCHIVE_VERSION
                or headers.get("Content-Range") != f"bytes {first}-{last}/{self.length}"
                or headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("MOT20 range response changed")
        self.total += expected
        self.version = ARCHIVE_VERSION
        return data


def archive():
    return MOTRangeArchive(URL, ARCHIVE_LENGTH, byte_limit=192 << 20)


def member(z, stream, name):
    info = z.getinfo(f"{PREFIX}/{name}")
    if info.file_size > 32 << 20:
        raise ValueError("Unexpected member length")
    data = z.read(info)
    if stream.version != ARCHIVE_VERSION:
        raise ValueError("Unexpected archive release")
    return data


def check_clock(text):
    parser = configparser.ConfigParser()
    parser.read_string(text)
    spec = parser["Sequence"]
    if (spec["name"], spec["imDir"], spec["imExt"], spec.getint("frameRate"), spec.getint("seqLength"),
            spec.getint("imWidth"), spec.getint("imHeight")) != (SEQUENCE, "img1", ".jpg", FPS, LENGTH, *RASTER):
        raise ValueError("Camera clock or dimensions changed")


def write_bytes(path, data):
    pending = path.with_suffix(".pending")
    pending.write_bytes(data)
    pending.replace(path)


def prepare(hybrid):
    with archive() as stream, ZipFile(stream) as z:
        check_clock(member(z, stream, "seqinfo.ini").decode("ascii"))
        names = {x.filename for x in z.infolist()}
        if not all(f"{PREFIX}/img1/{f:06d}.jpg" in names for f in range(1, LENGTH + 1)):
            raise ValueError("Published image clock has gaps")
        images = {}
        for start in starts():
            for step in range(OBS):
                frame = frame_id(start, step)
                path = LOCAL / f"{frame:06d}.jpg"
                name = f"img1/{path.name}"
                if not path.is_file():
                    write_bytes(path, member(z, stream, name))
                else:
                    info = z.getinfo(f"{PREFIX}/{name}")
                    body = path.read_bytes()
                    if len(body) != info.file_size or crc32(body) != info.CRC:
                        raise ValueError("Local source frame differs from the pinned ZIP")
                images[str(frame)] = hybrid.sha(path)
            print(f"Bound images through window starting at {start}; {stream.total} bytes transferred", flush=True)
    hybrid.immutable(PUBLIC / "registration.json", dict(
        bindings=bindings(hybrid), archive_url=URL, archive_etag=ARCHIVE_VERSION,
        archive_bytes=ARCHIVE_LENGTH, sequence=SEQUENCE, fps=FPS, sample_stride=STRIDE,
        observations=OBS, predictions=PRED, image_size=list(RASTER), starts=starts(),
        past_image_sha256=images, uses_public_detections=False, uses_gt_for_eligibility=False,
        role="new camera development feasibility; not a formal split",
    ))


def bindings(hybrid):
    key, *_ = hybrid.register()
    return {p.name: hybrid.sha(p) for p in (
        Path(__file__), ROOT / "src/data_unification/m3w_public_zip_range.py",
        ROOT / "scripts/run_m3w_acme_camera_roster.py", Path(hybrid.__file__),
    )} | {"prior_detector_registration": key}


def produce(hybrid, registration, key):
    torch, cv2 = hybrid.torch, hybrid.cv2
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    model = hybrid.cpu.detector()
    if torch.backends.mps.is_available():
        model.backbone = hybrid.Backbone(model.backbone)
        model.roi_heads.box_head = hybrid.BoxHead(model.roi_heads.box_head)
        backend = "cpu_mps"
    else:
        backend = "cpu"
    began = monotonic()
    for i, start in enumerate(starts(), 1):
        path = PRIVATE / f"window_{start:06d}.json"
        if path.exists():
            data = read_window(start, key)
            if data["backend"] != backend:
                raise ValueError("Cannot mix backends on resume")
        else:
            tracker, frames = StreamingTracks(), []
            for step in range(OBS):
                frame = frame_id(start, step)
                p = LOCAL / f"{frame:06d}.jpg"
                if hybrid.sha(p) != registration["past_image_sha256"][str(frame)]:
                    raise ValueError("Camera frame changed after registration")
                image = cv2.imread(str(p))
                if image is None or tuple(image.shape[:2][::-1]) != RASTER:
                    raise ValueError("Unexpected decoded image dimensions")
                detected = hybrid.old.detect(model, image)
                rows, _ = tracker.step(step, detected[detected[:, -1] == 1])
                frames.append(dict(camera_frame=frame, rows=rows.tolist()))
            hybrid.immutable(path, dict(registration=key, backend=backend,
                                        frames=frames, frames_digest=frame_digest(frames)))
        beat = dict(pid=os.getpid(), windows=i, total=len(starts()), backend=backend,
                    seconds=round(monotonic() - began, 2))
        write_bytes(PRIVATE / "heartbeat.json", json.dumps(beat).encode())
        print(beat, flush=True)


def frame_digest(frames):
    return hashlib.sha256(json.dumps(frames, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_window(start, key):
    data = json.loads((PRIVATE / f"window_{start:06d}.json").read_text())
    frames = data["frames"]
    if (data["registration"] != key or data["frames_digest"] != frame_digest(frames)
            or [f["camera_frame"] for f in frames] != [frame_id(start, k) for k in range(OBS)]):
        raise ValueError("Invalid frozen window cache")
    return data


def parse_gt(data):
    raw = np.loadtxt(data.decode("ascii").splitlines(), delimiter=",", ndmin=2)
    if (raw.shape[1] != 9 or not np.isfinite(raw).all() or (raw[:, :2] <= 0).any()
            or (raw[:, 0] > LENGTH).any() or (raw[:, 4:6] <= 0).any()
            or not np.array_equal(raw[:, [0, 1, 6, 7]], np.floor(raw[:, [0, 1, 6, 7]]))
            or len(np.unique(raw[:, :2], axis=0)) != len(raw)):
        raise ValueError("Malformed or duplicate MOT20 reference")
    raw[:, 4:6] += raw[:, 2:4]
    ticks, counts = np.unique(raw[:, 0], return_counts=True)
    by_frame = np.split(raw[np.argsort(raw[:, 0], kind="stable")], np.cumsum(counts)[:-1])
    return dict(zip(ticks.astype(int), by_frame, strict=True))


def match(rows, gt):
    iou = overlap(rows[:, 1:5], gt[:, 2:6])
    iou[iou < .5] = 0
    i, j = linear_sum_assignment(-iou)
    pairs, ignored = set(), set()
    for x, y in zip(i, j, strict=True):
        if iou[x, y] == 0:
            continue
        online, label, cls = int(rows[x, 0]), int(gt[y, 1]), int(gt[y, 7])
        if gt[y, 6] != 0 and cls == 1:
            pairs.add((online, label))
        if cls in DISTRACTORS:
            ignored.add((online, label))
    return pairs, ignored


def window_support(frames, gt, valid_ticks):
    online, matches, ignores = [], [], []
    for item in frames:
        rows = np.asarray(item["rows"]).reshape(-1, 7)
        t = item["camera_frame"]
        online.append(set(rows[:, 0].astype(int)))
        hits, skip = match(rows, gt.get(t, np.empty((0, 9))))
        matches.append(hits)
        ignores.append(skip)
    start = frames[0]["camera_frame"]
    # roster_counts uses tick units at stride four, so rescale to its grid.
    labels = set.union(*({ref for _, ref in f} for f in matches))
    local = {label: {(t - start) // STRIDE * 4 for t in valid_ticks[label] if (t - start) % STRIDE == 0}
             for label in labels}
    full = roster_counts(online, matches, local, 0)
    ignored = {i for i, _ in set.intersection(*ignores)}
    cleaned = roster_counts([ids - ignored for ids in online], matches, local, 0)
    past = {frame_id(start, k) for k in range(OBS)}
    future = {frame_id(start, k) for k in range(OBS, OBS + PRED)}
    references = sum((past | future) <= ticks for ticks in valid_ticks.values())
    framewise = set.intersection(*({label for _, label in pairs} for pairs in matches))
    complete_framewise = sum(future <= valid_ticks[label] for label in framewise)
    return dict(strict=full, ignore_only_diagnostic=cleaned, ignored=len(ignored),
                reference_complete=references, complete_framewise=complete_framewise)


def evaluate(hybrid, key):
    windows = [read_window(start, key) for start in starts()]
    backends = {w["backend"] for w in windows}
    if len(backends) != 1:
        raise ValueError("Mixed inference backends")
    label = LOCAL / "gt.txt"
    if not label.is_file():
        with archive() as stream, ZipFile(stream) as z:
            write_bytes(label, member(z, stream, "gt/gt.txt"))
    else:
        with archive() as stream, ZipFile(stream) as z:
            info = z.getinfo(f"{PREFIX}/gt/gt.txt")
            body = label.read_bytes()
            if stream.version != ARCHIVE_VERSION or len(body) != info.file_size or crc32(body) != info.CRC:
                raise ValueError("Reference snapshot changed")
    gt = parse_gt(label.read_bytes())
    ticks = {}
    for t, rows in gt.items():
        for i in rows[(rows[:, 6] != 0) & (rows[:, 7] == 1), 1].astype(int):
            ticks.setdefault(i, set()).add(t)
    result = {"last_box": [], "causal_transport": []}
    for w in windows:
        frames = w["frames"]
        for name, source in (("last_box", frames), ("causal_transport", hybrid.old.reassociate(frames))):
            result[name].append(dict(start=frames[0]["camera_frame"], **window_support(source, gt, ticks)))
    hybrid.immutable(PUBLIC / "analysis.json", dict(
        registration_sha256=key, backend=next(iter(backends)), label_sha256=hybrid.sha(label),
        result_source="fresh_run", independent_confirmation=False, forecast_scoring="not_run",
        reserved_outcomes_opened=False, original_2pct=True, windows=result,
    ))
    for name, rows in result.items():
        counts = {mode: {field: sum(row[mode][field] for row in rows) for field in rows[0][mode]}
                  for mode in ("strict", "ignore_only_diagnostic")}
        print(json.dumps(dict(tracker=name, **counts, reference_complete=sum(r["reference_complete"] for r in rows),
                              complete_framewise=sum(r["complete_framewise"] for r in rows))))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("prepare", "produce", "evaluate"))
    phase = cli.parse_args().phase
    from scripts import run_m3w_acme_hybrid_detector as hybrid

    for d in (PUBLIC, PRIVATE, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == "prepare":
            prepare(hybrid)
            return
        path = PUBLIC / "registration.json"
        registration = json.loads(path.read_text())
        if registration["bindings"] != bindings(hybrid) or registration["starts"] != starts():
            raise ValueError("Frozen registration changed")
        if phase == "produce":
            produce(hybrid, registration, hybrid.sha(path))
        else:
            evaluate(hybrid, hybrid.sha(path))


if __name__ == "__main__":
    main()
