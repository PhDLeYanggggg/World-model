"""Compare fixed detector/geometry variants on one exposed ACME camera."""

import hashlib
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
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import nms

from scripts.audit_m3w_acme_prefix_sources import track_clocks
from scripts.run_m3w_acme_video_prefix import VIDEO, VIDEO_SHA, PRIVATE, detect, detector, immutable, sha
from src.world_model.m3w_streaming_boxes import StreamingTracks


PUBLIC = ROOT / "outputs/publication_readiness_2026_09/acme_camera_repair_v1"
ASSETS = ROOT / "external_data/ACME_source_audit"
FLOOR = "01d516fe9990f0912c40b4b19c243e7d11b9c649a099bbed78478c96dbabb268"
ANNOTATIONS = "9d045f11d31374a2a4ec19013e13b5bd5aa8a123633f4bf5eb12503af3f5ecc8"
STARTS = (0, 1600)
STRIDE, COUNT = 4, 32
CROPS = ((740, 465, 1076, 801), (682, 545, 1018, 881))


def feet(boxes):
    return np.column_stack(((boxes[:, 0]+boxes[:, 2])/2, boxes[:, 3]))


def floor_filter(rows, polygons):
    return rows[np.array([any(cv2.pointPolygonTest(p, (float(x), float(y)), True) >= -12
                             for p in polygons) for x, y in feet(rows)], dtype=bool)]


def associate_labels(tracks, points):
    if not len(tracks) or not len(points):
        return []
    boxes = tracks[:, 1:5]
    xy = np.array([point for _, point in points])
    dist = np.linalg.norm(feet(boxes)[:, None, :] - xy[None, :, :], axis=-1)
    legal = dist <= np.maximum(12, .22*(boxes[:, 3]-boxes[:, 1]))[:, None]
    i, j = linear_sum_assignment(np.where(legal, dist, 1e6))
    return [(int(tracks[a, 0]), points[b][0]) for a, b in zip(i, j) if legal[a, b]]


@torch.inference_mode()
def distant_boxes(net, frame):
    images = [torch.from_numpy(frame[y:y2, x:x2, ::-1].copy()).permute(2, 0, 1).float()/255
              for x, y, x2, y2 in CROPS]
    rows = []
    for (x, y, x2, y2), result in zip(CROPS, net(images), strict=True):
        boxes = result["boxes"]
        # A truncated box does not provide a trustworthy floor anchor.
        keep = ((result["labels"] == 1) & (result["scores"] >= .35) &
                (boxes[:, 0] > 2) & (boxes[:, 1] > 2) &
                (boxes[:, 2] < x2-x-2) & (boxes[:, 3] < y2-y-2))
        box = boxes[keep] + torch.tensor([x, y, x, y])
        rows.append(torch.column_stack((box, result["scores"][keep], result["labels"][keep])))
    return torch.cat(rows).numpy().astype(np.float64)


def union(a, b):
    rows = np.concatenate((a, b))
    keep = nms(torch.from_numpy(rows[:, :4]), torch.from_numpy(rows[:, 4]), .5)
    return rows[keep.numpy()]


def label_rows():
    file = ASSETS/"umich.toml"
    if sha(file) != ANNOTATIONS:
        raise ValueError("Annotation revision changed")
    points = {frame: [] for start in STARTS for frame in range(start, start+STRIDE*COUNT, STRIDE)}
    for agent, track in tomllib.loads(file.read_text()).items():
        _, clock, _, _ = track_clocks(track)
        if clock:
            for frame, xy in zip(clock, track["trajectories"], strict=True):
                if frame in points:
                    points[frame].append((agent, xy))
    return points


def cached_detection(frame, raw, net):
    cache = PRIVATE/f"frame_{raw:05d}.npz"
    if raw < COUNT*STRIDE and cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            values = stored["detections"]
            token = json.loads(str(stored["binding"]))
            digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            if token["raw_frame"] != raw or token["frame_sha256"] != digest(frame) or str(stored["digest"]) != digest(values):
                raise ValueError("Original camera cache mismatch")
            return values
    rows = detect(net, frame)
    return rows[rows[:, -1] == 1]


def run_segment(net, video, start, polygons, truth):
    states = {arm: StreamingTracks() for arm in ("original", "floor_only", "floor_and_far_crop")}
    consecutive = {arm: {} for arm in states}
    metrics = {arm: dict(detections=0, point_matches=0, eight_step_tracks=0,
                         eight_step_same_label=0, frames=[]) for arm in states}
    label_streak = {}
    total_label = total_label8 = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range((COUNT-1)*STRIDE+1):
        ok, frame = video.read()
        raw = start+offset
        if not ok or round(video.get(cv2.CAP_PROP_POS_FRAMES)) != raw+1:
            raise OSError(f"Camera decode did not reach frame {raw}")
        if offset % STRIDE:
            continue
        k = offset//STRIDE
        base = cached_detection(frame, raw, net)
        floor = floor_filter(base, polygons)
        extra = floor_filter(distant_boxes(net, frame), polygons)
        inputs = dict(original=base, floor_only=floor, floor_and_far_crop=union(floor, extra))
        points = truth[raw]
        total_label += len(points)
        label_streak = {ident: label_streak.get(ident, 0)+1 for ident, _ in points}
        total_label8 += sum(value >= 8 for value in label_streak.values())
        for arm, values in inputs.items():
            rows, forecast = states[arm].step(k, values)
            match = associate_labels(rows, points)
            prev = consecutive[arm]
            consecutive[arm] = {(tid, lab): prev.get((tid, lab), 0)+1 for tid, lab in match}
            stable = sum(n >= 8 for n in consecutive[arm].values())
            m = metrics[arm]
            m["detections"] += len(rows)
            m["point_matches"] += len(match)
            m["eight_step_tracks"] += len(forecast)
            m["eight_step_same_label"] += stable
            m["frames"].append([raw, len(rows), len(match), len(forecast), stable])
        if k % 8 == 0 or k == COUNT-1:
            beat = dict(pid=os.getpid(), start=start, sampled=k+1, total=COUNT)
            (PRIVATE/"repair_heartbeat.json").write_text(json.dumps(beat)+"\n")
            print(beat, flush=True)
    return dict(start=start, label_points=total_label, eligible_label_histories=total_label8, variants=metrics)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    geo = ASSETS/"umich_floor_regions.json"
    if sha(geo) != FLOOR or sha(VIDEO) != VIDEO_SHA:
        raise ValueError("Video or producer floor polygons changed")
    polygons = [np.asarray(p["points"], dtype=np.float32) for p in json.loads(geo.read_text())
                if p["tag_id"] in (2, 5)]
    PUBLIC.mkdir(parents=True, exist_ok=True)
    immutable(PUBLIC/"registration.json", dict(
        source="one_exposed_Michigan_camera", video_sha256=VIDEO_SHA, label_sha256=ANNOTATIONS,
        floor_sha256=FLOOR, starts=STARTS, stride=STRIDE, samples_each=COUNT, crops=CROPS,
        code_sha256=sha(Path(__file__)), scope="detector_support_only_no_future_forecast_scoring",
        anchor="box_bottom_center", match_radius="max(12px,0.22*height)",
    ))
    truth = label_rows()
    model = detector()
    video = cv2.VideoCapture(str(VIDEO))
    if not video.isOpened():
        raise OSError("Cannot open pinned camera")
    try:
        results = [run_segment(model, video, s, polygons, truth) for s in STARTS]
    finally:
        video.release()
    immutable(PUBLIC/"analysis.json", dict(
        segments=results, result_source="fresh_run", evaluated_future_labels=False,
        same_camera_engineering_only=True, independent_confirmation=False,
    ))


if __name__ == "__main__":
    main()
