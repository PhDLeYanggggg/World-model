"""Measure whether past-only foreground motion recovers blurred pedestrians."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from scripts.run_m3w_acme_camera_repair import (
    ASSETS, COUNT, FLOOR, PUBLIC, STARTS, STRIDE, VIDEO, VIDEO_SHA,
    associate_labels, floor_filter, immutable, label_rows, sha,
)
from src.world_model.m3w_streaming_boxes import StreamingTracks


def foreground_boxes(mask):
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    n, _, regions, _ = cv2.connectedComponentsWithStats(cleaned)
    rows = []
    for x, y, w, h, area in regions[1:n]:
        if (18 <= h <= 340 and 6 <= w <= 180 and .12 <= w/h <= 1.4 and
                area >= 60 and x > 0 and y > 0 and y+h < mask.shape[0] and
                x+w < mask.shape[1]):
            rows.append((x, y, x+w, y+h, 1., 1.))
    return np.asarray(rows, dtype=float).reshape(-1, 6)


def probe(cap, start, polygons, labels):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start-80))
    first = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=160, varThreshold=24, detectShadows=False)
    tracker = StreamingTracks()
    streak = {}
    output = []
    for i in range(first, start+(COUNT-1)*STRIDE+1):
        ok, frame = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != i+1:
            raise OSError(f"Cannot read exact frame {i}")
        mask = subtractor.apply(frame, learningRate=.015)
        if i < start or (i-start) % STRIDE:
            continue
        raw = foreground_boxes(mask)
        rows, forecast = tracker.step((i-start)//STRIDE, floor_filter(raw, polygons))
        match = associate_labels(rows, labels[i])
        streak = {pair: streak.get(pair, 0)+1 for pair in match}
        output.append([i, len(rows), len(match), len(forecast), sum(n >= 8 for n in streak.values())])
    return dict(start=start, frames=output, detections=sum(x[1] for x in output),
                point_matches=sum(x[2] for x in output), eight_step_tracks=sum(x[3] for x in output),
                eight_step_same_label=sum(x[4] for x in output))


def main():
    geo = ASSETS/"umich_floor_regions.json"
    if sha(geo) != FLOOR or sha(VIDEO) != VIDEO_SHA:
        raise ValueError("Input video or floor polygons changed")
    floor = [np.asarray(p["points"], dtype=np.float32) for p in json.loads(geo.read_text())
             if p["tag_id"] in (2, 5)]
    labels = label_rows()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise OSError("Cannot open camera")
    try:
        segments = [probe(cap, start, floor, labels) for start in STARTS]
    finally:
        cap.release()
    immutable(PUBLIC/"foreground_probe.json", dict(
        code_sha256=sha(Path(__file__)), opencv=cv2.__version__, segments=segments,
        result_source="fresh_run", same_camera_exploratory=True, annotations_in_tracking=False,
        evaluated_future_labels=False, forecast_evaluation="not_run",
    ))
    print(json.dumps([{k: v for k, v in s.items() if k != "frames"} for s in segments]))


if __name__ == "__main__":
    main()
