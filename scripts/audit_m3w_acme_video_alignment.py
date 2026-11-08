"""Check direct pixel/clock compatibility on the exposed ACME camera probe."""

import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_acme_prefix_sources import track_clocks


def contained_pairs(points, boxes):
    if not len(points) or not len(boxes):
        return 0
    inside = ((points[:, None, :] >= boxes[None, :, :2]) &
              (points[:, None, :] <= boxes[None, :, 2:])).all(-1)
    i, j = linear_sum_assignment(inside, maximize=True)
    return int(inside[i, j].sum())


def main():
    path = ROOT / "outputs/publication_readiness_2026_09/acme_prefix_intake_v1"
    probe = json.loads((path / "video_probe.json").read_text())
    analysis = json.loads((path / "analysis.json").read_text())
    receipt = (ROOT / "external_data/ACME_source_audit/umich.toml").read_bytes()
    if hashlib.sha256(receipt).hexdigest() != analysis["sites"]["umich"]["sample_sha256"]:
        raise ValueError("Annotation revision changed")
    points = {r["raw_frame"]: [] for r in probe["frames"]}
    for track in tomllib.loads(receipt.decode()).values():
        _, frames, _, _ = track_clocks(track)
        for i, xy in zip(frames, track["trajectories"] if frames else (), strict=True):
            if i in points:
                points[i].append(xy)
    counts = dict(frames=len(points), frames_with_labels=0, label_rows=0,
                  detection_rows=0, direct_contained_pairs=0)
    distances = []
    private = ROOT / "data/stage_cvpr2027_experiments/acme_video_prefix_v1"
    for frame in probe["frames"]:
        i = frame["raw_frame"]
        with np.load(private / f"frame_{i:05d}.npz", allow_pickle=False) as z:
            detections = z["detections"]
            binding = json.loads(str(z["binding"]))
            digest = hashlib.sha256(np.ascontiguousarray(detections).tobytes()).hexdigest()
            if (binding != {k: frame[k] for k in ("raw_frame", "frame_sha256", "registration")} or
                    str(z["digest"]) != digest or len(detections) != frame["detections"]):
                raise ValueError("Cached detections disagree with camera probe")
        xy = np.asarray(points[i]).reshape(-1, 2)
        boxes = detections[:, :4]
        counts["frames_with_labels"] += bool(len(xy))
        counts["label_rows"] += len(xy)
        counts["detection_rows"] += len(boxes)
        counts["direct_contained_pairs"] += contained_pairs(xy, boxes)
        if len(xy) and len(boxes):
            centers = (boxes[:, :2] + boxes[:, 2:])/2
            distances.extend(np.linalg.norm(xy[:, None] - centers[None], axis=-1).min(1).tolist())
    result = dict(
        **counts, median_nearest_center_distance_px=float(np.median(distances)) if distances else None,
        role="exposed_camera_probe_geometry_only",
        tested_mapping="equal_frame_number_identity_pixels",
        transform_or_offset_fitted=False, raw_frame_annotation_alignment_verified=False,
        forecast_accuracy="not_run",
    )
    (path / "video_probe_alignment.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
