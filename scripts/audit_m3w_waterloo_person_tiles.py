"""Fixed finer-tiling detector diagnostic on the eight Waterloo prefix frames."""

from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_waterloo_ortho as base
from scripts.run_m3w_waterloo_raw_prefix import CHECKED_CLASSES, tv

cv2, np, torch, geo = base.cv2, base.np, base.torch, base.geo


def fine_grid(w, h):
    tw = int(np.ceil(w / (8 - 7 * .2)))
    th = int(np.ceil(h / (5 - 4 * .2)))
    return [(int(x), int(y), tw, th) for y in np.linspace(0, h - th, 5).round()
            for x in np.linspace(0, w - tw, 8).round()]


@torch.inference_mode()
def detect(net, frame):
    grid = fine_grid(frame.shape[1], frame.shape[0])
    boxes, scores, labels = [], [], []
    classes = torch.tensor(CHECKED_CLASSES)
    for start in range(0, len(grid), 3):
        locations = grid[start:start + 3]
        images = [torch.from_numpy(frame[y:y+h, x:x+w, ::-1].copy())
                  .permute(2, 0, 1).float() / 255. for x, y, w, h in locations]
        for out, (x, y, _, _) in zip(net(images), locations, strict=True):
            keep = (out["scores"] >= .35) & torch.isin(out["labels"], classes)
            boxes.append(out["boxes"][keep] + torch.tensor([x, y, x, y]))
            scores.append(out["scores"][keep])
            labels.append(out["labels"][keep])
    box, confidence, label = map(torch.cat, (boxes, scores, labels))
    keep = tv.ops.batched_nms(box, confidence, label, .5)
    return torch.column_stack((box[keep], confidence[keep], label[keep])).numpy().astype(np.float64)


def run():
    parent_path = base.PUBLIC / "analysis.json"
    bridge_path = base.PUBLIC / "causal_bridge.json"
    parent, bridge = [json.loads(p.read_text()) for p in (parent_path, bridge_path)]
    coarse = {row["frame"]: row for row in bridge["current_check"]}
    if bridge["direct_parent_sha256"] != base.sha(parent_path) or any(
        i not in coarse or "not_run" in coarse[i] for i in base.PREFIX
    ):
        raise ValueError("Missing verified eight-frame bridge")
    for path, expected in ((base.VIDEO, parent["source_pair"]["raw_video"]["sha256"]),
                           (base.ORTHO, parent["reference"]["sha256"]),
                           (base.DATA / "dataset.zip", parent["source_pair"]["annotation_archive"]["sha256"])):
        if base.sha(path) != expected:
            raise ValueError(f"Asset changed: {path.name}")
    weights = json.loads((base.PUBLIC.parent / "waterloo_raw_prefix_v1" /
                          "registration.json").read_text())["assets"]["weights"]
    if base.sha(base.WEIGHTS) != weights["sha256"]:
        raise ValueError("Detector weights changed")
    ref = cv2.imread(str(base.ORTHO), cv2.IMREAD_GRAYSCALE)
    cap, net, tracker = cv2.VideoCapture(str(base.VIDEO)), base.detector(), base.StreamingTracks()
    labels = base.early_labels(base.DATA / "dataset.zip")
    records, person_pairs = [], []
    try:
        for frame in range(base.PREFIX[-1] + 1):
            ok, raw = cap.read()
            if not ok:
                raise OSError("Video ended before registered prefix")
            if frame not in coarse:
                continue
            cur = cv2.cvtColor(cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2GRAY)
            if frame == 0:
                first = cur
                initial, fit = geo.fit(first, ref, list(base.PROBES.values()))
                h = initial
                passed = base.ground_pass(fit, geo.holdouts(first, ref, h, base.PROBES))
            else:
                image, fit = geo.fit(cur, first, list(base.PROBES.values()))
                h = initial @ image
                passed = (fit["inliers"] >= 40 and fit["inlier_quadrants"] == 4
                          and fit["support_hull_fraction"] >= .25 and fit["p95_ortho_pixels"] <= 2.
                          and sum(r["peak"] > .30 and r["delta_ortho_pixels"] <= 6.
                                  for r in geo.holdouts(cur, ref, h, base.PROBES)) >= 3)
            if not passed:
                raise ValueError(f"Bridge acceptance changed at frame {frame}")
            rows, _ = tracker.step(frame // 4, detect(net, raw))
            counts, links = base.associate(rows, labels[frame], h)
            if any(counts[k] != coarse[frame][k] for k in ("label_in_image", "label_person")):
                raise ValueError("Label support changed between tile settings")
            person_pairs.extend((track, label, distance) for track, label, distance, person in links if person)
            records.append(dict(frame=frame, **counts))
            print(json.dumps(dict(pid=os.getpid(), frame=frame, **counts)), flush=True)
    finally:
        cap.release()
    counts = Counter()
    for record in records:
        counts.update({k: v for k, v in record.items() if k != "frame"})
    person_tracks = Counter(track for track, _, _ in person_pairs)
    return dict(result_source="fresh_run", frame_records=records, fine_counts=counts,
                coarse_counts=bridge["counts"], grid=[8, 5], overlap=.2, confidence=.35, nms_iou=.5,
                person_repeated_tracks=sum(n > 1 for n in person_tracks.values()),
                person_distances_map_units=[d for _, _, d in person_pairs],
                bridge_sha256=base.sha(bridge_path), detector_weights_sha256=weights["sha256"],
                protocol_sha256=base.sha(base.PUBLIC / "tile_protocol.md"),
                implementation_sha256=base.sha(Path(__file__)),
                future_forecast_errors_read=False, independent_confirmation=False, role_assigned=False)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--replay"]):
        raise SystemExit("usage: audit_m3w_waterloo_person_tiles.py [--replay]")
    replay = bool(sys.argv[1:])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    path = base.PUBLIC / "fine_tiles.json"
    if replay and not path.is_file():
        raise FileNotFoundError("Run the diagnostic before replay")
    start = time.monotonic()
    base.immutable(path, run())
    base.immutable(base.PUBLIC / ("fine_tiles_replay.json" if replay else "fine_tiles_execution.json"),
                   dict(pid=os.getpid(), seconds=time.monotonic() - start, analysis_sha256=base.sha(path)))
