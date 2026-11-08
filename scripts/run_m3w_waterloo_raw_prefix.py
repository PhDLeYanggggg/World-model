"""Verify causal raw-video trajectories on one public Waterloo sample."""

import argparse
from collections import Counter
import fcntl
import hashlib
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
import cv2
import numpy as np
import torch
import torchvision as tv
from src.evaluation.m3w_waterloo_intake import sha
from src.world_model.m3w_streaming_boxes import StreamingTracks

DATA = ROOT/"external_data/Waterloo_WISE_annotations/roundabout_sample"
PRIVATE = ROOT/"data/stage_cvpr2027_experiments/waterloo_raw_prefix_v1"
PUBLIC = ROOT/"outputs/publication_readiness_2026_09/waterloo_raw_prefix_v1"
VIDEO = DATA/"DJI_0669.MP4"
LOG = DATA/"DJI_0669.tlgx"
WEIGHTS = ROOT/"external_data/pretrained/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
BASE = "https://wiselab.uwaterloo.ca/multi-agent-roundabout-traffic/raw_videos/Sample%20Dataset/"
CHECKED_CLASSES = (1, 2, 3, 4, 6, 8)
STRIDE, FRAMES, VERIFY = 4, 64, 24


def immutable(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f"Existing immutable artifact differs: {path.name}")
        return
    tmp = path.with_suffix(".pending")
    tmp.write_text(data)
    tmp.replace(path)


def asset(path, url, byte_size=None):
    if byte_size is not None and path.stat().st_size != byte_size:
        raise ValueError(f"Unexpected publisher length: {path.name}")
    return dict(url=url, bytes=path.stat().st_size, sha256=sha(path))


def register():
    published = {
        "video": asset(VIDEO, BASE+VIDEO.name, 3759114590),
        "tracking_log": asset(LOG, BASE+LOG.name, 7949400),
        "weights": asset(WEIGHTS, "https://download.pytorch.org/models/"+WEIGHTS.name),
    }
    if not published["weights"]["sha256"].startswith("907ea3f9"):
        raise ValueError("Torchvision model hash prefix changed")
    binding = dict(assets=published, stride=STRIDE, sampled_frames=FRAMES, cold_prefix=VERIFY,
        detector="Torchvision COCO MobileNetV3-Large Faster R-CNN FPN 320 checkpoint",
        detector_input_short_side=512, tile_grid=[4,3], tile_overlap=.2, confidence=.35, nms_iou=.5,
        classes=CHECKED_CLASSES, cpu_threads=4, workers=0, target_labels_opened=False,
        implementation={p.name:sha(p) for p in (Path(__file__), ROOT/"src/world_model/m3w_streaming_boxes.py")})
    immutable(PUBLIC/"registration.json", binding)
    return sha(PUBLIC/"registration.json")


def detector():
    make = tv.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
    net = make(weights=None, weights_backbone=None).eval()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.transform.min_size = (512,)
    net.transform.max_size = 768
    return net


def tiles(w, h):
    tw = int(np.ceil(w/(4-3*.2)))
    th = int(np.ceil(h/(3-2*.2)))
    return [(int(x), int(y), tw, th) for y in np.linspace(0, h-th, 3).round()
            for x in np.linspace(0, w-tw, 4).round()]


@torch.inference_mode()
def detect(net, frame):
    h, w = frame.shape[:2]
    boxes, scores, labels = [], [], []
    grid = tiles(w, h)
    for start in range(0, len(grid), 3):
        locations = grid[start:start+3]
        inputs = [torch.from_numpy(frame[y:y+hh,x:x+ww,::-1].copy()).permute(2,0,1).float()/255.
                  for x,y,ww,hh in locations]
        for out,(x,y,_,_) in zip(net(inputs), locations, strict=True):
            select = (out["scores"] >= .35) & torch.isin(out["labels"], torch.tensor(CHECKED_CLASSES))
            boxes.append(out["boxes"][select]+torch.tensor([x,y,x,y]))
            scores.append(out["scores"][select])
            labels.append(out["labels"][select])
    box, confidence, label = map(torch.cat, (boxes,scores,labels))
    keep = tv.ops.batched_nms(box,confidence,label,.5)
    return torch.column_stack((box[keep],confidence[keep],label[keep])).numpy().astype(np.float64)


def file_digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stream(count, reg, verify, resume):
    net = detector()
    video = cv2.VideoCapture(str(VIDEO))
    if not video.isOpened():
        raise OSError("Cannot open publisher MP4")
    facts = dict(width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 container_fps=video.get(cv2.CAP_PROP_FPS), container_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    state, all_rows = StreamingTracks(), []
    frame_reports, forecasts = [], []
    try:
        for i in range((count-1)*STRIDE+1):
            ok, frame = video.read()
            if not ok:
                raise OSError("Unexpected video truncation")
            if i % STRIDE:
                continue
            k = i//STRIDE
            prefix = dict(frame=i, registration_sha256=reg, image_sha256=file_digest_array(frame))
            cache = PRIVATE/f"frame_{i:05}.npz"
            if cache.exists() and not verify:
                if not resume:
                    raise FileExistsError("Existing sample cache requires --resume")
                with np.load(cache,allow_pickle=False) as z:
                    d = z["detections"]
                    if json.loads(str(z["binding"])) != prefix or str(z["digest"]) != file_digest_array(d):
                        raise ValueError("Cached past frame or detection payload changed")
            else:
                d = detect(net, frame)
                if not verify:
                    tmp = cache.with_suffix(".pending.npz")
                    np.savez(tmp,detections=d,binding=json.dumps(prefix),digest=file_digest_array(d))
                    tmp.replace(cache)
            rows, pred = state.step(k, d)
            all_rows.extend(rows.tolist())
            forecasts.extend((k,int(ident),cv,ols) for ident,cv,ols in pred)
            digest = hashlib.sha256()
            for value in (d, rows, np.asarray([np.stack((cv,ols)) for _,cv,ols in pred])):
                digest.update(file_digest_array(value).encode())
            frame_reports.append(dict(raw_frame=i, pixels=prefix["image_sha256"], causal_output=digest.hexdigest(),
                                      detections=len(rows), forecast_queries=len(pred)))
            if k % 8 == 0 or k == count-1:
                msg = dict(pid=os.getpid(), frame=i, sampled=k+1, total=count, detections=len(rows),
                           state="verified_prefix" if verify else "past_checkpoint")
                (PRIVATE/"heartbeat.json").write_text(json.dumps(msg)+"\n")
                print(json.dumps(msg),flush=True)
    finally:
        video.release()
    by_cls = Counter(str(int(r[-1])) for r in all_rows)
    pred_classes = {int(r[0]):str(int(r[-1])) for r in all_rows}
    queries_by_cls = Counter(pred_classes[key] for _,key,_,_ in forecasts)
    if not verify:
        p = PRIVATE/"causal_forecast_bank.npz"
        temp = p.with_suffix(".pending.npz")
        np.savez(temp, index=np.asarray([(k,agent) for k,agent,_,_ in forecasts],dtype=int).reshape(-1,2),
                 path=np.asarray([np.stack((cv,ols)) for _,_,cv,ols in forecasts]).reshape(-1,2,12,2))
        temp.replace(p)
    return dict(video=facts, frame_receipts=frame_reports, detection_rows=sum(by_cls.values()),
                tracks=state.next_id, detector_classes=by_cls, forecast_classes=queries_by_cls,
                forecasts=len(forecasts), frames_with_forecasts=sum(r["forecast_queries"]>0 for r in frame_reports))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("build","verify"))
    p.add_argument("--resume",action="store_true")
    a = p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg = register()
        verify = a.phase == "verify"
        result = stream(VERIFY if verify else FRAMES, reg, verify, a.resume)
        if verify:
            saved = json.loads((PUBLIC/"analysis.json").read_text())
            if result["frame_receipts"] != saved["frame_receipts"][:VERIFY] or result["video"] != saved["video"]:
                raise ValueError("Cold raw-video prefix replay disagrees")
            immutable(PUBLIC/"verification.json", dict(exact=True, compared_sampled_frames=VERIFY,
                      compared_forecasts=result["forecasts"], parent_analysis_sha256=sha(PUBLIC/"analysis.json"),
                      independent_confirmation=False))
        else:
            immutable(PUBLIC/"analysis.json",dict(**result, registration_sha256=reg, role="engineering_only",
                      raw_video_causal_inputs=True, track_label_alignment=False, model_scored=False,
                      original_rule_unchanged=True, no_external_roles_assigned=True,
                      version=dict(torch=torch.__version__,torchvision=tv.__version__,opencv=cv2.__version__)))
        immutable(PUBLIC/f"{a.phase}_execution.json",dict(pid=os.getpid(), seconds=time.monotonic()-start))


if __name__ == "__main__":
    main()
