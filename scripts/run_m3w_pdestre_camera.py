"""Matched past-only camera support pilot across P-DESTRE acquisition dates."""

import argparse
from collections import Counter
import fcntl
import io
import os
from pathlib import Path
import platform
import sys
import threading
from http.server import ThreadingHTTPServer

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from scripts import audit_m3w_pdestre as source
from scripts import probe_m3w_pdestre_video as video
from scripts import run_m3w_sdd_camera_support as detectors
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap

OUT = source.OUT.parent/"pdestre_camera_support_v1"
LOCAL = source.ROOT/"data/stage_cvpr2027_experiments/pdestre_camera_support_v1"
ARMS = ("rapid", "mobilenet")
MAX_VIDEO = 128*1024**2
IOU = .5


def plan(audit, index):
    out = {}
    for clip, r in sorted(audit["recordings"].items()):
        day = "-".join(clip.split("-")[:3])
        anchor = r["strides"]["1"]["first_selected_anchor"]
        if day not in out and anchor is not None:
            out[day] = dict(clip=clip, start=anchor-7, video=index["clips"][clip][".mp4"],
                            annotations=r["source"], annotation_sha256=r["sha256"])
    return out


def register():
    a, i = [source.OUT/x for x in ("analysis.json", "range_index.json")]
    s, index = source.load(a), source.load(i)
    if (s["index_sha256"] != source.digest(i.read_bytes()) or s["roles_assigned"]
            or (index["url"], index["size"], index["etag"]) != (source.URL, source.SIZE, source.ETAG)
            or len(s["shared_id_components"]) != 1):
        raise ValueError("Require the original linked source audit")
    d = dict(audit_sha256=source.digest(a.read_bytes()), code_sha256=source.digest(Path(__file__).read_bytes()),
        stream_sha256=source.digest(Path(video.__file__).read_bytes()), detector=detectors.detector_contract(),
        selection="lexical first history-eligible clip per date, earliest native nonoverlapping multi-agent query",
        obs=8, pred=12, stride=1, match_iou=IOU, max_video_bytes_per_clip=MAX_VIDEO, windows=plan(s, index),
        one_linked_source=True, training=False, future_features=False, independent_confirmation=False)
    source.save(OUT/"registration.json", d)
    return d, source.digest((OUT/"registration.json").read_bytes())


def observations(item, predictors):
    stream = video.MemberStream(item["video"], MAX_VIDEO)
    frames, paths = [], {}
    trackers = {a: StreamingTracks() for a in predictors}
    with ThreadingHTTPServer(("127.0.0.1", 0), stream.handler()) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        camera = cv2.VideoCapture()
        try:
            if not camera.open(f"http://127.0.0.1:{server.server_port}/clip.mp4", cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 45000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 45000]):
                raise OSError("Registered MP4 could not be opened")
            meta = dict(fps=camera.get(cv2.CAP_PROP_FPS), frames=int(camera.get(cv2.CAP_PROP_FRAME_COUNT)))
            if meta["frames"] < item["start"]+20:
                raise ValueError("Registered window exceeds video extent")
            if item["start"] and not camera.set(cv2.CAP_PROP_POS_FRAMES, item["start"]):
                raise OSError("Could not seek to registered camera frame")
            for j in range(8):
                ok, image = camera.read()
                f = item["start"]+j
                if not ok or int(camera.get(cv2.CAP_PROP_POS_FRAMES)) != f+1:
                    raise OSError(f"Camera did not decode registered frame {f}")
                row = dict(frame=f, sha256=source.digest(image.tobytes()), boxes={})
                for a, predict in predictors.items():
                    tr, pr = trackers[a].step(j, predict(image))
                    row["boxes"][a] = tr.tolist()
                    paths[a] = [[int(k), p.tolist(), q.tolist()] for k, p, q in pr]
                frames.append(row)
        finally:
            camera.release()
            server.shutdown()
            worker.join()
    return dict(frames=frames, paths=paths, video=meta)


def freeze(d, key):
    predictors, refs = None, {}
    for day, item in d["windows"].items():
        mark = LOCAL/f"{day}.json"
        if not mark.exists():
            if predictors is None:
                mobile = detectors.mobile.detector()
                def people(im):
                    b = detectors.mobile.detect(mobile, im)
                    return b[b[:, -1] == 1]

                predictors = dict(rapid=detectors.rapid.rapid(), mobilenet=people)
            result = observations(item, predictors)
            source.save(mark, dict(registration=key, source=item, **result))
        z = source.load(mark)
        if (z["registration"] != key or z["source"] != item
                or [f["frame"] for f in z["frames"]] != list(range(item["start"], item["start"]+8))):
            raise ValueError("Camera freeze does not match registered prefix")
        refs[day] = source.digest(mark.read_bytes())
        print(dict(pid=os.getpid(), phase="freeze", clips=len(refs), total=len(d["windows"]), day=day), flush=True)
    source.save(OUT/"freeze.json", dict(registration=key, receipts=refs, label_geometry_used=False,
                                       reserved_outcomes_opened=False))


def labels(b, start):
    frames = [{} for _ in range(20)]
    for line in io.BytesIO(b):
        if not line.strip():
            continue
        r = line.split(b",", 6)
        t, ident = int(r[0])-start, int(r[1])
        if not 0 <= t < 20 or ident < 0:
            continue
        x, y, h, w = map(float, r[2:6])
        if not np.isfinite([x, y, h, w]).all() or min(h, w) <= 0 or ident in frames[t]:
            raise ValueError("Invalid published box or duplicate known ID")
        frames[t][ident] = np.array([x, y, x+w, y+h])
    return frames


def associate(rows, truth):
    ids = sorted(truth)
    if not len(rows) or not ids:
        return set()
    d = overlap(rows[:, 1:5], np.stack([truth[k] for k in ids]))
    i, j = linear_sum_assignment(np.where(d >= IOU, -d, 1e6))
    return {(int(rows[a, 0]), ids[b]) for a, b in zip(i, j, strict=True) if d[a, b] >= IOU}


def metrics(z, truth, arm):
    f = [np.asarray(r["boxes"][arm], dtype=float).reshape(-1, 7) for r in z["frames"]]
    m = [associate(rows, t) for rows, t in zip(f, truth[:8], strict=True)]
    stable = dict(set.intersection(*m))
    reference = set.intersection(*(set(t) for t in truth[:8]))
    complete = set.intersection(*(set(t) for t in truth[8:]))
    future = set.union(*(set(t) for t in truth[8:]))
    cap = set.intersection(*({k for _, k in r} for r in m)) & complete
    paths = {k: np.asarray([c, o]) for k, c, o in z["paths"][arm]}
    if not stable.keys() <= paths.keys():
        raise ValueError("Stable match is missing its causal path")
    err = []
    for k, p in stable.items():
        if p in complete:
            box = np.stack([t[p] for t in truth[8:]])
            center = .5*(box[:, :2]+box[:, 2:])
            err.append(np.linalg.norm(paths[k]-center, axis=-1))
    n = len(err)
    return dict(label_boxes=sum(map(len, truth[:8])), detection_boxes=sum(map(len, f)),
                matched_boxes=sum(map(len, m)), reference_histories=len(reference),
                reference_complete=len(reference & complete), complete_match_ignoring_id=len(cap),
                online=len(paths), stable=len(stable), complete=n,
                absent=len(paths)-sum(p in future for p in stable.values()), full_roster=n >= 2 and n == len(paths),
                ade_px=np.mean(err, axis=(0, 2)).tolist() if n else None,
                fde_px=np.mean(np.stack(err)[:, :, -1], axis=0).tolist() if n else None)


def score(d, key):
    f = source.load(OUT/"freeze.json")
    if (f["registration"] != key or f["label_geometry_used"] or f["reserved_outcomes_opened"]
            or set(f["receipts"]) != set(d["windows"])):
        raise ValueError("Require complete outcome-free camera freeze")
    rows, totals = {}, {a: Counter() for a in ARMS}
    for day, item in d["windows"].items():
        p = LOCAL/f"{day}.json"
        if source.digest(p.read_bytes()) != f["receipts"][day]:
            raise ValueError("Camera result changed after freezing")
        mark = LOCAL/f"{day}.score.json"
        if not mark.exists():
            e = item["annotations"]
            b = source.byte_range(e["offset"], e["bytes"])
            if source.digest(b) != item["annotation_sha256"]:
                raise ValueError("Publisher annotation changed")
            t = labels(b, item["start"])
            z = source.load(p)
            source.save(mark, dict(freeze=f["receipts"][day], arms={a: metrics(z, t, a) for a in ARMS}))
        r = source.load(mark)
        if r["freeze"] != f["receipts"][day]:
            raise ValueError("Wrong score receipt")
        rows[day] = r["arms"]
        for a, m in r["arms"].items():
            totals[a].update({k: int(v) for k, v in m.items() if isinstance(v, int)})
        print(dict(pid=os.getpid(), phase="score", clips=len(rows), total=len(d["windows"]), day=day), flush=True)
    out = dict(windows=len(rows), acquisition_dates=len(rows), linked_sources=1, rows=rows, totals=totals,
               registration=key, freeze_sha256=source.digest((OUT/"freeze.json").read_bytes()),
               forecast_accuracy="complete-label diagnostic only", policy_risk_scoring=False,
               independent_confirmation=False, reserved_outcomes_opened=False)
    source.save(OUT/"analysis.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("freeze", "score"))
    phase = ap.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d, key = register()
        (freeze if phase == "freeze" else score)(d, key)


if __name__ == "__main__":
    main()
