"""Ground-control and current-frame correspondence audit, never forecast scoring."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_waterloo_raw_prefix import DATA, WEIGHTS, asset, detect, detector, immutable
from src.evaluation import m3w_ground_plane_registration as geo
from src.evaluation.m3w_waterloo_intake import sha
from src.world_model.m3w_streaming_boxes import StreamingTracks
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

PUBLIC = ROOT/"outputs/publication_readiness_2026_09/waterloo_ortho_registration_v1"
ORTHO = DATA/"ortho2020_932.png"
VIDEO = DATA/"DJI_0932.MP4"
BBOX = (534600,4810385,534800,4810550)
SHAPE = (1650,2000)
GRID_QUERY = ("https://gis.regionofwaterloo.ca/waimagery/rest/services/Imagery_2020/ImageServer/"
              "exportImage?bbox=534600,4810385,534800,4810550&bboxSR=32617&imageSR=32617&size=2000,1650&format=png")
PROBES = {"north_crossing":(503,127), "east_inbound":(902,176),
          "west_crossing":(374,536), "southeast_crossing":(995,545)}
KEYFRAMES = tuple(range(0,1201,120))+(1219,)
PREFIX = tuple(range(0,29,4))
TAU = 1001/24000


def ground_pass(stats, patches):
    return bool(stats["support_hull_fraction"] >= .18 and stats["inlier_quadrants"] >= 3
                and stats["inliers"] >= 8 and stats["p95_ortho_pixels"] <= 3.
                and sum(p["peak"] > .30 and p["delta_ortho_pixels"] <= 6. for p in patches) >= 3)


def early_labels(archive):
    """Current-frame positions in the declared prefix only, not future outcomes."""
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=DATA) as t:
        p = Path(t)/"working.db"
        with z.open("932.db") as f,p.open("wb") as g:
            shutil.copyfileobj(f,g,length=1024**2)
        with closing(sqlite3.connect(p.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("PRAGMA query_only=ON")
            out = {}
            for k in PREFIX:
                rows = db.execute("SELECT a.TRACK_ID,a.X,a.Y,b.TYPE FROM TRAJECTORIES AS a "
                    "JOIN TRACKS AS b USING(TRACK_ID) WHERE a.TIME BETWEEN ? AND ? ORDER BY a.TRACK_ID",
                    (k*TAU-1.e-6,k*TAU+1.e-6)).fetchall()
                if len(rows) != len(set(r[0] for r in rows)):
                    raise ValueError("Duplicated annotations in prefix")
                out[k] = rows
    return out


def ground_map_to_grid(xy):
    a = np.asarray(xy).reshape(-1,2)
    h,w = SHAPE
    return np.column_stack(((a[:,0]-BBOX[0])*w/(BBOX[2]-BBOX[0])-.5,
                            (BBOX[3]-a[:,1])*h/(BBOX[3]-BBOX[1])-.5))


def associate(det, lab, h):
    y = np.asarray([(x,y) for _,x,y,_ in lab],dtype=float).reshape(-1,2)
    lab_pixel = geo.project(ground_map_to_grid(y),np.linalg.inv(h))
    inside = ((lab_pixel > 0.) & (lab_pixel < [1280,720])).all(1)
    seen = np.flatnonzero(inside)
    past_pixels = (det[:,1:3]+det[:,3:5])/6.
    ground = geo.grid_to_utm(geo.project(past_pixels,h),BBOX,SHAPE)
    dist = np.linalg.norm(ground[:,None]-y[seen][None],axis=-1)
    ped_det = det[:,-1] == 1
    ped_lab = np.asarray([lab[i][3]=="Pedestrian" for i in seen],bool)
    legal = (dist<=3.) & (ped_det[:,None]==ped_lab[None])
    i,j = linear_sum_assignment(np.where(legal,dist,1e6))
    i,j = i[legal[i,j]],j[legal[i,j]]
    links = [(int(det[a,0]),int(lab[seen[b]][0]),float(dist[a,b]),bool(ped_det[a]))
             for a,b in zip(i,j,strict=True)]
    return dict(label_in_image=len(seen),label_person=int(ped_lab.sum()),detections=len(det),
                detector_person=int(ped_det.sum()),matched=len(links),
                person_matched=sum(r[3] for r in links)), links


def run():
    ref = cv2.imread(str(ORTHO),cv2.IMREAD_GRAYSCALE)
    if ref is None or ref.shape != SHAPE or sha(ORTHO) != "6a51e5d906ad185d84d70e2389d3ba2d36a3ed8c15c840998adcfa405d163a76":
        raise ValueError("Municipal image or grid changed")
    prior = json.loads((PUBLIC.parent/"waterloo_raw_prefix_v1/paired_source_preflight.json").read_text())
    if sha(VIDEO) != prior["assets"]["raw_video"]["sha256"] or sha(DATA/"dataset.zip") != prior["assets"]["annotation_archive"]["sha256"]:
        raise ValueError("Publisher candidate assets changed")
    if not sha(WEIGHTS).startswith("907ea3f9"):
        raise ValueError("Detector weights changed")
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise OSError("Original MP4 cannot be opened")
    transforms, checks, prefix = [], [], []
    ids = defaultdict(list)
    records, distances = [], []
    frames = set(KEYFRAMES+PREFIX)
    tracker, net = StreamingTracks(), detector()
    labels = None
    try:
        for i in range(KEYFRAMES[-1]+1):
            ok, raw = cap.read()
            if not ok:
                raise OSError("Raw video ends before final registered frame")
            if i not in frames:
                continue
            gray = cv2.cvtColor(cv2.resize(raw,(1280,720),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
            try:
                h, stats = geo.fit(gray,ref,list(PROBES.values()))
                patches = geo.holdouts(gray,ref,h,PROBES)
                passed = ground_pass(stats,patches)
                facts = dict(frame=i,passed=passed,fit=stats,holdouts=patches)
            except ValueError as e:
                passed = False
                facts = dict(frame=i,passed=False,geometry_error=str(e))
            checks.append(facts)
            if passed:
                transforms.append(dict(frame=i,h=h.tolist(),center=geo.grid_to_utm(
                    geo.project([[640,360]],h),BBOX,SHAPE)[0].tolist()))
            if i in PREFIX:
                d = detect(net,raw)
                current,_ = tracker.step(i//4,d)
                if passed:
                    if labels is None:
                        labels = early_labels(DATA/"dataset.zip")
                    counts,links = associate(current,labels[i],h)
                    prefix.append(dict(frame=i,**counts))
                    for track, label, err, ped in links:
                        ids[track].append(label)
                        distances.append(err)
                        records.append(ped)
                else:
                    prefix.append(dict(frame=i,not_run="static_control_failed"))
            print(json.dumps(dict(pid=os.getpid(), frame=i,static_geometry_passed=passed)),flush=True)
    finally:
        cap.release()
    common = [x for x in ids.values() if len(x)>1]
    c = Counter()
    for r in prefix:
        c.update({k:v for k,v in r.items() if k not in ("frame","not_run")})
    centers = np.asarray([x["center"] for x in transforms])
    return dict(reference=dict(url=GRID_QUERY,sha256=sha(ORTHO),bbox=BBOX,crs="EPSG:32617",shape=SHAPE),
        source_pair=prior["assets"], readout="fresh_run", keyframes=KEYFRAMES, current_check_frames=PREFIX,
        controls=checks, usable_static_frames=len(transforms),
        max_center_map_displacement=float(np.linalg.norm(centers-centers[0],axis=-1).max()) if len(centers) else None,
        calibration_prefix=prefix, counts=c,
        matched_distance_p50_p95_map_units=np.quantile(distances,[.5,.95]).tolist() if distances else None,
        matched_person_distance_p50_map_units=float(np.median(np.asarray(distances)[records])) if any(records) else None,
        repeated_detector_tracks=len(common), repeated_consistent_tracks=sum(len(set(a))==1 for a in common),
        estimated_translation_or_time_offset=False, dataset_datum_verified=False, target_definition_verified=False,
        future_forecast_errors_read=False, model_performance=False, role_assigned=False,
        independent_confirmation=False, original_rule_unchanged=True,
        implementation=dict(runner=sha(Path(__file__)),geometry=sha(Path(geo.__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    start = time.monotonic()
    path = PUBLIC/"analysis.json"
    if a.replay and not path.is_file():
        raise FileNotFoundError("Run the ground registration before replay")
    immutable(path,run())
    immutable(PUBLIC/("replay.json" if a.replay else "execution.json"),
              dict(pid=os.getpid(),seconds=time.monotonic()-start,analysis_sha256=sha(path)))
