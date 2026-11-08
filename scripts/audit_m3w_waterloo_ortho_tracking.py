"""Causal first-frame bridge for an orthophoto; calibration only."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_waterloo_ortho as base

np, cv2, torch, geo = base.np, base.cv2, base.torch, base.geo


def audit(parent):
    cap = cv2.VideoCapture(str(base.VIDEO))
    ref = cv2.imread(str(base.ORTHO),cv2.IMREAD_GRAYSCALE)
    net = base.detector()
    tracker = base.StreamingTracks()
    if not cap.isOpened() or base.sha(base.ORTHO) != parent["reference"]["sha256"] \
            or base.sha(base.VIDEO) != parent["source_pair"]["raw_video"]["sha256"] \
            or base.sha(base.DATA/"dataset.zip") != parent["source_pair"]["annotation_archive"]["sha256"]:
        raise ValueError("Unverified geometry assets")
    frames = set(base.KEYFRAMES+base.PREFIX)
    labels = base.early_labels(base.DATA/"dataset.zip")
    records, selected = [], []
    distances, ped_distances, centers = [],[],[]
    ids = defaultdict(list)
    try:
        for i in range(base.KEYFRAMES[-1]+1):
            ok, raw = cap.read()
            if not ok:
                raise OSError("Could not decode registered frames")
            if i not in frames:
                continue
            cur = cv2.cvtColor(cv2.resize(raw,(1280,720),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
            bridge = None
            if i == 0:
                first = cur
                initial, fit = geo.fit(first,ref,list(base.PROBES.values()))
                probe = geo.holdouts(first,ref,initial,base.PROBES)
                if not base.ground_pass(fit,probe):
                    raise ValueError("First-frame map anchor fails")
                h = initial
                passed = True
            else:
                image,bridge = geo.fit(cur,first,list(base.PROBES.values()))
                h = initial@image
                probe = geo.holdouts(cur,ref,h,base.PROBES)
                passed = (bridge["inliers"]>=40 and bridge["inlier_quadrants"]==4
                    and bridge["support_hull_fraction"]>=.25 and bridge["p95_ortho_pixels"]<=2.
                    and sum(r["peak"]>.30 and r["delta_ortho_pixels"]<=6. for r in probe)>=3)
            records.append(dict(frame=i,passed=passed,raw_reference_fit=bridge,holdouts=probe))
            if passed:
                centers.append(geo.grid_to_utm(geo.project([[640,360]],h),base.BBOX,base.SHAPE)[0])
            if i in base.PREFIX:
                d = base.detect(net,raw)
                rows,_ = tracker.step(i//4,d)
                if passed:
                    facts, links = base.associate(rows,labels[i],h)
                    selected.append(dict(frame=i,**facts))
                    for tr,lab,dist,ped in links:
                        ids[tr].append(lab)
                        distances.append(dist)
                        if ped:
                            ped_distances.append(dist)
                else:
                    selected.append(dict(frame=i,not_run="failed_geometry"))
            print(json.dumps(dict(pid=os.getpid(),frame=i,bridge_pass=passed)),flush=True)
    finally:
        cap.release()
    total = Counter()
    for row in selected:
        total.update({k:v for k,v in row.items() if k not in ("frame","not_run")})
    repeat = [x for x in ids.values() if len(x)>1]
    centers = np.asarray(centers)
    return dict(frames=records,checked=len(records),passed=sum(r["passed"] for r in records),
        keyframes_passed=sum(r["passed"] for r in records if r["frame"] in base.KEYFRAMES),
        current_check=selected,counts=total,
        matched_distance_p50_p95_map_units=np.quantile(distances,[.5,.95]).tolist() if distances else None,
        person_p50_map_units=float(np.median(ped_distances)) if ped_distances else None,
        repeated_tracks=len(repeat),consistent_repeated_tracks=sum(len(set(a))==1 for a in repeat),
        center_drift_max_map_units=float(np.linalg.norm(centers-centers[0],axis=-1).max()),
        direct_parent_sha256=base.sha(base.PUBLIC/"analysis.json"),
        map_uses_only_current_and_frame0=True, no_future_forecast_errors=True,
        no_independent_evaluation_or_role_change=True, original_rule_unchanged=True,
        implementation=base.sha(Path(__file__)))


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    base.immutable(base.PUBLIC/"causal_bridge.json",
                   audit(json.loads((base.PUBLIC/"analysis.json").read_text())))
