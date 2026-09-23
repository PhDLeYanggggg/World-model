"""Resumable multi-view source grouping; no predictors, roles or forecast errors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.m3w_dronecrowd_grouping import compare_clip_views,extract_features,group_evidence
from src.evaluation.m3w_dronecrowd_intake import sha
from scripts.audit_m3w_dronecrowd_metadata import plain_path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-new-pairs",type=int,default=0,help="Timing pilot only; zero means finish all pairs")
    args = parser.parse_args()
    if args.max_new_pairs<0:
        raise SystemExit("max-new-pairs cannot be negative")
    started = time.monotonic()
    cv2.setNumThreads(1)
    source = ROOT/"external_data/DroneCrowd_images"
    report = ROOT/"outputs/publication_readiness_2026_09/dronecrowd_grouping_v2"
    cache = plain_path(ROOT/"data/stage_cvpr2027_experiments/dronecrowd_grouping_v2")
    report.mkdir(parents=True,exist_ok=True)
    cache.mkdir(parents=True,exist_ok=True)
    lock = (cache/"run.lock").open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another source-grouping process holds the lock")
    config_path = ROOT/"configs/m3w_dronecrowd_grouping_v2.json"
    config = json.loads(config_path.read_text())
    prior_root = ROOT/"outputs/publication_readiness_2026_09/dronecrowd_image_audit_v1"
    prior_path, manifest_path = prior_root/"analysis.json",prior_root/"source_manifest.json"
    prior, sources = json.loads(prior_path.read_text()),json.loads(manifest_path.read_text())
    prior_execution = json.loads((prior_root/"execution.json").read_text())
    if digest_file(prior_path)!=prior_execution["analysis_sha256"]:
        raise ValueError("Prior audit hash changed")
    if digest_file(manifest_path)!=prior["public_source_manifest_sha256"]:
        raise ValueError("Prior image manifest changed")
    annotation = ROOT/"outputs/publication_readiness_2026_09/dronecrowd_annotations_v1/analysis.json"
    if digest_file(annotation)!=prior["annotation_analysis_sha256"]:
        raise ValueError("Annotation reference changed")
    permission = ROOT/"outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json"
    from src.evaluation.m3w_dronecrowd_image_intake import require_image_permission
    require_image_permission(json.loads(permission.read_text()))
    identity = {"config_sha256":digest_file(config_path),"image_manifest_sha256":digest_file(manifest_path),
                "prior_analysis_sha256":digest_file(prior_path),"authorization_sha256":digest_file(permission),
                "code_sha256":{name:digest_file(ROOT/name) for name in (
                    "scripts/audit_m3w_dronecrowd_grouping.py","src/evaluation/m3w_dronecrowd_grouping.py",
                    "src/evaluation/m3w_dronecrowd_image_geometry.py")},
                "opencv":cv2.__version__,"numpy":np.__version__,"threads":1}
    run_id = sha(json.dumps(identity,sort_keys=True).encode())
    identity_path = cache/"identity.json"
    if identity_path.exists():
        if not args.resume:
            raise SystemExit("Existing audit state; use --resume, never overwrite")
        if json.loads(identity_path.read_text())!={"run_id":run_id,**identity}:
            raise ValueError("Resume input/config/code/runtime identity mismatch")
    else:
        atomic_json(identity_path,{"run_id":run_id,**identity})
        atomic_json(report/"run_identity.json",{"run_id":run_id,**identity,"config":config})
    records = {(x["scene_id"],x["frame_one_based"]):x for x in sources["frames"]}
    scenes = sorted({s for s,_ in records})
    if len(scenes)!=112 or set(records)!={(s,f) for s in scenes for f in config["selected_frames_one_based"]}:
        raise ValueError("Source inventory differs from fixed 112-clip/three-frame audit")
    for row in records.values():
        if digest_file(source/row["local_name"])!=row["sha256"]:
            raise ValueError("Image payload changed")
    member_hashes = {x["name"]:x["sha256"] for x in json.loads(annotation.read_text())["members"]}
    feature_dir,pair_dir = cache/"features",cache/"pairs"
    feature_dir.mkdir(exist_ok=True)
    pair_dir.mkdir(exist_ok=True)
    feature_sources, features, summaries = [],{},[]

    def heartbeat(phase, **progress):
        value = {"phase":phase,"pid":os.getpid(),"run_id":run_id,
                 "utc":datetime.now(timezone.utc).isoformat(),
                 "session_seconds":round(time.monotonic()-started,3),**progress}
        atomic_json(cache/"heartbeat.json",value)
        print(json.dumps(value),flush=True)

    with zipfile.ZipFile(ROOT/"external_data/DroneCrowd_annotations/annotations.zip") as archive:
        for index,scene in enumerate(scenes):
            features[scene] = {}
            boxes = None
            for frame in config["selected_frames_one_based"]:
                path = feature_dir/f"{scene}_{frame:03d}.npz"
                receipt = path.with_suffix(".json")
                if path.exists() and receipt.exists():
                    evidence = json.loads(receipt.read_text())
                    if evidence["run_id"]!=run_id or digest_file(path)!=evidence["sha256"]:
                        raise ValueError("Cached descriptor checksum mismatch")
                    with np.load(path,allow_pickle=False) as arrays:
                        item = {"xy":arrays["xy"],"descriptors":arrays["descriptors"],
                                "background_fraction":float(arrays["background_fraction"])}
                        if len(item["descriptors"])==0:
                            item["descriptors"] = None
                    feature_sources.append("cached_verified")
                else:
                    if boxes is None:
                        name = f"annotations/{scene}.xml"
                        data = archive.read(name)
                        if sha(data)!=member_hashes[name]:
                            raise ValueError("XML changed")
                        boxes = {f-1:[] for f in config["selected_frames_one_based"]}
                        for track in ET.fromstring(data):
                            for box in track:
                                f = int(box.get("frame"))
                                if f in boxes and box.get("outside")==box.get("occluded")=="0":
                                    boxes[f].append([float(box.get(k)) for k in ("xtl","ytl","xbr","ybr")])
                    with Image.open(source/records[scene,frame]["local_name"]) as image:
                        item = extract_features(np.asarray(image.convert("RGB")),boxes[frame-1],config)
                    with path.with_suffix(".tmp").open("wb") as stream:
                        np.savez(stream,xy=item["xy"],descriptors=item["descriptors"] if item["descriptors"] is not None
                                 else np.empty((0,128),np.float32),background_fraction=item["background_fraction"])
                    path.with_suffix(".tmp").replace(path)
                    atomic_json(receipt,{"run_id":run_id,"sha256":digest_file(path)})
                    feature_sources.append("fresh_run")
                features[scene][frame] = item
                summaries.append({"scene":scene,"frame":frame,"keypoints":len(item["xy"]),
                                  "background_fraction":item["background_fraction"]})
            if (index+1)%16==0 or index+1==len(scenes):
                heartbeat("features",clips_complete=index+1,clips_total=len(scenes))
    results,new_pairs,cached_pairs,frame_pairs = [],0,0,0
    pair_checksums = {}
    for index,(a,b) in enumerate(itertools.combinations(scenes,2),1):
        path = pair_dir/f"{a}_{b}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("run_id")!=run_id or result.get("left")!=a or result.get("right")!=b:
                raise ValueError("Cached pair identity mismatch")
            cached_pairs += 1
        else:
            result = {"run_id":run_id,"left":a,"right":b,
                      "cross_release_split":records[a,1]["release_split"]!=records[b,1]["release_split"],
                      **compare_clip_views(features[a],features[b],config)}
            atomic_json(path,result)
            new_pairs += 1
        frame_pairs += result["frame_pairs_attempted"]
        pair_checksums[path.name] = digest_file(path)
        results.append(result)
        if index%100==0:
            heartbeat("pairs",clip_pairs_complete=index,clip_pairs_total=6216,
                      new_pairs=new_pairs,cached_pairs=cached_pairs,frame_pairs=frame_pairs)
        if args.max_new_pairs and new_pairs>=args.max_new_pairs:
            heartbeat("timing_pilot_complete_not_full_audit",clip_pairs_complete=index,clip_pairs_total=6216,
                      new_pairs=new_pairs,cached_pairs=cached_pairs,frame_pairs=frame_pairs)
            return
    pairs = [{k:v for k,v in x.items() if k!="run_id"} for x in results if x["status"]!="unsupported"]
    prior_edges = [(p["left"],p["right"]) for p in prior["cross_clip_matches_at_least_12"] if p["overlap_candidate"]]
    grouping = group_evidence(scenes,pairs,prior_edges)
    result = {"scope":"multiview_background_overlap_not_forecast_evaluation", "run_id":run_id,
              "recordings":len(scenes),"images":len(records),"clip_pairs_complete":len(results),
              "frame_pairs_attempted":frame_pairs,"feature_summary":summaries,
              "strong_multiview_pairs":sum(x["status"]=="strong_multiview" for x in results),
              "ambiguous_pairs":sum(x["status"]=="ambiguous" for x in results),
              "cross_release_strong_pairs":sum(x["status"]=="strong_multiview" and x["cross_release_split"] for x in results),
              "cross_release_ambiguous_pairs":sum(x["status"]=="ambiguous" and x["cross_release_split"] for x in results),
              "prior_positive_edges_preserved":prior_edges,"supported_pairs":pairs,**grouping,
              "pair_receipt_sha256":sha(json.dumps(pair_checksums,sort_keys=True).encode()),
              "physical_site_count":None,"scientific_roles_assigned":False,
              "forecast_errors_read":False,"training_run":False,"stage5c_executed":False,"smc_enabled":False}
    result_path = report/"analysis.json"
    if result_path.exists():
        if json.loads(result_path.read_text())!=result:
            raise ValueError("Completed result differs from cached evidence")
        result_source = "cached_verified"
    else:
        atomic_json(result_path,result)
        result_source = "fresh_run"
    execution = {"result_source":result_source,"new_pairs":new_pairs,"cached_pairs":cached_pairs,
                 "new_feature_frames":feature_sources.count("fresh_run"),
                 "cached_feature_frames":feature_sources.count("cached_verified"),
                 "analysis_sha256":digest_file(result_path),"run_id":run_id,
                 "completed_utc":datetime.now(timezone.utc).isoformat(),
                 "session_seconds":round(time.monotonic()-started,3)}
    execution_path = report/("verification_execution.json" if result_source=="cached_verified" else "execution.json")
    atomic_json(execution_path,execution)
    heartbeat("complete",clip_pairs_complete=len(results),**execution)
    print(json.dumps({k:result[k] for k in ("clip_pairs_complete","frame_pairs_attempted","strong_multiview_pairs",
                                           "ambiguous_pairs","cross_release_strong_pairs","cross_release_ambiguous_pairs")}),flush=True)


if __name__=="__main__":
    main()
