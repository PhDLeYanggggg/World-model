"""Audit frozen sparse images for clip overlap and camera motion, not forecasts."""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_dronecrowd_intake import sha
from src.evaluation.m3w_dronecrowd_image_geometry import background_features, match_background, candidate_components


def main():
    started = time.monotonic()
    cv2.setNumThreads(1)
    raw = ROOT / "external_data/DroneCrowd_images"
    report = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_image_audit_v1"
    report.mkdir(parents=True, exist_ok=True)
    figures = raw / "audit_contact_sheets"
    figures.mkdir(exist_ok=True)
    ref_path = ROOT / "outputs/publication_readiness_2026_09/dronecrowd_annotations_v1/analysis.json"
    reference = json.loads(ref_path.read_text())
    hashes = {m["name"]:m["sha256"] for m in reference["members"]}
    records, manifest_hashes, acquisitions = {}, {}, {}
    for split in ("train", "test"):
        path = raw / f"{split}_sparse_manifest.json"
        manifest = json.loads(path.read_text())
        if not manifest["status"].startswith("complete_sparse"):
            raise ValueError("Sparse acquisition not complete")
        manifest_hashes[split] = sha(path.read_bytes())
        acquisitions[split] = {k:v for k,v in manifest.items() if k != "frames"}
        for row in manifest["frames"]:
            if sha((raw / row["local_name"]).read_bytes()) != row["sha256"]:
                raise ValueError("Modified sampled image")
            if (row["scene_id"], row["frame_one_based"]) in records:
                raise ValueError("Duplicate sampled identity")
            records[row["scene_id"], row["frame_one_based"]] = {**row, "release_split":split}
    scenes = sorted({s for s,_ in records})
    if len(scenes)!=112 or set(records)!={(s,f) for s in scenes for f in (1,150,300)}:
        raise ValueError("Incomplete frozen sparse inventory")
    source_manifest = {"scope":"public_image_metadata_only_no_images",
                       "acquisitions":acquisitions,
                       "frames":[records[key] for key in sorted(records)]}
    source_manifest_path = report/"source_manifest.json"
    with source_manifest_path.open("x") as stream:
        stream.write(json.dumps(source_manifest,indent=2)+"\n")
    features, feature_summary, first_images = {}, [], {}
    with zipfile.ZipFile(ROOT / "external_data/DroneCrowd_annotations/annotations.zip") as archive:
        for scene in scenes:
            name = f"annotations/{scene}.xml"
            xml = archive.read(name)
            if sha(xml) != hashes[name]:
                raise ValueError("XML differs from annotation audit")
            root = ET.fromstring(xml)
            boxes = {0:[], 149:[], 299:[]}
            for track in root:
                for box in track:
                    frame = int(box.get("frame"))
                    if frame in boxes and box.get("outside")==box.get("occluded")=="0":
                        boxes[frame].append([float(box.get(k)) for k in ("xtl","ytl","xbr","ybr")])
            for frame in (1,150,300):
                with Image.open(raw / records[scene,frame]["local_name"]) as picture:
                    rgb = np.array(picture.convert("RGB"))
                features[scene,frame] = background_features(rgb, boxes[frame-1])
                item = features[scene,frame]
                feature_summary.append({"scene":scene,"frame":frame,"keypoints":len(item["xy"]),
                                        "background_fraction":item["background_fraction"]})
                if frame==1:
                    first_images[scene] = Image.fromarray(rgb).resize((320,180))
    for page, offset in enumerate(range(0,len(scenes),16),1):
        sheet = Image.new("RGB",(1280,816),"white")
        draw = ImageDraw.Draw(sheet)
        for index, scene in enumerate(scenes[offset:offset+16]):
            x,y = index%4*320,index//4*204
            sheet.paste(first_images[scene],(x,y))
            draw.text((x+3,y+183),f"{scene}  {records[scene,1]['release_split']}",fill="black")
        sheet.save(figures/f"first_frames_{page:02d}.jpg",quality=90)
    within = []
    for scene in scenes:
        for end in (150,300):
            within.append({"scene":scene,"from_frame":1,"to_frame":end,
                           **match_background(features[scene,1],features[scene,end])})
    pairs = []
    for index,(a,b) in enumerate(itertools.combinations(scenes,2),1):
        result = match_background(features[a,1],features[b,1])
        if result["ratio_matches"]>=12:
            pairs.append({"left":a,"right":b,"cross_release_split":records[a,1]["release_split"]!=records[b,1]["release_split"],
                          **result})
        if index%500==0:
            print(json.dumps({"pairs_checked":index,"seconds":round(time.monotonic()-started,1)}),flush=True)
    edges = [p for p in pairs if p["overlap_candidate"]]
    groups = candidate_components(scenes,[(p["left"],p["right"]) for p in edges])
    reliable = [p for p in within if p["overlap_candidate"]]
    result = {"result_source":"fresh_run", "image_manifest_sha256":manifest_hashes,
              "public_source_manifest_sha256":sha(source_manifest_path.read_bytes()),
              "annotation_analysis_sha256":sha(ref_path.read_bytes()), "images_verified":len(records),
              "recordings":len(scenes), "cross_clip_first_frame_pairs_checked":len(scenes)*(len(scenes)-1)//2,
              "feature_summary":feature_summary,"within_clip_registration":within,
              "cross_clip_matches_at_least_12":pairs,"overlap_candidate_pairs":len(edges),
              "cross_release_overlap_candidate_pairs":sum(p["cross_release_split"] for p in edges),
              "candidate_connected_components":groups,"candidate_component_count":len(groups),
              "candidate_components_are_not_verified_independent_sites":True,
              "reliable_within_clip_pairs":len(reliable),
              "reliable_pairs_above_5px_grid_motion":sum(p["median_grid_displacement_pixels"]>5 for p in reliable),
              "reliable_pairs_above_20px_grid_motion":sum(p["median_grid_displacement_pixels"]>20 for p in reliable),
              "registration_is_not_metric_homography":True,"effective_seconds":"unknown",
              "scientific_roles_assigned":False,"forecast_errors_read":False,
              "model_training":"not_run","stage5c_executed":False,"smc_enabled":False,
              "limitations":["First-frame cross-clip matches cannot certify distinct physical sites.",
                             "Sparse registrations do not measure every-frame camera motion.",
                             "Head masking is not complete moving-object segmentation.",
                             "Image homographies are not ground-plane or pixel-to-metre calibration.",
                             "No keyframe provenance can be inferred from image registration."]}
    target = report/"analysis.json"
    with target.open("x") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    with (report/"execution.json").open("x") as stream:
        json.dump({"completed_utc":datetime.now(timezone.utc).isoformat(),
                   "seconds":round(time.monotonic()-started,3),"analysis_sha256":sha(target.read_bytes()),
                   "opencv":cv2.__version__,"numpy":np.__version__,"threads":1,
                   "implementation_sha256":{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (
                       Path(__file__),ROOT/"src/evaluation/m3w_dronecrowd_image_geometry.py")}},stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in (
        "feature_summary","within_clip_registration","cross_clip_matches_at_least_12","candidate_connected_components")},indent=2))


if __name__=="__main__":
    main()
