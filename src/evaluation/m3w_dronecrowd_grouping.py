"""Conservative multi-view overlap screening, not site-independence certification."""
from __future__ import annotations

import cv2
import numpy as np

from src.evaluation.m3w_dronecrowd_image_geometry import candidate_components


def extract_features(rgb, boxes, config):
    width, height = config["resize"]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=config["clahe_clip_limit"],tileGridSize=(8,8)).apply(gray)
    sx,sy = width/rgb.shape[1],height/rgb.shape[0]
    mask = np.full((height,width),255,np.uint8)
    for x0,y0,x1,y1 in boxes:
        padding = max(x1-x0,y1-y0)*0.75+6
        a,c = np.clip(np.array([x0-padding,x1+padding])*sx,0,width).astype(int)
        b,d = np.clip(np.array([y0-padding,y1+padding])*sy,0,height).astype(int)
        mask[b:d,a:c] = 0
    points, descriptors = cv2.SIFT_create(nfeatures=config["max_features"]).detectAndCompute(gray,mask)
    xy = np.asarray([p.pt for p in points],dtype=np.float32).reshape(-1,2)
    if descriptors is not None:
        descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1,keepdims=True),1e-12))
    return {"xy":xy,"descriptors":descriptors,"background_fraction":float((mask>0).mean())}


def _ratio_matches(left, right, ratio):
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=48))
    return {p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(left,right,k=2)
            if len(p)==2 and p[0].distance<ratio*p[1].distance}


def pair_evidence(left, right, config):
    result = {"mutual_matches":0,"inliers":0,"inlier_fraction":0.0,
              "coverage_left":0.0,"coverage_right":0.0,"status":"unsupported"}
    if (left["descriptors"] is None or right["descriptors"] is None
            or len(left["descriptors"])<2 or len(right["descriptors"])<2):
        return result
    cv2.setRNGSeed(config["ransac_seed"])
    forward = _ratio_matches(left["descriptors"],right["descriptors"],config["descriptor_ratio"])
    reverse = _ratio_matches(right["descriptors"],left["descriptors"],config["descriptor_ratio"])
    pairs = [(a,b) for a,b in forward.items() if reverse.get(b)==a]
    result["mutual_matches"] = len(pairs)
    if len(pairs)<config["ambiguous_min_inliers"]:
        return result
    a = left["xy"][[p[0] for p in pairs]]
    b = right["xy"][[p[1] for p in pairs]]
    transform, support = cv2.findHomography(a,b,cv2.RANSAC,
        config["ransac_threshold_resized_px"],maxIters=3000,confidence=0.995)
    if transform is None or support is None or not np.isfinite(transform).all():
        return result
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return result
    keep = support.ravel().astype(bool)
    n = int(keep.sum())
    if n<4:
        return result
    area = np.prod(config["resize"])
    cover_a = float(cv2.contourArea(cv2.convexHull(a[keep]))/area)
    cover_b = float(cv2.contourArea(cv2.convexHull(b[keep]))/area)
    error_a = np.linalg.norm(cv2.perspectiveTransform(a[keep][None],transform)[0]-b[keep],axis=1)
    error_b = np.linalg.norm(cv2.perspectiveTransform(b[keep][None],inverse)[0]-a[keep],axis=1)
    error = float(np.median(np.maximum(error_a,error_b)))
    # Inliers must have support in both views; a degenerate small texture is insufficient.
    status = "unsupported"
    if np.isfinite(error) and error<=config["max_symmetric_median_error_resized_px"]:
        for level in ("ambiguous","strong"):
            if (n>=config[f"{level}_min_inliers"]
                    and n/len(pairs)>=config[f"{level}_min_inlier_fraction"]
                    and min(cover_a,cover_b)>=config[f"{level}_min_hull_coverage"]):
                status = level
    result.update(inliers=n,inlier_fraction=n/len(pairs),coverage_left=cover_a,
                  coverage_right=cover_b,symmetric_median_error_resized_px=error,
                  status=status,transform_960x540=transform.tolist())
    return result


def compare_clip_views(left, right, config):
    frames = config["selected_frames_one_based"]
    order = [(a,a) for a in frames]+[(a,b) for a in frames for b in frames if a!=b]
    supported, strong_left, strong_right, attempted = [],set(),set(),0
    for a,b in order:
        evidence = pair_evidence(left[a],right[b],config)
        attempted += 1
        if evidence["status"] in ("ambiguous","strong"):
            supported.append({"left_frame":a,"right_frame":b,**evidence})
        if evidence["status"]=="strong":
            strong_left.add(a)
            strong_right.add(b)
        if min(len(strong_left),len(strong_right))>=config["strong_frames_required"]:
            return {"status":"strong_multiview","frame_pairs_attempted":attempted,"support":supported}
    return {"status":"ambiguous" if supported else "unsupported",
            "frame_pairs_attempted":attempted,"support":supported}


def group_evidence(nodes, pairs, prior_edges=()):
    strong = [(p["left"],p["right"]) for p in pairs if p["status"]=="strong_multiview"]
    weak = [(p["left"],p["right"]) for p in pairs if p["status"]=="ambiguous"]
    prior = list(prior_edges)
    if any(a not in nodes or b not in nodes for a,b in strong+weak+prior):
        raise ValueError("Unknown recording in overlap evidence")
    return {"strong_components":candidate_components(nodes,strong+prior),
            "conservative_components":candidate_components(nodes,strong+weak+prior),
            "negative_matches_establish_independence":False}


def require_no_cross_role_edges(assignments, pairs, prior_edges=()):
    edges = [(p["left"],p["right"]) for p in pairs if p["status"]!="unsupported"]+list(prior_edges)
    for a,b in edges:
        if a not in assignments or b not in assignments:
            raise ValueError("Missing recording role")
        if assignments[a]!="excluded" and assignments[b]!="excluded" and assignments[a]!=assignments[b]:
            raise ValueError("Overlap/ambiguity relation crosses scientific roles")
