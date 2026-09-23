"""Image-to-image source overlap diagnostics, not physical calibration."""
from __future__ import annotations

import cv2
import numpy as np


def background_features(rgb, boxes):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
    mask = np.full(gray.shape, 255, np.uint8)
    for x0, y0, x1, y1 in boxes:
        # Enlarged head masks reduce pedestrian matches; not a segmentation claim.
        padding = max(x1-x0, y1-y0) * 0.75 + 6
        a, b = np.floor((np.array([x0, y0])-padding)/2).astype(int)
        c, d = np.ceil((np.array([x1, y1])+padding)/2).astype(int)
        a,c = np.clip([a,c],0,960)
        b,d = np.clip([b,d],0,540)
        mask[b:d, a:c] = 0
    points, descriptors = cv2.SIFT_create(nfeatures=1200).detectAndCompute(gray, mask)
    xy = np.array([p.pt for p in points], dtype=np.float32).reshape(-1,2)
    return {"xy": xy, "descriptors": descriptors,
            "background_fraction": float(np.mean(mask > 0))}


def match_background(left, right, seed=271828):
    result = {"ratio_matches": 0, "inliers": 0, "inlier_fraction": 0.0,
              "coverage_left": 0.0, "coverage_right": 0.0,
              "overlap_candidate": False, "transform_kind": "image_to_image_only"}
    if left["descriptors"] is None or right["descriptors"] is None:
        return result
    cv2.setRNGSeed(seed)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=40))
    matches = matcher.knnMatch(left["descriptors"], right["descriptors"], k=2)
    candidates = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance < 0.7*pair[1].distance]
    # Unique target descriptors avoid many-to-one repeated-texture support.
    candidates.sort(key=lambda m: m.distance)
    good, seen = [], set()
    for match in candidates:
        if match.trainIdx not in seen:
            good.append(match)
            seen.add(match.trainIdx)
    result["ratio_matches"] = len(good)
    if len(good) < 12:
        return result
    a = left["xy"][[m.queryIdx for m in good]]
    b = right["xy"][[m.trainIdx for m in good]]
    transform, inlier = cv2.findHomography(a, b, cv2.RANSAC, 2.5, maxIters=2000, confidence=0.995)
    if transform is None or inlier is None or not np.isfinite(transform).all():
        return result
    keep = inlier.ravel().astype(bool)
    n = int(keep.sum())
    if n < 4:
        return result
    coverage_a = float(cv2.contourArea(cv2.convexHull(a[keep]))/(960*540))
    coverage_b = float(cv2.contourArea(cv2.convexHull(b[keep]))/(960*540))
    grid = np.array([[x,y] for x in (96,480,864) for y in (54,270,486)], np.float32)
    warped = cv2.perspectiveTransform(grid[None], transform)[0]
    displacement = np.linalg.norm(warped-grid,axis=1)*2
    reprojection = np.linalg.norm(cv2.perspectiveTransform(a[keep][None],transform)[0]-b[keep],axis=1)*2
    result.update(inliers=n, inlier_fraction=n/len(good), coverage_left=coverage_a,
                  coverage_right=coverage_b, median_reprojection_pixels=float(np.median(reprojection)),
                  median_grid_displacement_pixels=float(np.median(displacement)),
                  maximum_grid_displacement_pixels=float(displacement.max()),
                  transform_960x540=transform.tolist(),
                  overlap_candidate=bool(n>=25 and n/len(good)>=0.35
                                         and min(coverage_a,coverage_b)>=0.08))
    return result


def candidate_components(nodes, pairs):
    parents = {n:n for n in nodes}
    def root(n):
        while parents[n] != n:
            parents[n] = parents[parents[n]]
            n = parents[n]
        return n
    for a,b in pairs:
        parents[root(a)] = root(b)
    groups = {}
    for n in sorted(nodes):
        groups.setdefault(root(n), []).append(n)
    return sorted(groups.values(), key=lambda group: group[0])
