"""Static-image control for an aerial video to an orthophoto grid."""

import cv2
import numpy as np


def project(xy, h):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if not len(xy):
        return xy
    return cv2.perspectiveTransform(xy[None], np.asarray(h, dtype=float))[0]


def candidate_pairs(frame, reference, *, ratio=.77, nfeatures=8000):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    p, x = sift.detectAndCompute(frame, None)
    q, y = sift.detectAndCompute(reference, None)
    if x is None or y is None or min(len(x), len(y)) < 2:
        raise ValueError("Not enough candidate keypoints")
    match = cv2.BFMatcher()
    a = {m.queryIdx:m.trainIdx for m,n in match.knnMatch(x,y,k=2) if m.distance < ratio*n.distance}
    b = {m.queryIdx:m.trainIdx for m,n in match.knnMatch(y,x,k=2) if m.distance < ratio*n.distance}
    pairs = [(p[i].pt,q[j].pt) for i,j in a.items() if b.get(j) == i]
    if len(pairs) < 4:
        raise ValueError("Not enough mutually matching keypoints")
    return tuple(map(np.asarray, zip(*pairs, strict=True)))


def fit(frame, reference, probes, *, exclusion=48):
    a, b = candidate_pairs(frame, reference)
    use = (np.abs(a[:, None]-np.asarray(probes)[None]).max(2)>exclusion).all(1)
    if use.sum() < 8:
        raise ValueError("Not enough non-holdout points")
    h, mask = cv2.findHomography(a[use], b[use], cv2.USAC_MAGSAC, 3.)
    if h is None or not np.isfinite(h).all():
        raise ValueError("Cannot solve static planar projection")
    p = a[use][mask.ravel()>0]
    q = b[use][mask.ravel()>0]
    if len(p) < 8:
        raise ValueError("Not enough mutually matching inlier points")
    error = np.linalg.norm(project(p,h)-q, axis=1)
    hull = cv2.convexHull(p.astype(np.float32))
    stats = dict(candidate_pairs=len(a), fit_pairs=int(use.sum()), inliers=len(p),
                 p50_ortho_pixels=float(np.median(error)), p95_ortho_pixels=float(np.quantile(error,.95)),
                 support_hull_fraction=float(cv2.contourArea(hull)/np.prod(frame.shape)),
                 inlier_quadrants=int(len(np.unique((p[:,0]>=frame.shape[1]/2)+2*(p[:,1]>=frame.shape[0]/2)))))
    return h, stats


def gradients(im):
    f = im.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(f,-1,1,0), cv2.Sobel(f,-1,0,1))


def holdouts(frame, reference, h, probes, *, radius=28, search=16):
    """Check excluded road-marking patches using locally translated edge maps."""
    a = gradients(frame)
    b = gradients(cv2.warpPerspective(reference,np.linalg.inv(h),frame.shape[::-1]))
    out = []
    for name, (x,y) in probes.items():
        x, y = int(x),int(y)
        r = radius+search
        if not r <= x < frame.shape[1]-r or not r <= y < frame.shape[0]-r:
            raise ValueError("Road-marking holdout exceeds field of view")
        score = cv2.matchTemplate(b[y-r:y+r+1,x-r:x+r+1],
            a[y-radius:y+radius+1,x-radius:x+radius+1], cv2.TM_CCOEFF_NORMED)
        _, corr, _, loc = cv2.minMaxLoc(score)
        delta = np.asarray(loc)-search
        p = project([[x,y], [x+delta[0],y+delta[1]]],h)
        out.append(dict(patch=name, delta_pixels=delta.tolist(), peak=float(corr),
                        delta_ortho_pixels=float(np.linalg.norm(p[0]-p[1]))))
    return out


def grid_to_utm(points, bbox, shape):
    p = np.asarray(points, dtype=float)
    h,w = shape
    x0,y0,x1,y1 = bbox
    return np.column_stack((x0+(p[:,0]+.5)*(x1-x0)/w, y1-(p[:,1]+.5)*(y1-y0)/h))
