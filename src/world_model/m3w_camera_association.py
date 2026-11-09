"""Online box association with optional image-plane camera compensation."""

from collections import deque

import cv2
import numpy as np

from src.world_model.m3w_streaming_boxes import match


def background_mask(shape, boxes):
    h, w = shape
    mask = np.full(shape, 255, dtype=np.uint8)
    for box in boxes:
        x1, y1 = np.maximum(0, np.floor(box[:2]-4).astype(int))
        x2, y2 = np.minimum([w, h], np.ceil(box[2:4]+4).astype(int))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask


def camera_step(previous, current, previous_boxes, current_boxes):
    """Estimate a two-frame similarity from background points; otherwise use identity."""
    if previous.ndim != 2 or previous.shape != current.shape:
        raise ValueError('Camera motion requires equal-size grayscale past/current images')
    h, w = previous.shape
    result = dict(tracks=0, inliers=0, quadrants=0, accepted=False, reason='no_corners')
    identity = np.eye(3)
    points = cv2.goodFeaturesToTrack(previous, maxCorners=900, qualityLevel=.01, minDistance=12,
                                     mask=background_mask(previous.shape, previous_boxes))
    if points is None:
        return identity, result
    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    moved, fwd, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **lk)
    back, bwd, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, **lk)
    a, b, back = (p.reshape(-1, 2) for p in (points, moved, back))
    finite = np.isfinite(b).all(axis=1) & np.isfinite(back).all(axis=1)
    good = finite & fwd.ravel().astype(bool) & bwd.ravel().astype(bool)
    good &= np.linalg.norm(a-back, axis=1) <= 1.
    good &= (b >= 0).all(axis=1) & (b < [w, h]).all(axis=1)
    a, b = a[good], b[good]
    bx, by = b.astype(int).T
    keep = background_mask(current.shape, current_boxes)[by, bx] != 0
    a, b = a[keep], b[keep]
    result.update(tracks=len(a), reason='insufficient_tracks')
    if len(a) < 24:
        return identity, result
    motion, inlier = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.,
                                                maxIters=2000, confidence=.99, refineIters=10)
    if motion is None or not np.isfinite(motion).all():
        result['reason'] = 'fit_failed'
        return identity, result
    consensus = a[inlier.ravel().astype(bool)]
    q = np.unique((consensus[:, 0] >= w/2).astype(int) + 2*(consensus[:, 1] >= h/2))
    result.update(inliers=len(consensus), quadrants=len(q), reason='weak_consensus')
    if len(consensus) < 24 or len(consensus)/len(a) < .60 or len(q) < 3:
        return identity, result
    scale = float(np.linalg.norm(motion[:, 0]))
    center = np.array([w/2, h/2])
    shift = float(np.linalg.norm(motion @ np.r_[center, 1.] - center))
    if not .8 <= scale <= 1.25 or shift > .25*np.hypot(w, h):
        result['reason'] = 'implausible_similarity'
        return identity, result
    result.update(accepted=True, reason='accepted', scale=scale, center_shift_px=shift)
    return np.vstack((motion, [0., 0., 1.])), result


class MotionAssociation:
    """Predict positions only for assignment; emit unmodified observed boxes."""

    def __init__(self, velocity):
        self.velocity = velocity
        self.index = -1
        self.next_id = 0
        self.pose = np.eye(3)
        self.active = {}

    def projected(self, history, index):
        last_index, box, old_inv = history[-1]
        transform = self.pose @ old_inv
        center = transform[:2] @ np.r_[(box[:2]+box[2:4])/2., 1.]
        half = .5*(box[2:4]-box[:2])*np.linalg.norm(transform[:2, 0])
        if self.velocity and len(history) == 2:
            older_index, older, older_inv = history[0]
            older_center = (self.pose @ older_inv)[:2] @ np.r_[(older[:2]+older[2:4])/2., 1.]
            center += (center-older_center)*(index-last_index)/(last_index-older_index)
        return np.r_[center-half, center+half, box[4:]]

    def step(self, index, detection, motion=None):
        if index != self.index+1:
            raise ValueError('Track updates must consume every sampled frame in order')
        if (detection.ndim != 2 or detection.shape[1] != 6 or not np.isfinite(detection).all()
                or (detection[:, 2:4] <= detection[:, :2]).any()):
            raise ValueError('Expected finite xyxy, confidence, class')
        if motion is not None:
            if motion.shape != (3, 3) or not np.isfinite(motion).all() or not np.array_equal(motion[2], [0, 0, 1]):
                raise ValueError('Expected a finite image-plane transform')
            self.pose = motion @ self.pose
        inverse = np.linalg.inv(self.pose)
        self.index = index
        self.active = {key: h for key, h in self.active.items() if index-h[-1][0] <= 2}
        keys = list(self.active)
        lookup = {}
        if keys and len(detection):
            predicted = np.stack([self.projected(self.active[k], index) for k in keys])
            lookup = {j: keys[i] for i, j in match(predicted, detection)}
        rows = []
        for j, box in enumerate(detection):
            key = lookup.get(j)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.active[key] = deque(maxlen=2)
            self.active[key].append((index, box.copy(), inverse))
            rows.append((key, *box))
        return np.asarray(rows, dtype=float).reshape(-1, 7)
