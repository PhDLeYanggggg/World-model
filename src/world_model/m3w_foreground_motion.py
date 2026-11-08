"""Measure independent local pixel motion inside past annotation boxes."""

import cv2
import numpy as np


NAMES = ("coverage", "last_supported", "fb", "photometric_error",
         "last_residual_forward", "last_residual_side", "mean_residual_forward",
         "mean_residual_side", "mean_residual_size", "motion_agreement",
         "foreground_speed", "foreground_speed_std")
OFFSETS = np.array([[0., 0.], [-.23, -.23], [-.23, .23], [.23, -.23], [.23, .23]])


def flow(previous, current, box):
    """Track five sampling locations per previous box; no next-box fit."""
    a, b = previous["gray"], current["gray"]
    s = np.asarray(previous["resize_scale"])
    if (a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8
            or not np.array_equal(s, current["resize_scale"])):
        raise ValueError("Mismatched past image geometry")
    n = len(box)
    out = np.zeros((n, 5), dtype=float)
    if not n:
        return out
    center = box[:, :2] + box[:, 2:] / 2.
    loc = (center[:, None] + box[:, None, 2:] * OFFSETS) * s
    points = loc.reshape(-1, 1, 2).astype(np.float32)
    opts = dict(winSize=(13, 13), maxLevel=2, minEigThreshold=.0001,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, .015))
    nxt, status, error = cv2.calcOpticalFlowPyrLK(a, b, points, None, **opts)
    if nxt is None:
        return out
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **opts)
    if back is None:
        return out
    x, y = points[:, 0], nxt[:, 0]
    size = np.array([a.shape[1], a.shape[0]])
    fb = np.linalg.norm(back[:, 0] - x, axis=1)
    inside = ((x >= 1.) & (x < size-1.) & (y >= 1.) & (y < size-1.)).all(1)
    ok = (inside & status.ravel().astype(bool) & reverse.ravel().astype(bool)
          & np.isfinite(fb) & (fb <= 1.5) & np.isfinite(error.ravel()))
    if not ok.any():
        return out
    velocity = ((y-x)/s).reshape(n, len(OFFSETS), 2)
    fb, error, ok = (v.reshape(n, len(OFFSETS)) for v in (fb, error, ok))
    for i in np.flatnonzero(ok.sum(1) >= 3):
        m = ok[i]
        out[i] = (*np.median(velocity[i, m], axis=0), 1., float(np.median(fb[i, m])),
                  float(np.median(error[i, m])))
    return out


def features(history, observed_flow):
    if history.shape != (len(history), 8, 2) or observed_flow.shape != (len(history), 7, 5):
        raise ValueError("Expected seven past transitions")
    motion = np.diff(history, axis=1)
    speed = np.linalg.norm(motion, axis=-1)
    scale = speed.mean(1).clip(1.)
    axis = motion[:, -1] / speed[:, -1, None].clip(1e-8)
    axis[speed[:, -1] <= 1e-8] = (1., 0.)
    left = np.column_stack((-axis[:, 1], axis[:, 0]))
    v, valid = observed_flow[:, :, :2], observed_flow[:, :, 2] == 1.
    residual = v - motion
    count = valid.sum(1).clip(1)

    def average(x):
        return np.where(valid, x, 0.).sum(1)/count

    fwd = np.einsum("nti,ni->nt", residual, axis)/scale[:, None]
    side = np.einsum("nti,ni->nt", residual, left)/scale[:, None]
    pixel_speed = np.linalg.norm(v, axis=-1)
    agreement = (v*motion).sum(2)/(pixel_speed*speed).clip(1e-8)
    mean_speed = average(pixel_speed)
    out = np.column_stack((valid.mean(1), valid[:, -1], average(observed_flow[:, :, 3]),
        average(observed_flow[:, :, 4])/32., fwd[:, -1]*valid[:, -1], side[:, -1]*valid[:, -1],
        average(fwd), average(side), average(np.linalg.norm(residual, axis=-1))/scale,
        average(agreement), mean_speed/scale,
        np.sqrt(average((pixel_speed-mean_speed[:, None])**2))/scale))
    return out.clip(-16., 16.).astype(np.float32)
