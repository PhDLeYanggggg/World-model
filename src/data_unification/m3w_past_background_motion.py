"""Past-image motion measurement; not camera calibration or metric geometry."""
import csv
import io

import cv2
import numpy as np

QUALITY_NAMES = ('last_supported', 'history_supported_fraction', 'last_inlier_fraction',
    'last_match_fraction', 'last_residual_resized', 'last_fb_resized',
    'last_spatial_coverage', 'previous_background_fraction', 'current_background_fraction',
    'last_log_scale', 'last_rotation_cos', 'last_rotation_sin')
MOTION_NAMES = ('last_background_forward', 'last_background_side', 'last_background_speed',
    'background_heading_cos', 'mean_background_forward', 'mean_background_side',
    'std_background_forward', 'std_background_side', 'last_agent_residual_forward',
    'last_agent_residual_side', 'mean_agent_residual_forward', 'mean_agent_residual_side',
    'std_agent_residual_forward', 'std_agent_residual_side',
    'last_background_accel_forward', 'last_background_accel_side')
STABLE_NAMES = ('stable_last_forward', 'stable_last_side', 'stable_ols4_forward',
    'stable_ols4_side', 'stable_ols8_forward', 'stable_ols8_side', 'stable_turn_cos',
    'stable_turn_sin', 'stable_mean_speed', 'stable_speed_std', 'stable_path_length',
    'stable_net_to_path')
ROLLOUT_NAMES = ('rollout_first_residual_forward', 'rollout_first_residual_side',
    'rollout_end_residual_forward', 'rollout_end_residual_side',
    'rollout_first_residual_speed', 'rollout_end_residual_speed',
    'rollout_first_past_residual_cos', 'rollout_end_past_residual_cos')
NAMES = QUALITY_NAMES+MOTION_NAMES+STABLE_NAMES+ROLLOUT_NAMES


def boxes_by_frame(body):
    """Index released boxes; each image mask later selects one exact past frame."""
    out = {}
    for row in csv.reader(io.StringIO(body.decode('utf-8-sig'))):
        if not row or all(not x.strip() for x in row):
            continue
        if len(row) != 10:
            raise ValueError('Ten-column released MOT rows required')
        a = np.asarray(row, dtype=float)
        if not np.isfinite(a).all() or a[0] != int(a[0]) or a[0] < 0 or np.any(a[4:6] <= 0):
            raise ValueError('Invalid published box geometry')
        x, y, w, h = a[2:6]
        out.setdefault(int(a[0]), []).append((x, y, x+w, y+h))
    return {k: np.asarray(v, float).reshape(-1, 4) for k, v in out.items()}


def prepare(rgb, boxes, cfg):
    rgb, boxes = np.asarray(rgb), np.asarray(boxes)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or boxes.shape != (len(boxes), 4) or not np.isfinite(boxes).all()):
        raise ValueError('Native RGB and same-frame finite boxes required')
    height, width = rgb.shape[:2]; scale = min(1., cfg['long_edge']/max(height, width))
    w, h = max(1, round(width*scale)), max(1, round(height*scale))
    sx, sy = w/width, h/height
    gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA)
    mask = np.full((h, w), 255, np.uint8)
    for x0, y0, x1, y1 in boxes:
        padding = max(x1-x0, y1-y0)*cfg['box_padding_fraction']+cfg['box_padding_pixels']
        a, c = np.clip(np.floor(np.array([x0-padding, x1+padding])*sx), 0, w).astype(int)
        b, d = np.clip(np.floor(np.array([y0-padding, y1+padding])*sy), 0, h).astype(int)
        mask[b:d, a:c] = 0
    center = (boxes[:, :2]+boxes[:, 2:])/2
    inside = ((center[:, 0] >= 0)&(center[:, 0] < width)&(center[:, 1] >= 0)&(center[:, 1] < height))
    return dict(gray=gray, mask=mask, native_size=np.array([width, height]), resize_scale=np.array([sx, sy]),
        centers=len(boxes), centers_in_bounds=int(inside.sum()))


def estimate(previous, current, cfg):
    if (not np.array_equal(previous['native_size'], current['native_size'])
            or previous['gray'].shape != current['gray'].shape
            or not np.array_equal(previous['resize_scale'], current['resize_scale'])):
        raise ValueError('Adjacent images must share native/resized coordinate geometry')
    a, b = previous['gray'], current['gray']; h, w = a.shape
    if any(x.dtype != np.uint8 or x.shape != (h, w) for x in (a, b, previous['mask'], current['mask'])):
        raise ValueError('Lossless 8-bit raster/mask pair required')
    result = dict(supported=False, reason='insufficient_corners', corners=0, matches=0,
        inliers=0, inlier_fraction=0., spatial_coverage=0., residual_median=None,
        fb_median=None, previous_background_fraction=float((previous['mask'] > 0).mean()),
        current_background_fraction=float((current['mask'] > 0).mean()),
        native_affine=np.eye(3).tolist(), log_scale=0., rotation_cos=0., rotation_sin=0.)
    p0 = cv2.goodFeaturesToTrack(a, cfg['max_corners'], cfg['corner_quality'], cfg['min_corner_distance'],
        mask=previous['mask'], blockSize=7)
    if p0 is None:
        return result
    result['corners'] = len(p0)
    options = dict(winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01), minEigThreshold=.0001)
    p1, status1, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **options)
    if p1 is None:
        result['reason'] = 'forward_tracking_failed'; return result
    back, status2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **options)
    if back is None:
        result['reason'] = 'backward_tracking_failed'; return result
    x, y, recovered = p0[:, 0], p1[:, 0], back[:, 0]
    finite = np.isfinite(y).all(1)&np.isfinite(recovered).all(1)
    rounded = np.rint(np.where(np.isfinite(y), y, -1)).astype(np.int64)
    inside = finite&(rounded[:, 0] >= 0)&(rounded[:, 0] < w)&(rounded[:, 1] >= 0)&(rounded[:, 1] < h)
    background = np.zeros(len(x), bool)
    background[inside] = current['mask'][rounded[inside, 1], rounded[inside, 0]] > 0
    fb = np.linalg.norm(recovered-x, axis=1)
    use = status1[:, 0].astype(bool)&status2[:, 0].astype(bool)&background&finite&(fb <= cfg['fb_max'])
    result['matches'] = int(use.sum())
    if use.any():
        result['fb_median'] = float(np.median(fb[use]))
    if use.sum() < cfg['minimum_inliers']:
        result['reason'] = 'insufficient_tracks'; return result
    cv2.setRNGSeed(cfg['ransac_seed'])
    transform, inlier = cv2.estimateAffinePartial2D(x[use], y[use], method=cv2.RANSAC,
        ransacReprojThreshold=cfg['ransac_threshold'], maxIters=2000, confidence=.99, refineIters=10)
    if transform is None or not np.isfinite(transform).all() or inlier is None:
        result['reason'] = 'affine_estimation_failed'; return result
    keep = inlier[:, 0].astype(bool); xx, yy = x[use][keep], y[use][keep]
    result['inliers'] = int(keep.sum()); result['inlier_fraction'] = float(keep.mean())
    if len(xx) >= 3:
        result['spatial_coverage'] = float(cv2.contourArea(cv2.convexHull(xx))/(w*h))
    residual = np.linalg.norm(xx@transform[:, :2].T+transform[:, 2]-yy, axis=1)
    result['residual_median'] = float(np.median(residual)) if len(residual) else None
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    if scale <= 0:
        result['reason'] = 'degenerate_scale'; return result
    S = np.diag([*previous['resize_scale'], 1.]); A = np.eye(3); A[:2] = transform
    result.update(native_affine=(np.linalg.inv(S)@A@S).tolist(), log_scale=float(np.log(scale)),
        rotation_cos=float(transform[0, 0]/scale), rotation_sin=float(transform[1, 0]/scale))
    valid = (keep.sum() >= cfg['minimum_inliers'] and keep.mean() >= cfg['minimum_inlier_fraction']
        and result['spatial_coverage'] >= cfg['minimum_coverage']
        and result['residual_median'] is not None and result['residual_median'] <= cfg['maximum_median_residual']
        and cfg['minimum_scale'] <= scale <= cfg['maximum_scale'])
    result.update(supported=bool(valid), reason='supported' if valid else 'quality_not_supported')
    return result


def contextual_features(history, rollout, query_frame, flows):
    """One query, absolute native past history, relative causal rollout; no labels."""
    h, p = np.asarray(history, float), np.asarray(rollout, float)
    if (h.shape != (len(h), 8, 2) or p.shape != (len(h), 12, 2) or not len(h)
            or not np.isfinite(h).all() or not np.isfinite(p).all() or query_frame < 7):
        raise ValueError('Finite causal history and rollout required')
    requested = list(range(query_frame-6, query_frame+1))
    if set(flows) != set(requested):
        raise ValueError('Exactly seven observed transitions,ending at query,required')
    n = len(h); records = [flows[f] for f in requested]
    supported = np.array([r['supported'] for r in records], bool)
    T = np.stack([np.array(r['native_affine'], float) if r['supported'] else np.eye(3) for r in records])
    v = np.diff(h, axis=1); speed = np.linalg.norm(v, axis=-1); scale = np.maximum(speed.mean(1), 1.)
    direction = np.divide(v[:, -1], speed[:, -1, None], out=np.tile([1., 0.], (n, 1)), where=speed[:, -1, None] > 1e-12)
    def project(x):
        return np.stack(((x*direction[:, None]).sum(-1),
            direction[:, 0, None]*x[..., 1]-direction[:, 1, None]*x[..., 0]), -1)/scale[:, None, None]
    background = np.einsum('tij,ntj->nti', T[:, :2, :2], h[:, :-1])+T[None, :, :2, 2]-h[:, :-1]
    bg, residual = project(background), project(v-background)
    last = records[-1]; quality = np.zeros((n, 12))
    quality[:, 1] = supported.mean()
    if last['supported']:
        quality[:] = np.array([1., supported.mean(), last['inlier_fraction'], last['matches']/1000,
            last['residual_median'], last['fb_median'], last['spatial_coverage'],
            last['previous_background_fraction'], last['current_background_fraction'],
            last['log_scale'], last['rotation_cos'], last['rotation_sin']])
    length = np.linalg.norm(bg[:, -1], axis=1)
    cosine = np.divide(bg[:, -1, 0], length, out=np.zeros(n), where=length > 0)
    motion = np.c_[bg[:, -1], length, cosine, bg.mean(1), bg.std(1), residual[:, -1],
        residual.mean(1), residual.std(1), bg[:, -1]-bg[:, -2]]
    aligned = h.copy(); composition = np.eye(3)
    for t in range(6, -1, -1):
        composition = composition@T[t]
        aligned[:, t] = h[:, t]@composition[:2, :2].T+composition[:2, 2]
    stable = project(np.diff(aligned, axis=1)); sv = stable[:, -1]
    ols = []
    for k in (4, 8):
        times = np.arange(k)-(k-1)/2
        velocity = np.einsum('ntj,t->nj', aligned[:, -k:], times)/np.sum(times*times)
        ols.append(project(velocity[:, None])[:, 0])
    aa, bb = stable[:, -2], stable[:, -1]; turn_den = np.linalg.norm(aa, axis=1)*np.linalg.norm(bb, axis=1)
    turn_cos = np.divide((aa*bb).sum(1), turn_den, out=np.zeros(n), where=turn_den > 0)
    turn_sin = np.divide(aa[:, 0]*bb[:, 1]-aa[:, 1]*bb[:, 0], turn_den, out=np.zeros(n), where=turn_den > 0)
    s = np.linalg.norm(stable, axis=-1); path = s.sum(1)
    net = np.linalg.norm((aligned[:, -1]-aligned[:, 0])/scale[:, None], axis=1)
    stable_features = np.c_[sv, *ols, turn_cos, turn_sin, s.mean(1), s.std(1), path,
        np.divide(net, path, out=np.zeros(n), where=path > 0)]
    pp = project(np.stack((p[:, 0]-background[:, -1], p[:, -1]/12-background[:, -1]), 1))
    norm = np.linalg.norm(pp, axis=-1); past_residual = residual[:, -1]; past_norm = np.linalg.norm(past_residual, axis=1)
    angle = np.divide((pp*past_residual[:, None]).sum(-1), norm*past_norm[:, None],
        out=np.zeros_like(norm), where=norm*past_norm[:, None] > 0)
    rollout_features = np.c_[pp[:, 0], pp[:, 1], norm, angle]
    out = np.c_[quality, motion, stable_features, rollout_features]
    if out.shape != (n, 48) or not np.isfinite(out).all() or len(NAMES) != 48:
        raise ValueError('Finite fixed48-column causal context required')
    return out.astype(np.float32)
