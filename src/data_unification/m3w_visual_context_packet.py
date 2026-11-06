"""Action-aligned past-image features with a quality-only motion ablation."""
import numpy as np
from src.data_unification.m3w_past_background_motion import contextual_features
from src.world_model.m3w_visual_risk_cost import edge_features, ARMS


def packet(history, paths, recordings, frames, recording_names, flow_loader, size_loader):
    h, p, rec, frame = map(np.asarray, (history, paths, recordings, frames))
    n = len(h)
    if (h.shape != (n, 8, 2) or p.shape != (n, 7, 12, 2) or rec.shape != (n,) or frame.shape != (n,)
            or not np.isfinite(h).all() or not np.isfinite(p).all() or (frame < 7).any()):
        raise ValueError('Original causal history,seven rollouts and exact query metadata required')
    motion = np.empty((n, 7, 56), np.float32); quality = np.empty_like(motion)
    for ri in np.unique(rec):
        node = recording_names[int(ri)]
        for q in np.unique(frame[rec == ri]):
            ix = np.flatnonzero((rec == ri)&(frame == q)); q = int(q)
            flows = {f: flow_loader(node, f) for f in range(q-6, q+1)}
            suppressed = {f: dict(r, native_affine=np.eye(3).tolist(), log_scale=0.,
                rotation_cos=1. if r['supported'] else 0., rotation_sin=0.) for f, r in flows.items()}
            size = size_loader(node, q)
            for action in range(7):
                edge = edge_features(h[ix], p[ix, action], size)
                motion[ix, action] = np.c_[contextual_features(h[ix], p[ix, action], q, flows), edge]
                quality[ix, action] = np.c_[contextual_features(h[ix], p[ix, action], q, suppressed), edge]
    for x in (motion, quality):
        for columns in (slice(0, 40), slice(48, 54)):
            np.testing.assert_array_equal(x[:, :, columns], np.broadcast_to(x[:, :1, columns], x[:, :, columns].shape))
    return dict(motion=motion, quality=quality)


def select(table, arm):
    if arm not in ARMS or table['motion'].shape != table['quality'].shape or table['motion'].shape[-1] != 56:
        raise ValueError('Matched fixed image context arms required')
    if arm == 'motion_geometry': return table['motion']
    if arm == 'quality_geometry': return table['quality']
    out = np.zeros_like(table['motion'])
    if arm == 'image_bounds': out[..., 48:] = table['motion'][..., 48:]
    return out
