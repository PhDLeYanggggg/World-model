"""Reference-consistent labels with an unchanged CV-defined evaluation event."""
import numpy as np
import torch
from src.world_model.m3w_european_source_intervention import causal_cost_features, paired_cost_labels
from src.world_model.m3w_geometric_cost_head import rollout_envelope
from src.world_model.m3w_cross_moment_rank import cross_pairs
from src.world_model.m3w_native_forecast import draw_batch


def assert_producer_exclusion(train, target, outer, parent):
    train, target, outer, parent = map(set, (train, target, outer, parent))
    if (len(train) != 2 or len(target) != 2 or len(parent) != 4 or not outer
            or train & target or (train | target) & outer or train | target != parent):
        raise ValueError('Two disjoint inner halves within four fitting sources; all outer sources excluded')


def relative_targets(cv, floor, candidate, *, reference, event, easy_cut):
    cv, floor, candidate = [np.asarray(v, float) for v in (cv, floor, candidate)]
    if (reference not in ('cv', 'floor') or event not in ('all', 'easy')
            or not np.isfinite(easy_cut) or easy_cut <= 0):
        raise ValueError('Registered reference and unchanged fitting-only CV easy cut required')
    paired_cost_labels(cv, floor)
    ref = cv if reference == 'cv' else floor
    utility = paired_cost_labels(ref, candidate)
    known = np.isfinite(cv)
    mask = known if event == 'all' else known & (cv > 0) & (cv <= easy_cut)
    risk = np.zeros_like(utility)
    risk[mask, 0] = ref[mask]
    risk[mask, 1] = utility[mask, 1]
    risk[~known] = np.nan
    return utility, risk


def matched_features(geometry, cv_rollout, floor_rollout, candidate, floor_switch):
    b, d, n = map(np.asarray, (cv_rollout, floor_rollout, candidate))
    switch = np.asarray(floor_switch)
    if switch.dtype != bool or switch.shape != (len(b),) or not np.array_equal(d[~switch], b[~switch]):
        raise ValueError('Explicit causal floor decisions and unchanged nonselected CV rollouts required')
    x, scale = causal_cost_features(geometry, d, n)
    x = np.column_stack((x, (b/scale[:, None, None]).reshape(-1, 24), switch)).astype(np.float32)
    envelope = np.maximum(rollout_envelope(b, n), rollout_envelope(d, n))
    if x.shape != (len(b), 380) or not np.isfinite(x).all():
        raise ValueError('Finite matched causal schema required')
    return x, envelope


def fixed_rank_scale(y, sites, pr, *, seed, batch_size, batches=40):
    known = pr['known']
    target = torch.from_numpy(np.where(known[:, None], y/pr['cost_scale'], 0).astype(np.float32))
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    values = []
    for _ in range(batches):
        ids = draw_batch(groups, batch_size, rng)
        _, _, delta = cross_pairs(target[ids], sites[ids])
        values.append(float(delta.abs().sum()))
    scale = float(np.mean(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('No positive fitting-only pair weight; do not silently invent a denominator')
    return scale
