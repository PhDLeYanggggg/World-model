"""Causal incumbent-relative costs and label-free incremental intervention."""
import numpy as np
from src.world_model.m3w_floor_relative import relative_targets
from src.world_model.m3w_fixed_producer_roles import label_free_replay


def checked_choice(choice, n):
    choice = np.asarray(choice)
    if choice.dtype != bool or choice.shape != (n,):
        raise ValueError('Aligned boolean causal incumbent choice required')
    return choice


def features(x, incumbent):
    x = np.asarray(x)
    incumbent = checked_choice(incumbent, len(x))
    if x.ndim != 2 or x.shape[1] != 380 or not np.isfinite(x).all():
        raise ValueError('Verified 380-dimensional causal schema required')
    return np.column_stack((x, incumbent)).astype(np.float32)


def targets(cv, floor, neural, incumbent, *, arm, event, easy_cut):
    """Future errors are supervision only; event stays anchored to producer CV."""
    floor, neural = np.asarray(floor), np.asarray(neural)
    incumbent = checked_choice(incumbent, len(floor))
    if arm == 'floor_reference':
        ref, alt = floor, neural
    elif arm == 'incumbent_reference':
        ref = np.where(incumbent, neural, floor)
        alt = np.where(incumbent, floor, neural)
    else:
        raise ValueError('Unregistered reference arm')
    return relative_targets(cv, ref, alt, reference='floor', event=event, easy_cut=easy_cut)


def choices(utility, risk, moving, incumbent, *, arm, direction='both'):
    incumbent = checked_choice(incumbent, len(moving))
    utility, risk = np.asarray(utility), np.asarray(risk)
    if (utility.shape != (len(moving), 2) or risk.shape != utility.shape
            or not np.isfinite(utility).all() or not np.isfinite(risk).all()
            or (utility < 0).any() or (risk < 0).any()):
        raise ValueError('Finite nonnegative predicted costs required')
    if np.any(incumbent & ~np.asarray(moving, bool)):
        raise ValueError('Incumbent must preserve the latest-step stopping guard')
    allowed = np.asarray(moving, bool) & (utility[:, 0] > utility[:, 1]) & (risk[:, 1] <= .02*risk[:, 0])
    if arm == 'floor_reference' and direction == 'both':
        return allowed
    if arm != 'incumbent_reference' or direction not in ('both', 'add', 'remove'):
        raise ValueError('Unregistered deployment rule')
    if direction == 'add': allowed &= ~incumbent
    if direction == 'remove': allowed &= incumbent
    return incumbent ^ allowed


def replay(utility, risk, moving, incumbent, *, arm, direction='both'):
    """Scalar decision check uses scores and history, never error labels."""
    gate = label_free_replay((utility, risk), moving)
    if arm == 'floor_reference': return gate
    return np.asarray([not old if switch and (direction == 'both' or
        (direction == 'add' and not old) or (direction == 'remove' and old)) else old
        for old, switch in zip(incumbent, gate)], bool)
