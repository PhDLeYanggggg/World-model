"""Exact allocation and shared-recipient components of a frozen contrast."""

import numpy as np


PARTS = ('joint_only', 'control_only', 'shared')
METRICS = ('observed_pp', 'missing_debit_pp', 'lower_pp', 'upper_pp')


def decompose(joint, control, y, valid, joint_rows, control_rows, cv_mass):
    n = len(joint)
    if (joint.shape != control.shape or y.shape != joint.shape or joint.shape != (n, 12, 2)
            or valid.shape != (n, 12) or valid.dtype != bool or not np.isfinite(y[valid]).all()
            or not np.isfinite(joint).all() or not np.isfinite(control).all()
            or not np.isfinite(cv_mass) or cv_mass <= 0.):
        raise ValueError('Need aligned paths, label masks and a positive CV denominator')
    a, b = map(np.asarray, (joint_rows, control_rows))
    if any(t.shape != (2,) or not np.issubdtype(t.dtype, np.integer) or np.any((t < 0) | (t >= n))
           or t[0] == t[1] for t in (a, b)):
        raise ValueError('Need two distinct recipients in each arm')
    left, right = set(a), set(b)
    changed = np.any(joint != control, axis=(1, 2))
    if not set(np.flatnonzero(changed)) <= left | right:
        raise ValueError('An untouched recipient has different forecasts')
    scale = 100./(n*cv_mass)
    known = np.where(valid, np.linalg.norm(control-y, axis=-1)-np.linalg.norm(joint-y, axis=-1), 0.)
    debit = np.where(valid, 0., np.linalg.norm(joint-control, axis=-1))
    result = {}
    for tag, people in zip(PARTS, (left-right, right-left, left & right), strict=True):
        people = np.asarray(sorted(people), dtype=int)
        o, d = float(known[people].sum()*scale), float(debit[people].sum()*scale)
        result[tag] = dict(rows=len(people), different_paths=int(changed[people].sum()),
                           observed_pp=o, missing_debit_pp=d, lower_pp=o-d, upper_pp=o+d)
    return result
