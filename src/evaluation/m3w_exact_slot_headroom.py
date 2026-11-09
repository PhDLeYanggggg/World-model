"""Hindsight upper bound and verified witnesses on a fixed displacement menu."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_cv_joint_feasibility import POSITIVE_PP
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_slate_shared_menu import legal_assignments, slot_menu


def lower(ref, path, target, valid):
    return paired(ref, path, target, valid)[1].sum(-1)


def make_design(cv, ols, bank, additive, native, y, v, q, easy_cut, *, double_only=False):
    peak = np.linalg.norm(native-cv, axis=-1).max(-1)
    add_peak = np.linalg.norm(additive-cv, axis=-1).max(-1)
    _, group, counts = np.unique(q, return_inverse=True, return_counts=True)
    w = 1./counts[group]
    order = np.argsort(group, kind='stable')
    chunks = np.split(order, counts.cumsum()[:-1])
    ref = float(w @ np.where(v, np.linalg.norm(cv-y, axis=-1), 0.).sum(-1))
    if not np.any(peak > 0):
        if np.any(add_peak > 0):
            raise ValueError('Unmatched no-edit record')
        return None, 'no_edits'
    if ref <= 0:
        return None, 'no_reference'
    norm = 100./ref
    base = norm * np.array([0., w @ lower(ols, cv, y, v), w @ lower(additive, cv, y, v)])
    cv_ade, _ = errors(cv, y, v)
    easy = v.any(1) & (cv_ade > 0.) & (cv_ade <= easy_cut)
    easy_cap = .02 * float(cv_ade[easy].sum())
    rows, slots, paths, gains, risks, allowed = [], [], [], [], [], []
    upper = base.copy()
    n_slots = 0
    for ix in chunks:
        radii = peak[ix][peak[ix] > 0]
        np.testing.assert_allclose(np.sort(add_peak[ix][add_peak[ix] > 0]), np.sort(radii), atol=1e-9, rtol=1e-9)
        if not len(radii):
            continue
        options, choices = slot_menu(cv[ix], bank[ix], radii)
        at = np.concatenate([r for r, _ in choices])
        ids = ix[at]
        cols = np.concatenate([c+j*bank.shape[1] for j, (_, c) in enumerate(choices)])
        actions = options[at, cols]
        enabled = np.ones(len(ids), dtype=bool)
        np.testing.assert_allclose(np.linalg.norm(actions-cv[ids], axis=-1).max(-1),
                                   np.repeat(radii, [len(r) for r, _ in choices]), atol=1e-9, rtol=1e-9)
        # Every edited additive path must be an actual menu member, not just have the same radius.
        for r in ix[add_peak[ix] > 0]:
            members = np.flatnonzero((ids == r) & np.isclose(actions, additive[r], atol=1e-9, rtol=1e-9).all((1, 2)))
            if not len(members):
                raise ValueError('Additive control lies outside the frozen path menu')
            if double_only and len(radii) == 1:
                enabled[:] = False
                enabled[members[0]] = True
        _, _, action_gain, _ = candidate_terms(cv[ids], ols[ids], actions[:, None], y[ids], v[ids])
        direct = lower(additive[ids], actions, y[ids], v[ids])-lower(additive[ids], cv[ids], y[ids], v[ids])
        value = np.column_stack((action_gain[:, 0], direct))*w[ids, None]*norm
        risk = risk_excess(cv[ids], actions[:, None], y[ids], v[ids], easy_cut)[:, 0]
        signed = np.where(easy[ids], errors(actions, y[ids], v[ids])[0]-cv_ade[ids], 0.)
        legal = legal_assignments(choices[0][0], choices[1][0] if len(choices) == 2 else None)
        legal = legal[enabled[legal[:, 0]]]
        if not len(legal):
            raise ValueError('Frozen menu has no legal option')
        local_upper = (value[legal[:, 0]] + np.where(legal[:, 1, None] >= 0, value[legal[:, 1]], 0.)).max(0)
        upper += local_upper
        rows.append(ids)
        slots.append(np.repeat(np.arange(n_slots, n_slots+len(radii)), [len(r) for r, _ in choices]))
        paths.append(actions)
        gains.append(value)
        risks.append(np.column_stack((risk, signed)))
        allowed.append(enabled)
        n_slots += len(radii)
    row = np.concatenate(rows)
    enabled = np.concatenate(allowed)
    return dict(row=row, slot=np.concatenate(slots), slots=n_slots, path=np.concatenate(paths), enabled=enabled,
                gain=np.concatenate(gains), risk=np.concatenate(risks), easy=easy[row] & enabled, easy_cap=easy_cap,
                offset=base, upper=upper, reference=ref), None


def solve(d, seconds):
    """Maximize direct paired utility subject to exact slots and original risk."""
    if not d['easy'].any():
        return None, dict(status='no_easy_support', solver_status=None, upper_pp=None)
    if min(d['upper'][:2]) < POSITIVE_PP:
        return None, dict(status='nonpositive_reference_bound', solver_status=None, upper_pp=None)
    row, slot = d['row'], d['slot']
    _, r = np.unique(row, return_inverse=True)
    n = len(row)
    cols = np.arange(n)
    by_row = coo_array((np.ones(n), (r, cols)), shape=(r.max()+1, n))
    by_slot = coo_array((np.ones(n), (slot, cols)), shape=(d['slots'], n))
    scale = np.abs(d['risk']).max(0).clip(1e-6)
    extra = csc_array(np.vstack((d['risk'].T/scale[:, None], -d['easy'].astype(float), -d['gain'][:, :2].T)))
    matrix = vstack((by_row, by_slot, extra), format='csc')
    lb = np.r_[np.full(by_row.shape[0], -np.inf), np.ones(d['slots']), np.full(6, -np.inf)]
    ub = np.r_[np.ones(by_row.shape[0]+d['slots']), 0., 0., d['easy_cap']/scale[2], -1., d['offset'][:2]-POSITIVE_PP]
    result = milp(-d['gain'][:, 2], integrality=np.ones(n), bounds=Bounds(0., d['enabled'].astype(float)),
                  constraints=LinearConstraint(matrix, lb, ub),
                  options=dict(time_limit=seconds, mip_rel_gap=0.))
    upper = getattr(result, 'mip_dual_bound', None)
    upper = min(float(d['upper'][2]), float(d['offset'][2]-upper)) if upper is not None and np.isfinite(upper) else None
    status = {0: 'optimal', 1: 'limited', 2: 'infeasible'}.get(result.status, 'unresolved')
    use = result.x > .5 if result.x is not None else None
    if use is not None:
        if not np.array_equal(np.bincount(slot[use], minlength=d['slots']), np.ones(d['slots'])):
            raise ValueError('Integer result failed exact slot coverage')
        if len(set(row[use])) != d['slots']:
            raise ValueError('Integer result assigned two slots to one agent')
    return use, dict(status=status, solver_status=int(result.status), upper_pp=upper)
