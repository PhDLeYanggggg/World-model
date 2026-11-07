"""Hindsight support and utility under sparse, count-matched agent exchanges."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, vstack

from src.evaluation.m3w_frozen_amplitude import shrink
from src.evaluation.m3w_prefix_capacity import incidence
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments


CAPS = (1, 4, 16)
SECONDS = 4.


def prepare(floor, bank, choices, groups, target, valid, easy_cut):
    n, families = bank.shape[:2]
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n)):
        raise ValueError('Queries must partition the recording')
    qbank = floor[:, None]+.25*(bank-floor[:, None])
    m = [moments(floor, qbank[:, a], target, valid, easy_cut) for a in range(families)]
    keys = ('all_excess', 'easy_excess', 'easy_signed')
    costs = np.stack([np.column_stack([z[k] for z in m]) for k in keys], axis=-1)
    radius = np.column_stack([z['radius'] for z in m])
    diff = np.column_stack([z['diff'] for z in m])
    known, easy = m[0]['known'], m[0]['easy']
    selected = np.flatnonzero(choices >= 0)
    old = choices[selected]
    if not (radius[selected, old] > 0).all():
        raise ValueError('Original choice must change its forecast')
    base = np.r_[costs[selected, old].sum(0), int(easy[selected].sum())]
    before = floor.copy()
    before[selected] = shrink(floor[selected], bank[selected, old], .25)
    events, moves, changes, gains = [], [], [], []
    for j, q in enumerate(groups):
        remove = q[choices[q] >= 0]
        if not len(remove):
            continue
        insert, a = np.where((choices[q, None] < 0) & (radius[q] > 0))
        if not len(insert):
            continue
        i = np.tile(q[insert], len(remove))
        f = np.tile(a, len(remove))
        d = np.repeat(remove, len(insert))
        fd = choices[d]
        events.append(np.full(len(i), j, dtype=np.int32))
        moves.append(np.column_stack((i, f, d)))
        delta = costs[i, f]-costs[d, fd]
        changes.append(np.column_stack((delta, easy[i].astype(int)-easy[d])))
        gains.append((np.where(known[i], diff[i, f], -radius[i, f])
                      +np.where(known[d], -diff[d, fd], -radius[d, fd]))/len(q))
    return dict(before=before, floor=floor, quarter_bank=qbank, base=base, limits=np.array(
        [0., 0., BUDGET*m[0]['ref_easy'].sum(), 1.]),
        query=np.concatenate(events) if events else np.empty(0, dtype=int),
        moves=np.concatenate(moves) if moves else np.empty((0, 3), dtype=int),
        changes=np.concatenate(changes) if changes else np.empty((0, 4)),
        gains=np.concatenate(gains) if gains else np.empty(0), queries=len(groups))


def passes(mass, limits):
    return (mass[..., :3] <= limits[:3]).all(-1) & (mass[..., 3] >= limits[3])


def solve(p, cap, seconds=SECONDS):
    if cap not in CAPS:
        raise ValueError('Unsupported replacement cap')
    gains = p['gains']
    empty = np.empty(0, dtype=int)
    base = bool(passes(p['base'], p['limits']))
    if cap == 1 or not len(gains):
        good = np.flatnonzero(passes(p['base']+p['changes'], p['limits']) & (gains >= 0.))
        if len(good) and (not base or gains[good].max() > 0.):
            take = good[np.argmax(gains[good])][None]
            return take, 'optimal', float(gains[take[0]])
        return (empty, 'optimal', 0.) if base else (None, 'infeasible', None)
    if p['base'][3]+min(cap, len(np.unique(p['query'][p['changes'][:, 3] > 0]))) < 1.:
        return None, 'infeasible', None
    unit = np.abs(p['changes']).max(0).clip(1.)
    score_unit = max(1., np.abs(gains).max())
    target = (p['limits']-p['base'])/unit
    A = vstack((incidence(p['query'], p['queries']), csc_array(np.ones((1, len(gains)))),
                csc_array((p['changes']/unit).T), csc_array(gains[None]/score_unit)), format='csc')
    low = np.r_[np.full(p['queries']+4, -np.inf), target[3], 0.]
    high = np.r_[np.ones(p['queries']), cap, target[:3], np.inf, np.inf]
    res = milp(-gains/score_unit, integrality=np.ones(len(gains)), bounds=Bounds(0., 1.),
               constraints=LinearConstraint(A, low, high),
               options=dict(time_limit=seconds, mip_rel_gap=0.))
    dual = getattr(res, 'mip_dual_bound', None)
    bound = None if dual is None or not np.isfinite(dual) else float(-dual*score_unit)
    if res.x is None:
        return None, 'infeasible' if res.status == 2 else 'unresolved', bound
    take = np.flatnonzero(res.x > .5)
    if (len(take) > cap or len(np.unique(p['query'][take])) != len(take)
            or not passes(p['base']+p['changes'][take].sum(0), p['limits'])
            or gains[take].sum() < 0.):
        return None, 'unresolved_postcheck', bound
    return take, 'optimal' if res.status == 0 else 'feasible_unproven', bound


def construct(p, picks):
    pred = p['before'].copy()
    i, a, d = p['moves'][picks].T
    pred[d] = p['floor'][d]
    pred[i] = p['quarter_bank'][i, a]
    return pred
