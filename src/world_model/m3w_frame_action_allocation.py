"""Contemporaneous multi-action budget pooling; no outcome labels accepted."""
from collections import Counter
import hashlib
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.world_model.m3w_multiaction_cost import select_actions

ARMS = ('independent', 'frame_uniform', 'joint_matched', 'joint_free')


def validate(prediction, allowed):
    p, ok = np.asarray(prediction, float), np.asarray(allowed)
    if (p.ndim != 3 or p.shape[1:] != (7, 5) or not len(p)
            or ok.shape != p.shape[:2] or ok.dtype != bool
            or not np.isfinite(p).all() or (p < 0).any()):
        raise ValueError('Finite seven-action predicted moments and Boolean causal availability required')
    return p, ok


def positive_actions(p, ok):
    return ok & (p[..., 0] > p[..., 1]) & (p[..., 2] > 0) & (p[..., 3] > 0)


def terms(p, choice):
    at = np.flatnonzero(choice >= 0); v = p[at, choice[at]]
    gain = math.fsum((v[:, 0]-v[:, 1]).tolist())
    q = [math.fsum((v[:, h]-.02*v[:, r]).tolist()) for h, r in ((1, 2), (4, 3))]
    return gain, q


def feasible(prediction, allowed, choice):
    p, ok = validate(prediction, allowed); c = np.asarray(choice)
    if (c.shape != (len(p),) or not np.issubdtype(c.dtype, np.integer)
            or (c < -1).any() or (c >= 7).any()):
        return False
    at = np.flatnonzero(c >= 0)
    return bool(positive_actions(p, ok)[at, c[at]].all() and max(terms(p, c)[1]) <= 0)


def solve_frame(prediction, allowed, parent, *, exact_count, seconds):
    """Maximize predicted gain under both2% budgets, retaining the parent on failure."""
    p, ok = validate(prediction, allowed); parent = np.asarray(parent)
    if (not feasible(p, ok, parent) or not np.isfinite(seconds) or seconds <= 0
            or (exact_count is not None and (not isinstance(exact_count, int)
                or exact_count != int((parent >= 0).sum())))):
        raise ValueError('Feasible frozen parent, unchanged matched count and positive runtime required')
    gain = p[..., 0]-p[..., 1]; legal = positive_actions(p, ok)
    parent_gain = terms(p, parent)[0]

    def result(choice, *, reason, optimal, accepted, solver_status=None):
        value, risk = terms(p, choice)
        if not feasible(p, ok, choice):
            raise ValueError('No numerical risk violation may escape to inference')
        return choice.copy(), dict(reason=reason, optimal=optimal, accepted=accepted,
            solver_status=solver_status, selected=int((choice >= 0).sum()),
            predicted_gain=value, predicted_budget_excess=risk,
            predicted_gain_increment=value-parent_gain, labels_used=False)

    if exact_count == 0:
        return result(parent, reason='zero_count_exact', optimal=True, accepted=True)
    best = np.where(legal, gain, -np.inf).argmax(1)
    candidate = np.where(legal.any(1), best, -1)
    if exact_count is not None:
        rows = np.flatnonzero(candidate >= 0)
        keep = rows[np.argsort(-gain[rows, candidate[rows]], kind='stable')[:exact_count]]
        keep_mask = np.zeros(len(p), bool); keep_mask[keep] = True; candidate[~keep_mask] = -1
    if feasible(p, ok, candidate):
        return result(candidate, reason='unconstrained_optimum_feasible', optimal=True, accepted=True)
    if len(p) == 1:
        return result(parent, reason='singleton_pointwise_exact', optimal=True, accepted=True)

    row, action = np.nonzero(legal); m = len(row)
    per_agent = coo_array((np.ones(m), (row, np.arange(m))), shape=(len(p), m)).tocsc()
    risks = np.stack((p[row, action, 1]-.02*p[row, action, 2],
                      p[row, action, 4]-.02*p[row, action, 3]))
    # Scale each constraint independently, preserving even very small easy moments.
    risk_scale = np.max(np.abs(risks), axis=1).clip(1e-300)
    matrices = [per_agent, csc_array(risks/risk_scale[:, None])]
    low, high = [np.full(len(p), -np.inf), np.full(2, -np.inf)], [np.ones(len(p)), np.zeros(2)]
    if exact_count is not None:
        matrices.append(csc_array(np.ones((1, m))))
        low.append(np.array([exact_count])); high.append(np.array([exact_count]))
    value = milp(c=-gain[row, action]/gain[row, action].max(), integrality=np.ones(m),
        bounds=Bounds(np.zeros(m), np.ones(m)),
        constraints=LinearConstraint(vstack(matrices, format='csc'), np.concatenate(low), np.concatenate(high)),
        options=dict(presolve=True, mip_rel_gap=0., time_limit=seconds))
    if value.status != 0 or value.x is None:
        return result(parent, reason=f'solver_status_{value.status}_parent_fallback',
            optimal=False, accepted=False, solver_status=int(value.status))
    selected = np.flatnonzero(value.x > .5); out = np.full(len(p), -1, np.int64)
    out[row[selected]] = action[selected]
    if (len(np.unique(row[selected])) != len(selected) or not feasible(p, ok, out)
            or (exact_count is not None and len(selected) != exact_count)
            or terms(p, out)[0] < parent_gain):
        return result(parent, reason='postcheck_failed_parent_fallback', optimal=False,
            accepted=False, solver_status=0)
    return result(out, reason='exact_milp', optimal=True, accepted=True, solver_status=0)


def query_groups(recordings, frames, ids):
    rec, frame, ids = map(np.asarray, (recordings, frames, ids))
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or rec.shape != ids.shape or frame.shape != ids.shape or not np.isfinite(frame).all()):
        raise ValueError('Unique causal row IDs and aligned recording/current-frame metadata required')
    keys = np.rec.fromarrays([rec.astype(str), frame], names='recording,frame')
    _, inverse = np.unique(keys, return_inverse=True)
    order = np.lexsort((ids, inverse)); cuts = np.flatnonzero(np.diff(inverse[order]))+1
    return np.split(order, cuts)


def uniform_frame(p, ok):
    best, score = np.full(len(p), -1, np.int64), 0.
    for action in range(7):
        c = np.full(len(p), action, np.int64)
        if feasible(p, ok, c) and terms(p, c)[0] > score:
            best, score = c, terms(p, c)[0]
    return best


def allocate(prediction, allowed, recordings, frames, ids, *, seconds, heartbeat=lambda **_: None,
             checkpoint=lambda _: None, resume_state=None):
    p, ok = validate(prediction, allowed); groups = query_groups(recordings, frames, ids)
    parent, rank = select_actions(p, ok, arm='physical_neural')
    choices = {a: np.full(len(p), -1, np.int64) for a in ARMS}; choices['independent'] = parent.copy()
    reasons = {a: Counter() for a in ARMS[2:]}; nonoptimal = Counter(); changed = Counter()
    bindings = {k: hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
        +np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in
        zip(('prediction', 'allowed', 'recordings', 'frames', 'ids'), (p, ok, recordings, frames, ids))}
    bindings.update(seconds=seconds, version=1)
    start = 0
    if resume_state is not None:
        if (resume_state['bindings'] != bindings or set(resume_state['choices']) != set(ARMS)
                or not 0 <= resume_state['completed_queries'] <= len(groups)):
            raise ValueError('Same frozen causal inputs and allocation design required for resume')
        start = resume_state['completed_queries']
        choices = {a: np.asarray(v).copy() for a, v in resume_state['choices'].items()}
        np.testing.assert_array_equal(choices['independent'], parent)
        for at in groups[:start]:
            for arm in ARMS:
                if not feasible(p[at], ok[at], choices[arm][at]):
                    raise ValueError('Invalid processed-query checkpoint')
            if (choices['joint_matched'][at] >= 0).sum() != (parent[at] >= 0).sum():
                raise ValueError('Matched count changed in checkpoint')
        reasons = {a: Counter(v) for a, v in resume_state['reason_counts'].items()}
        nonoptimal, changed = Counter(resume_state['nonoptimal_queries']), Counter(resume_state['changed_queries'])

    def save(completed):
        checkpoint(dict(bindings=bindings, completed_queries=completed,
            choices={a: v.copy() for a, v in choices.items()},
            reason_counts={a: dict(v) for a, v in reasons.items()},
            nonoptimal_queries=dict(nonoptimal), changed_queries=dict(changed)))

    for index in range(start, len(groups)):
        rows = groups[index]
        x, valid, frozen = p[rows], ok[rows], parent[rows]
        choices['frame_uniform'][rows] = uniform_frame(x, valid)
        for arm, count in (('joint_matched', int((frozen >= 0).sum())), ('joint_free', None)):
            chosen, info = solve_frame(x, valid, frozen, exact_count=count, seconds=seconds)
            choices[arm][rows] = chosen; reasons[arm][info['reason']] += 1
            nonoptimal[arm] += int(not info['optimal']); changed[arm] += int(np.any(chosen != frozen))
        if index == 0 or (index+1) % 250 == 0 or index+1 == len(groups):
            save(index+1)
            heartbeat(queries=index+1, total_queries=len(groups))
    ranks = {}
    for arm, c in choices.items():
        at = np.flatnonzero(c >= 0); r = np.zeros(len(p)); r[at] = p[at, c[at], 0]-p[at, c[at], 1]
        ranks[arm] = r
    np.testing.assert_array_equal(ranks['independent'], rank)
    for rows in groups:
        if (choices['joint_matched'][rows] >= 0).sum() != (parent[rows] >= 0).sum():
            raise AssertionError('Every query, not just the pooled roster, must match intervention count')
    return choices, ranks, dict(queries=len(groups), multiagent_queries=sum(len(v) > 1 for v in groups),
        maximum_agents=max(map(len, groups)), reason_counts={a: dict(v) for a, v in reasons.items()},
        nonoptimal_queries=dict(nonoptimal), changed_queries=dict(changed),
        labels_used=False, all_visible_agents_claim=False, pair_interactions_optimized=False,
        empty_is_not_a_safety_pass=True)
