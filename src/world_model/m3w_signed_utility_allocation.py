"""Relax only the per-action positive-utility filter,not the risk budget."""
from collections import Counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack
from src.world_model import m3w_frame_action_allocation as base

NAMES = ('positive_matched', 'signed_matched', 'positive_free', 'signed_free')


def legal(p, ok):
    return ok & (p[..., 2] > 0) & (p[..., 3] > 0)


def feasible(prediction, allowed, choice):
    p, ok = base.validate(prediction, allowed); c = np.asarray(choice)
    if (c.shape != (len(p),) or not np.issubdtype(c.dtype, np.integer)
            or ((c < -1) | (c >= 7)).any()): return False
    at = np.flatnonzero(c >= 0); gain, risk = base.terms(p, c)
    return bool(legal(p, ok)[at, c[at]].all() and gain >= 0 and max(risk) <= 0)


def solve(prediction, allowed, parent, *, exact_count, seconds):
    p, ok = base.validate(prediction, allowed); parent = np.asarray(parent)
    if (not feasible(p, ok, parent) or not np.isfinite(seconds) or seconds <= 0
            or exact_count is not None and (type(exact_count) is not int
                or exact_count != int((parent >= 0).sum()))):
        raise ValueError('A feasible anchor with the exact registered count is required')
    gain = p[..., 0]-p[..., 1]; anchor_gain = base.terms(p, parent)[0]
    real = legal(p, ok)

    def result(c, reason, optimal, accepted):
        score, risks = base.terms(p, c); at = np.flatnonzero(c >= 0)
        if not feasible(p, ok, c) or score < anchor_gain: raise ValueError('Anchor nesting violated')
        return c.copy(), dict(reason=reason, optimal=optimal, accepted=accepted,
            selected=len(at), predicted_gain=score, predicted_gain_increment=score-anchor_gain,
            predicted_budget_excess=risks, selected_nonpositive_utility_actions=int((gain[at, c[at]] <= 0).sum()),
            labels_used=False, positive_harm_not_net_credit=True)

    if exact_count == 0 or not real.any(): return result(parent, 'zero_count_or_no_action_exact', True, True)
    best = np.where(real, gain, -np.inf).argmax(1)
    c = np.where(real.any(1) & (gain[np.arange(len(p)), best] > 0), best, -1)
    if exact_count is not None:
        rows = np.flatnonzero(c >= 0)
        keep = rows[np.argsort(-gain[rows, c[rows]], kind='stable')[:exact_count]]
        c[np.setdiff1d(np.arange(len(p)), keep)] = -1
    if ((exact_count is None or int((c >= 0).sum()) == exact_count)
            and feasible(p, ok, c) and base.terms(p, c)[0] >= anchor_gain):
        return result(c, 'unconstrained_optimum_feasible', True, True)
    row, action = np.nonzero(real); m = len(row)
    agents = coo_array((np.ones(m), (row, np.arange(m))), shape=(len(p), m)).tocsc()
    risk = np.stack((p[row, action, 1]-.02*p[row, action, 2], p[row, action, 4]-.02*p[row, action, 3]))
    matrices = [agents, csc_array(risk/np.max(np.abs(risk), axis=1).clip(1e-300)[:, None])]
    low = [np.full(len(p), -np.inf), np.full(2, -np.inf)]
    high = [np.ones(len(p)), np.zeros(2)]
    if exact_count is not None:
        matrices.append(csc_array(np.ones((1, m)))); low.append(np.array([exact_count])); high.append(np.array([exact_count]))
    scale = max(float(np.abs(gain[row, action]).max()), 1e-300)
    value = milp(c=-gain[row, action]/scale, integrality=np.ones(m), bounds=Bounds(np.zeros(m), np.ones(m)),
        constraints=LinearConstraint(vstack(matrices, format='csc'), np.concatenate(low), np.concatenate(high)),
        options=dict(presolve=True, mip_rel_gap=0., time_limit=seconds))
    if value.status != 0 or value.x is None:
        return result(parent, f'solver_status_{value.status}_anchor_fallback', False, False)
    take = np.flatnonzero(value.x > .5); out = np.full(len(p), -1, np.int64); out[row[take]] = action[take]
    if (len(set(row[take])) != len(take) or not feasible(p, ok, out)
            or exact_count is not None and len(take) != exact_count or base.terms(p, out)[0] < anchor_gain):
        return result(parent, 'postcheck_failed_anchor_fallback', False, False)
    return result(out, 'exact_milp', True, True)


def allocate(prediction, allowed, recordings, frames, ids, anchor, *, seconds,
             checkpoint=lambda _: None, heartbeat=lambda **_: None, resume_state=None):
    p, ok = base.validate(prediction, allowed); anchor = np.asarray(anchor)
    groups = base.query_groups(recordings, frames, ids)
    bind = {k: base.hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
        +np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in zip(
            ('prediction', 'allowed', 'recordings', 'frames', 'ids', 'anchor'),
            (p, ok, recordings, frames, ids, anchor))}
    bind.update(seconds=seconds, version=1)
    choices = {a: np.full(len(p), -1, np.int64) for a in NAMES}; choices['positive_matched'] = anchor.copy()
    reasons = {a: Counter() for a in NAMES[1:]}; nonoptimal = Counter(); start = 0
    if resume_state is not None:
        if resume_state['bindings'] != bind: raise ValueError('Frozen causal inputs changed')
        start = resume_state['completed_queries']
        if not 0 <= start <= len(groups): raise ValueError('Invalid resume position')
        choices = {a: np.asarray(resume_state['choices'][a]).copy() for a in NAMES}
        np.testing.assert_array_equal(choices['positive_matched'], anchor)
        reasons = {a: Counter(v) for a, v in resume_state['reasons'].items()}
        nonoptimal = Counter(resume_state['nonoptimal'])
        for g in groups[:start]:
            if any(not (base.feasible if a.startswith('positive') else feasible)(p[g], ok[g], choices[a][g]) for a in NAMES):
                raise ValueError('Processed allocation no longer feasible')
            if int((choices['signed_matched'][g] >= 0).sum()) != int((anchor[g] >= 0).sum()):
                raise ValueError('Processed matched count changed')
    for i in range(start, len(groups)):
        g = groups[i]; q, allowed_g, old = p[g], ok[g], anchor[g]
        if not base.feasible(q, allowed_g, old): raise ValueError('Original positive-action anchor required')
        k = int((old >= 0).sum())
        for name, solver, incumbent, count in (
                ('signed_matched', solve, old, k), ('positive_free', base.solve_frame, old, None),
                ('signed_free', solve, None, None)):
            if incumbent is None: incumbent = choices['signed_matched'][g]
            c, info = solver(q, allowed_g, incumbent, exact_count=count, seconds=seconds)
            choices[name][g] = c; reasons[name][info['reason']] += 1; nonoptimal[name] += int(not info['optimal'])
        if i == 0 or (i+1) % 250 == 0 or i+1 == len(groups):
            checkpoint(dict(bindings=bind, completed_queries=i+1, choices=choices,
                reasons={a: dict(v) for a, v in reasons.items()}, nonoptimal=dict(nonoptimal)))
            heartbeat(queries=i+1, total_queries=len(groups))
    return choices, dict(queries=len(groups), reasons={a: dict(v) for a, v in reasons.items()},
        nonoptimal=dict(nonoptimal), labels_used=False, matched_count_every_query=True,
        positive_harm_budget_unchanged=True, risk_budget=.02)
