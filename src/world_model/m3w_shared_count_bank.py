"""Reoptimize physical/neural action banks at one causal physical query count."""
from collections import Counter
import hashlib
import numpy as np

from src.world_model import m3w_frame_action_allocation as base

ARMS = ('physical_independent', 'neural_trimmed', 'physical_joint', 'neural_joint')


def physical_mask(allowed):
    out = np.asarray(allowed).copy()
    if out.ndim != 2 or out.shape[1] != 7 or out.dtype != bool:
        raise ValueError('Seven Boolean causal action columns required')
    out[:, 6] = False
    return out


def trimmed_neural(p, independent, count):
    chosen = np.full(len(p), -1, np.int64)
    selected = np.flatnonzero(independent >= 0)
    if len(selected) < count:
        raise ValueError('Adding an action cannot shrink pointwise admission support')
    gain = p[selected, independent[selected], 0]-p[selected, independent[selected], 1]
    keep = selected[np.argsort(-gain, kind='stable')[:count]]
    chosen[keep] = independent[keep]
    return chosen


def allocate(prediction, allowed, recordings, frames, ids, *, seconds,
             heartbeat=lambda **_: None, checkpoint=lambda _: None, resume_state=None):
    p, ok = base.validate(prediction, allowed); physical = physical_mask(ok)
    groups = base.query_groups(recordings, frames, ids)
    pc, _ = base.select_actions(p, physical, arm='physical_neural')
    nc, _ = base.select_actions(p, ok, arm='physical_neural')
    if np.any((pc >= 0) & (nc < 0)):
        raise ValueError('Physical admissibility must be a subset of neural-bank admissibility')
    choices = {a: np.full(len(p), -1, np.int64) for a in ARMS}
    choices['physical_independent'] = pc.copy()
    reasons = {a: Counter() for a in ARMS[2:]}; nonoptimal = Counter(); changed = Counter(); warm = Counter()
    bindings = {k: hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
        +np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in
        zip(('prediction', 'allowed', 'recordings', 'frames', 'ids'), (p, ok, recordings, frames, ids))}
    bindings.update(seconds=seconds, version=1, count_anchor='physical_independent')
    start = 0
    if resume_state is not None:
        if (resume_state['bindings'] != bindings or set(resume_state['choices']) != set(ARMS)
                or not 0 <= resume_state['completed_queries'] <= len(groups)):
            raise ValueError('Same frozen predictions and common-count design required')
        start = resume_state['completed_queries']
        choices = {a: np.asarray(c).copy() for a, c in resume_state['choices'].items()}
        np.testing.assert_array_equal(choices['physical_independent'], pc)
        for at in groups[:start]:
            count = int((pc[at] >= 0).sum())
            for a, c in choices.items():
                legal = physical[at] if a.startswith('physical_') else ok[at]
                if not base.feasible(p[at], legal, c[at]) or (c[at] >= 0).sum() != count:
                    raise ValueError('Processed common-count checkpoint is not feasible')
        reasons = {a: Counter(v) for a, v in resume_state['reason_counts'].items()}
        nonoptimal, changed, warm = (Counter(resume_state[k]) for k in
            ('nonoptimal_queries', 'changed_queries', 'neural_warm_start_counts'))

    def save(completed):
        checkpoint(dict(bindings=bindings, completed_queries=completed,
            choices={a: c.copy() for a, c in choices.items()},
            reason_counts={a: dict(v) for a, v in reasons.items()},
            nonoptimal_queries=dict(nonoptimal), changed_queries=dict(changed),
            neural_warm_start_counts=dict(warm)))

    for index in range(start, len(groups)):
        at = groups[index]; x, original = p[at], pc[at]
        count = int((original >= 0).sum())
        trimmed = trimmed_neural(x, nc[at], count)
        physical_joint, pi = base.solve_frame(x, physical[at], original, exact_count=count, seconds=seconds)
        # The expanded bank retains the strongest available feasible warm start.
        if base.terms(x, physical_joint)[0] > base.terms(x, trimmed)[0]:
            anchor, name = physical_joint, 'optimized_physical'
        else:
            anchor, name = trimmed, 'trimmed_neural_independent'
        neural_joint, ni = base.solve_frame(x, ok[at], anchor, exact_count=count, seconds=seconds)
        choices['neural_trimmed'][at] = trimmed
        choices['physical_joint'][at] = physical_joint; choices['neural_joint'][at] = neural_joint
        warm[name] += 1
        for arm, info, c, comparator in (('physical_joint', pi, physical_joint, original),
                ('neural_joint', ni, neural_joint, trimmed)):
            reasons[arm][info['reason']] += 1; nonoptimal[arm] += int(not info['optimal'])
            changed[arm] += int(np.any(c != comparator))
        if index == 0 or (index+1) % 250 == 0 or index+1 == len(groups):
            save(index+1); heartbeat(queries=index+1, total_queries=len(groups))
    ranks = {}
    for arm, c in choices.items():
        at = np.flatnonzero(c >= 0); r = np.zeros(len(p))
        r[at] = p[at, c[at], 0]-p[at, c[at], 1]; ranks[arm] = r
    for at in groups:
        count = int((pc[at] >= 0).sum())
        if any((c[at] >= 0).sum() != count for c in choices.values()):
            raise ValueError('Every policy must match the common count in every query')
        if base.terms(p[at], choices['neural_joint'][at])[0] < base.terms(p[at], choices['physical_joint'][at])[0]:
            raise ValueError('Neural expansion must preserve best feasible predicted gain')
    return choices, ranks, dict(queries=len(groups), multiagent_queries=sum(len(v) > 1 for v in groups),
        maximum_agents=max(map(len, groups)), common_anchor_selected=int((pc >= 0).sum()),
        legacy_neural_selected=int((nc >= 0).sum()), zero_count_queries=sum(not (pc[v] >= 0).any() for v in groups),
        same_count_every_query=True, predicted_bank_dominance_every_query=True,
        reason_counts={a: dict(v) for a, v in reasons.items()}, nonoptimal_queries=dict(nonoptimal),
        changed_queries=dict(changed), neural_warm_start_counts=dict(warm),
        labels_used=False, observed_risk_guarantee=False, empty_is_not_a_safety_pass=True,
        all_visible_agents_claim=False, pair_interactions_optimized=False)
