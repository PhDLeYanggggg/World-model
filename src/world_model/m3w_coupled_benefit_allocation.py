"""Frozen-reference allocation; protected full-policy choices are verified controls."""
from collections import Counter
import hashlib
import numpy as np
from src.world_model.m3w_critic_allocation import base

MODES = ('protected_benefit', 'released_benefit')
NAMES = tuple(a+'_'+b for a in ('protected', 'released') for b in ('independent', 'joint', 'matched'))


def allocate(predictions, allowed, recordings, frames, ids, control, *, seconds,
             checkpoint=lambda _: None, heartbeat=lambda **_: None, resume_state=None):
    if set(predictions) != set(MODES): raise ValueError('Both matched prediction arms required')
    p = {k: base.validate(v, allowed)[0] for k, v in predictions.items()}
    np.testing.assert_array_equal(p[MODES[0]][:, :, [2, 3]], p[MODES[1]][:, :, [2, 3]])
    ok = np.asarray(allowed); groups = base.query_groups(recordings, frames, ids)
    choices = {a: np.full(len(ids), -1, np.int64) for a in NAMES}
    for prefix, mode in zip(('protected', 'released'), MODES):
        choices[prefix+'_independent'], _ = base.select_actions(p[mode], ok, arm='physical_neural')
    np.testing.assert_array_equal(choices['protected_independent'], control['set_independent'])
    choices['protected_joint'] = np.asarray(control['set_joint']).copy()
    if choices['protected_joint'].shape != (len(ids),): raise ValueError('Full cached control required')
    for g in groups:
        if not base.feasible(p[MODES[0]][g], ok[g], choices['protected_joint'][g]):
            raise ValueError('Original full-policy control is not feasible')
    binds = {k: hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
        +np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in
        zip(('protected', 'released', 'allowed', 'recordings', 'frames', 'ids', 'control_joint'),
            (p[MODES[0]], p[MODES[1]], ok, recordings, frames, ids, choices['protected_joint']))}
    binds.update(seconds=seconds, version=1)
    start = 0; reasons = {a: Counter() for a in NAMES if a.endswith(('joint', 'matched'))}; nonoptimal = Counter(); calls = 0
    if resume_state is not None:
        if resume_state['bindings'] != binds: raise ValueError('Allocation resume inputs changed')
        start = resume_state['completed_queries']
        if not 0 <= start <= len(groups): raise ValueError('Invalid resume point')
        for a in ('protected_independent', 'released_independent', 'protected_joint'):
            np.testing.assert_array_equal(resume_state['choices'][a], choices[a])
        choices = {a: v.copy() for a, v in resume_state['choices'].items()}
        reasons = {a: Counter(v) for a, v in resume_state['reasons'].items()}
        nonoptimal = Counter(resume_state['nonoptimal']); calls = resume_state['solver_calls']

    def save(completed):
        checkpoint(dict(bindings=binds, completed_queries=completed, choices={a: c.copy() for a, c in choices.items()},
            reasons={a: dict(v) for a, v in reasons.items()}, nonoptimal=dict(nonoptimal), solver_calls=calls))

    for index in range(start, len(groups)):
        g = groups[index]; counts = {a: int((choices[a+'_independent'][g] >= 0).sum()) for a in ('protected', 'released')}
        common = min(counts.values())
        for prefix, mode in zip(('protected', 'released'), MODES):
            q = p[mode][g]; anchor = choices[prefix+'_independent'][g]
            if prefix == 'protected':
                own = choices['protected_joint'][g]; own_info = dict(reason='cached_verified_original_full_policy', optimal=False)
            else:
                own, own_info = base.solve_frame(q, ok[g], anchor, exact_count=counts[prefix], seconds=seconds)
                calls += 1; choices[prefix+'_joint'][g] = own
                nonoptimal[prefix+'_joint'] += int(not own_info['optimal'])
            reasons[prefix+'_joint'][own_info['reason']] += 1
            if common == counts[prefix]: matched, info = own, own_info
            else:
                at = np.flatnonzero(anchor >= 0); gain = q[at, anchor[at], 0]-q[at, anchor[at], 1]
                keep = at[np.argsort(-gain, kind='stable')[:common]]; trimmed = np.full(len(g), -1, np.int64); trimmed[keep] = anchor[keep]
                matched, info = base.solve_frame(q, ok[g], trimmed, exact_count=common, seconds=seconds); calls += 1
            choices[prefix+'_matched'][g] = matched; reasons[prefix+'_matched'][info['reason']] += 1
            if info['reason'] != 'cached_verified_original_full_policy': nonoptimal[prefix+'_matched'] += int(not info['optimal'])
        if index == 0 or (index+1) % 250 == 0 or index+1 == len(groups):
            save(index+1); heartbeat(queries=index+1, total_queries=len(groups))
    if any((choices['protected_matched'][g] >= 0).sum() != (choices['released_matched'][g] >= 0).sum() for g in groups):
        raise AssertionError('Matched utility controls must preserve every query count')
    ranks = {}
    for name, c in choices.items():
        q = p[MODES[0] if name.startswith('protected') else MODES[1]]; at = np.flatnonzero(c >= 0); rank = np.zeros(len(c))
        rank[at] = q[at, c[at], 0]-q[at, c[at], 1]; ranks[name] = rank
    return choices, ranks, dict(queries=len(groups), solver_calls=calls, reason_counts={a: dict(v) for a, v in reasons.items()},
        nonoptimal_queries=dict(nonoptimal), same_count_controls_every_query=True, labels_used=False,
        protected_full_choices='cached_verified_not_reoptimized', original_references_frozen=True,
        matched_utility_controls_do_not_replace_full_record_risk=True)
