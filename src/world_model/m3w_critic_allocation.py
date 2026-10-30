"""Same allocator for two risk critics; common-count controls are utility only."""
from collections import Counter
import hashlib
import numpy as np
from src.world_model import m3w_frame_action_allocation as base
from src.world_model.m3w_multiaction_cost import exact

NAMES = ('row_independent', 'set_independent', 'row_joint', 'set_joint', 'row_matched', 'set_matched')


def allocate(predictions, allowed, recordings, frames, ids, *, seconds,
             checkpoint=lambda _: None, heartbeat=lambda **_: None, resume_state=None):
    if set(predictions) != {'row_local', 'selected_set'}:
        raise ValueError('Both fixed matched critic predictions required')
    p = {k: base.validate(v, allowed)[0] for k, v in predictions.items()}
    np.testing.assert_array_equal(p['row_local'][:, :, [0, 2, 3]], p['selected_set'][:, :, [0, 2, 3]])
    ok = np.asarray(allowed); groups = base.query_groups(recordings, frames, ids)
    choices = {a: np.full(len(ids), -1, np.int64) for a in NAMES}
    for prefix, arm in (('row', 'row_local'), ('set', 'selected_set')):
        choices[prefix+'_independent'], _ = base.select_actions(p[arm], ok, arm='physical_neural')
    reasons = {a: Counter() for a in NAMES[2:]}; nonoptimal = Counter()
    binds = {k: hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
        +np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in
        zip(('row_prediction', 'set_prediction', 'allowed', 'recordings', 'frames', 'ids'),
            (p['row_local'], p['selected_set'], ok, recordings, frames, ids))}
    binds.update(seconds=seconds, version=1)
    start, calls, reused = 0, 0, 0
    if resume_state is not None:
        if resume_state['bindings'] != binds or not 0 <= resume_state['completed_queries'] <= len(groups):
            raise ValueError('Same frozen critic predictions/keys/settings required')
        for arm in NAMES[:2]: np.testing.assert_array_equal(resume_state['choices'][arm], choices[arm])
        start = resume_state['completed_queries']
        choices = {a: np.asarray(resume_state['choices'][a]).copy() for a in NAMES}
        reasons = {a: Counter(v) for a, v in resume_state['reason_counts'].items()}
        nonoptimal = Counter(resume_state['nonoptimal_queries'])
        calls, reused = resume_state['solver_function_calls'], resume_state['matched_reuses']
        for g in groups[:start]:
            for prefix, arm in (('row', 'row_local'), ('set', 'selected_set')):
                for suffix in ('independent', 'joint', 'matched'):
                    if not base.feasible(p[arm][g], ok[g], choices[prefix+'_'+suffix][g]):
                        raise ValueError('Processed query is not feasible')

    def save(completed):
        checkpoint(dict(bindings=binds, completed_queries=completed, choices={a: c.copy() for a, c in choices.items()},
            reason_counts={a: dict(v) for a, v in reasons.items()}, nonoptimal_queries=dict(nonoptimal),
            solver_function_calls=calls, matched_reuses=reused))

    for index in range(start, len(groups)):
        g = groups[index]
        counts = {a: int((choices[a+'_independent'][g] >= 0).sum()) for a in ('row', 'set')}
        common = min(counts.values())
        for prefix, arm in (('row', 'row_local'), ('set', 'selected_set')):
            q = p[arm][g]; anchor = choices[prefix+'_independent'][g]
            own, own_info = base.solve_frame(q, ok[g], anchor, exact_count=counts[prefix], seconds=seconds)
            calls += 1; choices[prefix+'_joint'][g] = own
            reasons[prefix+'_joint'][own_info['reason']] += 1
            nonoptimal[prefix+'_joint'] += int(not own_info['optimal'])
            if common == counts[prefix]:
                matched, info = own, own_info; reused += 1
            else:
                at = np.flatnonzero(anchor >= 0)
                gain = q[at, anchor[at], 0]-q[at, anchor[at], 1]
                keep = at[np.argsort(-gain, kind='stable')[:common]]
                trimmed = np.full(len(g), -1, np.int64); trimmed[keep] = anchor[keep]
                matched, info = base.solve_frame(q, ok[g], trimmed, exact_count=common, seconds=seconds)
                calls += 1
            choices[prefix+'_matched'][g] = matched
            reasons[prefix+'_matched'][info['reason']] += 1
            nonoptimal[prefix+'_matched'] += int(not info['optimal'])
        if index == 0 or (index+1) % 250 == 0 or index+1 == len(groups):
            save(index+1); heartbeat(queries=index+1, total_queries=len(groups))
    ranks = {}
    for a, c in choices.items():
        q = p['row_local' if a.startswith('row_') else 'selected_set']; at = np.flatnonzero(c >= 0)
        rank = np.zeros(len(ids)); rank[at] = q[at, c[at], 0]-q[at, c[at], 1]; ranks[a] = rank
    count_match = all((choices['row_matched'][g] >= 0).sum() == (choices['set_matched'][g] >= 0).sum() for g in groups)
    if not count_match: raise AssertionError('Counts must match in every current query')
    return choices, ranks, dict(queries=len(groups), same_count_controls_every_query=True,
        solver_function_calls=calls, matched_reuses=reused,
        reason_counts={a: dict(v) for a, v in reasons.items()}, nonoptimal_queries=dict(nonoptimal),
        labels_used=False, matched_role='utility_only_not_deployment_repair',
        original_full_recording_risk_unchanged=True, original_references_frozen=True)
