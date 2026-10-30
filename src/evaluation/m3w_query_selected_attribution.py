"""Post-choice diagnostics for pooled budgets, not pointwise admission claims."""
import numpy as np
from src.world_model import m3w_frame_action_allocation as allocation
from src.evaluation import m3w_selected_action_attribution as old
from src.evaluation import m3w_active_bank_cost_readout as bank


def incremental_neural_use(packet, independent, joint):
    """Difference of leave-neural-use-out effects; shared unknown uses cancel."""
    a, b = np.asarray(independent), np.asarray(joint)
    n = len(packet['ids'])
    if (a.shape != (n,) or b.shape != (n,) or any(not np.issubdtype(v.dtype, np.integer)
            or ((v < -1) | (v >= 7)).any() for v in (a, b))):
        raise ValueError('Aligned immutable causal bank choices required')
    coefficient = (b == 6).astype(int)-(a == 6).astype(int)
    known = np.isfinite(packet['floor_ade']); reference = float(packet['floor_ade'][known].sum())
    delta = float((coefficient[known]*(packet['floor_ade'][known]-packet['ade'][known, 6])).sum())
    distance = np.linalg.norm(packet['rollouts'][:, 6]-packet['floor_rollout'], axis=-1).max(1)
    missing = float((np.abs(coefficient[~known])*distance[~known]).sum())
    return dict(known_pp=100*delta/reference if reference > 0 else None,
        lower_pp=100*(delta-missing)/reference if reference > 0 else None,
        upper_pp=100*(delta+missing)/reference if reference > 0 else None,
        known_delta_mass=delta, unknown_disagreement_mass=missing,
        newly_neural_rows=int((coefficient > 0).sum()), no_longer_neural_rows=int((coefficient < 0).sum()),
        shared_neural_rows=int(((a == 6) & (b == 6)).sum()), shared_unknown_neural_uses_cancel=True,
        not_difference_of_independent_lower_bounds=True, causal_identification=False)


def channel_names(value):
    for channel in value['channels'].values():
        channel['observed_positive_excess_rows'] = channel.pop('false_admitted_known_rows')
    return value


def analyze(packet, prediction, choices, *, easy_cut):
    p, allowed = allocation.validate(prediction, packet['allowed'])
    y, c = np.asarray(packet['y']), np.asarray(choices)
    if y.shape != p.shape or c.shape != (len(p),) or not np.issubdtype(c.dtype, np.integer):
        raise ValueError('Aligned frozen choices and separate evaluation labels required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all():
        raise ValueError('Wholly unknown labels must not be partly imputed')
    groups = allocation.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    if any(not allocation.feasible(p[at], allowed[at], c[at]) for at in groups):
        raise ValueError('Original complete-query predicted budgets required, not pointwise budgets')
    cv, floor, env, rec = [np.asarray(packet[k]) for k in ('cv_ade', 'floor_ade', 'envelope', 'recordings')]
    if not np.array_equal(known, np.isfinite(cv)) or not np.array_equal(known, np.isfinite(floor)):
        raise ValueError('Unchanged known-label cohort required')
    out = channel_names(old._group(y, p, c, cv, floor, known, env, easy_cut))
    out['records'] = {str(r): channel_names(old._group(y[at], p[at], c[at], cv[at], floor[at],
        known[at], env[at], easy_cut)) for r in sorted(set(rec)) for at in (rec == r,)}
    out['actions'] = {str(a): channel_names(old._group(y[at], p[at], c[at], cv[at], floor[at],
        known[at], env[at], easy_cut)) for a in range(7) for at in (c == a,)}
    yy, pp, empty, mixed = [], [], 0, 0
    for at in groups:
        selected = at[c[at] >= 0]
        if not len(selected):
            empty += 1
        elif not known[selected].all():
            mixed += 1
        else:
            yy.append(y[selected, c[selected]].sum(0)); pp.append(p[selected, c[selected]].sum(0))
    actual = np.asarray(yy).reshape(-1, 5); predicted = np.asarray(pp).reshape(-1, 5)
    query_channels = {}
    for name, h, r in (('all', 1, 2), ('easy', 4, 3)):
        v = old._channel(actual, predicted, h, r)
        v['fully_known_selected_queries_with_observed_excess'] = v.pop('false_admitted_known_rows')
        query_channels[name] = v
    removed = c.copy(); removed[c == 6] = -1
    rank = np.zeros(len(c)); active = np.flatnonzero(c >= 0)
    rank[active] = p[active, c[active], 0]-p[active, c[active], 1]
    removed_rank = rank.copy(); removed_rank[c == 6] = 0.
    use = bank.pair(packet, removed, removed_rank, packet, c, rank)
    return dict(out, query_channels=query_channels, fully_known_selected_queries=len(yy),
        empty_queries=empty, selected_unknown_queries=mixed,
        row_excess_not_necessarily_a_query_violation=True,
        query_diagnostic_excludes_selected_unknown_only_for_arithmetic_comparison=True,
        full_policy_unknowns_retained=True, actual_neural_use=use,
        neural_deleted_policy=bank.policy_metrics(packet, removed),
        neural_use_ablation='replace_only_selected_neural_actions_by_floor_keep_every_other_action',
        optimized_physical_only_comparator=False, deployment_policy_constructed=False,
        future_informed_diagnostic_only=True, causal_identification=False)
