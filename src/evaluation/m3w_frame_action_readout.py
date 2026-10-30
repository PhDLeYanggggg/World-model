"""Unchanged recording-risk readout for fixed contemporaneous allocations."""
import numpy as np
from src.world_model import m3w_frame_action_allocation as allocation
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_detector_forecast_readout import tail_stats

NAMES = (*allocation.ARMS, 'floor')
CONTRASTS = dict(joint_matched_minus_independent=('independent', 'joint_matched'),
    joint_free_minus_independent=('independent', 'joint_free'),
    uniform_minus_independent=('independent', 'frame_uniform'),
    joint_matched_minus_uniform=('frame_uniform', 'joint_matched'))


def oracle_frame_check(packet, groups, choice):
    out = dict(queries=len(groups), algebraic_budget_failures=0,
        defined_all_support=0, defined_easy_support=0, positive_gain_queries=0,
        label_aware_not_deployable=True, matched_rate_optimum=False)
    for rows in groups:
        active = rows[choice[rows] >= 0]
        y = packet['y'][active, choice[active]]
        if not np.isfinite(y).all():
            raise ValueError('The known-label oracle cannot silently impute outcomes')
        v = y.sum(0)
        out['defined_all_support'] += int(v[2] > 0)
        out['defined_easy_support'] += int(v[3] > 0)
        out['positive_gain_queries'] += int(v[0] > v[1])
        valid = all(v[h] <= (.02+1e-12)*v[r] if v[r] > 0 else v[h] == 0
            for h, r in ((1, 2), (4, 3)))
        out['algebraic_budget_failures'] += int(not valid)
    return out


def pair_proxy(packet, groups, choices, origins):
    origin = np.asarray(origins, float)
    if origin.shape != (len(packet['ids']), 2) or not np.isfinite(origin).all():
        raise ValueError('Aligned causal origins in the same source coordinate frame required')
    floor = packet['floor_rollout']+origin[:, None]
    paths = {a: bank.legacy.chosen_values(packet['rollouts'], packet['floor_rollout'], c)+origin[:, None]
             for a, c in choices.items()}
    totals = {a: 0. for a in paths}; count = 0
    for rows in groups:
        i, j = np.triu_indices(len(rows), 1); i, j = rows[i], rows[j]
        if not len(i):
            continue
        reference = np.linalg.norm(floor[i]-floor[j], axis=-1)
        scale = np.sqrt(packet['scales'][i]*packet['scales'][j])[:, None]
        for a, p in paths.items():
            distance = np.linalg.norm(p[i]-p[j], axis=-1)
            totals[a] += float((np.maximum(reference-distance, 0)/scale).sum())
        count += reference.size
    return dict(pair_requested_step_views=count,
        mean_positive_distance_contraction_over_frozen_scale={a: v/count if count else None for a, v in totals.items()},
        query_targets_only=True, all_visible_agents_claim=False, physical_collision_or_safety_certificate=False,
        optimized_by_this_controller=False)


def readout(packet, choices, ranks, info, easy_cut, origins, oracle):
    if set(choices) != set(allocation.ARMS) or set(ranks) != set(allocation.ARMS):
        raise ValueError('All four fixed decoder arms required')
    groups = allocation.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    all_choices = dict(choices, floor=np.full(len(packet['ids']), -1, np.int64))
    policies = {a: bank.policy_metrics(packet, c) for a, c in all_choices.items()}
    records = {str(r): {a: bank.legacy.compact_score(packet['y'][at], packet['envelope'][at], c[at])
        for a, c in all_choices.items()} for r in sorted(set(packet['recordings']))
        for at in (packet['recordings'] == r,)}
    comparisons = {name: bank.pair(packet, choices[left], ranks[left], packet, choices[right], ranks[right])
        for name, (left, right) in CONTRASTS.items()}
    ade = {a: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], c) for a, c in all_choices.items()}
    known = np.isfinite(packet['floor_ade'])
    return dict(rows=len(known), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        policies=policies, recordings=records, comparisons=comparisons,
        subsets=subsets(packet, all_choices, easy_cut), allocation=info,
        oracle_frame_check=oracle_frame_check(packet, groups, oracle),
        pair_proxy=pair_proxy(packet, groups, all_choices, origins),
        recording_tails={str(r): {a: tail_stats(v, packet['recordings'] == r) for a, v in ade.items()}
            for r in sorted(set(packet['recordings']))},
        exact_count_per_query=all(int((choices['independent'][at] >= 0).sum()) ==
            int((choices['joint_matched'][at] >= 0).sum()) for at in groups),
        inference_label_filter=False, pair_interaction_model_trained=False)


def summarize(rows, contexts, cfg):
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != expected or len(contexts) != 6:
        raise ValueError('Complete original72-group six-context fitting roster required')
    interval = lambda getter: bank.legacy.interval(rows, getter, cfg)
    contrasts = {name: {mode+'_'+bound: interval(lambda r, name=name, mode=mode, bound=bound:
        r['comparisons'][name][mode][bound]) for mode in ('full', 'matched')
        for bound in ('known_pp', 'lower_pp', 'upper_pp')} for name in CONTRASTS}
    risk = {a: bank.safety(rows, a) for a in NAMES}
    primary = contrasts['joint_matched_minus_independent']
    positive = lambda value: value['CI95'] is not None and value['CI95'][0] > 0
    gates = dict(complete72groups=len(rows) == 72,
        all_frozen_controls_exact=all(r['control_exact'] and r['record_oracle_exact'] for r in rows),
        same_intervention_count_every_query=all(r['result']['exact_count_per_query'] for r in rows),
        all_primary_queries_solved_optimally=all(not r['result']['allocation']['nonoptimal_queries'].get('joint_matched', 0) for r in rows),
        primary_known_CI_positive=positive(primary['full_known_pp']),
        primary_completion_lower_CI_positive=positive(primary['full_lower_pp']),
        complete_primary_recording_risk=risk['joint_matched']['supported_views'] == 72 and
            risk['joint_matched']['supported_record_views'] == risk['joint_matched']['required_record_views'])
    return dict(groups=72, trained_models=0, cached_verified_cost_heads=18, new_training_updates=0,
        primary='joint_matched_minus_independent_full_policy',
        accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        safety=risk, comparisons=contrasts, subsets=pooled_subsets(rows),
        shared_bank_regret={a: interval(lambda r, a=a: r['policies'][a]['known_regret_over_cost_scale']) for a in NAMES},
        query_views=sum(r['result']['allocation']['queries'] for r in rows),
        multiagent_query_views=sum(r['result']['allocation']['multiagent_queries'] for r in rows),
        changed_primary_query_views=sum(r['result']['allocation']['changed_queries'].get('joint_matched', 0) for r in rows),
        primary_solver_nonoptimal_query_views=sum(r['result']['allocation']['nonoptimal_queries'].get('joint_matched', 0) for r in rows),
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        result_source='fresh_fixed_current_query_allocation_on_cached_verified_SOURCE_fitting_models_and_inputs',
        whole_pipeline_OOF=False, independent_confirmation=False, independent_roles_opened=False,
        matched_controls_utility_only=True, empty_risk_not_estimable=True, held_scored=False,
        deployment_changed=False, submission_ready=False, metric_claim=False, seconds_claim=False,
        stage5c_executed=False, smc_enabled=False)
