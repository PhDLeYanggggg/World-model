"""Matched reoptimized-bank contrasts under the unchanged complete-policy risk."""
import numpy as np
from src.world_model import m3w_shared_count_bank as allocation
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_frame_action_readout import pair_proxy
from src.evaluation.m3w_detector_forecast_readout import tail_stats

NAMES = (*allocation.ARMS, 'legacy_neural_independent', 'legacy_neural_joint', 'floor')
CONTRASTS = dict(neural_joint_minus_physical_joint=('physical_joint', 'neural_joint'),
    neural_trimmed_minus_physical_independent=('physical_independent', 'neural_trimmed'),
    physical_joint_minus_physical_independent=('physical_independent', 'physical_joint'),
    neural_joint_minus_neural_trimmed=('neural_trimmed', 'neural_joint'))


def interaction_contrast(packet, choices):
    """Four-policy contrast with its own cancellation-aware unknown bound."""
    if set(choices) != set(allocation.ARMS):
        raise ValueError('Both banks and both decoder controls required')
    n = len(packet['ids'])
    for c in choices.values():
        if (np.asarray(c).shape != (n,) or not np.issubdtype(np.asarray(c).dtype, np.integer)
                or ((c < -1) | (c >= 7)).any()):
            raise ValueError('Aligned immutable forecast-bank choices required')
    errors = {a: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], c) for a, c in choices.items()}
    paths = {a: bank.legacy.chosen_values(packet['rollouts'], packet['floor_rollout'], c) for a, c in choices.items()}
    known = np.isfinite(packet['floor_ade']); ref = float(packet['floor_ade'][known].sum())
    delta = float((errors['physical_joint'][known]-errors['neural_joint'][known]
        -errors['physical_independent'][known]+errors['neural_trimmed'][known]).sum())
    def distance(a, b):
        return np.linalg.norm(paths[a][~known]-paths[b][~known], axis=-1).max(1)
    # The two matchings pair positive and negative terms; each is a valid bound.
    first = distance('physical_joint', 'neural_joint')+distance('physical_independent', 'neural_trimmed')
    second = distance('physical_joint', 'physical_independent')+distance('neural_joint', 'neural_trimmed')
    missing = float(np.minimum(first, second).sum())
    return dict(known_pp=100*delta/ref if ref > 0 else None,
        lower_pp=100*(delta-missing)/ref if ref > 0 else None,
        upper_pp=100*(delta+missing)/ref if ref > 0 else None,
        known_delta_mass=delta, unknown_disagreement_mass=missing,
        own_four_policy_bound=True, not_difference_of_lower_bounds=True,
        secondary_not_primary=True, causal_identification=False)


def readout(packet, choices, ranks, info, legacy_independent, legacy_joint, easy_cut, origins):
    if set(choices) != set(allocation.ARMS) or set(ranks) != set(allocation.ARMS):
        raise ValueError('All four common-count controls required')
    groups = allocation.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    if any((choices[a] == 6).any() for a in ('physical_independent', 'physical_joint')):
        raise ValueError('A physical-only policy cannot use neural action6')
    count_match = all(len({int((c[at] >= 0).sum()) for c in choices.values()}) == 1 for at in groups)
    if not count_match:
        raise ValueError('Every common-count policy must match in every query')
    all_choices = dict(choices, legacy_neural_independent=legacy_independent,
        legacy_neural_joint=legacy_joint, floor=np.full(len(packet['ids']), -1, np.int64))
    policies = {a: bank.policy_metrics(packet, c) for a, c in all_choices.items()}
    records = {str(r): {a: bank.legacy.compact_score(packet['y'][at], packet['envelope'][at], c[at])
        for a, c in all_choices.items()} for r in sorted(set(packet['recordings']))
        for at in (packet['recordings'] == r,)}
    comparisons = {name: bank.pair(packet, choices[left], ranks[left], packet, choices[right], ranks[right])
        for name, (left, right) in CONTRASTS.items()}
    ade = {a: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], c) for a, c in all_choices.items()}
    return dict(rows=len(packet['ids']), known_rows=int(np.isfinite(packet['floor_ade']).sum()),
        unknown_rows=int((~np.isfinite(packet['floor_ade'])).sum()), policies=policies, recordings=records,
        comparisons=comparisons, interaction_contrast=interaction_contrast(packet, choices),
        subsets=subsets(packet, all_choices, easy_cut), allocation=info, same_count_every_query=count_match,
        pair_proxy=pair_proxy(packet, groups, all_choices, origins),
        recording_tails={str(r): {a: tail_stats(v, packet['recordings'] == r) for a, v in ade.items()}
            for r in sorted(set(packet['recordings']))},
        full_cohort_preserved=True, inference_label_filter=False, new_training_updates=0,
        pair_interactions_optimized=False, common_count_not_retroactive_deployment_repair=True)


def summarize(rows, contexts, cfg):
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != expected or len(contexts) != 6:
        raise ValueError('Complete72-group original fitting roster required')
    interval = lambda getter: bank.legacy.interval(rows, getter, cfg)
    contrasts = {name: {mode+'_'+bound: interval(lambda r, name=name, mode=mode, bound=bound:
        r['comparisons'][name][mode][bound]) for mode in ('full', 'matched')
        for bound in ('known_pp', 'lower_pp', 'upper_pp')} for name in CONTRASTS}
    safety = {a: bank.safety(rows, a) for a in NAMES}
    primary = contrasts['neural_joint_minus_physical_joint']
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    gates = dict(complete72groups=len(rows) == 72,
        all144legacy_policy_scores_exact=all(r['legacy_controls_exact'] for r in rows),
        same_count_every_current_query=all(r['result']['same_count_every_query'] for r in rows),
        predicted_bank_dominance=all(r['result']['allocation']['predicted_bank_dominance_every_query'] for r in rows),
        all_new_queries_solved_optimally=all(not any(r['result']['allocation']['nonoptimal_queries'].values()) for r in rows),
        primary_known_CI_positive=positive(primary['full_known_pp']),
        primary_completion_lower_CI_positive=positive(primary['full_lower_pp']),
        complete_primary_recording_risk=safety['neural_joint']['supported_views'] == 72 and
            safety['neural_joint']['supported_record_views'] == safety['neural_joint']['required_record_views'])
    return dict(groups=72, cached_verified_cost_heads=18, new_training_updates=0,
        primary='neural_joint_minus_physical_joint_at_physical_independent_query_count',
        accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        comparisons=contrasts, safety=safety, subsets=pooled_subsets(rows),
        joint_specific_neural_increment={bound: interval(lambda r, bound=bound:
            r['interaction_contrast'][bound]) for bound in ('known_pp', 'lower_pp', 'upper_pp')},
        shared_bank_regret={a: interval(lambda r, a=a: r['policies'][a]['known_regret_over_cost_scale']) for a in NAMES},
        query_views=sum(r['result']['allocation']['queries'] for r in rows),
        multiagent_query_views=sum(r['result']['allocation']['multiagent_queries'] for r in rows),
        zero_count_query_views=sum(r['result']['allocation']['zero_count_queries'] for r in rows),
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        result_source='fresh_reoptimized_shared_physical_count_bank_readout_on_cached_verified_SOURCE_fitting_assets',
        legacy_joint_choices_cached_verified_not_resolved_again=True,
        common_count_comparison_not_main_deployment_repair=True, whole_pipeline_OOF=False,
        independent_confirmation=False, independent_roles_opened=False, matched_controls_utility_only=True,
        empty_risk_not_estimable=True, held_scored=False, deployment_changed=False,
        submission_ready=False, metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
