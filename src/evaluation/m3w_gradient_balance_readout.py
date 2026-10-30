"""Fixed matched norm-parity evaluation, preserving the complete risk contract."""
import numpy as np
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation import m3w_selected_action_attribution as attribution
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_budget_score_readout import score_diagnostics
from src.evaluation.m3w_detector_forecast_readout import tail_stats

OBJECTIVES = ('upper_score', 'admitted_score', 'balanced_score')


def readout(packet, predictions, choices, ranks, easy_cut, rms):
    if any(set(v) != set(OBJECTIVES) for v in (predictions, choices, ranks)):
        raise ValueError('Fresh balanced and both frozen matched controls required')
    policies = {o: bank.policy_metrics(packet, choices[o]) for o in OBJECTIVES}
    records = {str(r): {o: bank.legacy.compact_score(packet['y'][at], packet['envelope'][at], choices[o][at])
        for o in OBJECTIVES} for r in sorted(set(packet['recordings'])) for at in (packet['recordings'] == r,)}
    comparisons = {label: bank.pair(packet, choices[a], ranks[a], packet, choices[b], ranks[b])
        for label, a, b in (('balanced_minus_admitted', 'admitted_score', 'balanced_score'),
                            ('balanced_minus_upper', 'upper_score', 'balanced_score'))}
    restricted, rank = bank.cost.select_actions(predictions['balanced_score'], packet['allowed'], arm='physical_only')
    nonneural = choices['balanced_score'] != 6
    np.testing.assert_array_equal(restricted[nonneural], choices['balanced_score'][nonneural])
    paths = {o: bank.legacy.chosen_values(packet['rollouts'], packet['floor_rollout'], choices[o]) for o in OBJECTIVES}
    ade = {o: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], choices[o]) for o in OBJECTIVES}
    return dict(policies=policies, recordings=records, comparisons=comparisons,
        subsets=subsets(packet, choices, easy_cut),
        attribution={o: attribution.analyze(packet, predictions[o], choices[o], easy_cut=easy_cut) for o in OBJECTIVES},
        score_diagnostics={o: score_diagnostics(packet, predictions[o], choices[o], 'new_neural', rms) for o in OBJECTIVES},
        actual_neural_use=bank.pair(packet, restricted, rank, packet, choices['balanced_score'], ranks['balanced_score']),
        restricted_physical_policy=bank.policy_metrics(packet, restricted),
        recording_tails={str(r): {o: tail_stats(ade[o], packet['recordings'] == r) for o in OBJECTIVES}
            for r in sorted(set(packet['recordings']))},
        validity_proxies={o: dict(finite=bool(np.isfinite(p).all()),
            requested_step_second_difference_mean=float(np.linalg.norm(np.diff(p, n=2, axis=1), axis=-1).mean()),
            verified_physical_safety=False, metric_or_seconds=False) for o, p in paths.items()},
        inference_label_filter=False, scene_joint_decisions_trained=False)


def summarize(rows, contexts, cfg, training):
    expected = {(c, s, seed) for c, sources in contexts.items() for s in sources for seed in cfg['seeds']}
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != expected or len(contexts) != 6:
        raise ValueError('Complete72group fixed fitting roster required')
    interval = lambda getter: bank.legacy.interval(rows, getter, cfg)
    contrasts = {name: {mode+'_'+bound: interval(
        lambda r, name=name, mode=mode, bound=bound: r['comparisons'][name][mode][bound])
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
        for name in ('balanced_minus_admitted', 'balanced_minus_upper')}
    risk = {o: bank.safety(rows, o) for o in OBJECTIVES}
    use = {mode+'_'+bound: interval(lambda r, mode=mode, bound=bound: r['actual_neural_use'][mode][bound])
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    gates = dict(complete18fresh36cached_endpoints72groups=training['count'] == 18
            and len(training['cached_verified_controls']) == 36,
        cached_control_inference_and_scores_exact=all(r['controls_exact'] for r in rows),
        complete_recording_risk=risk['balanced_score']['supported_views'] == risk['balanced_score']['required_views']
            and risk['balanced_score']['supported_record_views'] == risk['balanced_score']['required_record_views'],
        primary_known_CI_positive=positive(contrasts['balanced_minus_admitted']['full_known_pp']),
        primary_completion_lower_CI_positive=positive(contrasts['balanced_minus_admitted']['full_lower_pp']),
        actual_neural_use_completion_lower_CI_positive=positive(use['full_lower_pp']))
    return dict(groups=72, fresh_models=18, cached_verified_control_models=36,
        primary='balanced_score_minus_admitted_score',
        accuracy={o: interval(lambda r, o=o: r['policies'][o]['known_ADE_gain_floor_percent']) for o in OBJECTIVES},
        FDE={o: interval(lambda r, o=o: r['policies'][o]['known_FDE_gain_floor_percent']) for o in OBJECTIVES},
        safety=risk, comparisons=contrasts, subsets=pooled_subsets(rows), actual_neural_use=use,
        shared_bank_regret={o: interval(lambda r, o=o: r['policies'][o]['known_regret_over_cost_scale']) for o in OBJECTIVES},
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        result_source='fresh_fixed_gradient_norm_balancing_inference_on_cached_verified_SOURCE_fitting_inputs',
        independent_confirmation=False, whole_pipeline_OOF=False, scene_joint_decisions_trained=False,
        count_matching_utility_only=True, empty_comparison_not_estimable=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
