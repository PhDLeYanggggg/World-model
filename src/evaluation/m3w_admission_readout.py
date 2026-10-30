"""Complete unchanged-risk readout of current-selected-action loss learning."""
import numpy as np
from src.world_model.m3w_admission_cost import OBJECTIVES
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation import m3w_selected_action_attribution as attribution
from src.evaluation.m3w_budget_score_readout import score_diagnostics
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_detector_forecast_readout import tail_stats


def readout(packet, predictions, choices, ranks, easy_cut, rms):
    if any(set(v) != set(OBJECTIVES) for v in (predictions, choices, ranks)):
        raise ValueError('All three complete objective controls required')
    policies = {o: bank.policy_metrics(packet, choices[o]) for o in OBJECTIVES}
    records = {str(r): {o: bank.legacy.compact_score(packet['y'][at], packet['envelope'][at], choices[o][at])
        for o in OBJECTIVES} for r in sorted(set(packet['recordings'])) for at in (packet['recordings'] == r,)}
    comparisons = {label: bank.pair(packet, choices[a], ranks[a], packet, choices[b], ranks[b])
        for label, a, b in (('admitted_minus_upper', 'upper_score', 'admitted_score'),
                            ('admitted_minus_original', 'original', 'admitted_score'))}
    restricted, rank = bank.cost.select_actions(predictions['admitted_score'], packet['allowed'], arm='physical_only')
    nonneural = choices['admitted_score'] != 6
    np.testing.assert_array_equal(restricted[nonneural], choices['admitted_score'][nonneural])
    selected = {o: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], choices[o]) for o in OBJECTIVES}
    return dict(rows=len(packet['ids']), policies=policies, recordings=records, comparisons=comparisons,
        subsets=subsets(packet, choices, easy_cut),
        attribution={o: attribution.analyze(packet, predictions[o], choices[o], easy_cut=easy_cut) for o in OBJECTIVES},
        score_diagnostics={o: score_diagnostics(packet, predictions[o], choices[o], 'new_neural', rms) for o in OBJECTIVES},
        actual_neural_use=bank.pair(packet, restricted, rank, packet, choices['admitted_score'], ranks['admitted_score']),
        neural_restricted_policy=bank.policy_metrics(packet, restricted),
        recording_tails={str(r): {o: tail_stats(selected[o], packet['recordings'] == r) for o in OBJECTIVES}
            for r in sorted(set(packet['recordings']))},
        inference_label_filter=False, scene_joint_decisions_trained=False)


def summarize(rows, contexts, cfg, training):
    expected = {(c, s, seed) for c, sources in contexts.items() for s in sources for seed in cfg['seeds']}
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != expected or len(contexts) != 6:
        raise ValueError('All72original context/source/seed groups required')
    interval = lambda getter: bank.legacy.interval(rows, getter, cfg)
    comparisons = {name: {mode + '_' + bound: interval(
        lambda r, name=name, mode=mode, bound=bound: r['comparisons'][name][mode][bound])
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
        for name in ('admitted_minus_upper', 'admitted_minus_original')}
    risk = {o: bank.safety(rows, o) for o in OBJECTIVES}
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    gates = dict(complete54heads72groups=training['count'] == 54,
        exact36parent_controls=training['exact_parent_controls'] == 36 and all(r['controls_exact'] for r in rows),
        complete_recording_risk=risk['admitted_score']['supported_views'] == risk['admitted_score']['required_views']
            and risk['admitted_score']['supported_record_views'] == risk['admitted_score']['required_record_views'],
        primary_known_CI_positive=positive(comparisons['admitted_minus_upper']['full_known_pp']),
        primary_completion_lower_CI_positive=positive(comparisons['admitted_minus_upper']['full_lower_pp']),
        actual_neural_use_completion_lower_CI_positive=positive(interval(lambda r: r['actual_neural_use']['full']['lower_pp'])))
    return dict(groups=len(rows), models=54, primary='admitted_score_minus_upper_score',
        accuracy={o: interval(lambda r, o=o: r['policies'][o]['known_ADE_gain_floor_percent']) for o in OBJECTIVES},
        FDE={o: interval(lambda r, o=o: r['policies'][o]['known_FDE_gain_floor_percent']) for o in OBJECTIVES},
        safety=risk, comparisons=comparisons, subsets=pooled_subsets(rows),
        actual_neural_use={mode + '_' + bound: interval(lambda r, mode=mode, bound=bound: r['actual_neural_use'][mode][bound])
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')},
        shared_bank_regret={o: interval(lambda r, o=o: r['policies'][o]['known_regret_over_cost_scale']) for o in OBJECTIVES},
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        result_source='fresh_fixed_online_admission_cost_inference_SOURCE_fitting_resubstitution',
        independent_confirmation=False, whole_pipeline_OOF=False, scene_joint_decisions_trained=False,
        count_matching_utility_only=True, empty_comparison_not_estimable=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
