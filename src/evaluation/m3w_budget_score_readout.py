"""Fixed complete-policy risk and paired utility for budget-score objectives."""
import numpy as np

from src.evaluation import m3w_active_bank_cost_readout as bank
from src.world_model.m3w_budget_score_cost import OBJECTIVES


def score_diagnostics(packet, prediction, choices, variant, rms):
    known = np.isfinite(packet['y']).all((1, 2))
    legal = packet['allowed'].copy() & known[:, None]
    if variant == 'physical_only':
        legal[:, 6] = False
    scale = np.sqrt(np.square(rms[[1, 4]]) + np.square(.02 * rms[[2, 3]]))
    predicted = prediction[..., [1, 4]] - .02 * prediction[..., [2, 3]]
    observed = packet['y'][..., [1, 4]] - .02 * packet['y'][..., [2, 3]]
    error = (predicted - observed) / packet['scales'][:, None, None] / scale
    at = np.flatnonzero(known & (choices >= 0))
    selected = error[at, choices[at]]
    return dict(known_legal_action_views=int(legal.sum()), known_selected_views=len(at),
        legal_normalized_score_MSE=np.square(error[legal]).mean(0).tolist() if legal.any() else None,
        selected_normalized_score_bias=selected.mean(0).tolist() if len(at) else None,
        selected_score_underprediction_views=(selected < 0).sum(0).tolist(),
        unit='frozen_physical_fitting_RMS_quadrature', unknown_outcomes_imputed=False,
        probability_calibration_or_risk_guarantee=False)


def objective_readouts(packets, predictions, choices, ranks, easy_cut, rms):
    if any(set(v) != set(OBJECTIVES) for v in (predictions, choices, ranks)):
        raise ValueError('Every original,symmetric and upper prediction required')
    out = {}
    for objective in OBJECTIVES:
        result = bank.group_readout(packets, choices[objective], ranks[objective], predictions[objective], easy_cut)
        against = {v: bank.pair(packets[v], choices['original'][v], ranks['original'][v],
            packets[v], choices[objective][v], ranks[objective][v]) for v in bank.VARIANTS}
        for name, v in (('new_minus_original_old', 'old_neural'), ('new_minus_original_physical', 'physical_only')):
            against[name] = bank.pair(packets[v], choices['original'][v], ranks['original'][v],
                packets['new_neural'], choices[objective]['new_neural'], ranks[objective]['new_neural'])
        symmetric = {v: bank.pair(packets[v], choices['symmetric_score'][v], ranks['symmetric_score'][v],
            packets[v], choices[objective][v], ranks[objective][v]) for v in bank.VARIANTS}
        out[objective] = dict(result=result, against_original=against, against_symmetric=symmetric,
            budget_diagnostics={v: score_diagnostics(packets[v], predictions[objective][v],
                choices[objective][v], v, rms) for v in bank.VARIANTS})
    return out


def intervals(rows, field, cfg):
    prepared = [dict(r, result=r[field]) for r in rows]
    return {name: {mode + '_' + bound: bank.legacy.interval(prepared,
        lambda value, name=name, mode=mode, bound=bound: value[name][mode][bound], cfg)
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
        for name in rows[0][field]}


def objective_summary(rows, cfg):
    result = dict(
        accuracy={v: bank.legacy.interval(rows, lambda r, v=v: r['policies'][v]['known_ADE_gain_floor_percent'], cfg)
            for v in bank.VARIANTS},
        FDE={v: bank.legacy.interval(rows, lambda r, v=v: r['policies'][v]['known_FDE_gain_floor_percent'], cfg)
            for v in bank.VARIANTS},
        safety={v: bank.safety(rows, v) for v in bank.VARIANTS}, subsets=bank.pooled_subsets(rows),
        against_original=intervals(rows, 'against_original', cfg),
        against_symmetric=intervals(rows, 'against_symmetric', cfg),
        bank_contrasts={name: {mode + '_' + bound: bank.legacy.interval(rows,
            lambda value, name=name, mode=mode, bound=bound: value['comparisons'][name][mode][bound], cfg)
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
            for name in ('new_minus_old', 'new_minus_physical', 'old_minus_physical')},
        actual_neural_use={mode + '_' + bound: bank.legacy.interval(rows,
            lambda value, mode=mode, bound=bound: value['neural_usage_diagnostic']['main_minus_restricted'][mode][bound], cfg)
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')},
        neural_use_counts={k: sum(r['result']['neural_usage_diagnostic'][k] for r in rows)
            for k in ('neural_selected', 'known_neural_selected', 'unknown_neural_selected')},
        shared_union_oracle_regret={v: bank.legacy.interval(rows,
            lambda value, v=v: value['shared_union_oracle_regret_over_cost_scale'][v], cfg) for v in bank.VARIANTS},
        matched_empty_groups={k: sum(r['against_original'][k]['matched_empty'] for r in rows)
            for k in rows[0]['against_original']})
    diagnostics = {}
    for v in bank.VARIANTS:
        values = [r['budget_diagnostics'][v] for r in rows]
        selected = sum(p['known_selected_views'] for p in values)
        legal = sum(p['known_legal_action_views'] for p in values)
        def mean(field, count, total):
            return [sum(p[field][i] * p[count] for p in values if p[count]) / total for i in range(2)] if total else None
        diagnostics[v] = dict(known_selected_repeated_views=selected, legal_repeated_views=legal,
            legal_normalized_score_MSE=mean('legal_normalized_score_MSE', 'known_legal_action_views', legal),
            selected_normalized_score_bias=mean('selected_normalized_score_bias', 'known_selected_views', selected),
            selected_score_underprediction_views=[sum(p['selected_score_underprediction_views'][i] for p in values)
                for i in range(2)], probability_calibration_or_risk_guarantee=False)
    return dict(result, budget_diagnostics=diagnostics)


def summarize(rows, contexts, cfg, training):
    expected = {(c, s, seed, o) for c, sources in contexts.items() for s in sources
                for seed in cfg['seeds'] for o in OBJECTIVES}
    keys = [(r['context'], r['source'], r['seed'], r['objective_arm']) for r in rows]
    if (len(rows) != 216 or len(set(keys)) != 216 or set(keys) != expected or len(contexts) != 6
            or any(len(v) != 4 or len(set(v)) != 4 for v in contexts.values())):
        raise ValueError('All216original context/source/seed/objective groups required')
    by = {o: [r for r in rows if r['objective_arm'] == o] for o in OBJECTIVES}
    summaries = {o: objective_summary(by[o], cfg) for o in OBJECTIVES}
    main = summaries['upper_score']
    risk = main['safety']['new_neural']
    positive = lambda p: p['CI95'] is not None and p['CI95'][0] > 0
    gates = dict(complete162fits216groups=training['count'] == 162,
        exact54original_states72readouts=training['exact_original_controls'] == 54
            and all(r['original_readout_exact'] for r in by['original']),
        complete_primary_recording_risk=risk['supported_views'] == risk['required_views']
            and risk['supported_record_views'] == risk['required_record_views'],
        primary_known_CI_positive=positive(main['against_original']['new_neural']['full_known_pp']),
        primary_completion_lower_CI_positive=positive(main['against_original']['new_neural']['full_lower_pp']),
        versus_original_old_neural_completion_lower_CI_positive=positive(main['against_original']['new_minus_original_old']['full_lower_pp']),
        actual_neural_use_completion_lower_CI_positive=positive(main['actual_neural_use']['full_lower_pp']))
    return dict(result_source='fresh_full_fixed_budget_score_cost_inference_SOURCE_fitting_only',
        groups=216, models=162, primary='upper_score_new_neural_minus_original_new_neural',
        objectives=summaries, fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        known_score_diagnostics_not_calibration=True, original_risk_rule_unchanged=True,
        count_matching_utility_only=True, empty_comparison_not_estimable=True,
        independent_confirmation=False, whole_pipeline_OOF=False, scene_joint_decisions_trained=False,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
