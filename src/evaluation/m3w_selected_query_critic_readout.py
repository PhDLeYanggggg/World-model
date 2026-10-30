"""Complete-policy comparisons of matched selected-set risk recalibrators."""
import numpy as np
from src.world_model import m3w_critic_allocation as decoding
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_frame_action_readout import pair_proxy
from src.evaluation.m3w_detector_forecast_readout import tail_stats

NAMES = (*decoding.NAMES, 'parent_independent', 'parent_joint', 'floor')
CONTRASTS = dict(set_joint_minus_row_joint=('row_joint', 'set_joint'),
    set_matched_minus_row_matched=('row_matched', 'set_matched'),
    set_joint_minus_parent_joint=('parent_joint', 'set_joint'),
    row_joint_minus_parent_joint=('parent_joint', 'row_joint'),
    set_joint_minus_set_independent=('set_independent', 'set_joint'))


def diagnostics(packet, prediction, original, initial, rms, groups):
    known = np.isfinite(packet['y']).all((1, 2)); legal = packet['allowed'] & known[:, None]
    scale = packet['scales']; target = packet['y']
    out = {}
    for name, p in dict(original=original, **prediction).items():
        err = (p[:, :, [1, 4]]-target[:, :, [1, 4]])/scale[:, None, None]/rms[[1, 4]]
        errors = []; complete = 0; unknown = 0
        for g in groups:
            selected = g[initial[g] >= 0]
            if not len(selected): continue
            if not known[selected].all(): unknown += 1; continue
            ix, actions = selected, initial[selected]
            difference = (p[ix, actions][:, [1, 4]]-.02*p[ix, actions][:, [2, 3]]
                -target[ix, actions][:, [1, 4]]+.02*target[ix, actions][:, [2, 3]])
            score_rms = np.sqrt(rms[[1, 4]]**2+(.02*rms[[2, 3]])**2)
            errors.append((difference.sum(0)/scale[ix].sum()/score_rms)**2); complete += 1
        at = np.flatnonzero(known & (initial >= 0)); actions = initial[at]
        out[name] = dict(all_known_legal_risk_MSE=float(np.square(err[legal]).mean()) if legal.any() else None,
            frozen_selected_query_score_MSE=float(np.mean(errors)) if errors else None,
            complete_frozen_selected_queries=complete, selected_unknown_queries=unknown,
            frozen_selected_known_rows=len(at),
            predicted_easy_harm=float(p[at, actions, 4].sum()),
            observed_easy_harm=float(target[at, actions, 4].sum()),
            inference_label_filter=False, raw_mass_bookkeeping_not_physical_metric=True)
    return out


def readout(packet, choices, ranks, allocation, predictions, original, initial, legacy_independent,
            easy_cut, origins, rms):
    all_choices = dict(choices, parent_independent=legacy_independent, parent_joint=initial,
        floor=np.full(len(packet['ids']), -1, np.int64))
    old_at = np.flatnonzero(initial >= 0); old_rank = np.zeros(len(initial))
    old_rank[old_at] = original[old_at, initial[old_at], 0]-original[old_at, initial[old_at], 1]
    legacy_rank = np.zeros(len(initial)); at = np.flatnonzero(legacy_independent >= 0)
    legacy_rank[at] = original[at, legacy_independent[at], 0]-original[at, legacy_independent[at], 1]
    ranks = dict(ranks, parent_joint=old_rank, parent_independent=legacy_rank)
    groups = decoding.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    policies = {a: bank.policy_metrics(packet, c) for a, c in all_choices.items()}
    records = {str(rec): {a: bank.legacy.compact_score(packet['y'][mask], packet['envelope'][mask], c[mask])
        for a, c in all_choices.items()} for rec in sorted(set(packet['recordings']))
        for mask in (packet['recordings'] == rec,)}
    comparisons = {name: bank.pair(packet, all_choices[left], ranks[left], packet, all_choices[right], ranks[right])
        for name, (left, right) in CONTRASTS.items()}
    reference = float(packet['floor_ade'][np.isfinite(packet['floor_ade'])].sum())
    ade = {a: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], c) for a, c in all_choices.items()}
    return dict(rows=len(initial), known_rows=int(np.isfinite(packet['floor_ade']).sum()),
        unknown_rows=int((~np.isfinite(packet['floor_ade'])).sum()), policies=policies, recordings=records,
        comparisons=comparisons, calibration=diagnostics(packet, predictions, original, initial, rms, groups),
        floor_completion_lower_pp={a: 100*v['policy']['selected_net_gain_lower_mass']/reference
            if reference > 0 else None for a, v in policies.items()},
        subsets=subsets(packet, all_choices, easy_cut), allocation=allocation,
        pair_proxy=pair_proxy(packet, groups, all_choices, origins),
        recording_tails={str(rec): {a: tail_stats(v, packet['recordings'] == rec) for a, v in ade.items()}
            for rec in sorted(set(packet['recordings']))},
        full_cohort_preserved=True, inference_label_filter=False, original_benefit_references_unchanged=True,
        pair_interaction_dynamics_trained=False, unknown_targets_imputed=False)


def summarize(rows, contexts, cfg):
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != expected:
        raise ValueError('Complete original72-group fitting roster required')
    interval = lambda getter: bank.legacy.interval(rows, getter, cfg)
    comparisons = {name: {mode+'_'+bound: interval(lambda r, name=name, mode=mode, bound=bound:
        r['comparisons'][name][mode][bound]) for mode in ('full', 'matched')
        for bound in ('known_pp', 'lower_pp', 'upper_pp')} for name in CONTRASTS}
    safety = {a: bank.safety(rows, a) for a in NAMES}
    floor_lower = {a: interval(lambda r, a=a: r['floor_completion_lower_pp'][a]) for a in NAMES}
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    risk = safety['set_joint']
    gates = dict(complete72groups=True, all144parent_full_policy_controls_exact=all(r['parent_controls_exact'] for r in rows),
        paired_feature_initialization_and_training_draws=all(r['paired_training_verified'] for r in rows),
        same_count_controls_every_query=all(r['result']['allocation']['same_count_controls_every_query'] for r in rows),
        context_minus_row_full_lower_CI_positive=positive(comparisons['set_joint_minus_row_joint']['full_lower_pp']),
        context_vs_floor_completion_lower_CI_positive=positive(floor_lower['set_joint']),
        complete_context_recording_risk=risk['supported_record_views'] == risk['required_record_views'] and risk['supported_views'] == 72,
        all_main_query_optima_certified=all(not r['result']['allocation']['nonoptimal_queries'].get('set_joint', 0) for r in rows))
    return dict(groups=72, fresh_critic_fits=36, critic_training_updates=72000, new_trajectory_forecaster_fits=0,
        primary='selected_set_joint_minus_matched_row_local_joint_full_policy',
        accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        floor_completion_lower_pp=floor_lower, comparisons=comparisons, safety=safety, subsets=pooled_subsets(rows),
        calibration_risk_MSE={a: interval(lambda r, a=a: r['calibration'][a]['all_known_legal_risk_MSE'])
            for a in ('original', 'row_local', 'selected_set')},
        query_views=sum(r['result']['allocation']['queries'] for r in rows),
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        result_source='fresh_full_critic_fit_and_risk_allocation_SOURCE_fitting_resubstitution_only',
        original_full_recording_risk_unchanged=True, matched_controls_utility_only=True,
        empty_comparison_not_estimable=True, whole_pipeline_OOF=False,
        independent_roles_opened=False, independent_confirmation=False, held_scored=False,
        deployment_changed=False, submission_ready=False, metric_claim=False, seconds_claim=False,
        stage5c_executed=False, smc_enabled=False)
