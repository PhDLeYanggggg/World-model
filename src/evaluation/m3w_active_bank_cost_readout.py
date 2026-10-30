"""Cross-bank trajectory-aware comparisons, never stale action-index costs."""
import numpy as np
from src.world_model.m3w_active_bank_cost import VARIANTS, BANKS
from src.world_model.m3w_completion_cost import base as cost
from src.evaluation import m3w_multiaction_cost_readout as legacy
from src.evaluation.m3w_completion_cost_readout import safety
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_detector_forecast_readout import tail_stats
from src.evaluation.m3w_multiaction_support import solve_action_bank


def aligned(left, right):
    for key in ('ids', 'sites', 'recordings', 'frames', 'scales', 'floor_ade', 'floor_fde',
                'floor_rollout', 'cv_ade', 'hard'):
        np.testing.assert_array_equal(left[key], right[key])
    np.testing.assert_array_equal(np.isfinite(left['ade']), np.isfinite(right['ade']))
    np.testing.assert_array_equal(np.isfinite(left['fde']), np.isfinite(right['fde']))
    return len(left['ids'])


def cross_bank_bounds(left_packet, left, right_packet, right, take=None):
    """Action6 can denote two different paths; compare their actual coordinates."""
    n = aligned(left_packet, right_packet)
    take = np.ones(n, bool) if take is None else np.asarray(take)
    if take.shape != (n,) or take.dtype != bool:
        raise ValueError('Aligned fixed evaluation subset required')
    a = legacy.chosen_values(left_packet['ade'], left_packet['floor_ade'], left)
    b = legacy.chosen_values(right_packet['ade'], right_packet['floor_ade'], right)
    pa = legacy.chosen_values(left_packet['rollouts'], left_packet['floor_rollout'], left)
    pb = legacy.chosen_values(right_packet['rollouts'], right_packet['floor_rollout'], right)
    known = np.isfinite(a)
    np.testing.assert_array_equal(known, np.isfinite(b))
    reference = float(left_packet['floor_ade'][take & known].sum())
    delta = float((a[take & known] - b[take & known]).sum())
    missing = float(np.linalg.norm(pa[take & ~known] - pb[take & ~known], axis=-1).max(1).sum())
    return dict(known_delta_mass=delta, unknown_disagreement_mass=missing,
        known_pp=100 * delta / reference if reference > 0 else None,
        lower_pp=100 * (delta - missing) / reference if reference > 0 else None,
        upper_pp=100 * (delta + missing) / reference if reference > 0 else None,
        full_reference_mass_counted_once=reference)


def pair(left_packet, left, left_rank, right_packet, right, right_rank):
    aligned(left_packet, right_packet)
    a, b = legacy.matched_choices(left, right, left_rank, right_rank,
        left_packet['recordings'], left_packet['frames'], left_packet['ids'])
    matched = cross_bank_bounds(left_packet, a, right_packet, b)
    count = int((a >= 0).sum())
    if not count:
        # No interventions to compare: do not turn empty controls into a pass.
        matched.update(known_pp=None, lower_pp=None, upper_pp=None, reason='not_estimable_empty_count_match')
    return dict(full=cross_bank_bounds(left_packet, left, right_packet, right), matched=matched,
        matched_count=count, matched_empty=count == 0, matched_role='utility_only_not_safety',
        by_record={str(record): cross_bank_bounds(left_packet, left, right_packet, right,
                    left_packet['recordings'] == record) for record in sorted(set(left_packet['recordings']))})


def policy_metrics(packet, choice):
    """Keep the original policy score byte-for-byte comparable to its control."""
    known = np.isfinite(packet['y']).all((1, 2))
    reference = float(packet['floor_ade'][known].sum())
    cv = float(packet['cv_ade'][known].sum())
    fde_known = np.isfinite(packet['floor_fde'])
    fde_reference = float(packet['floor_fde'][fde_known].sum())
    ade = legacy.chosen_values(packet['ade'], packet['floor_ade'], choice)
    fde = legacy.chosen_values(packet['fde'], packet['floor_fde'], choice)
    oracle = np.minimum(packet['floor_ade'], np.min(np.where(packet['allowed'], packet['ade'], np.inf), 1))
    return dict(policy=legacy.compact_score(packet['y'], packet['envelope'], choice),
        known_ADE_gain_floor_percent=100 * (1 - ade[known].sum() / reference) if reference > 0 else None,
        known_ADE_gain_CV_percent=100 * (1 - ade[known].sum() / cv) if cv > 0 else None,
        known_FDE_gain_floor_percent=100 * (1 - fde[fde_known].sum() / fde_reference) if fde_reference > 0 else None,
        known_regret_over_cost_scale=float(np.mean((ade[known] - oracle[known]) / packet['scales'][known])),
        neural_interventions=int((choice == 6).sum()),
        action_counts={str(i): int((choice == i).sum()) for i in range(-1, 7)})


def moment_diagnostics(packet, prediction, choice, variant):
    ids = np.array(cost.ARMS[BANKS[variant]])
    known = np.isfinite(packet['y']).all((1, 2))
    legal = packet['allowed'][:, ids] & known[:, None]
    target = packet['y'][:, ids]
    errors = (prediction[:, ids] - target) / packet['scales'][:, None, None]
    selected = known & (choice >= 0)
    rows = np.flatnonzero(selected)
    chosen_error = (prediction[rows, choice[rows]] - packet['y'][rows, choice[rows]]) / packet['scales'][rows, None]
    return dict(known_legal_action_rows=int(legal.sum()),
        normalized_moment_MSE=np.square(errors[legal]).mean(0).tolist() if legal.any() else None,
        known_selected_rows=len(rows), selected_normalized_bias=chosen_error.mean(0).tolist() if len(rows) else None,
        selected_harm_underprediction_rows=int((chosen_error[:, 1] < 0).sum()),
        selected_easy_harm_underprediction_rows=int((chosen_error[:, 4] < 0).sum()),
        inactive_actions_excluded=True, unknown_costs_not_imputed=True,
        probability_calibration_or_safety_certificate=False)


def opportunity(packet, variant):
    output = {}
    for record in sorted(set(packet['recordings'])):
        at = packet['recordings'] == record
        allowed = packet['allowed'][at].copy()
        if variant == 'physical_only':
            allowed[:, 6] = False
        value, _ = solve_action_bank(packet['y'][at], packet['envelope'][at], allowed)
        output[str(record)] = value
    return output


def group_readout(packets, choices, ranks, predictions, easy_cut):
    if any(set(v) != set(VARIANTS) for v in (packets, choices, ranks, predictions)):
        raise ValueError('All three frozen banks,choices and predictions required')
    original = packets['physical_only']
    for packet in packets.values():
        aligned(original, packet)
    known = np.isfinite(original['floor_ade'])
    union_cost = np.column_stack((original['ade'][:, :7], packets['new_neural']['ade'][:, 6]))
    union_allowed = np.column_stack((original['allowed'][:, :7], packets['new_neural']['allowed'][:, 6]))
    oracle = np.minimum(original['floor_ade'], np.where(union_allowed, union_cost, np.inf).min(1))
    policies, slices, diagnostics, regrets, tails, proxies, oracles = {}, {}, {}, {}, {}, {}, {}
    for variant in VARIANTS:
        p, c = packets[variant], choices[variant]
        policies[variant] = policy_metrics(p, c)
        slices[variant] = subsets(p, {variant: c}, easy_cut)
        diagnostics[variant] = moment_diagnostics(p, predictions[variant], c, variant)
        ade = legacy.chosen_values(p['ade'], p['floor_ade'], c)
        selected = legacy.chosen_values(p['rollouts'], p['floor_rollout'], c)
        regrets[variant] = float(np.mean((ade[known] - oracle[known]) / p['scales'][known]))
        tails[variant] = {str(r): dict(selected=tail_stats(ade, p['recordings'] == r),
            floor=tail_stats(p['floor_ade'], p['recordings'] == r)) for r in sorted(set(p['recordings']))}
        proxies[variant] = dict(finite=bool(np.isfinite(selected).all()),
            requested_step_second_difference_mean=float(np.linalg.norm(np.diff(selected, n=2, axis=1), axis=-1).mean()),
            verified_physical_safety=False, metric_or_seconds=False)
        oracles[variant] = opportunity(p, variant)
    subset_rows = {}
    for name in ('hard', 'easy'):
        subset_rows[name] = dict(slices['physical_only'][name], policies={v: slices[v][name]['policies'][v] for v in VARIANTS})
    records = {str(r): {v: legacy.compact_score(packets[v]['y'][original['recordings'] == r],
        packets[v]['envelope'][original['recordings'] == r], choices[v][original['recordings'] == r])
        for v in VARIANTS} for r in sorted(set(original['recordings']))}
    comparisons = {label: pair(packets[a], choices[a], ranks[a], packets[b], choices[b], ranks[b])
        for label, a, b in (('new_minus_old', 'old_neural', 'new_neural'),
                            ('new_minus_physical', 'physical_only', 'new_neural'),
                            ('old_minus_physical', 'physical_only', 'old_neural'))}
    restricted, restricted_rank = cost.select_actions(predictions['new_neural'], packets['new_neural']['allowed'], arm='physical_only')
    non_neural = choices['new_neural'] != 6
    np.testing.assert_array_equal(restricted[non_neural], choices['new_neural'][non_neural])
    usage = dict(restricted_physical_policy=policy_metrics(packets['new_neural'], restricted),
        main_minus_restricted=pair(packets['new_neural'], restricted, restricted_rank,
            packets['new_neural'], choices['new_neural'], ranks['new_neural']),
        neural_selected=int((choices['new_neural'] == 6).sum()),
        known_neural_selected=int((known & (choices['new_neural'] == 6)).sum()),
        unknown_neural_selected=int((~known & (choices['new_neural'] == 6)).sum()),
        same_frozen_new_head_no_refit=True, not_a_fourth_training_variant=True)
    return dict(rows=len(known), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=float(original['floor_ade'][known].sum()), policies=policies,
        recordings=records, comparisons=comparisons, subsets=subset_rows,
        calibration=diagnostics, shared_union_oracle_regret_over_cost_scale=regrets,
        bank_local_regret_in_policy_metrics_not_cross_bank_comparable=True,
        shared_union_oracle_diagnostic_not_available_action_bank=True,
        recording_tails=tails, validity_proxies=proxies, bank_opportunity_diagnostic=oracles,
        neural_usage_diagnostic=usage, inference_label_filter=False, scene_joint_decisions_trained=False)


def summarize(rows, contexts, cfg, training):
    expected = {(context, source, seed) for context, sources in contexts.items()
                for source in sources for seed in cfg['seeds']}
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != expected or len(contexts) != 6:
        raise ValueError('All72original context/source/controller-seed groups required')
    if any(len(v) != 4 or len(set(v)) != 4 for v in contexts.values()):
        raise ValueError('Four original fitting sources per context required')
    primary_risk = safety(rows, 'new_neural')
    accuracy = {v: legacy.interval(rows, lambda r, v=v: r['policies'][v]['known_ADE_gain_floor_percent'], cfg) for v in VARIANTS}
    fde = {v: legacy.interval(rows, lambda r, v=v: r['policies'][v]['known_FDE_gain_floor_percent'], cfg) for v in VARIANTS}
    comparisons = {label: {mode + '_' + bound: legacy.interval(rows,
        lambda r, label=label, mode=mode, bound=bound: r['comparisons'][label][mode][bound], cfg)
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
        for label in ('new_minus_old', 'new_minus_physical', 'old_minus_physical')}
    positive = lambda x: x['CI95'] is not None and x['CI95'][0] > 0
    gates = dict(full54fits_and72groups=training['count'] == 54,
        legacy36endpoints_and72readouts_exact=training['legacy_exact_endpoints'] == 36 and all(r['legacy_readout_exact'] for r in rows),
        complete_new_policy_recording_risk=primary_risk['supported_views'] == primary_risk['required_views']
            and primary_risk['supported_record_views'] == primary_risk['required_record_views'],
        new_minus_old_known_CI_positive=positive(comparisons['new_minus_old']['full_known_pp']),
        new_minus_old_completion_lower_CI_positive=positive(comparisons['new_minus_old']['full_lower_pp']),
        matched_new_minus_old_completion_lower_CI_positive=positive(comparisons['new_minus_old']['matched_lower_pp']),
        new_minus_physical_completion_lower_CI_positive=positive(comparisons['new_minus_physical']['full_lower_pp']))
    oracle_rows = [r for r in rows if r['seed'] == cfg['seeds'][0]]
    for reference in oracle_rows:
        for row in rows:
            if row['context'] == reference['context'] and row['source'] == reference['source']:
                if row['result']['bank_opportunity_diagnostic'] != reference['result']['bank_opportunity_diagnostic']:
                    raise ValueError('Frozen-bank opportunity cannot depend on controller seed')
    oracles = {v: dict(required_record_views=sum(len(r['result']['bank_opportunity_diagnostic'][v]) for r in oracle_rows),
        feasible_record_views=sum(o['budget_feasible'] for r in oracle_rows for o in r['result']['bank_opportunity_diagnostic'][v].values()),
        positive_record_views=sum(o['positive_feasible'] for r in oracle_rows for o in r['result']['bank_opportunity_diagnostic'][v].values()),
        label_aware_not_deployable=True) for v in VARIANTS}
    return dict(result_source='fresh_fixed_frozen_active_bank_cost_head_inference_SOURCE_development',
        primary='new_neural_minus_old_neural', groups=72, heads=54, primary_optimizer_updates=108000,
        accuracy=accuracy, FDE=fde, safety={v: safety(rows, v) for v in VARIANTS},
        comparisons=comparisons, subsets=pooled_subsets(rows), bank_opportunity_diagnostic=oracles,
        shared_union_oracle_regret={v: legacy.interval(rows,
            lambda r, v=v: r['shared_union_oracle_regret_over_cost_scale'][v], cfg) for v in VARIANTS},
        new_head_neural_usage_increment={mode + '_' + bound: legacy.interval(rows,
            lambda r, mode=mode, bound=bound: r['neural_usage_diagnostic']['main_minus_restricted'][mode][bound], cfg)
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')},
        new_head_neural_usage_counts={key: sum(r['result']['neural_usage_diagnostic'][key] for r in rows)
            for key in ('neural_selected', 'known_neural_selected', 'unknown_neural_selected')},
        matched_empty_groups={label: sum(r['result']['comparisons'][label]['matched_empty'] for r in rows) for label in comparisons},
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        scope='original_SOURCE_controller_fitting_resubstitution', independent_confirmation=False,
        whole_pipeline_OOF=False, count_matching_utility_only=True, empty_comparison_not_estimable=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
