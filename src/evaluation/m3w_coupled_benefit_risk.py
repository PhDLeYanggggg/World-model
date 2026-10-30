"""Full-cohort SOURCE evaluation; benefit repair never alters observed risk targets."""
import numpy as np
from src.world_model import m3w_coupled_benefit_allocation as decoder
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_frame_action_readout import pair_proxy
from src.evaluation.m3w_detector_forecast_readout import tail_stats

NAMES = (*decoder.NAMES, 'floor')


def readout(packet, choices, ranks, allocation, predictions, easy_cut, origins, rms):
    choices = dict(choices, floor=np.full(len(packet['ids']), -1, np.int64))
    policies = {a: bank.policy_metrics(packet, c) for a, c in choices.items()}
    records = {str(rec): {a: bank.legacy.compact_score(packet['y'][at], packet['envelope'][at], c[at])
        for a, c in choices.items()} for rec in sorted(set(packet['recordings'])) for at in (packet['recordings'] == rec,)}
    contrasts = {mode: bank.pair(packet, choices['protected_'+mode], ranks['protected_'+mode],
        packet, choices['released_'+mode], ranks['released_'+mode]) for mode in ('joint', 'matched')}
    reference = packet['floor_ade'][np.isfinite(packet['floor_ade'])].sum()
    known = np.isfinite(packet['y']).all((1, 2)); legal = packet['allowed'] & known[:, None]
    calibration = {a: {str(k): float(np.square((p[:, :, k]-packet['y'][:, :, k])/
        packet['scales'][:, None]/rms[k])[legal].mean()) if legal.any() else None for k in (0, 1, 4)}
        for a, p in predictions.items()}
    ade = {a: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], c) for a, c in choices.items()}
    return dict(rows=len(known), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        policies=policies, recordings=records, contrasts=contrasts, calibration=calibration,
        floor_lower={a: 100*v['policy']['selected_net_gain_lower_mass']/reference if reference > 0 else None for a, v in policies.items()},
        subsets=subsets(packet, choices, easy_cut), allocation=allocation,
        pair_proxy=pair_proxy(packet, decoder.base.query_groups(packet['recordings'], packet['frames'], packet['ids']), choices, origins),
        record_tails={str(rec): {a: tail_stats(e, packet['recordings'] == rec) for a, e in ade.items()} for rec in records},
        original_evaluation_reference_and_targets_unchanged=True, future_inputs_used=False,
        world_dynamics_training=False, whole_pipeline_OOF=False)


def summarize(rows, contexts, cfg):
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != expected:
        raise ValueError('All72 original fitting source/context/seed groups required')
    interval = lambda get: bank.legacy.interval(rows, get, cfg)
    effects = {mode: {bound: interval(lambda r, m=mode, b=bound: r['contrasts'][m]['full'][b])
        for bound in ('known_pp', 'lower_pp', 'upper_pp')} for mode in ('joint', 'matched')}
    safety = {a: bank.safety(rows, a) for a in NAMES}
    lower = {a: interval(lambda r, a=a: r['floor_lower'][a]) for a in NAMES}
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    risk = safety['released_joint']
    gates = dict(complete36fits72groups=True,
        all72_old_prediction_and_full_policy_controls_exact=all(r['old_control_exact'] for r in rows),
        matched_features_initialization_and_draws=all(r['paired_training_verified'] for r in rows),
        same_count_utility_controls=all(r['result']['allocation']['same_count_controls_every_query'] for r in rows),
        primary_released_minus_protected_lower_CI_positive=positive(effects['joint']['lower_pp']),
        released_floor_lower_CI_positive=positive(lower['released_joint']),
        complete_original_recording_risk=risk['supported_views'] == 72 and risk['supported_record_views'] == risk['required_record_views'])
    return dict(groups=72, fresh_heads=36, primary_optimizer_updates=72000, exact_refit_updates=4000,
        primary='released_minus_protected_full_joint_completion_lower', effects=effects,
        accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        safety=safety, floor_lower=lower, subsets=pooled_subsets(rows), fitting_gates=gates,
        fitting_advancement_gate=all(gates.values()),
        result_source='fresh_coupled_cost_head_training_on_cached_verified_exposed_SOURCE_fitting',
        new_neural_trajectory_training=False, independent_confirmation=False, whole_pipeline_OOF=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
