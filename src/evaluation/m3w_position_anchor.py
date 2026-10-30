"""Forecast-error evaluation is separate from causal position estimation."""
import numpy as np
from src.world_model import m3w_position_anchor as model
from src.world_model.m3w_fixed_floor_probe import targets
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask, causal_bank_envelope
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_frame_action_readout import pair_proxy
from src.evaluation.m3w_detector_forecast_readout import tail_stats
from src.world_model.m3w_frame_action_allocation import query_groups

NAMES = ('ols4_raw', 'ols4_fitted', 'ols8_raw', 'ols8_fitted')


def evaluate(history, origin, future, future_valid, packet, eligible, easy_cut):
    n = len(history); eligible = np.asarray(eligible)
    if eligible.shape != (n,) or eligible.dtype != bool: raise ValueError('Frozen past-only eligibility required')
    predictions = {a: model.predict(history, window=k, anchor=mode)-origin[:, None]
        for a, k, mode in zip(NAMES, (4, 4, 8, 8), ('raw_last', 'causal_fitted')*2)}
    known = np.isfinite(packet['floor_ade'])
    ade, fde, env, labels, moving = [], [], [], [], []
    for i, a in enumerate(NAMES):
        path = predictions[a]
        error, endpoint = native_errors(path+origin[:, None], future, future_valid, np.ones(n))
        np.testing.assert_array_equal(np.isfinite(error), known)
        if a.endswith('_raw'):
            index = 4 if i == 0 else 5
            np.testing.assert_array_equal(path, packet['rollouts'][:, index])
            np.testing.assert_array_equal(error, packet['ade'][:, index])
            np.testing.assert_array_equal(endpoint, packet['fde'][:, index])
        ade.append(error); fde.append(endpoint)
        env.append(causal_bank_envelope(packet['rollouts'][:, 1], packet['floor_rollout'], path))
        labels.append(targets(packet['cv_ade'], packet['floor_ade'], error, easy_cut))
        moving.append(eligible & real_intervention_mask(path, packet['floor_rollout']))
    local = dict(packet, rollouts=np.stack(list(predictions.values()), 1), ade=np.stack(ade, 1),
        fde=np.stack(fde, 1), envelope=np.stack(env, 1), y=np.stack(labels, 1), allowed=np.stack(moving, 1))
    choices = {a: np.where(moving[i], i, -1) for i, a in enumerate(NAMES)}
    reference = float(packet['floor_ade'][known].sum()); fknown = np.isfinite(packet['floor_fde'])
    fref = float(packet['floor_fde'][fknown].sum()); policies = {}; full = {}; errors = {}
    for i, a in enumerate(NAMES):
        e = np.where(moving[i], ade[i], packet['floor_ade']); f = np.where(moving[i], fde[i], packet['floor_fde'])
        errors[a] = (e, f)
        policy = bank.legacy.compact_score(local['y'], local['envelope'], choices[a])
        policies[a] = dict(policy=policy, known_ADE_gain_floor_percent=100*(1-e[known].sum()/reference) if reference > 0 else None,
            known_FDE_gain_floor_percent=100*(1-f[fknown].sum()/fref) if fref > 0 else None)
        full[a] = dict(known_ADE_gain_floor_percent=100*(1-ade[i][known].sum()/reference) if reference > 0 else None,
            known_FDE_gain_floor_percent=100*(1-fde[i][fknown].sum()/fref) if fref > 0 else None,
            role='ungated_forecast_diagnostic_not_deployment')
    records = {str(rec): {a: bank.legacy.compact_score(local['y'][at], local['envelope'][at], choices[a][at]) for a in NAMES}
        for rec in sorted(set(packet['recordings'])) for at in (packet['recordings'] == rec,)}
    comparisons = {str(k): bank.legacy.paired_bounds(local, choices[f'ols{k}_raw'], choices[f'ols{k}_fitted']) for k in (4, 8)}
    floor = np.full(n, -1, np.int64)
    for a in NAMES: comparisons[a+'_minus_floor'] = bank.legacy.paired_bounds(local, floor, choices[a])
    shifts = {}
    for k in (4, 8):
        d = np.linalg.norm(predictions[f'ols{k}_fitted'][:, 0]-predictions[f'ols{k}_raw'][:, 0], axis=1)
        shifts[str(k)] = dict(mean_pixels=float(d.mean()), p95_pixels=float(np.quantile(d, .95)),
            zero_shift_rows=int((d == 0).sum()), label_used=False)
    return dict(rows=n, known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        policies=policies, ungated_forecasts=full, recordings=records, comparisons=comparisons,
        subsets=subsets(local, choices, easy_cut), causal_position_shift=shifts,
        record_tails={str(rec): {a: dict(ADE=tail_stats(e, at), FDE=tail_stats(f, at)) for a, (e, f) in errors.items()}
            for rec in sorted(set(packet['recordings'])) for at in (packet['recordings'] == rec,)},
        pair_proxy=pair_proxy(local, query_groups(packet['recordings'], packet['frames'], packet['ids']), choices, origin),
        neutral_raw_paths_ADE_FDE_exact=True, common_causal_eligibility=True,
        same_intervention_count_claim=False, inference_labels_used=False, new_training_updates=0)


def summarize(rows, contexts, cfg):
    expected = {(c, s) for c, ss in contexts.items() for s in ss}
    if len(rows) != 24 or {(r['context'], r['source']) for r in rows} != expected:
        raise ValueError('All24 original fitting context/source groups required')
    interval = lambda get: bank.legacy.interval(rows, get, dict(cfg, seeds=[43]))
    effects = {str(k): {b: interval(lambda r, k=k, b=b: r['comparisons'][str(k)][b])
        for b in ('known_pp', 'lower_pp', 'upper_pp')} for k in (4, 8)}
    safety = {a: bank.safety(rows, a) for a in NAMES}
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    floor_lower = interval(lambda r: r['comparisons']['ols8_fitted_minus_floor']['lower_pp'])
    risk = safety['ols8_fitted']
    gates = dict(complete24=len(rows) == 24, all24_original_identity_controls_exact=all(r['identity_control_exact'] for r in rows),
        all48_raw_path_and_cost_controls_exact=all(r['result']['neutral_raw_paths_ADE_FDE_exact'] for r in rows),
        common_causal_eligibility=all(r['result']['common_causal_eligibility'] for r in rows),
        primary_lower_CI_positive=positive(effects['8']['lower_pp']), fitted_vs_floor_lower_CI_positive=positive(floor_lower),
        complete_original_recording_risk=risk['supported_record_views'] == risk['required_record_views'] and risk['supported_views'] == 24)
    return dict(groups=24, new_training_updates=0, primary='gated_OLS8_fitted_minus_raw_completion_lower',
        effects=effects, accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        ungated_accuracy={a: interval(lambda r, a=a: r['ungated_forecasts'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        safety=safety, subsets=pooled_subsets(rows), fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        prediction_geometry_proxy_not_physical_safety=True,
        result_source='fresh_causal_position_estimation_on_cached_verified_exposed_SOURCE_fitting',
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
