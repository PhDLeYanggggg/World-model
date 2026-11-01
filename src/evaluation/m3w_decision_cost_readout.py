"""Original full-policy estimands, with explicit regression/decision arm names."""
from src.evaluation import m3w_coupled_benefit_risk as previous

NAMES = tuple(a+'_'+b for a in ('regression', 'decision') for b in ('independent', 'joint', 'matched'))+('floor',)


def rename(value):
    if isinstance(value, dict):
        return {('regression_'+k[len('protected_'):] if k.startswith('protected_') else
            'decision_'+k[len('released_'):] if k.startswith('released_') else k): rename(v) for k, v in value.items()}
    if isinstance(value, list): return [rename(v) for v in value]
    return value


def readout(*args):
    return rename(previous.readout(*args))


def summarize(rows, contexts, cfg):
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != expected:
        raise ValueError('All72 original groups required')
    interval = lambda get: previous.bank.legacy.interval(rows, get, cfg)
    effects = {mode: {bound: interval(lambda r, m=mode, b=bound: r['contrasts'][m]['full'][b])
        for bound in ('known_pp', 'lower_pp', 'upper_pp')} for mode in ('joint', 'matched')}
    safety = {a: previous.bank.safety(rows, a) for a in NAMES}
    lower = {a: interval(lambda r, a=a: r['floor_lower'][a]) for a in NAMES}
    positive = lambda v: v['CI95'] is not None and v['CI95'][0] > 0
    risk = safety['decision_joint']
    gates = dict(complete18newfits72groups=True, all72_original_full_policy_controls_exact=all(r['old_control_exact'] for r in rows),
        same_features_architecture_draws_updates=all(r['paired_training_verified'] for r in rows),
        same_count_utility_controls=all(r['result']['allocation']['same_count_controls_every_query'] for r in rows),
        decision_minus_regression_lower_CI_positive=positive(effects['joint']['lower_pp']),
        decision_floor_lower_CI_positive=positive(lower['decision_joint']),
        complete_original_recording_risk=risk['supported_views'] == 72 and risk['supported_record_views'] == risk['required_record_views'])
    return dict(groups=72, fresh_heads=18, primary_optimizer_updates=36000, exact_refit_updates=2000,
        primary='decision_minus_regression_full_joint_completion_lower', effects=effects,
        accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        safety=safety, floor_lower=lower, subsets=previous.pooled_subsets(rows), fitting_gates=gates,
        fitting_advancement_gate=all(gates.values()), result_source='fresh_SOURCE_decision_cost_training_cached_verified_forecasts',
        new_neural_trajectory_training=False, independent_confirmation=False, whole_pipeline_OOF=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
