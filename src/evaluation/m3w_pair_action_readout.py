"""Unchanged ADE and original full recording-risk evaluation for pair decoding."""
import numpy as np
from src.world_model import m3w_pair_action_allocation as api
from src.evaluation import m3w_active_bank_cost_readout as bank
from src.evaluation.m3w_frame_action_readout import pair_proxy
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.evaluation.m3w_detector_forecast_readout import tail_stats

NAMES = (*api.NAMES, 'floor')


def readout(packet, choices, ranks, allocation, easy_cut, origins):
    choices = dict(choices, floor=np.full(len(packet['ids']), -1, np.int64))
    policies = {a: bank.policy_metrics(packet, c) for a, c in choices.items()}
    records = {str(r): {a: bank.legacy.compact_score(packet['y'][at], packet['envelope'][at], c[at])
        for a, c in choices.items()} for r in sorted(set(packet['recordings'])) for at in (packet['recordings'] == r,)}
    proxy = pair_proxy(packet, api.base.query_groups(packet['recordings'], packet['frames'], packet['ids']), choices, origins)
    proxy['optimized_by_this_controller'] = 'sparse_past_4nearest_graph_only_not_complete_readout'
    mass = allocation['graph_edges']
    reference = packet['floor_ade'][np.isfinite(packet['floor_ade'])].sum()
    ade = {a: bank.legacy.chosen_values(packet['ade'], packet['floor_ade'], c) for a, c in choices.items()}
    return dict(rows=len(packet['ids']), policies=policies, recordings=records, allocation=allocation,
        contrasts=bank.pair(packet, choices['unary_joint'], ranks['unary_joint'], packet, choices['pair_joint'], ranks['pair_joint']),
        floor_lower={a: 100*v['policy']['selected_net_gain_lower_mass']/reference if reference > 0 else None for a, v in policies.items()},
        subsets=subsets(packet, choices, easy_cut), pair_proxy=proxy,
        graph_pair_change=(allocation['graph_pair_mass_after']-allocation['graph_pair_mass_before'])/mass if mass else None,
        record_tails={r: {a: tail_stats(v, packet['recordings'] == r) for a, v in ade.items()} for r in records},
        future_inputs_used=False, new_neural_training=False, independent_confirmation=False)


def summarize(rows, contexts, cfg):
    expected = {(c, s, seed) for c, ss in contexts.items() for s in ss for seed in cfg['seeds']}
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != expected:
        raise ValueError('Complete original72 fitting groups required')
    interval = lambda get: bank.legacy.interval(rows, get, cfg)
    effects = {k: interval(lambda r, k=k: r['contrasts']['full'][k]) for k in ('known_pp','lower_pp','upper_pp')}
    graph = interval(lambda r: r['graph_pair_change'])
    risk = {a: bank.safety(rows, a) for a in NAMES}
    positive = lambda x: x['CI95'] is not None and x['CI95'][0] > 0
    gates = dict(complete72groups=True, exact72_frozen_policy_controls=all(r['old_control_exact'] for r in rows),
        same_count_and_budgets=all(r['result']['allocation']['same_count_every_query'] for r in rows),
        primary_lower_CI_positive=positive(effects['lower_pp']),
        graph_proxy_reduced=graph['CI95'] is not None and graph['CI95'][1] < 0,
        original_full_recording_risk=risk['pair_joint']['supported_views'] == 72 and
            risk['pair_joint']['supported_record_views'] == risk['pair_joint']['required_record_views'])
    return dict(groups=72, accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in NAMES},
        FDE={a: interval(lambda r, a=a: r['policies'][a]['known_FDE_gain_floor_percent']) for a in NAMES},
        effects=effects, graph_pair_change=graph, safety=risk, subsets=pooled_subsets(rows),
        query_views=sum(r['result']['allocation']['queries'] for r in rows),
        changed_queries=sum(r['result']['allocation']['changed_queries'] for r in rows),
        solver_calls=sum(r['result']['allocation']['solver_calls'] for r in rows),
        nonoptimal_solver_queries=sum(r['result']['allocation']['nonoptimal_solver_queries'] for r in rows),
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        primary='pair_minus_unary_full_policy_completion_lower', new_training_updates=0,
        result_source='fresh_pair_decoder_on_cached_verified_exposed_SOURCE_fitting_models',
        independent_confirmation=False, whole_pipeline_OOF=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
