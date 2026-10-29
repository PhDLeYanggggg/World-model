"""Fixed fitting-roster contrasts; original unknown labels stay in scoring."""
import numpy as np

from src.evaluation import m3w_multiaction_cost_readout as base
from src.evaluation.m3w_multiaction_forest_readout import subsets, pooled_subsets
from src.world_model import m3w_completion_cost as model


def expected_roster(contexts, seeds):
    expected = set()
    for context, items in contexts.items():
        names = [item['packet']['identity']['source'] for item in items]
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError('Four distinct original fitting sources per context required')
        expected.update((context, source, seed, mode) for source in names for seed in seeds for mode in model.MODES)
    return expected


def paired(packet, left, right, left_rank, right_rank):
    a, b = base.matched_choices(left, right, left_rank, right_rank,
        packet['recordings'], packet['frames'], packet['ids'])
    return dict(full=base.paired_bounds(packet, left, right), matched=base.paired_bounds(packet, a, b),
        matched_count=int((a >= 0).sum()), matched_role='utility_only_not_safety')


def proxy_diagnostics(packet, prediction, bank):
    endpoint, known = model.completion_endpoints(packet['y'], packet['envelope'])
    ids = np.array(model.base.ARMS[bank]); available = packet['allowed'][:, ids]
    output = {}
    for label, mask in dict(all=np.ones(len(known), bool), known=known, unknown=~known).items():
        take = available & mask[:, None]
        error = ((prediction[:, ids]-endpoint[:, ids])/packet['scales'][:, None, None])**2
        output[label] = dict(query_rows=int(mask.sum()), real_action_rows=int(take.sum()),
            component_MSE=error[take].mean(0).tolist() if take.any() else None,
            status='defined' if take.any() else 'not_estimable')
    return dict(finite_endpoint_proxy_MSE=output, actual_unknown_errors_observed=False,
        columns=list(model.BOUND_COLUMNS), inactive_actions_excluded=True,
        conditional_risk_calibration_certificate=False)


def safety(rows, name):
    policies = [r['result']['policies'][name]['policy'] for r in rows]
    records = [value[name] for r in rows for value in r['result']['recordings'].values()]
    return dict(required_views=len(policies), supported_views=sum(p['budget_status']['supported'] for p in policies),
        required_record_views=len(records), supported_record_views=sum(p['budget_status']['supported'] for p in records),
        undefined_easy_record_views=sum(p['easy_selected_risk_upper'] is None for p in records),
        known_all_risk_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk'] > .02+1e-12 for p in records),
        known_easy_risk_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in records),
        selected=sum(p['selected_count'] for p in policies), selected_unknown=sum(p['selected_unknown'] for p in policies))


def summarize(rows, contexts, cfg):
    expected = expected_roster(contexts, cfg['seeds'])
    keys = [(r['context'], r['source'], r['seed'], r['mode']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or len(rows) != 144:
        raise ValueError('All 144 original mode/context/source/seed groups required')
    output = {}
    for mode in model.MODES:
        group = [r for r in rows if r['mode'] == mode]
        names = list(group[0]['result']['policies'])
        output[mode] = dict(groups=72,
            unique_fitting_recordings=len({(r['source'], key) for r in group for key in r['result']['recordings']}),
            safety={name: safety(group, name) for name in names},
            accuracy={name: base.interval(group, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent'], cfg) for name in names},
            FDE={name: base.interval(group, lambda r, n=name: r['policies'][n]['known_FDE_gain_floor_percent'], cfg) for name in names},
            parent_contrasts={bank: {m+'_'+b: base.interval(group,
                lambda r, n=bank, mode=m, bound=b: r['parent_contrasts'][n][mode][bound], cfg)
                for m in ('full', 'matched') for b in ('known_pp', 'lower_pp', 'upper_pp')} for bank in model.BANKS},
            neural_bank_increment={m+'_'+b: base.interval(group,
                lambda r, mode=m, bound=b: r['comparisons']['physical_neural_minus_physical_only'][mode][bound], cfg)
                for m in ('full', 'matched') for b in ('known_pp', 'lower_pp', 'upper_pp')},
            subsets=pooled_subsets(group))
    group = [r for r in rows if r['mode'] == 'completion_bounds']
    contrasts = {bank: {m+'_'+b: base.interval(group,
        lambda r, n=bank, mode=m, bound=b: r['supervision_contrasts'][n][mode][bound], cfg)
        for m in ('full', 'matched') for b in ('known_pp', 'lower_pp', 'upper_pp')} for bank in model.BANKS}
    primary = output['completion_bounds']['safety']['physical_neural']
    change, neural = contrasts['physical_neural'], output['completion_bounds']['neural_bank_increment']
    positive = lambda c: c['CI95'] is not None and c['CI95'][0] > 0
    gates = dict(full144fitting_groups=True,
        full_mixed_recording_risk=primary['supported_views'] == primary['required_views']
            and primary['supported_record_views'] == primary['required_record_views'],
        full_completion_minus_control_gain_CI_positive=positive(change['full_lower_pp']),
        matched_completion_minus_control_gain_CI_positive=positive(change['matched_lower_pp']),
        mixed_minus_physical_completion_CI_positive=positive(neural['full_lower_pp']))
    advance = all(gates.values())
    return dict(result_source='fresh_paired_Torch_fitting_readout_original_unknown_labels_preserved',
        modes=output, completion_minus_known=contrasts, fitting_gates=gates,
        fitting_advancement_gate=advance, heads=72, primary_optimizer_updates=144000,
        scope='fitting_resubstitution_only', held_scored=False, independent_confirmation=False,
        whole_pipeline_OOF=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
