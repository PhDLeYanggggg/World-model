import copy

import pytest

from scripts import report_m3w_decoder_risk_factorial as report
from tests.test_m3w_decoder_risk_readout import predictions
from tests.test_m3w_shared_train_risk import packet


def fixture():
    result, _ = report.evaluation.evaluate(packet(), predictions())
    row = dict(scope='held', result=result, outer_development_read=False,
               independent_roles_read=False, deployment_changed=False)
    partition = dict(held_recordings=list(result['recordings']), fit_recordings=['other'])
    return row, partition


def test_frozen_recording_partition_and_factorial_interaction():
    row, partition = fixture()
    report.check_view(row, partition)
    changed = copy.deepcopy(row)
    changed['result']['contrasts']['decoder_by_risk_MSE_interaction'] = 1.
    with pytest.raises(ValueError):
        report.check_view(changed, partition)
    with pytest.raises(ValueError):
        report.check_view(row, dict(partition, held_recordings=['unregistered']))


def test_new_data_role_cannot_be_hidden():
    row, partition = fixture()
    row['independent_roles_read'] = True
    with pytest.raises(ValueError):
        report.check_view(row, partition)


def test_missing_model_identity_rejected():
    reg = dict(contexts={'a': []}, seeds=[17])
    with pytest.raises(ValueError):
        report.check_roster(reg, {})


def test_changed_capacity_or_partition_cannot_pass_roster_check():
    partition = dict(fit_recordings=['1'], held_recordings=['2'])
    reg = dict(contexts={'a': [dict(packet=dict(identity=dict(source='source')), partition=partition)]}, seeds=[17])
    docs = {}
    for arm in report.api.ARMS:
        count = 13644 if arm.startswith('source_') else 13149
        docs['a', 17, arm] = dict(identity=dict(context='a', seed=17, partitions={'source': partition}),
            parameter_count=count, primary_parameter_count=count-792,
            previous_control_exact=True if arm == 'shared_risk' else None)
    report.check_roster(reg, docs)
    changed = copy.deepcopy(docs)
    changed['a', 17, 'source_risk']['parameter_count'] -= 1
    with pytest.raises(ValueError):
        report.check_roster(reg, changed)
    changed = copy.deepcopy(docs)
    changed['a', 17, 'source_utility']['identity']['partitions']['source']['fit_recordings'] = ['2']
    with pytest.raises(ValueError):
        report.check_roster(reg, changed)


def test_positive_utility_cannot_override_failed_full_policy_support():
    contexts = {str(c): [dict(packet=dict(identity=dict(source=str((4*c+i) % 12)))) for i in range(4)]
                for c in range(6)}
    reg = dict(contexts=contexts, seeds=[17, 29, 43])
    contrasts = {right+'_minus_'+left+'_'+kind: (-1. if kind == 'MSE' else 1.)
                 for left, right in report.evaluation.PAIRS for kind in ('full', 'matched', 'MSE')}
    contrasts['decoder_by_risk_MSE_interaction'] = 0.
    policies = {a: dict(selected_count=1, known_easy_positive_risk=0., easy_selected_risk_upper=0.,
                       finite_completion_supported=True) for a in report.evaluation.ARMS}
    rows = []
    for scope in ('fit', 'held'):
        for context, items in contexts.items():
            for item in items:
                for seed in reg['seeds']:
                    rows.append(dict(scope=scope, context=context, source=item['packet']['identity']['source'], seed=seed,
                        result=dict(rows=1, unknown_rows=0, contrasts=contrasts, policies=copy.deepcopy(policies))))
    cfg = dict(bootstrap_draws=30, bootstrap_seed=1)
    good = report.evaluation.summarize(rows, reg, cfg)
    assert good['routing']['source_risk']['eligible_for_later_development_design']
    rows[-1]['result']['policies']['source_risk']['finite_completion_supported'] = False
    bad = report.evaluation.summarize(rows, reg, cfg)
    assert bad['routing']['source_risk']['failures'] == ['full_policy_risk_support_failed']
    assert not bad['routing']['source_risk']['eligible_for_later_development_design']


def test_report_preserves_undefined_and_negative_interval():
    assert report.interval(dict(CI95=None)) == 'not estimable'
    assert report.interval(dict(mean=-2., CI95=[-3., -1.])) == '-2.000000 [-3.000000, -1.000000]'
