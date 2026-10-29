import copy

import pytest

from scripts import report_m3w_isolated_harm as report
from tests.test_m3w_isolated_harm_readout import fixture


def grid():
    reg = dict(contexts={str(c): [dict(packet=dict(identity=dict(source=str((4*c+i) % 12))))
        for i in range(4)] for c in range(6)}, seeds=[17, 29, 43])
    contrasts = {right+'_minus_'+left+'_'+kind: (-1. if kind == 'MSE' else 1.)
        for left, right in report.evaluation.PAIRS for kind in ('full', 'matched', 'MSE')}
    p = dict(selected_count=1, known_easy_positive_risk=0., easy_selected_risk_upper=0.,
        easy_degradation_upper=0., finite_completion_supported=True)
    rows = [dict(scope=scope, context=context, source=item['packet']['identity']['source'], seed=seed,
        result=dict(rows=1, unknown_rows=0, contrasts=copy.deepcopy(contrasts),
            policies={a: copy.deepcopy(p) for a in report.evaluation.ARMS}, protected_outputs_exact=True))
        for scope in ('fit', 'held') for context, items in reg['contexts'].items()
        for item in items for seed in reg['seeds']]
    return rows, reg, dict(bootstrap_draws=30, bootstrap_seed=1)


def test_all_comparators_and_full_safety_required():
    rows, reg, cfg = grid()
    assert report.evaluation.summarize(rows, reg, cfg)['eligible_for_later_development_design']
    rows[-1]['result']['policies']['isolated_risk']['finite_completion_supported'] = False
    out = report.evaluation.summarize(rows, reg, cfg)
    assert out['failures'] == ['full_policy_risk_support_failed']
    assert not out['scientific_success'] and not out['deployment_changed']
    for r in rows:
        r['result']['contrasts']['isolated_risk_minus_joint_risk_matched'] = -1.
    assert 'matched_utility_vs_joint_risk_not_supported' in report.evaluation.summarize(rows, reg, cfg)['failures']


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'identity'])
def test_incomplete_roster_rejected(change):
    rows, reg, cfg = grid()
    if change == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    elif change == 'missing':
        rows.pop()
    else:
        rows[-1]['source'] = 'not_registered'
    with pytest.raises(ValueError):
        report.evaluation.summarize(rows, reg, cfg)


def test_fixed_cohort_not_confused_with_own_selection():
    r = report.evaluation.evaluate(*fixture())
    rows = [dict(scope=s, source='a', result=copy.deepcopy(r)) for s in ('fit', 'held')]
    detail = report.details(rows)
    assert detail['held']['anchor']['fixed_anchor_cohort']['known_selected_occurrences'] == detail['held']['isolated_risk']['fixed_anchor_cohort']['known_selected_occurrences']
    assert detail['held']['anchor']['diagnostics']['known_selected_occurrences'] > detail['held']['isolated_risk']['diagnostics']['known_selected_occurrences']
    assert detail['sources']['a']['isolated_risk']['known_easy_violations'] == 0


def test_undefined_and_negative_intervals_remain_visible():
    assert report.interval(dict(CI95=None)) == 'not estimable'
    assert report.number(None) == 'undefined'
    assert report.interval(dict(mean=-2., CI95=[-3., -1.])) == '-2.000000 [-3.000000, -1.000000]'
