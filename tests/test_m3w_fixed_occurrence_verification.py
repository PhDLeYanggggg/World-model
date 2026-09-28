import pytest
from scripts.verify_m3w_fixed_occurrence import check_record_sets, run, verdict_for


@pytest.mark.parametrize('ci,expected', [
    (None, 'undefined_primary'), ([1., 2.], 'advantage_but_screen_failed'),
    ([-2., -1.], 'negative_primary'), ([-1., 1.], 'no_resolved_primary_advantage'),
    ([0., 1.], 'no_resolved_primary_advantage')])
def test_accuracy_cannot_override_failed_risk_screen(ci, expected):
    assert verdict_for(ci, False) == expected


def test_screen_pass_remains_exploratory():
    assert verdict_for([1., 2.], True) == 'exploratory_screen_pass_only'
    with pytest.raises(AssertionError): verdict_for([-1., 1.], True)


def complete():
    keys = dict(rows=run.POLICIES, qualities=run.api.ARMS,
                contrasts=[a+'_vs_'+b for a, b in run.CONTRASTS])
    return {k: [dict(group=str(g), site=str(s), policy=p)
                for g in range(108) for s in range(2) for p in policies]
            for k, policies in keys.items()}


def test_complete_records():
    check_record_sets(complete())


@pytest.mark.parametrize('error', ['missing', 'duplicate', 'wrong_policy', 'wrong_view'])
def test_incomplete_or_replaced_contrast_fails(error):
    data = complete()
    if error == 'missing': data['contrasts'].pop()
    elif error == 'duplicate': data['contrasts'][-1] = data['contrasts'][0].copy()
    elif error == 'wrong_policy': data['contrasts'][-1]['policy'] = 'invented'
    else:
        for r in data['contrasts']:
            if r['site'] == '1': r['site'] = 'unregistered_site'
    with pytest.raises(AssertionError): check_record_sets(data)
