import pytest
from scripts.report_m3w_temporal_auxiliary_readout import interval, render, diagnose, COMPARATORS


def fixture():
    contrast = dict(mean=-1., CI95=[-2., -.5], status='development')
    s = dict(groups=72, localities=12, trained_cost_heads=216, independent_confirmation=False,
             deployment_changed=False, transfer_evaluated=False, stage5c_executed=False, smc_enabled=False,
             advance_to_transfer_design=False, failure_reasons=['unsupported_selected'], safety={}, contrasts={})
    for a in COMPARATORS:
        for suffix in ('all_MSE', 'original_selected_MSE', 'full_paired_lower_percent',
                       'matched_paired_lower_percent', 'full_lower_proxy_delta_percent',
                       'matched_lower_proxy_delta_percent'):
            s['contrasts'][a+'_'+suffix] = contrast.copy()
    c = dict(independent_scalar_checks=10, seconds=12., peak_RSS_bytes=2**30)
    return s, c


def test_undefined_never_rendered_as_zero_or_dropped():
    s, c = fixture()
    s['contrasts']['none_original_selected_MSE'] = dict(mean=None, CI95=None, status='undefined_support_no_dropping')
    text = render(s, c)
    assert 'undefined (undefined_support_no_dropping)' in text
    assert 'Advance to transfer design: **false**' in text and 'unsupported_selected' in text
    assert 'not raw FDE improvement' in text and 'not 72' in text


def test_sign_and_interval_preserved():
    assert interval(dict(mean=-.5, CI95=[-1., .1])) == '-0.500000 [-1.000000, +0.100000]'


@pytest.mark.parametrize('key', ['independent_confirmation', 'deployment_changed', 'transfer_evaluated', 'stage5c_executed', 'smc_enabled'])
def test_report_rejects_claim_expansion(key):
    s, c = fixture(); s[key] = True
    with pytest.raises(ValueError): render(s, c)


def diagnostic_fixture():
    rows = []
    for i in range(72):
        policies = {a: dict(known_easy_positive_risk=.03 if i % 2 else .01, easy_selected_risk_upper=.04)
                    for a in (*COMPARATORS, 'temporal')}
        pairs = {a+'_'+m: dict(known_difference_mass=-1., unknown_exchanged_envelope_mass=.5)
                 for a in ('original', 'none', 'rowmean') for m in ('full', 'matched')}
        rows.append(dict(group=str(i), source=str(i % 12), head_seed=17,
            result=dict(policies=policies, pairs=pairs, full_known_reference_mass=100.,
                scores=dict(temporal=dict(original_selected=dict(conditional_MSE=None if i == 0 else 1.))))))
    return rows


def test_diagnosis_separates_known_harm_from_unknown_penalty():
    d = diagnose(diagnostic_fixture())
    assert d['risk']['temporal']['known_violations'] == 36
    assert d['risk']['temporal']['upper_violations_with_known_safe'] == 36
    assert d['paired_utility_decomposition']['none']['full']['paired_lower_mean'] == -1.5
    assert len(d['undefined_common_cohort']) == 1


def test_diagnosis_rejects_dropping_or_duplicate_views():
    rows = diagnostic_fixture()
    with pytest.raises(ValueError): diagnose(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError): diagnose(rows)
