from scripts import report_m3w_risk_aggregation as report


def test_report_preserves_population_and_safety_limits():
    record = dict(query_excess_views=1, mass_excess_views=2,
        query_pass_mass_fail_views=2, mass_pass_query_fail_views=1,
        unsupported_reference_views=3)
    summary = {mode:{gate:{kind:record for kind in ('all','easy')}
                     for gate in ('soft','hard')}
               for mode in ('unit_control','budget_scaled')}
    text = report.content(dict(seconds=2), summary)
    assert 'not independent observations' in text
    assert 'Zero reference is undefined, not safe' in text
    assert '| budget_scaled | hard | easy | 1 | 2 | 2 | 1 | 3 |' in text
    assert 'No arm is selected or promoted' in text
