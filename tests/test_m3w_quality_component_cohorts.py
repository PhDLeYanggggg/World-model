from scripts.report_m3w_quality_component_cohorts import cohort_summary


def row(site, reference, harm):
    c = dict(selected=3, unknown=1, known_truth_moments=[0., harm, reference, reference, harm],
             easy=dict(predicted_excess_mass=-.01*reference, harm_underestimate_mass=harm-.01*reference,
                       reference_inflation_budget_mass=0.))
    return dict(source=site, result=dict(arms=dict(quality=dict(cohorts=dict(added=c)))))


def test_repeated_heads_not_additional_localities():
    rows = [row('a', 1., .03), row('b', 1., .07)]
    assert cohort_summary(rows, 'added')['equal_locality_mean'] == cohort_summary(rows*3, 'added')['equal_locality_mean']
    assert cohort_summary(rows*3, 'added')['defined_localities'] == 2


def test_zero_reference_is_unknown_not_zero_risk():
    r = cohort_summary([row('a', 0., 0.)], 'added')
    assert r['defined_localities'] == 0
    assert r['unknown_selected'] == 1
    assert r['equal_locality_mean']['actual_known_easy_risk_percent'] is None
