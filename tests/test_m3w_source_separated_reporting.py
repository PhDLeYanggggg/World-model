from scripts.report_m3w_source_separated_critics import per_source_value, pooled_cost_diagnostics


def row(site, value):
    return dict(site=site, nonempty_matched=dict(disjoint_data=dict(contrast=dict(known_pp=value))))


def test_unavailable_source_does_not_hide_other_sources_or_become_zero():
    rows = [row('a', 1.), row('a', 3.), row('b', None), row('b', 5.)]
    assert per_source_value(rows, 'a', 'disjoint_data', 'known_pp') == '2.0'
    assert per_source_value(rows, 'b', 'disjoint_data', 'known_pp') == 'NA'
    assert per_source_value(rows, 'missing', 'disjoint_data', 'known_pp') == 'NA'


def test_pooled_harm_ratios_use_identical_known_rows_not_unknown_bound_mass():
    score = dict(realized_selected_known_moments=[1., 4., 5., 2., 3.],
        predicted_on_selected_known_moments=[1., 2., 5., 2., 1.], selected_known=7,
        bounded_label_selected_moments=[9., 99., 9., 9., 999.])
    groups = [dict(cost_cells=[dict(model='critic', region='self', scores=score)])]*2
    cell = pooled_cost_diagnostics(groups)['critic__self']
    assert cell['actual_over_predicted_H'] == 2.
    assert cell['actual_over_predicted_EH'] == 3.
    assert cell['selected_known_repeated_rows'] == 14
