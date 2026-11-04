from scripts.report_m3w_source_separated_critics import per_source_value


def row(site, value):
    return dict(site=site, nonempty_matched=dict(disjoint_data=dict(contrast=dict(known_pp=value))))


def test_unavailable_source_does_not_hide_other_sources_or_become_zero():
    rows = [row('a', 1.), row('a', 3.), row('b', None), row('b', 5.)]
    assert per_source_value(rows, 'a', 'disjoint_data', 'known_pp') == '2.0'
    assert per_source_value(rows, 'b', 'disjoint_data', 'known_pp') == 'NA'
    assert per_source_value(rows, 'missing', 'disjoint_data', 'known_pp') == 'NA'
