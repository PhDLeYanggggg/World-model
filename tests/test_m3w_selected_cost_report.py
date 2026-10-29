from scripts.report_m3w_selected_cost_audit import median, render


def test_median_uses_both_middle_values_and_keeps_undefined_out():
    records = [dict(diagnosis={'selected':{'value':v}}) for v in [None, 1., 2., 3., 4.]]
    assert median(records, 'selected', 'value') == 2.5


def test_selected_report_does_not_claim_uniform_underprediction_or_safety():
    text = render({'families':{}})
    assert 'not the universal failure' in text
    assert 'not prove selection caused the error' in text
    assert 'do not' in text and 'replace full-completion' in text
    assert 'Independent' in text and 'unchanged' in text
