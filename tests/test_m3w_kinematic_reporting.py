from scripts import report_m3w_moving_kinematic_risk as report


def test_all_controls_and_undefined_support_are_visible():
    row = dict(known=dict(mean=-.5), hard=dict(mean=None), switch_percent=dict(mean=0.),
        safety=dict(supported=0, required=17, known_all_violations=13,
            known_easy_violations=15, undefined_selected_easy=2))
    text = report.table({name: row for name in report.NAMES})
    assert text.count('0/17') == 6 and text.count('13/15') == 6
    assert text.count('NA') == 6 and text.count('-0.500000') == 6
    assert 'masked_veto' in text and 'kinematic_veto' in text
