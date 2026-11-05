from scripts.report_m3w_moving_admission_priority import table, NAMES


def test_failed_and_undefined_views_are_not_hidden():
    record = dict(known=dict(mean=None), hard=dict(mean=-2.), switch_percent=dict(mean=0.),
        safety=dict(supported=0, required=17, known_all_violations=3,
                    known_easy_violations=4, undefined_selected_easy=2))
    text = table({name: record for name in NAMES})
    assert text.count('NA') == len(NAMES)
    assert text.count('0/17') == len(NAMES)
    assert text.count('3/4') == len(NAMES)
    assert text.count('| 2 |') == len(NAMES)
