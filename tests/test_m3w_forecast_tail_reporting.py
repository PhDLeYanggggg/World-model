from scripts import report_m3w_moving_forecast_tail as r


def test_empty_positive_mass_remains_na_in_report():
    scores={k:None for k in r.run.method.SCORES}
    a={'contexts':[{'summary':{'source_macro_capture_at_10percent':{'easy':scores},
        'passes_information_screen':False},'folds':[]}]}
    table,detail=r.tables(a)
    assert table.count('NA')==4 and '| fail |' in table
    assert 'not an actual deployed subset' in detail


def test_tail_capture_is_not_formatted_as_trajectory_gain():
    scores={k:.5 for k in r.run.method.SCORES}
    a={'contexts':[{'summary':{'source_macro_capture_at_10percent':{'easy':scores},
        'passes_information_screen':False},'folds':[]}]}
    table,_=r.tables(a)
    assert '50.000%' in table and 'Label oracle (diagnostic)' in table
    assert 'ADE improvement' not in table
