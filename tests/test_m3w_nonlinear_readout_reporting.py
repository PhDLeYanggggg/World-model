from scripts import report_m3w_moving_nonlinear_readout as report


def test_report_preserves_old_controls_failures_and_undefined_support():
    p = dict(known=dict(mean=.25), hard=dict(mean=None), switch_percent=dict(mean=0.),
        safety=dict(supported=0, required=17, known_all_violations=4,
            known_easy_violations=8, undefined_selected_easy=9))
    text = report.table({name: p for name in ('conditioned_veto', 'mean_veto',
        'quantile90_veto', 'readout_veto', 'score_only_veto', 'latent_score_veto')})
    assert text.count('NA') == 6 and text.count('0/17') == 6
    assert 'score_only_veto' in text and 'latent_score_veto' in text
    assert text.count('0.250000') == 6
