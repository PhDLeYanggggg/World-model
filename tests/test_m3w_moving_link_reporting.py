import json
import pytest

from scripts import report_m3w_moving_link_repair as report


def test_missing_comparison_is_not_reported_as_a_zero_effect():
    assert report.interval(dict(point=None, CI95=None)) == 'not estimable'
    assert report.interval(dict(point=0., CI95=[0., 0.])) == '+0.0000 [+0.0000, +0.0000]'


def test_fitting_mass_summary_requires_all18views_in_each_region(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'PUBLIC', tmp_path)
    destination = tmp_path/'fitting_check'
    destination.mkdir()
    rows = [dict(arm=arm, region=region, scores=dict(selected=1,
        realized_selected_known_moments=[0., 4., 10., 5., 2.],
        predicted_on_selected_known_moments=[0., 2., 10., 5., 1.]))
        for arm in ('frozen_parent','continue_original','decoder_auxiliary')
        for region in ('all_neural','frozen_parent_independent','continue_original_independent','decoder_auxiliary_independent')
        for _ in range(18)]
    (destination/'analysis.json').write_text(json.dumps(dict(rows=rows)))
    (destination/'verify_execution.json').write_text(json.dumps(dict(exact_scalar_replay=True)))
    summary = report.fitting_summary()
    assert len(summary) == 12
    assert all(r['H_over_predicted_H'] == 2 for r in summary)
    assert all(r['known_easy_risk_percent'] == 40 for r in summary)
    assert all(r['selected_row_views'] == 18 for r in summary)
    rows.pop()
    (destination/'analysis.json').write_text(json.dumps(dict(rows=rows)))
    with pytest.raises(ValueError, match='All18fitting'):
        report.fitting_summary()
