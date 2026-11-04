import json
import pytest
from scripts import report_m3w_moving_amplitude_cost as report


def test_empty_same_count_utility_stays_not_estimable_with_zero_amplitude_mean():
    records=[]
    for site,value in [('a',0.),('b',1.)]:
        records.append(dict(site=site,policies={},amplitudes={},matched_utility={}))
        r=records[-1]
        for arm in ('binary_only','amplitude_bank','amplitude_bank_binary_deploy'):
            r['policies'][arm]=dict(joint_free=dict(switch_rate=value))
            r['amplitudes'][arm]=dict(joint_free=dict(mean_amplitude_all_rows=value/2,action_counts=[1,0,0,0,0,0,0,1]))
            r['matched_utility'][arm]=dict(contrast=dict(known_pp=None if value==0 else 1.))
    rows=report.usage(records)
    assert all(r['mean_source_switch_percent']==50 for r in rows)
    assert all(r['mean_source_amplitude_all_rows']==.25 for r in rows)
    assert all(not r['full_scope_same_count_utility_estimable']for r in rows)
    assert report.interval(dict(point=None,CI95=None))=='not estimable'


def test_fitting_summary_requires_all18views_and_preserves_unknown_denominators(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'PUBLIC',tmp_path)
    p=tmp_path/'fitting_check';p.mkdir()
    rows=[dict(arm=arm,region=region,scores=dict(realized_selected_known_moments=[0,4,10,5,2],
        predicted_on_selected_known_moments=[0,2,10,5,0]))
        for arm in ('binary_only','amplitude_bank')
        for region in ('all_legal_full_amplitude','binary_only_independent','amplitude_bank_independent','amplitude_bank_binary_independent')
        for _ in range(18)]
    (p/'analysis.json').write_text(json.dumps(dict(rows=rows)))
    (p/'verify_execution.json').write_text(json.dumps(dict(all_scalars_exact=True)))
    values=report.fitting_summary()
    assert len(values)==8
    assert all(r['H_over_predicted_H']==2 and r['EH_over_predicted_EH']is None for r in values)
    rows.pop();(p/'analysis.json').write_text(json.dumps(dict(rows=rows)))
    with pytest.raises(ValueError,match='All18'):
        report.fitting_summary()
