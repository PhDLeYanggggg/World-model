import json

import pytest

from scripts import report_m3w_songdo_secondary as report
from tests.test_m3w_songdo_confirmation import Fixture, runner, specification


def test_calibrated_receipts_report_risk_and_native_errors_without_new_calls(tmp_path):
    data, spec = Fixture(), specification()
    for phase in ('register', 'calibrate', 'confirm'):
        runner.execute(tmp_path, phase, spec, data)
    calls = list(data.calls)
    r = report.read_finished(tmp_path)
    assert data.calls == calls
    assert r['flights'] == 2 and r['flight_seed_views'] == 6
    assert r['matched']['joint']['original_risk']['original_risk_pass'] == 3
    j = r['released']['joint']
    assert j['risk']['edited'] == 6 and j['risk']['measured_risk_violation'] == 3
    assert j['risk_by_source']['C']['original_risk_pass'] == 3
    assert j['metrics']['available_ade_native']['worst_estimable_source'] == 'D'
    assert j['metrics']['available_fde_native']['defined_seed_views'] == 6
    assert j['edited_only']['hard_known_gain_percent']['status'] == 'required_component_not_estimable'


def test_fallback_is_measured_for_accuracy_but_not_certified(tmp_path):
    data, spec = Fixture(), specification()
    spec.update(mode='zero_shot', site_roles={s: 'confirmation' for s in 'ABCD'},
                release=dict(method='fixed_scores', threshold_source='synthetic fixture',
                             thresholds={str(s): {a: None for a in runner.protocol.ARMS}
                                         for s in runner.protocol.SEEDS}))
    runner.execute(tmp_path, 'register', spec, data)
    runner.execute(tmp_path, 'confirm', spec, data)
    r = report.read_finished(tmp_path)
    j = r['released']['joint']
    assert j['risk']['edited'] == j['risk']['original_risk_pass'] == 0
    assert j['risk']['no_edit_not_certified'] == 15
    assert j['metrics']['available_ade_native'] == r['full_roster_motion_reference']['cv']['metrics']['available_ade_native']
    assert j['edited_only']['easy_known_gain_percent']['point'] is None


def test_matched_only_never_reports_release_and_tampering_fails(tmp_path):
    data = Fixture()
    spec = dict(mode='zero_shot', approval_reference='synthetic only',
                site_roles={s: 'confirmation' for s in 'ABCD'}, cutoff_rule='not_used',
                release=dict(method='matched_only'))
    runner.execute(tmp_path, 'register', spec, data)
    runner.execute(tmp_path, 'confirm', spec, data)
    r = report.read_finished(tmp_path)
    assert 'released' not in r and 'original_risk' not in r['matched']['joint']
    p = tmp_path/'confirmation'/'a.json'
    x = json.loads(p.read_text())
    x['scores']['17']['matched']['arms']['cv']['available_ade_native'] = 123.
    p.write_text(json.dumps(x))
    with pytest.raises(ValueError, match='do not reproduce'):
        report.read_finished(tmp_path)
