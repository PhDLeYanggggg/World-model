import csv
import io
import json

import pytest

from scripts import report_m3w_dronecrowd_calibration as report


def outcome(chosen):
    risk = dict(required_views=120, scored_views=120, supported_views=100, undefined_easy_views=20,
                all_risk_over_2_views=2, signed_easy_over_2_views=0, worst_signed_easy_percent=.4,
                worst_recording_p95_known_ade_pixel=30., switches=200, switch_rate=.01)
    by_arm = {a: dict(risk) for a in report.ARMS}
    by_arm[report.BASE].update(switches=0, switch_rate=0., supported_views=0, undefined_easy_views=120)
    if chosen:
        by_arm[chosen].update(supported_views=120, undefined_easy_views=0, all_risk_over_2_views=0)
    effects = {f'{a}_vs_cv': dict(completion_lower_pp=dict(point=.2, interval95=[.05, .35]))
               for a in report.ARMS if a != report.BASE}
    return dict(summary=dict(recordings=40, source_groups=15, views=120, seeds=[17, 29, 43],
                             bootstrap_draws=3000, risk=by_arm, effects=effects),
                confirmation_outcome_access=False, independent_confirmation=False, original_rules_unchanged=True,
                calibration_outcomes_opened=True, calibration_pass=bool(chosen), selected_arm=chosen,
                selected_normalized_gain_margin=.01 if chosen else None, submission_ready=False,
                deployment_changed=False)


def test_replay_summary_reports_a_support_failure_despite_positive_gain():
    r = report.summarize(outcome(None))
    assert not r['eligible_arms'] and not r['independent_confirmation']
    assert 'No registered threshold qualifies' in report.render(r)
    cells = list(csv.DictReader(io.StringIO(report.table(r))))
    row = next(v for v in cells if v['arm'] == 'gate_1')
    assert row['disqualification'] == 'recording_risk_or_support'
    assert row['vs_cv_ci_low'] == '0.05'
    assert not row['vs_ols4_ci_low']


def test_reporting_preserves_a_locked_winner_and_rejects_changed_threshold():
    d = outcome('gate_1')
    r = report.summarize(d)
    assert r['eligible_arms'] == ['gate_1']
    assert r['selected_margin'] == .01
    d['selected_normalized_gain_margin'] = .03
    with pytest.raises(ValueError, match='winner rule'):
        report.summarize(d)
    d['summary']['risk']['gate_1']['required_views'] = 119
    with pytest.raises(ValueError, match='recording risk support'):
        report.summarize(d)


def test_full_support_does_not_override_a_nonpositive_lower_endpoint():
    d = outcome(None)
    d['summary']['risk']['gate_0'].update(supported_views=120, undefined_easy_views=0)
    d['summary']['effects']['gate_0_vs_cv']['completion_lower_pp']['interval95'][0] = 0.
    row = next(r for r in report.summarize(d)['rows'] if r['arm'] == 'gate_0')
    assert row['disqualification'] == ['cv_lower_ci_not_positive']


def test_generation_requires_complete_replay_and_the_same_frozen_roster(tmp_path):
    doc = outcome(None)
    role = {'assignments': {'calibration': {'recordings': ['a']}, 'confirmation': {'recordings': ['b']}},
            'input_only_counts': {'calibration': {'recordings': 1, 'queries': 2, 'eligible': 10}}}
    m = tmp_path/'roles.json'
    m.write_text(json.dumps(role))
    with pytest.raises(ValueError, match='replay must finish'):
        report.verify_artifacts(tmp_path, tmp_path, m)

    def artifact(path):
        return dict(path=path.name, sha256=report.digest(path), bytes=path.stat().st_size)

    (tmp_path/'registration.json').write_text(json.dumps(dict(parents=[artifact(m)])))
    reg = report.digest(tmp_path/'registration.json')
    f = tmp_path/'forecast_freeze.json'
    f.write_text(json.dumps(dict(registration=reg, future_labels_read=False,
        clips=[dict(sequence='a', queries=2, rows=10, registration=reg)])))
    a = tmp_path/'analysis.json'
    a.write_text(json.dumps(doc))
    for phase, path in (('infer', f), ('score', a), ('replay', a)):
        receipt = dict(artifact=artifact(path))
        (tmp_path/f'{phase}_execution.json').write_text(json.dumps(receipt))
    verified, _ = report.verify_artifacts(tmp_path, tmp_path, m)
    assert report.summarize(verified)['selected_arm'] is None
    role['assignments']['confirmation']['recordings'].append('a')
    m.write_text(json.dumps(role))
    with pytest.raises(ValueError, match='does not bind'):
        report.verify_artifacts(tmp_path, tmp_path, m)
    a.write_text('{}')
    with pytest.raises(ValueError, match='does not bind'):
        report.verify_artifacts(tmp_path, tmp_path, m)
