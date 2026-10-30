import copy
import json
import pytest
from scripts import report_m3w_position_anchor as report


def test_complete_registered_readouts_and_posthoc_replay_are_required():
    value = report.build()
    s = value['summary']
    assert s['groups'] == len(value['rows']) == 24
    assert s['unique_target_rows'] == 149510 and s['unique_recordings'] == 74
    assert s['record_context_views'] == 148
    assert not s['fitting_advancement_gate'] and not s['deployment_changed']
    assert value['mechanism']['summary']['all24_parent_readouts_exact']
    assert value['mechanism']['verification']['all24groups_exact']
    assert all(v['supported_record_views'] == 0 for v in s['safety'].values())
    assert all(v['point'] is None for v in s['FDE'].values())


def test_reports_preserve_negative_primary_unknowns_and_diagnostic_role():
    v = report.build()
    text = report.render(v)
    for required in ('-21.078204', '-19.349820', 'not_estimable', '0/148',
                     'not a matched-count selector', 'not independent', 'no new neural training'):
        assert required in text
    diagnosis = report.render_mechanism(v)
    assert 'post-hoc' in diagnosis and 'not ADE' in diagnosis
    assert 'measurement noise' in diagnosis and 'not identified' in diagnosis
    assert '292694' in diagnosis and '6326' in diagnosis


def test_canonical_validation_does_not_silently_accept_metadata_or_metric_changes():
    v = report.build()
    changed = copy.deepcopy(v['summary']); changed['effects']['8']['lower_pp']['point'] = 0.
    with pytest.raises(ValueError, match='Canonical'):
        report.validate_summary(v['rows'], changed)
    changed = copy.deepcopy(v['summary']); changed['record_context_views'] -= 1
    with pytest.raises(ValueError, match='Recording'):
        report.validate_summary(v['rows'], changed)


def test_report_evidence_and_all_record_rows_match_frozen_summary():
    v = report.build(); p = report.run.PUBLIC
    evidence = json.loads((p/'evidence.json').read_text())
    assert evidence['summary_sha256'] == report.run.sha(p/'summary.json')
    assert evidence['mechanism_summary_sha256'] == report.run.sha(p/'mechanism/summary.json')
    assert evidence['new_training_updates'] == 0 and not evidence['submission_ready']
    assert evidence['record_tail_rows'] == 4*148
    assert evidence['source_proxy_rows'] == 24
    assert (p/'readout.md').read_text() == report.render(v)
    for name, expected in (('record_tail_table.csv', 593), ('pair_proxy_table.csv', 25)):
        content = (p/name).read_bytes()
        assert b'\r' not in content and len(content.splitlines()) == expected
