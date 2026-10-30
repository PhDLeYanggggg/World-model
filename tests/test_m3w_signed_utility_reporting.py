from scripts import report_m3w_signed_utility_allocation as report
from tests.test_m3w_signed_utility_reduction import synthetic


def test_report_preserves_unknown_risk_and_no_new_training_claim(monkeypatch):
    rows, cfg = synthetic(monkeypatch); s = report.run.summarize(rows, cfg)
    value = dict(summary=s, complete=dict(pid=1, seconds=1., peak_RSS_bytes=1),
        replay_runtime=dict(pid=2, seconds=1., peak_RSS_bytes=1), reasons={}, oracle_gain_equality={})
    text = report.render(value)
    assert 'not_estimable' in text and 'Deployment unchanged' in text
    assert 'not_run:new training' in text and 'complete_original_risk: fail' in text
    assert 'not adaptive-search-adjusted confirmation' in text
    assert 'not newly discovered' in text and 'does not newly score interaction-path/tail proxies' in text


def test_complete_real_signed_readout_and_bundle():
    value = report.build(); p = report.run.PUBLIC; s = value['summary']
    assert s['groups'] == 72 and s['new_training_updates'] == 0
    assert s['fitting_gates']['all72_original_controls_exact'] and s['fitting_gates']['oracle_control_exact']
    assert s['oracle_admissibility']['required_record_contexts'] == 148
    assert (p/'readout.md').read_text() == report.render(value)
    bundle = report.verify(report.ROOT, p)
    assert bundle['groups'] == 72 and bundle['bytes_verified'] and not bundle['raw_data_or_weights_included']
    assert not s['independent_confirmation'] and not s['deployment_changed'] and not s['submission_ready']
