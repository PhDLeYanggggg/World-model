from copy import deepcopy
from scripts import report_m3w_european_oof_magnitude as report


def fake(monkeypatch):
    m = {k:dict(positive=6, negative=0, overlap=0, not_estimable=0)
         for k in (report.PRIMARY, *report.GUARDS)}
    s = {k:dict(full=deepcopy(m)) for k in report.COMPARISONS}
    monkeypatch.setattr(report, 'paired_contrasts', lambda *_:{})
    monkeypatch.setattr(report, 'summarize_contrasts', lambda _:s)
    return s


def test_matched_control_required_not_only_identity(monkeypatch):
    s = fake(monkeypatch)
    assert report.aggregate([], {})['gates']['advance_gate']
    s['scaled_true_vs_scaled_cost']['full'][report.PRIMARY]['positive'] = 0
    g = report.aggregate([], {})['gates']
    assert not g['advance_gate'] and not g['deployment_changed']


def test_negative_guard_and_missing_data_block(monkeypatch):
    s = fake(monkeypatch)
    s['scaled_true_vs_raw_true']['full'][report.GUARDS[0]]['negative'] = 1
    assert not report.aggregate([], {})['gates']['advance_gate']
    s['scaled_true_vs_raw_true']['full'][report.GUARDS[0]]['negative'] = 0
    s['scaled_true_vs_raw_true']['full'][report.GUARDS[0]]['not_estimable'] = 1
    assert not report.aggregate([], {})['gates']['advance_gate']
