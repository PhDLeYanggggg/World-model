import json
from src.evaluation.m3w_risk_moment_crossfit import MEASURES


def test_reporting_reads_relative_artifact_paths_and_replays(tmp_path, monkeypatch):
    from scripts import report_m3w_risk_moment_crossfit as report
    public = tmp_path/'public'; public.mkdir()
    record = dict(fit=dict(step=2000, seconds=1.5, unknown_rows_sampled=0, parameters=22914))
    (tmp_path/'complete.json').write_text(json.dumps(record))
    (public/'prediction_freeze.json').write_text(json.dumps(dict(heads=[dict(path='complete.json')])))
    measures = {prefix+'_'+key: dict(point=1., ci95=[0., 2.])
                for prefix in ('fit','held','gap') for key in MEASURES}
    payload = dict(by_candidate=dict(dimensionless=measures, damped=measures), by_seed={})
    (public/'summary.json').write_text(json.dumps(payload))
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report.run, 'PUBLIC', public)
    report.main()
    before = {p.name: p.read_bytes() for p in public.iterdir()}
    report.main()
    assert before == {p.name: p.read_bytes() for p in public.iterdir()}
    assert json.loads((public/'training.json').read_text())['updates'] == 2000
    assert not json.loads((public/'gates.json').read_text())['diagnostic_training_complete']
