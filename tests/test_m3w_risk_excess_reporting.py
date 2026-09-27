import json
import numpy as np
from src.evaluation.m3w_risk_excess import metric


def test_report_paths_and_replay(tmp_path, monkeypatch):
    from scripts import report_m3w_risk_excess as report
    p = tmp_path/'public'; p.mkdir()
    values = metric(np.ones(2), np.ones(2)*2, [[1, 1], [2, 0]], [2, 1], 0, 1, 1.5, 1.8)
    s = {scope+'_'+k: dict(point=v, ci95=[v, v] if v is not None else None)
         for scope in ('fit','held') for k, v in values.items()}
    (p/'summary.json').write_text(json.dumps(dict(by_candidate=dict(dimensionless=s,damped=s), by_seed={}, gates={'deployment_changed':False})))
    r = dict(fit=dict(step=2000, seconds=1., unknown_rows_sampled=0), matched_sampling_exact=True, control_inference_exact=True)
    (tmp_path/'complete.json').write_text(json.dumps(r))
    (p/'prediction_freeze.json').write_text(json.dumps(dict(heads=[dict(path='complete.json')])))
    monkeypatch.setattr(report, 'ROOT', tmp_path); monkeypatch.setattr(report.run, 'PUBLIC', p)
    report.main(); before = {f.name: f.read_bytes() for f in p.iterdir()}
    report.main(); assert before == {f.name: f.read_bytes() for f in p.iterdir()}
    assert 'undefined' in (p/'results.md').read_text()
    assert json.loads((p/'training.json').read_text())['fresh_control_inferences_exact'] == 1
