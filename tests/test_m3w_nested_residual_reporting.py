import json
import numpy as np
from scripts import run_m3w_european_nested_residual as run


def test_support_report_serializes_numpy_comparison(tmp_path, monkeypatch):
    sites = np.repeat(['a','b','c'], 8)
    cv = np.tile(np.linspace(.1,4,8),3)
    row = dict(x=np.column_stack((cv,cv)),raw=np.column_stack((cv,np.full(len(cv),.2))),
               cv=cv,sites=sites,outer='d',ids=np.arange(len(cv)),tag='synthetic',pair='full')
    (tmp_path/'registration_lock.json').write_text('{}')
    monkeypatch.setattr(run,'PUBLIC',tmp_path)
    monkeypatch.setattr(run,'views',lambda _:iter([row]*144))
    monkeypatch.setattr(run,'beat',lambda *a,**kw:None)
    monkeypatch.setattr(run,'artifact',lambda p: {'synthetic':p.name})
    run.support({'pairs':['full']},{})
    doc=json.loads((tmp_path/'support_report.json').read_text())
    assert len(doc['rows']) == 432 and doc['training_allowed'] is True
    assert all(type(r['numerically_supported']) is bool for r in doc['rows'])
