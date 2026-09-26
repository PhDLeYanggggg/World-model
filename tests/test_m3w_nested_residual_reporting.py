import json
import numpy as np
import pytest
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


@pytest.mark.parametrize('variant',run.method.VARIANTS)
def test_probe_provenance_metadata_does_not_change_numbers(variant):
    rng=np.random.default_rng(5); n=60
    context=rng.normal(size=(n,7)); sites=np.repeat(['a','b','c'],20)
    p=np.tile([3.,2.,1.,.2],(n,1)); y=p.copy(); y[:,3]=rng.uniform(0,2,n)
    w=np.ones(n)/n
    prior=run.residual.fit(context,p,y,w,sites,'d',ridge=.1)
    current=run.fit_residual_bank(context,p,y,w,sites,'d',ridge=.1,variant=variant)
    for arm in prior:
        assert current[arm]['fitted_with_in_sample_base_predictions'] is (variant!='oof')
        assert current[arm]['residual_prediction_source']==variant
        for k in prior[arm]:
            if k!='fitted_with_in_sample_base_predictions': assert current[arm][k]==prior[arm][k]
        np.testing.assert_array_equal(run.residual.predict(current[arm],context,p),run.residual.predict(prior[arm],context,p))
