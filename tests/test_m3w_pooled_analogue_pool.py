import numpy as np
import pytest
from src.evaluation import m3w_pooled_analogue as api


def packets():
    return {s:(np.zeros((2,380)),np.ones(2),np.ones((2,5)),np.zeros((2,12,5)),
        np.repeat(s,2),np.repeat('same',2),np.arange(2),{'training_site':s})
        for s in ('a','b','c','d')}


def test_namespace_and_pool_alignment(monkeypatch):
    monkeypatch.setattr(api,'eligible',lambda args:np.ones(len(args[0]),bool))
    b,spans,census=api.pool(packets())
    assert spans=={'a':(0,2),'b':(2,4),'c':(4,6),'d':(6,8)}
    assert len(set(b['recordings']))==4
    assert b['sources'].tolist()==['a','a','b','b','c','c','d','d']
    assert all(v['min_pooled_donors_after_record_exclusion']==6 for v in census.values())
    assert not any(v['pooled_supported_all_recordings'] for v in census.values())


def test_unknown_rows_retained_in_bank_not_supervised_donor_count(monkeypatch):
    monkeypatch.setattr(api,'eligible',lambda args:np.ones(len(args[0]),bool))
    p=packets();p['a'][2][0]=np.nan
    b,_,c=api.pool(p)
    assert len(b['labels'])==8 and np.isnan(b['labels'][0]).all()
    assert c['a']['unknown_eligible']==1 and c['b']['min_pooled_donors_after_record_exclusion']==5


def test_wrong_source_metadata_rejected(monkeypatch):
    monkeypatch.setattr(api,'eligible',lambda args:np.ones(len(args[0]),bool))
    p=packets();p['a'][-1]['training_site']='wrong'
    with pytest.raises(ValueError,match='identity'):api.pool(p)
