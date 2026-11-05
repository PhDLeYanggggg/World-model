import json
import numpy as np
from scripts import run_m3w_moving_ego_frame as r


def test_cell_serializes_numpy_scale_without_changing_metric_values(monkeypatch):
    d=dict(sites=np.array(['a','b']),xs={'original120':np.zeros((2,120))},
        target=np.ones((2,2)),weights=np.array([.5,.5]),ids=np.array(['x','y']),query=np.array([0,1]))
    scale=np.array([.3,.7])
    monkeypatch.setattr(r.source,'sha',lambda _: 'fixed_hash')
    monkeypatch.setattr(r.parent.old,'cached_control',lambda *args,**kwargs: (dict(target_scale=scale),{'path':'test'}))
    monkeypatch.setattr(r.method,'predict',lambda state,x:np.zeros((len(x),2)))
    monkeypatch.setattr(r.method.previous,'metrics',lambda p,y,w,s:dict(scale_used=s.tolist()))
    record=r.cell(d,{'key':'synthetic'},'b','original120',lambda:None)
    assert json.loads(json.dumps(record))['target_scale']==[.3,.7]
    assert record['metrics']['scale_used']==[.3,.7]
    assert record['result_source']=='cached_verified'
