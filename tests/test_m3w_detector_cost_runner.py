import json
import time

import torch

from scripts import run_m3w_detector_cost as run


def test_pilot_resumes_its_own_ten_step_checkpoint_without_external_resume(monkeypatch,tmp_path):
    cfg=dict(representations=['history'],banks=['physical_neural'])
    monkeypatch.setattr(run,'PUBLIC',tmp_path/'public');run.PUBLIC.mkdir()
    monkeypatch.setattr(run,'PRIVATE',tmp_path/'private')
    monkeypatch.setattr(run,'admission',lambda:(cfg,[None,None,dict(contexts=dict(ctx=[]))],None,None))
    monkeypatch.setattr(run,'runtime',lambda _:(time.monotonic(),lambda **_:None,lambda:None))
    monkeypatch.setattr(run.parent,'packets',lambda *_:{})
    monkeypatch.setattr(run.parent.api,'training_data',lambda *_:{})
    monkeypatch.setattr(run,'data_for',lambda *_:{})
    calls=[]
    def fit(data,cfg,context,seed,bank,arm,path,beat,guard,*,stop_at=None,resume=False):
        if path.exists() and not resume:raise ValueError('Existing internal checkpoint requires resume')
        path.parent.mkdir(parents=True,exist_ok=True);path.touch();calls.append((stop_at,resume))
        return dict(parameters=30405,unknown_rows_drawn=1,unknown_rows_with_loss=0,
                    seconds=.1,model=dict(detector_weight=torch.ones(1)))
    monkeypatch.setattr(run,'fit',fit)
    run.pilot()
    assert calls==[(10,False),(20,True),(20,False)]
    assert json.loads((run.PUBLIC/'pilot.json').read_text())['engineering_pass']
