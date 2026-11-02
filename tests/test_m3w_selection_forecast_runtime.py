import json
import numpy as np
from scripts import run_m3w_selection_forecast as run


def test_prediction_pilot_writes_complete_fixed_metadata_without_duplicate_kwargs(tmp_path,monkeypatch):
    public=tmp_path/'public';private=tmp_path/'private';public.mkdir();private.mkdir()
    monkeypatch.setattr(run,'PUBLIC',public);monkeypatch.setattr(run,'PRIVATE',private)
    run.write(public/'feature_freeze.json',{})
    run.write(public/'feature_verification.json',dict(all28_raw_reparsed_all_features_exact=True,
        feature_freeze_sha256=run.sha(public/'feature_freeze.json')))
    metadata=dict(pid=1,seconds=0.,future_targets_constructed=False,forecast_errors_opened=False,new_training=False)
    monkeypatch.setattr(run,'runtime',lambda *a:(lambda *a,**k:None,lambda:None,lambda:metadata.copy()))
    monkeypatch.setattr(run,'load_features',lambda:dict(ids=np.arange(1024)))
    def predict(ref,data,path,cfg,beat,guard,**kw):
        path.parent.mkdir(parents=True,exist_ok=True)
        if kw.get('stop_at') is None:np.save(path,np.zeros((1024,12,2)))
        return {}
    monkeypatch.setattr(run,'predict_one',predict)
    monkeypatch.setattr(run,'fresh_prediction',lambda *a,**k:np.zeros((1024,12,2)))
    monkeypatch.setattr(run,'parent_prediction',lambda *a,**k:np.zeros((1024,12,2)))
    reg=dict(parent_jobs=[dict(key='single0_seed17')],lineage=[
        dict(trial='single0_seed17',representation=a,checkpoint={}) for a in ('history','disabled')])
    run.prediction_stage('prediction_pilot',dict(pilot_rows=1024,pilot_interrupt_rows=256),None,None,None,reg)
    result=json.loads((public/'prediction_pilot.json').read_text())
    assert result['engineering_pass'] and result['exact_interrupted_prediction_recovery']
    assert not result['forecast_errors_opened'] and not result['new_training']
