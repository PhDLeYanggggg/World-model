import json
import time
import pytest
from scripts import run_m3w_detector_forecaster as run


def test_resume_verifies_completed_models_without_refitting_or_rewriting_receipt(monkeypatch,tmp_path):
    public,private=tmp_path/'public',tmp_path/'private';public.mkdir();private.mkdir()
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run,'PUBLIC',public);monkeypatch.setattr(run,'PRIVATE',private)
    reg=public/'registration.json';reg.write_text('{}');monkeypatch.setattr(run,'REG',reg)
    (public/'pilot.json').write_text(json.dumps(dict(engineering_pass=True)))
    cp=private/'checkpoint.pt';cp.touch();key='single0_seed17';ti=dict(seed=17)
    row=dict(trial=key,representation='disabled',original_producer=ti,
        registration_sha256=run.sha(reg),fit=dict(step=4000),
        checkpoint=dict(path=str(cp),sha256=run.sha(cp)))
    (public/'fits').mkdir();rp=public/'fits'/(key+'_disabled.json');rp.write_text(json.dumps(row));before=rp.read_bytes()
    monkeypatch.setattr(run,'admission',lambda:(dict(representations=['disabled']),{},[(key,{},ti)],{},{}))
    class Verified(Exception):pass
    def beat(**event):
        if event.get('state')=='completed_producer_verified':raise Verified
    monkeypatch.setattr(run,'runtime',lambda _:(time.monotonic(),beat,lambda:None))
    def forbidden(*a,**kw):raise AssertionError('Completed model must not be refit')
    monkeypatch.setattr(run,'fit',forbidden)
    with pytest.raises(Verified):run.train(resume=True)
    assert before==rp.read_bytes()
