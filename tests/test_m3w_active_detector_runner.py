import json
from scripts import run_m3w_active_detector_forecaster as run


def test_resume_verifies_and_skips_all_completed_active_receipts(tmp_path,monkeypatch):
    public=tmp_path/'public';private=tmp_path/'private';public.mkdir();private.mkdir()
    reg=public/'registration.json';reg.write_text('{}')
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'PUBLIC',public)
    monkeypatch.setattr(run,'PRIVATE',private);monkeypatch.setattr(run,'REG',reg)
    jobs=[dict(key=f'single{f}_seed{s}',old_identity=dict(kind='single',fold=f,seed=s)) for f in range(3) for s in (17,29,43)]
    (public/'pilot.json').write_text(json.dumps(dict(engineering_pass=True)))
    for job in jobs:
        for arm in run.api.REPRESENTATIONS:
            cp=run.directory(job['key'],arm)/'checkpoint.pt';cp.parent.mkdir(parents=True)
            cp.write_bytes(b'unit-metadata-fixture-not-a-real-trained-checkpoint')
            row=dict(trial=job['key'],representation=arm,parent_trial=job['old_identity'],
                registration_sha256=run.sha(reg),fit=dict(step=4000),
                checkpoint=dict(path=str(cp.relative_to(tmp_path)),sha256=run.sha(cp)))
            run.once(public/'fits'/(job['key']+'_'+arm+'.json'),row)
            if job==jobs[0] and arm=='history':
                run.once(public/'first_full_refit.json',dict(scientific_state_exact=True,checkpoint_sha256=run.sha(cp)))
    before={p:p.read_bytes() for p in (public/'fits').glob('*.json')}
    monkeypatch.setattr(run,'admission',lambda:(dict(representations=list(run.api.REPRESENTATIONS)),None,jobs,None,{}))
    monkeypatch.setattr(run,'runtime',lambda cfg:(run.time.monotonic(),lambda **_:None,lambda:None))
    monkeypatch.setattr(run,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Completed model refit')))
    run.train(resume=True)
    frozen=json.loads((public/'training_freeze.json').read_text())
    assert frozen['count']==36 and frozen['disabled_exact_endpoints']==9
    assert all(p.read_bytes()==v for p,v in before.items())
