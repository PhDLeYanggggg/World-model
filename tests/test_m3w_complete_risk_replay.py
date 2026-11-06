from scripts import replay_m3w_complete_risk_donors as replay


def test_replay_twice_preserves_prior_receipts_and_never_fits(tmp_path,monkeypatch):
    run=replay.run
    private=tmp_path/'private';public=tmp_path/'public';public.mkdir()
    (public/'result.json').write_text('{}')
    monkeypatch.setattr(run,'PRIVATE',private);monkeypatch.setattr(run,'PUBLIC',public)
    monkeypatch.setattr(run,'register',lambda: ({},{}));monkeypatch.setattr(run,'committed',lambda:None)
    monkeypatch.setattr(run,'runtime',lambda *_:(0,None,lambda **_:None))
    calls=[]
    def infer(*_args,**kwargs):
        assert kwargs['replay']
        try:run.seeds.method.fit()
        except RuntimeError as e:assert 'forbidden' in str(e)
        else:raise AssertionError('Fitting unexpectedly permitted')
        calls.append('infer')
    monkeypatch.setattr(run,'infer',infer)
    monkeypatch.setattr(run,'evaluate',lambda *_args,**kwargs:calls.append('evaluate'))
    fit=run.seeds.method.fit
    replay.main();replay.main()
    assert calls==['infer','evaluate','infer','evaluate']
    assert len(list((private/'replays').glob('*.json')))==2
    assert (public/'result.json').read_text()=='{}'
    assert run.seeds.method.fit is fit
