import json

from scripts import run_m3w_budget_score_cost as run


def test_all162completed_receipts_are_verified_without_repeating_optimizer_work(tmp_path, monkeypatch):
    public, private, parent = (tmp_path / v for v in ('public', 'private', 'parent'))
    for folder in (public, private, parent):
        folder.mkdir()
    contexts = {f'context{i}': [] for i in range(6)}
    reg = public / 'registration.json'
    reg.write_text(json.dumps(dict(contexts=contexts)))
    (parent / 'training_freeze.json').write_text('{}')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'PUBLIC', public)
    monkeypatch.setattr(run, 'PRIVATE', private)
    monkeypatch.setattr(run, 'REG', reg)
    monkeypatch.setattr(run.frozen, 'PUBLIC', parent)
    cfg = dict(seeds=[17, 29, 43], variants=list(run.api.active.VARIANTS), objectives=list(run.api.OBJECTIVES))
    values = (None, None, dict(contexts=contexts))
    (public / 'pilot.json').write_text(json.dumps(dict(engineering_pass=True)))
    for context in contexts:
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                for objective_arm in cfg['objectives']:
                    path = run.checkpoint(context, seed, variant, objective_arm)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'unit-fixture-not-a-scientific-training-run')
                    run.once(public / 'fits' / f'{context}_seed{seed}_{variant}_{objective_arm}.json',
                        dict(context=context, seed=seed, variant=variant, objective_arm=objective_arm, step=2000,
                            registration_sha256=run.sha(reg), original_scientific_state_exact=objective_arm == 'original',
                            checkpoint=dict(path=str(path.relative_to(tmp_path)), sha256=run.sha(path))))
    original = run.checkpoint('context0', 17, 'new_neural', 'upper_score')
    refit = run.checkpoint('context0', 17, 'new_neural', 'upper_score', 'full_refit')
    refit.parent.mkdir(parents=True)
    refit.write_bytes(b'separate-unit-fixture')
    run.once(public / 'first_full_refit.json', dict(scientific_state_exact=True,
        checkpoint_sha256=run.sha(original), refit_checkpoint_sha256=run.sha(refit)))
    before = {p: p.read_bytes() for p in (public / 'fits').glob('*.json')}
    monkeypatch.setattr(run, 'admission', lambda: (cfg, values, {}, None, {}))
    monkeypatch.setattr(run, 'runtime', lambda _: (run.time.monotonic(), lambda **kw: None, lambda: None))
    monkeypatch.setattr(run.frozen, 'prepared_context', lambda *args: (None, None, None, None))
    monkeypatch.setattr(run.frozen.api.completion.base, 'read_checkpoint',
        lambda _: dict(step=2000, parameters=30405, unknown_rows_with_loss=0))
    monkeypatch.setattr(run.frozen.api, 'assert_common_design', lambda *_: None)
    monkeypatch.setattr(run, 'fit', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Completed fit repeated')))
    run.train(resume=True)
    value = json.loads((public / 'training_freeze.json').read_text())
    assert value['count'] == 162 and value['exact_original_controls'] == 54
    assert value['primary_optimizer_updates'] == 324000
    assert all(p.read_bytes() == b for p, b in before.items())
    monkeypatch.setattr(run.frozen.engine, 'committed', lambda _: None)
    assert len(run.models()) == 162
