import json
from scripts import run_m3w_active_bank_cost as run


def test_all_completed_matched_bank_receipts_are_verified_without_new_optimizer_work(tmp_path, monkeypatch):
    public, private = tmp_path / 'public', tmp_path / 'private'
    public.mkdir()
    private.mkdir()
    reg = public / 'registration.json'
    reg.write_text('{}')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'PUBLIC', public)
    monkeypatch.setattr(run, 'PRIVATE', private)
    monkeypatch.setattr(run, 'REG', reg)
    contexts = {f'context{i}': [] for i in range(6)}
    cfg = dict(seeds=[17, 29, 43], variants=list(run.api.VARIANTS))
    values = (None, None, dict(contexts=contexts))
    (public / 'pilot.json').write_text(json.dumps(dict(engineering_pass=True)))
    (public / 'bank_complete.json').write_text('{}')
    for context in contexts:
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                path = run.checkpoint(context, seed, variant)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'unit-fixture-not-real-training')
                row = dict(context=context, seed=seed, variant=variant, step=2000,
                    registration_sha256=run.sha(reg), legacy_scientific_state_exact=variant != 'new_neural',
                    checkpoint=dict(path=str(path.relative_to(tmp_path)), sha256=run.sha(path)))
                run.once(public / 'fits' / f'{context}_seed{seed}_{variant}.json', row)
                if context == 'context0' and seed == 17 and variant == 'new_neural':
                    refit = run.checkpoint(context, seed, variant, 'full_refit')
                    refit.parent.mkdir(parents=True)
                    refit.write_bytes(b'separate-unit-fixture')
                    run.once(public / 'first_full_refit.json', dict(scientific_state_exact=True,
                        checkpoint_sha256=run.sha(path), refit_checkpoint_sha256=run.sha(refit)))
    before = {p: p.read_bytes() for p in (public / 'fits').glob('*.json')}
    monkeypatch.setattr(run, 'admission', lambda: (cfg, values, {}, None, None))
    monkeypatch.setattr(run, 'cache_refs', lambda: {})
    monkeypatch.setattr(run, 'runtime', lambda cfg: (run.time.monotonic(), lambda **_: None, lambda: None))
    monkeypatch.setattr(run, 'prepared_context', lambda *args: (None, None, None, None))
    monkeypatch.setattr(run.api.completion.base, 'read_checkpoint', lambda _: dict(step=2000, parameters=30405, unknown_rows_with_loss=0))
    monkeypatch.setattr(run.api, 'assert_common_design', lambda *_: None)
    monkeypatch.setattr(run, 'fit', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Completed fit repeated')))
    run.train(resume=True)
    frozen = json.loads((public / 'training_freeze.json').read_text())
    assert frozen['count'] == 54 and frozen['legacy_exact_endpoints'] == 36
    assert frozen['primary_optimizer_updates'] == 108000
    assert all(p.read_bytes() == value for p, value in before.items())
