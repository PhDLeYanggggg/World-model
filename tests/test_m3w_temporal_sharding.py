import json
from pathlib import Path

import pytest

from src.world_model import m3w_temporal_sharding as shards


def manifest(n=72):
    return {'heads': [{'group': 'group'+str(i//3), 'identity': {
        'group': 'group'+str(i//3), 'source': 'site'+str(i//3),
        'seed': [17, 29, 43][i % 3]}, 'bytes': 10, 'sha256': 'a'*64}
        for i in range(n)]}


def test_partition_preserves_identity_order_and_complete_arms():
    m = manifest()
    parts = shards.partition(m, 4)
    assert all(len(p['heads']) == 18 for p in parts)
    assert parts[2]['heads'] == m['heads'][2::4]
    assert len({shards.identity_key(h['identity']) for p in parts for h in p['heads']}) == 72
    for p in parts:
        assert [h['identity']['seed'] for h in p['heads']].count(17) == 6
        assert len(shards.fit_keys(p)) == 54


@pytest.mark.parametrize('kind', ['duplicate', 'bad_count', 'group_mismatch'])
def test_partition_rejects_ambiguous_assignment(kind):
    m = manifest()
    if kind == 'duplicate': m['heads'][-1] = m['heads'][0]
    if kind == 'bad_count': m['heads'].pop()
    if kind == 'group_mismatch': m['heads'][0]['identity']['group'] = 'wrong'
    with pytest.raises(ValueError): shards.partition(m, 4)


def test_time_admission_does_not_relabel_single_job_pilot():
    p = dict(exact_interrupted_resume=True, matched_draws=True, local_time_feasible=False,
             estimated_full_seconds=156435.58068771847, peak_RSS_bytes=1404203008)
    cfg = dict(source_heads=72, neural_fits=216, hard_runtime_limit_seconds=43200)
    out = shards.admission(p, cfg)
    assert out['estimated_seconds_per_shard'] < 43200
    assert out['single_job_time_feasible'] is False and p['local_time_feasible'] is False
    with pytest.raises(ValueError): shards.admission({**p, 'estimated_full_seconds': 200000}, cfg)
    with pytest.raises(ValueError): shards.admission({**p, 'exact_interrupted_resume': False}, cfg)


def test_accounting_requires_all_four_successful_tasks_not_parent():
    rows = '\n'.join(f'123_{i}|COMPLETED|0:0' for i in range(4))
    assert len(shards.array_accounting(rows, '123')) == 4
    for bad in (rows.replace('123_3|COMPLETED|0:0', ''),
                rows.replace('123_1|COMPLETED|0:0', '123_1|FAILED|1:0'),
                rows+'\n123_0|COMPLETED|0:0', '123|COMPLETED|0:0'):
        with pytest.raises(ValueError): shards.array_accounting(bad, '123')


@pytest.mark.parametrize('arm', ['none', 'rowmean', 'temporal'])
def test_scoped_execution_and_resume_preserve_optimizer_exactly(tmp_path, arm):
    from scripts import run_m3w_temporal_auxiliary as run
    from test_m3w_temporal_auxiliary import data, settings
    api = run.api
    kwargs = dict(arm=arm, settings=settings(), seed=17, identity={'synthetic': True},
                  heartbeat=lambda **kw: None)
    direct = api.fit(*data(), path=tmp_path/'direct.pt.gz', **kwargs)
    old_save, old_beat, old_once = api.core.save_checkpoint, run.beat, run.once
    saved_guard_calls = []
    with shards.scoped_execution(run, tmp_path, tmp_path, 1, '123', 'b'*64,
                                 lambda: saved_guard_calls.append(True)):
        partial = api.fit(*data(), path=tmp_path/'resume.pt.gz', stop_at=3, **kwargs)
        assert partial['step'] == 3
        resumed = api.fit(*data(), path=tmp_path/'resume.pt.gz', resume=True, **kwargs)
        run.beat(state='test')
        run.once(tmp_path/'training_freeze.json', {'complete': True})
        assert not (tmp_path/'training_freeze.json').exists()
    assert saved_guard_calls
    assert (run.beat, run.once, api.core.save_checkpoint) == (old_beat, old_once, old_save)
    for k in direct:
        if k != 'seconds': api.core.exact(direct[k], resumed[k])
    assert json.loads((tmp_path/'training_shards/shard1.json').read_text())['array_job_id'] == '123'


def test_scoped_execution_restores_patches_and_rechecks_quota(tmp_path):
    from scripts import run_m3w_temporal_auxiliary as run
    old = run.api.core.save_checkpoint
    def blocked(): raise OSError('quota')
    with pytest.raises(OSError, match='quota'):
        with shards.scoped_execution(run, tmp_path, tmp_path, 0, '123', 'b'*64, blocked):
            run.api.core.save_checkpoint(tmp_path/'blocked.pt.gz', {})
    assert run.api.core.save_checkpoint is old
    assert not (tmp_path/'blocked.pt.gz').exists()


def test_join_rejects_missing_or_partial_fit(tmp_path):
    cfg = {'head_training': {'steps': 2000}, 'checkpoint_cap_bytes': 268435456, 'neural_fits': 12}
    m = manifest(4)
    with pytest.raises((ValueError, FileNotFoundError)):
        shards.verify_completed_shards(tmp_path, m, cfg, '123', 'b'*64)


def build_complete_fixture(root):
    m = manifest(4)
    cfg = dict(neural_fits=12, head_training={'steps': 2000}, checkpoint_cap_bytes=268435456)
    for i, part in enumerate(shards.partition(m)):
        refs = []
        for head in part['heads']:
            for arm in shards.ARMS:
                key = shards.fit_key(head['identity'], arm)
                cp = root/shards.PRIVATE/'heads'/key/'checkpoint.pt.gz'
                cp.parent.mkdir(parents=True, exist_ok=True); cp.write_bytes(b'hashed fixture only')
                doc = root/shards.PUBLIC/'fits'/(key+'.json'); doc.parent.mkdir(parents=True, exist_ok=True)
                doc.write_text(json.dumps(dict(identity=head['identity'], arm=arm, step=2000,
                    validation_labels_scored=False, checkpoint=dict(path=str(cp.relative_to(root)),
                    sha256=shards.sha(cp), bytes=cp.stat().st_size))))
                refs.append(dict(path=str(doc.relative_to(root)), sha256=shards.sha(doc)))
        path = root/shards.PUBLIC/'training_shards'/('shard'+str(i)+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(shard=i, array_job_id='123', execution_registration_sha256='b'*64,
            freeze=dict(fits=refs, neural_fits=3, validation_labels_scored=False,
                        independent_roles_read=False, deployment_changed=False, new_forecasters=False))))
    return m, cfg


def test_complete_join_preserves_grid_and_rejects_checksum_change(tmp_path):
    m, cfg = build_complete_fixture(tmp_path)
    refs, envelopes = shards.verify_completed_shards(tmp_path, m, cfg, '123', 'b'*64)
    assert len(refs) == 12 and len(envelopes) == 4
    cp = next((tmp_path/shards.PRIVATE).rglob('*.pt.gz')); cp.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        shards.verify_completed_shards(tmp_path, m, cfg, '123', 'b'*64)


@pytest.mark.parametrize('kind', ['partial', 'duplicate', 'identity', 'scored', 'path', 'registration'])
def test_join_checks_semantics_even_with_rehashed_receipts(tmp_path, kind):
    m, cfg = build_complete_fixture(tmp_path)
    path = tmp_path/shards.PUBLIC/'training_shards/shard0.json'; envelope = json.loads(path.read_text())
    ref = envelope['freeze']['fits'][0]; docpath = tmp_path/ref['path']; doc = json.loads(docpath.read_text())
    if kind == 'partial': doc['step'] = 100
    if kind == 'duplicate': envelope['freeze']['fits'][1] = ref.copy()
    if kind == 'identity': doc['identity']['source'] = 'different'
    if kind == 'scored': envelope['freeze']['validation_labels_scored'] = True
    if kind == 'path': doc['checkpoint']['path'] = '../outside.pt.gz'
    if kind == 'registration': envelope['execution_registration_sha256'] = 'c'*64
    docpath.write_text(json.dumps(doc)); ref['sha256'] = shards.sha(docpath); path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError): shards.verify_completed_shards(tmp_path, m, cfg, '123', 'b'*64)


def test_real_orchestration_serial_and_four_shards_match(tmp_path, monkeypatch):
    from scripts import run_m3w_temporal_auxiliary as run
    from test_m3w_temporal_auxiliary import data, settings
    m = manifest(4)
    cfg = dict(source_heads=4, neural_fits=12, head_training=settings(), hard_runtime_limit_seconds=300)
    monkeypatch.setattr(run, 'guard', lambda config: None)
    monkeypatch.setattr(run, 'beat', lambda **kw: None)
    def configure(root):
        monkeypatch.setattr(run, 'ROOT', root)
        monkeypatch.setattr(run, 'PUBLIC', root/shards.PUBLIC)
        monkeypatch.setattr(run, 'PRIVATE', root/shards.PRIVATE)
        run.PUBLIC.mkdir(parents=True); run.PRIVATE.mkdir(parents=True)
    def inputs(part): return iter((h['identity'], data()) for h in part['heads'])
    configure(tmp_path/'serial'); run.train(cfg, inputs(m), pilot=False, resume=False)
    configure(tmp_path/'parallel')
    for i, part in enumerate(shards.partition(m)):
        with shards.scoped_execution(run, run.PRIVATE, run.PUBLIC, i, '123', 'b'*64, lambda: None):
            run.train({**cfg, 'source_heads': 1, 'neural_fits': 3}, inputs(part), pilot=False, resume=False)
    for key in shards.fit_keys(m):
        path = shards.PRIVATE/'heads'/key/'checkpoint.pt.gz'
        a = run.api.core.read_checkpoint(tmp_path/'serial'/path)
        b = run.api.core.read_checkpoint(tmp_path/'parallel'/path)
        for k in a:
            if k != 'seconds': run.api.core.exact(a[k], b[k])
    assert not (tmp_path/'parallel'/shards.PUBLIC/'training_freeze.json').exists()


def test_submission_is_held_receipted_and_non_numerical():
    from scripts import manage_m3w_temporal_parallel_create as manager
    assert "'--hold'" in manager.SUBMIT and "'--dependency=afterok:'" in manager.SUBMIT
    assert manager.SUBMIT.index('once(root/') < manager.SUBMIT.index("'scontrol','release'")
    assert 'unknown_timeout_do_not_resubmit' in manager.SUBMIT
    assert 'torch' not in manager.SUBMIT and 'numpy' not in manager.SUBMIT
    assert 'independent' not in manager.STATUS or 'False' in manager.STATUS
