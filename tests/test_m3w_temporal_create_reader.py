import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import read_m3w_temporal_auxiliary_create as reader
from test_m3w_temporal_auxiliary import settings
from test_m3w_temporal_auxiliary_readout import fixture as policy_fixture


def fixture(root):
    cfg = dict(neural_fits=3, head_training={'steps': 8}, checkpoint_cap_bytes=2**20)
    expected = [dict(group='synthetic-locality', source='site', head_seed=17)]
    files, refs, cps = {}, [], {}
    for arm in reader.original.training.api.ARMS:
        rel = reader.HEADS+'/synthetic-locality_head17_'+arm+'/checkpoint.pt.gz'
        raw = b'synthetic-manifest-only'; path = root/rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        cp = dict(path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        doc = dict(identity=dict(group='synthetic-locality', source='site', seed=17, registration_sha256='a'*64),
                   arm=arm, step=8, validation_labels_scored=False, checkpoint=cp)
        rel = reader.HOME+'/fits/'+arm+'.json'; text = json.dumps(doc)
        path = root/rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        files[rel] = text; refs.append(dict(path=rel, sha256=reader.digest(text.encode()))); cps[cp['path']] = cp
    freeze = dict(neural_fits=3, fits=refs, validation_labels_scored=False,
                  independent_roles_read=False, deployment_changed=False, new_forecasters=False)
    raw = json.dumps(freeze); (root/reader.HOME/'training_freeze.json').write_text(raw)
    bundle = dict(freeze=raw, files=files, checkpoints=cps, job_id='123', accounting='COMPLETED|0:0',
                  manifest_sha256='b'*64, registration_sha256='a'*64, code_hashes_verified=True)
    return cfg, expected, bundle


def test_metadata_grid_matches_original_local_admission(tmp_path):
    cfg, expected, b = fixture(tmp_path)
    docs = reader.check_bundle(b, expected, cfg, 'a'*64, 'b'*64, '123')
    local = reader.original.training_documents(tmp_path, tmp_path/reader.HOME, expected, cfg)
    assert docs == local and len(docs) == 3


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'partial', 'scored', 'outside',
                                  'registration', 'hash', 'inventory', 'job', 'manifest', 'code', 'cap'])
def test_bad_training_fails_before_scientific_arrays_or_deserialization(tmp_path, monkeypatch, change):
    cfg, expected, b = fixture(tmp_path)
    monkeypatch.setattr(reader.original.training.api.torch, 'load', lambda *_a, **_kw: pytest.fail('No tensors before admission'))
    f = json.loads(b['freeze']); rel = f['fits'][0]['path']; d = json.loads(b['files'][rel])
    if change == 'missing': f['fits'].pop()
    elif change == 'duplicate': f['fits'][-1] = f['fits'][0]
    elif change == 'extra': b['files'][reader.HOME+'/fits/extra.json'] = '{}'
    elif change == 'partial': d['step'] = 4
    elif change == 'scored': d['validation_labels_scored'] = True
    elif change == 'outside': d['checkpoint']['path'] = '../foreign/checkpoint.pt.gz'
    elif change == 'registration': d['identity']['registration_sha256'] = 'f'*64
    elif change == 'hash': f['fits'][0]['sha256'] = 'f'*64
    elif change == 'inventory': b['checkpoints'] = {}
    elif change == 'job': b['accounting'] = 'RUNNING|0:0'
    elif change == 'manifest': b['manifest_sha256'] = 'c'*64
    elif change == 'code': b['code_hashes_verified'] = False
    elif change == 'cap': cfg['checkpoint_cap_bytes'] = 1
    if change in ('partial', 'scored', 'outside', 'registration'):
        b['files'][rel] = json.dumps(d); f['fits'][0]['sha256'] = reader.digest(b['files'][rel].encode())
    b['freeze'] = json.dumps(f)
    with pytest.raises(ValueError): reader.check_bundle(b, expected, cfg, 'a'*64, 'b'*64, '123')


def test_missing_freeze_does_not_contact_remote_or_load_data(tmp_path, monkeypatch):
    monkeypatch.setattr(reader.original.training, 'PUBLIC', tmp_path)
    monkeypatch.setattr(reader, 'INVENTORY', tmp_path/'inventory.json')
    monkeypatch.setattr(reader.manager, 'remote', lambda *_a, **_kw: pytest.fail('No remote contact'))
    monkeypatch.setattr(reader.original.parent.inner.old, 'load', lambda: pytest.fail('No scientific arrays'))
    assert reader.local_admission({}, {}) == (None, None)


def test_loader_restored_even_on_failure():
    prior = reader.original.training.api.core.read_checkpoint
    def replacement(_): return 'synthetic'
    with pytest.raises(RuntimeError):
        with reader.bound_loader(replacement):
            assert reader.original.training.api.core.read_checkpoint is replacement
            raise RuntimeError('test')
    assert reader.original.training.api.core.read_checkpoint is prior


def test_hash_rejected_before_deserialization(monkeypatch):
    monkeypatch.setattr(reader.original.training.api.torch, 'load', lambda *_a, **_kw: pytest.fail('Hash must reject first'))
    with pytest.raises(ValueError, match='checksum'):
        reader.decode_checkpoint(b'corrupt', {'bytes': 7, 'sha256': 'a'*64})


def test_streamed_synthetic_models_reproduce_original_predictions_and_readout(tmp_path, monkeypatch):
    api = reader.original.training.api
    _, _, y, env, moving, _, sites, rec, frames, ids = policy_fixture()
    x = np.column_stack((np.arange(len(y))/10, np.sin(np.arange(len(y)))))
    series = np.stack([np.tile([r[1]-r[0], r[2]], (12, 1)) for r in y])
    pr = api.core.preprocess(x, env, y, sites, rec, frames, training_site='a')
    args = x, env, y, series, sites, rec, frames, pr
    direct, streamed, requests = {}, {}, []
    monkeypatch.setattr(reader, 'write_payload', lambda fd, raw: requests.append(json.loads(raw)))
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz')
        state = api.fit(*args, arm=arm, settings=settings(), seed=17, identity={'synthetic': True},
                        path=path, heartbeat=lambda **kw: None)
        raw = path.read_bytes(); rel = reader.HEADS+'/synthetic-locality_head17_'+arm+'/checkpoint.pt.gz'
        ref = dict(path=rel, bytes=len(raw), sha256=reader.digest(raw))
        stream = reader.NeuralReader({'checkpoints': {rel: ref}})
        stream.p = SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: 999),
                                   stdout=io.BytesIO((json.dumps(ref)+'\n').encode()+raw))
        loaded = stream.load(reader.ROOT/rel)
        api.core.exact(state, loaded)
        assert stream.fetched_bytes == len(raw) and requests[-1] == ref
        direct[arm], _, support = api.predict(state, x, env)
        streamed[arm], _, _ = api.predict(loaded, x, env)
        assert not (reader.ROOT/rel).exists(), 'No checkpoint disk cache'
    for control in reader.original.api.CONTROLS:
        direct[control] = direct['none']; streamed[control] = streamed['none']
    extra = y, env, moving, support, sites, rec, frames, ids
    result, actions = reader.original.api.evaluate(pr, streamed, *extra)
    expected_result, expected_actions = reader.original.api.evaluate(pr, direct, *extra)
    api.core.exact(result, expected_result); api.core.exact(actions, expected_actions)
    assert reader.original.scalar.verify(pr, streamed, *extra, result, actions) > 0


def test_remote_programs_parse_and_do_no_numerical_work_on_login():
    for code in (reader.COLLECT, reader.SERVER):
        tree = ast.parse(code)
        imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not set(imports) & {'torch', 'numpy', 'scipy', 'joblib'}
        assert 'sbatch' not in code


def test_write_exact_does_not_replace_existing_values(tmp_path):
    p = tmp_path/'record.json'; reader.write_exact(p, b'{"a":1}\n')
    reader.write_exact(p, b'{"a":1}\n')
    with pytest.raises(ValueError): reader.write_exact(p, b'{"a":2}\n')
    assert p.read_bytes() == b'{"a":1}\n'
