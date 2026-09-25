import numpy as np
import pytest
import json
import hashlib
from types import SimpleNamespace
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_frozen_selection import require_selection, validate_inputs, motion_floor, infer, POLICIES


def data():
    h = np.stack([np.column_stack((np.arange(8), np.zeros(8))), np.zeros((8, 2))]).astype(float)
    g = np.zeros((2, 476), np.float32); g[:, :16] = (h-h[:, -1, None]).reshape(2, 16)
    g[:, 16:24] = np.arange(-7, 1)/12
    return dict(geometry=g, history=h, origin=h[:, -1])


@pytest.mark.parametrize('role', ['source_training', 'risk_calibration_reserved', 'confirmation_reserved'])
def test_closed_roles(role):
    with pytest.raises(PermissionError):
        require_selection(dict(recordings=[dict(source_member='x', role=role, training_access=False)]),
                          'x', purpose='frozen_model_selection_readout')


def test_selection_not_training():
    m = dict(recordings=[dict(source_member='x', role='model_selection_reserved', training_access=False)])
    assert require_selection(m, 'x', purpose='frozen_model_selection_readout')['source_member'] == 'x'
    for p in ('training', 'calibration', 'confirmation'):
        with pytest.raises(PermissionError): require_selection(m, 'x', purpose=p)


@pytest.mark.parametrize('forbidden', ['target', 'target_eval', 'valid', 'future_endpoint', 'easy_label'])
def test_reject_labels(forbidden):
    with pytest.raises(ValueError): validate_inputs(dict(data(), **{forbidden: np.zeros(2)}))


def test_label_free_guard_and_policy_shapes():
    d = data()
    def u(x, e): return np.column_stack((np.ones(len(x)), np.zeros(len(x))))
    def r(x, e): return np.column_stack((np.ones(len(x)), np.zeros(len(x))))
    floor = motion_floor(d, u, r)
    assert floor[2].tolist() == [True, False]
    heads = {(arm, t): u if t == 'utility' else r for arm in ('old_stop', 'previous_matched',
        'floor_reference', 'incumbent_reference', 'ridge_incumbent') for t in ('utility', 'risk')}
    out, x, env = infer(d, floor[0]+.1, floor, heads)
    assert x.shape == (2, 380) and env.shape == (2,)
    for p in POLICIES:
        assert out[p].dtype == bool and out[p].shape == (2,)
        if p != 'raw_neural': assert not out[p][1]
    assert out['add_only'][0] and not out['remove_only'][0]


def test_loader_separates_future_labels(tmp_path, monkeypatch):
    directory = tmp_path/'packed'; directory.mkdir()
    arrays = {}
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for key in ('geometry', 'target_eval', 'valid'):
        p = directory/(key+'.npy'); np.save(p, np.zeros((2, 3)))
        arrays[key] = dict(name=p.name, sha256=digest(p))
    identity = dict(version='test')
    (directory/'receipt.json').write_text(json.dumps(dict(identity=identity, arrays=arrays,
        input_fields=['geometry'], label_fields=['target_eval', 'valid'])))
    original = np.load
    def guarded(path, **kw):
        assert path.name == 'geometry.npy', 'inference attempted future label access'
        return original(path, **kw)
    monkeypatch.setattr(np, 'load', guarded)
    loaded = adapter.load(SimpleNamespace(PRIVATE=tmp_path, digest=digest), identity)
    assert set(loaded) == {'geometry'}
    with pytest.raises(ValueError): adapter.load(SimpleNamespace(PRIVATE=tmp_path, digest=digest), dict(version='changed'))


def test_configuration_keeps_calibration_and_confirmation_closed():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/m3w_european_selection_readout_v1.json').read_text())
    assert cfg['seeds'] == [17, 29, 43] and cfg['groups'] == 36 and cfg['recordings'] == 28
    assert cfg['bootstrap_resamples'] >= 2000 and cfg['risk_budget'] == .02
    for key in ('new_fitting', 'threshold_refit', 'calibration_access', 'confirmation_access',
                'deployment_changed', 'stage5c_executed', 'smc_enabled'): assert cfg[key] is False
    roles = json.loads((root/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json').read_text())
    selected = [r for r in roles['recordings'] if r['role'] == cfg['data_role']]
    assert len(selected) == cfg['recordings']
    assert sorted({r['locality_group'] for r in selected}) == cfg['localities']
    assert not {r['locality_group'] for r in roles['recordings'] if r['role'] != cfg['data_role']} & set(cfg['localities'])
