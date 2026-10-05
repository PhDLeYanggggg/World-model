import copy
import json
from pathlib import Path
import numpy as np
import pytest

from scripts import run_m3w_temporal_auxiliary_readout as runner
from scripts import verify_m3w_temporal_auxiliary_readout as scalar
from src.world_model import m3w_temporal_auxiliary as trained
from test_m3w_temporal_auxiliary_readout import fixture, summary_fixture


def manifest(tmp_path):
    home = tmp_path/'outputs'/'experiment'
    private = tmp_path/'data/stage_cvpr2027_experiments'/runner.training.NAME/'heads'
    cfg = dict(neural_fits=3, head_training=dict(steps=8))
    expected = [dict(group='g', source='s', head_seed=17)]
    refs = []
    for arm in trained.ARMS:
        cp = private/arm/'checkpoint.pt.gz'; cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(b'synthetic-manifest-fixture')
        doc = dict(identity=dict(group='g', source='s', seed=17), arm=arm, step=8,
            validation_labels_scored=False, checkpoint=dict(path=str(cp.relative_to(tmp_path)),
            bytes=cp.stat().st_size, sha256=runner.sha(cp)))
        path = home/'fits'/(arm+'.json'); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        refs.append(dict(path=str(path.relative_to(tmp_path)), sha256=runner.sha(path)))
    freeze = dict(neural_fits=3, fits=refs, validation_labels_scored=False, independent_roles_read=False, deployment_changed=False)
    (home/'training_freeze.json').write_text(json.dumps(freeze))
    return home, cfg, expected, freeze


def test_complete_manifest_verified_without_loading_any_model(tmp_path, monkeypatch):
    home, cfg, expected, _ = manifest(tmp_path)
    monkeypatch.setattr(trained.core, 'read_checkpoint', lambda *_: pytest.fail('No model read at admission'))
    result = runner.training_documents(tmp_path, home, expected, cfg)
    assert len(result) == 3


@pytest.mark.parametrize('change', ['checkpoint', 'missing', 'duplicate', 'partial', 'already_scored', 'outside'])
def test_bad_or_incomplete_training_blocks_before_validation(tmp_path, change):
    home, cfg, expected, freeze = manifest(tmp_path)
    path = tmp_path/freeze['fits'][0]['path']; doc = json.loads(path.read_text())
    if change == 'checkpoint': (tmp_path/doc['checkpoint']['path']).write_bytes(b'corrupt')
    if change == 'missing': freeze['fits'].pop()
    if change == 'duplicate': freeze['fits'][-1] = freeze['fits'][0]
    if change == 'partial': doc['step'] = 4
    if change == 'already_scored': doc['validation_labels_scored'] = True
    if change == 'outside': doc['checkpoint']['path'] = str(home/'unowned.pt.gz')
    if change in ('partial', 'already_scored', 'outside'):
        path.write_text(json.dumps(doc)); freeze['fits'][0]['sha256'] = runner.sha(path)
    (home/'training_freeze.json').write_text(json.dumps(freeze))
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.training_documents(tmp_path, home, expected, cfg)


def test_missing_training_does_not_probe_remote_or_load_science(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.training, 'PUBLIC', tmp_path)
    monkeypatch.setattr(runner.training, 'storage', lambda *_: {'allowed':False})
    monkeypatch.setattr(runner.parent.inner.old, 'load', lambda: pytest.fail('No science before training'))
    monkeypatch.setattr(runner.controls, 'FrozenReader', lambda *_: pytest.fail('No remote probe needed'))
    result = runner.admission({}, {'expected_source_heads': []})
    assert not result['readout_allowed'] and not result['new_validation_predictions_read']


def test_neural_optimizer_to_readout_on_synthetic_only(tmp_path):
    pr_unused, pred, y, env, moving, support, sites, rec, frames, ids = fixture()
    x = np.column_stack((ids/10, np.arange(len(ids))/10))
    series = np.stack([np.tile([r[1]-r[0], r[2]], (12, 1)) for r in y])
    pr = trained.core.preprocess(x, env, y, sites, rec, frames, training_site='a')
    settings = dict(width=4, steps=4, learning_rate=.001, query_batch_size=2, gradient_clip=5.,
        checkpoint_every=2, heartbeat_every=2, auxiliary_weight=.1)
    states = []
    for arm in trained.ARMS:
        state = trained.fit(x, env, y, series, sites, rec, frames, pr, arm=arm, settings=settings,
            identity={'fixture':'not_real_validation'}, seed=17, path=tmp_path/(arm+'.pt.gz'), heartbeat=lambda **kw:None)
        pred[arm], _, _ = trained.predict(state, x, env); states.append(state)
    trained.assert_matched(states)
    args = pr, pred, y, env, moving, support, sites, rec, frames, ids
    result, actions = runner.api.evaluate(*args)
    assert scalar.verify(*args, result, actions) > 500


def test_summary_reverified_with_scalar_locality_resampling():
    rows, expected, cfg = summary_fixture()
    report = runner.api.summarize(rows, expected, cfg)
    assert scalar.verify_summary(rows, cfg, report) > 100
    report['advance_to_transfer_design'] = True
    with pytest.raises(AssertionError): scalar.verify_summary(rows, cfg, report)
