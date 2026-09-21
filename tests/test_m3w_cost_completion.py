import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.world_model.m3w_oof_identity import oof_feature_identity
from test_m3w_experiment_contract import artifact, fixture_contract

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'code_sha256': ROOT / 'src/world_model/m3w_supervised_intervention.py',
    'oof_identity_source_sha256': ROOT / 'src/world_model/m3w_oof_identity.py',
    'script_sha256': ROOT / 'scripts/train_m3w_oof_cost_head.py',
}


def completed_fixture(tmp_path):
    p = fixture_contract(tmp_path)
    a = artifact(tmp_path, p, 'a_model', fit=('b',))
    b = artifact(tmp_path, p, 'b_model', fit=('a',))
    c = ExperimentContract(p, tmp_path, [a, b])
    output = tmp_path / 'head'
    output.mkdir()
    run = dict(protocol_sha256=c.digest, fold_models={'0': 'a_model', '1': 'b_model'},
               baseline='constant_velocity_causal_fd', alpha=1., device='cpu', batch_size=128,
               predictor_sha256={x['id']: x['sha256'] for x in (a, b)},
               **{k: file_digest(v) for k, v in SOURCES.items()})
    (output / 'run_identity.json').write_text(json.dumps(run))
    groups = []
    for fold, (recording, model) in enumerate((('a', a), ('b', b))):
        group = dict(predictor_id=model['id'], predictor_sha256=model['sha256'],
            protocol_sha256=c.digest, metric='ade', baseline_name=run['baseline'],
            feature_source='past_and_frozen_rollouts_only', target_source='held_fold_realized_relative_costs')
        x = np.array([[1., fold], [2., fold]])
        y = np.array([[1., 0.], [0., 2.]])
        rows = [dict(recording_id=recording, physical_scene=recording, agent_id=1,
                     frame_id=t, horizon_raw=120, protocol='observation_steps', data_role='fit') for t in (70, 80)]
        np.savez(output / f'fold_{fold}.npz', features=x, targets=y, identities_json=json.dumps(rows))
        receipt = dict(run_identity=run, cache_sha256=file_digest(output / f'fold_{fold}.npz'), group_metadata=group)
        (output / f'fold_{fold}.json').write_text(json.dumps(receipt))
        groups.append(dict(group, features=x, targets=y, identities=rows))
    x = np.concatenate([g['features'] for g in groups])
    np.savez(output / 'linear_cost_head.npz', mean=x.mean(0), scale=np.maximum(x.std(0), 1e-6),
             coef=np.zeros((2, 2)), intercept=np.zeros(2))
    report = dict(alpha=1., fit_recordings=['a', 'b'], parents=['a_model', 'b_model'],
        protocol_sha256=c.digest, normalization_source='fit_OOF_rows_only', baseline_name=run['baseline'],
        metric='ade', oof_feature_identity=oof_feature_identity(groups), training_rows=4,
        oof_total_folds=2, feature_dimension=2, calibrated_policy=False, test_evaluated=False,
        checkpoint_sha256=file_digest(output / 'linear_cost_head.npz'), code_sha256=run['code_sha256'])
    (output / 'fit_report.json').write_text(json.dumps(report))
    head = dict(id='head', kind='risk_head', path='head/linear_cost_head.npz', sha256=report['checkpoint_sha256'],
                protocol_sha256=c.digest, lineage_complete=True, fit_recordings=['a', 'b'],
                selection_recordings=[], calibration_recordings=[], parents=['a_model', 'b_model'])
    (output / 'artifact.json').write_text(json.dumps(head))
    return c, output, run


def verify(c, output, run, **kwargs):
    from src.evaluation.m3w_cost_completion import verify_completed_ridge
    return verify_completed_ridge(c, output, source_paths=SOURCES, expected_run_identity=run, **kwargs)


def test_completed_cache_report_and_normalization_revalidate(tmp_path):
    c, output, run = completed_fixture(tmp_path)
    result = verify(c, output, run)
    assert result['training_rows'] == 4
    assert result['normalization_exact']
    assert result['target_source_status'] == 'receipt_verified_not_independently_anchored'
    replay = verify(c, output, run, expected_files=result['file_bindings'])
    assert replay['target_source_status'] == 'externally_bound_bytes_verified'


@pytest.mark.parametrize('change', ['cache_bytes', 'cache_missing', 'receipt', 'report_rows',
                                   'report_feature_identity', 'report_parents', 'artifact', 'run_settings'])
def test_completed_resume_rejects_changed_dependencies(tmp_path, change):
    c, out, run = completed_fixture(tmp_path)
    if change == 'cache_bytes':
        with (out / 'fold_0.npz').open('ab') as f:
            f.write(b'changed')
    elif change == 'cache_missing':
        (out / 'fold_0.npz').unlink()
    else:
        path = out / {'receipt': 'fold_0.json', 'artifact': 'artifact.json',
                      'run_settings': 'run_identity.json'}.get(change, 'fit_report.json')
        value = json.loads(path.read_text())
        if change == 'receipt': value['group_metadata']['predictor_id'] = 'b_model'
        elif change == 'artifact': value['selection_recordings'] = ['d']
        elif change == 'run_settings': value['alpha'] = 2.
        elif change == 'report_rows': value['training_rows'] = 5
        elif change == 'report_feature_identity': value['oof_feature_identity']['sha256'] = 'changed'
        else: value['parents'] = []
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, OSError)):
        verify(c, out, run)


def test_rehashed_targets_require_an_independent_completion_binding(tmp_path):
    c, out, run = completed_fixture(tmp_path)
    anchor = verify(c, out, run)['file_bindings']
    cache = out / 'fold_0.npz'
    with np.load(cache, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    arrays['targets'] = arrays['targets'] * 2
    np.savez(cache, **arrays)
    receipt = json.loads((out / 'fold_0.json').read_text())
    receipt['cache_sha256'] = file_digest(cache)
    (out / 'fold_0.json').write_text(json.dumps(receipt))
    # Feature identity deliberately excludes targets: do not call it a target anchor.
    assert verify(c, out, run)['target_source_status'] == 'receipt_verified_not_independently_anchored'
    with pytest.raises(ValueError, match='binding'):
        verify(c, out, run, expected_files=anchor)


@pytest.mark.parametrize('change', ['normalizer', 'nonfinite'])
def test_nonfinite_checkpoint_and_incorrect_normalizer_rejected(tmp_path, change):
    c, out, run = completed_fixture(tmp_path)
    checkpoint = out / 'linear_cost_head.npz'
    with np.load(checkpoint, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    if change == 'normalizer': arrays['mean'] += 1
    else: arrays['coef'][0, 0] = np.nan
    np.savez(checkpoint, **arrays)
    for filename, key in [('fit_report.json', 'checkpoint_sha256'), ('artifact.json', 'sha256')]:
        path = out / filename
        value = json.loads(path.read_text()); value[key] = file_digest(checkpoint)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='normalization|Malformed'):
        verify(c, out, run)


def test_real_synthetic_training_legacy_bug_and_v2_resume(tmp_path):
    import torch
    from test_m3w_supervised_intervention import datasets, register, training_config, train_forecaster
    torch.set_num_threads(2)
    c, a, b = datasets(tmp_path)
    arch, settings = training_config()
    manifests = []
    for name, dataset in [('model_a', a), ('model_b', b)]:
        r = train_forecaster(dataset, architecture=arch, settings=settings, output_dir=tmp_path / name)
        artifact_data = register(c, r).artifacts['fold_a']
        artifact_data['id'] = name
        path = tmp_path / (name+'.json'); path.write_text(json.dumps(artifact_data))
        manifests.append(str(path))
    protocol, mapping = tmp_path / 'protocol.json', tmp_path / 'mapping.json'
    protocol.write_text(json.dumps(c.protocol)); mapping.write_text(json.dumps({'0': 'model_b', '1': 'model_a'}))
    common = ['--protocol', str(protocol), '--workspace-root', str(tmp_path), '--artifacts', *manifests,
              '--fold-models', str(mapping), '--baseline', 'constant_velocity_causal_fd', '--alpha', '1',
              '--threads', '2', '--output-dir', str(tmp_path / 'cost')]
    old = [sys.executable, str(ROOT / 'scripts/train_m3w_oof_cost_head.py'), *common]
    new = [sys.executable, str(ROOT / 'scripts/train_m3w_oof_cost_head_v2.py'), *common]
    subprocess.run(new + ['--stop-after-folds', '1'], check=True, capture_output=True, timeout=60)
    assert not (tmp_path / 'cost/completion_v2.json').exists()
    subprocess.run(new + ['--resume'], check=True, capture_output=True, timeout=60)
    report = tmp_path / 'cost/fit_report.json'
    before = {p.name: file_digest(p) for p in report.parent.iterdir() if p.is_file()}
    good = subprocess.run(new + ['--resume'], check=True, capture_output=True, text=True, timeout=60)
    assert 'complete_all_dependencies_verified' in good.stdout
    assert '"torch_imported": false' in good.stdout
    assert before == {p.name: file_digest(p) for p in report.parent.iterdir() if p.is_file()}
    cache = tmp_path / 'cost/fold_0.npz'
    original = cache.read_bytes()
    with np.load(cache, allow_pickle=False) as a:
        changed = {k: a[k] for k in a.files}
    changed['targets'][0, 0] += .25
    np.savez(cache, **changed)
    legacy = subprocess.run(old + ['--resume'], capture_output=True, text=True, timeout=60)
    assert legacy.returncode == 0 and 'already_complete_cached_verified' in legacy.stdout
    fixed = subprocess.run(new + ['--resume'], capture_output=True, text=True, timeout=60)
    assert fixed.returncode != 0 and 'binding' in fixed.stderr
    cache.write_bytes(original)
    (tmp_path / 'cost/completion_v2.json').unlink()
    unanchored = subprocess.run(new + ['--resume'], capture_output=True, text=True, timeout=60)
    assert unanchored.returncode != 0 and 'trusted completion' in unanchored.stderr
