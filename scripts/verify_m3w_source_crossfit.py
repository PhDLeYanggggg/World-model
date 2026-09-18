"""Verify cold-start fold lineage, OOF targets and completed-resume immutability."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_crossfit import load_config, load_data, CrossfitCorpus
from scripts.run_m3w_source_continuation import immutable_json
from scripts.run_m3w_source_start_probe import array_hash
import numpy as np
import torch
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    report = json.loads((public/'report.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    assert replay['identity'] == report['identity']
    assert len(set(replay['exact_replays'])) == 12 and replay['new_training_updates'] == 0
    data = CrossfitCorpus(load_data(Path(reg['data_registration'])))
    expected = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    pieces = {seed:[] for seed in reg['seeds']}
    before = {}
    poisoned_input_checks = 0
    total_updates = 0
    for trial in report['trials']:
        site, seed = trial['site'], trial['seed']
        train, weights, held, outer, scale = data.configure(site)
        identity = trial['identity']
        assert identity['fold']['training_ids_sha256'] == array_hash(train)
        assert identity['fold']['held_ids_sha256'] == array_hash(held)
        assert identity['fold']['normalizer_sha256'] == array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant'])
        assert identity['fold']['cost_scale'] == scale
        for name in ('parent', 'checkpoint', 'prediction'):
            path = ROOT/trial[name+'_path']
            assert file_digest(path) == trial[name+'_sha256']
            before[str(path.relative_to(ROOT))] = file_digest(path)
        parent = torch.load(ROOT/trial['parent_path'], map_location='cpu', weights_only=False)
        final = torch.load(ROOT/trial['checkpoint_path'], map_location='cpu', weights_only=False)
        for state in (parent, final):
            assert state['identity'] == identity
            np.testing.assert_array_equal(state['train_ids'], train)
            assert not np.intersect1d(state['train_ids'], held).size
            assert not np.intersect1d(state['train_ids'], outer).size
            assert state['normalizer'] == scale
            assert state['draw_counts'].sum() == state['step'] * reg['initial_training']['batch_size']
            assert all(torch.isfinite(v).all() for v in state['model'].values())
        assert parent['step'] == 2000 and final['step'] == 10000
        assert final['schedule'] == 'cosine' and final['arm'] == 'mask_only'
        total_updates += final['step']
        with np.load(ROOT/trial['prediction_path'], allow_pickle=False) as a:
            np.testing.assert_array_equal(a['train_ids'], train)
            np.testing.assert_array_equal(a['held_ids'], held)
            pred = a['held_prediction'].copy()
        support = data.support[held-data.nmain]
        assert np.isfinite(pred).all() and not pred[~support].any()
        bound = data.radius[held-data.nmain]
        assert np.all(np.linalg.norm(pred.astype(float), axis=-1) <= bound[:,None]*1.00001 + 1e-7)
        pieces[seed].append(dict(ids=held, prediction=pred, cost_scale=np.full(len(held), scale)))
        sample = held[:8]
        original = data.dynamics_inputs(sample)
        target = data.target.copy()
        try:
            data.target[:] = np.nan
            changed = data.dynamics_inputs(sample)
        finally:
            data.target[:] = target
        for left, right in zip(original, changed):
            for a, b in zip(left, right):
                assert torch.equal(a, b)
        poisoned_input_checks += len(sample)
    assert total_updates == 120000
    for receipt in report['oof_labels']:
        pred, scale = assemble_oof(expected, pieces[receipt['seed']])
        labels = cost_labels(pred, data.target[expected-data.nmain], scale)
        path = ROOT/receipt['path']
        assert file_digest(path) == receipt['sha256']
        before[str(path.relative_to(ROOT))] = file_digest(path)
        with np.load(path, allow_pickle=False) as a:
            for name, value in dict(ids=expected, prediction=pred, cost_scale=scale, **labels).items():
                np.testing.assert_array_equal(a[name], value)
    for path in list((private/'trials').glob('*.json')) + [private/'identity.json', public/'input_checks.json', public/'report.json', public/'replay.json']:
        before[str(path.relative_to(ROOT))] = file_digest(path)
    process = subprocess.run([sys.executable, 'scripts/run_m3w_source_crossfit.py', '--registration', str(args.registration)],
        cwd=ROOT, capture_output=True, text=True, check=True)
    event = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'crossfit_complete' and event['new_training_updates'] == 0
    assert before == {name:file_digest(ROOT/name) for name in before}
    evidence = dict(result_source='fresh_run_lineage_target_and_readonly_resume_verification',
        identity=report['identity'], exact_train_and_held_replayed_models=12,
        cold_start_fold_lineages_verified=12, total_updates=total_updates,
        oof_rows_per_seed=len(expected), seeds=reg['seeds'],
        loaded_future_target_poison_input_checks=poisoned_input_checks,
        counterfactual_scope='loaded_label_array_only_not_raw_annotation_acquisition',
        immutable_artifacts=len(before), artifact_hashes=before, completed_resume=event,
        new_updates_on_resume=0, outer_rows_scored=0, main_rows_scored=0,
        independent_confirmation=False, new_deployment=False)
    immutable_json(public/'verification.json', evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('artifact_hashes','identity')}, indent=2))


if __name__ == '__main__':
    main()
