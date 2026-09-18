"""Verify fixed-loss intervention lineage, matched exposure and immutable resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_motion_candidate import load_config, context, immutable_json, array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    _, controls, data = context(reg)
    report = json.loads((public/'report.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    assert report['identity'] == replay['identity'] and len(set(replay['exact_replays'])) == 12
    assert report['optimizer_updates'] == 120000 and replay['new_updates'] == 0
    expected = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    pieces = {seed:[] for seed in reg['seeds']}
    before, poisoned = {}, 0
    for trial in report['trials']:
        train, _, held, outer, scale = data.configure(trial['site'])
        control = next(t for t in controls['trials'] if t['site'] == trial['site'] and t['seed'] == trial['seed'])
        fold = trial['identity']['fold']
        assert array_hash(train) == fold['training_ids_sha256'] and array_hash(held) == fold['held_ids_sha256']
        assert array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant']) == fold['normalizer_sha256']
        assert scale == fold['cost_scale']
        for name in ('checkpoint','prediction'):
            for receipt in (trial,control):
                path = ROOT/receipt[name+'_path']
                assert file_digest(path) == receipt[name+'_sha256']
                before[str(path.relative_to(ROOT))] = file_digest(path)
        state = torch.load(ROOT/trial['checkpoint_path'],map_location='cpu',weights_only=False)
        old = torch.load(ROOT/control['checkpoint_path'],map_location='cpu',weights_only=False)
        assert state['identity'] == trial['identity'] and state['step'] == old['step'] == 10000
        assert state['suppress_zero_targets'] is True and state['scale'] == old['normalizer'] == scale
        np.testing.assert_array_equal(state['train_ids'], train)
        np.testing.assert_array_equal(state['train_ids'], old['train_ids'])
        np.testing.assert_array_equal(state['draw_counts'], old['draw_counts'])
        assert state['draw_counts'].sum() == 10000 * 64
        assert torch.equal(state['sampler_rng'], old['sampler_rng'])
        assert not np.intersect1d(train,held).size and not np.intersect1d(train,outer).size
        assert all(torch.isfinite(value).all() for value in state['model'].values())
        with np.load(ROOT/trial['prediction_path'],allow_pickle=False) as a:
            np.testing.assert_array_equal(a['train_ids'],train); np.testing.assert_array_equal(a['held_ids'],held)
            prediction = a['held_prediction'].copy()
        support, radius = data.support[held-data.nmain], data.radius[held-data.nmain]
        assert np.isfinite(prediction).all() and not prediction[~support].any()
        assert np.all(np.linalg.norm(prediction.astype(float),axis=-1) <= radius[:,None]*1.00001+1e-7)
        pieces[trial['seed']].append(dict(ids=held,prediction=prediction,cost_scale=np.full(len(held),scale)))
        sample = held[:8]; original = data.dynamics_inputs(sample); target = data.target.copy()
        try:
            data.target[:] = np.nan; changed = data.dynamics_inputs(sample)
        finally: data.target[:] = target
        assert all(torch.equal(a,b) for left,right in zip(original,changed) for a,b in zip(left,right))
        poisoned += len(sample)
    for receipt in report['oof_labels']:
        prediction, scale = assemble_oof(expected,pieces[receipt['seed']])
        path = ROOT/receipt['path']; assert file_digest(path) == receipt['sha256']
        before[str(path.relative_to(ROOT))] = file_digest(path)
        labels = cost_labels(prediction,data.target[expected-data.nmain],scale)
        with np.load(path,allow_pickle=False) as a:
            for name,value in dict(ids=expected,prediction=prediction,cost_scale=scale,**labels).items():
                np.testing.assert_array_equal(a[name],value)
    for path in list((out/'trials').glob('*.json'))+[out/'identity.json',public/'input_checks.json',public/'report.json',public/'replay.json',ROOT/reg['control_report']]:
        before[str(path.relative_to(ROOT))] = file_digest(path)
    proc = subprocess.run([sys.executable,'scripts/run_m3w_source_motion_candidate.py','--registration',str(args.registration)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    event = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'motion_candidate_complete' and event['new_updates'] == 0
    assert before == {name:file_digest(ROOT/name) for name in before}
    result = dict(result_source='fresh_run_lineage_sampler_target_and_completed_resume_checks',identity=report['identity'],
        exact_replayed_models=12, matched_control_sampling_streams=12, total_updates=120000,
        oof_rows_per_seed=len(expected), seeds=reg['seeds'], loaded_target_poison_checks=poisoned,
        poison_scope='loaded_future_label_array_not_raw_annotation_acquisition',
        immutable_artifacts=len(before), artifact_hashes=before, completed_resume=event,
        new_updates_on_resume=0, outer_rows_scored=0, main_rows_scored=0, new_deployment=False)
    immutable_json(public/'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_hashes','identity')},indent=2))


if __name__ == '__main__':
    main()
