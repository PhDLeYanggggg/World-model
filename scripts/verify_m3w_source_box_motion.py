"""Verify observed-flow replay, fixed real training, OOF alignment and resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_box_motion import registration
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import array_hash
from scripts.verify_m3w_source_motion_quality import preserve_verification
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_source_box_motion import pair_features
from src.world_model.m3w_source_box_motion_head import BoxMotionCorpus, BoxMotionDynamics
from src.world_model.m3w_source_conditioned_readout import training_readout_gain
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_importance_sampling import uniform_risk_factors
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = registration(args.registration); base = load_config(Path(reg['source_registration']))
    _, data, cached, _, _, ids, groups, _ = context(base)
    private, public = ROOT/reg['output'], ROOT/reg['reports']
    extraction = json.loads((public/'extraction.json').read_text())
    assert extraction['artifact']['sha256'] == file_digest(ROOT/extraction['artifact']['path'])
    flow = BoxMotionCorpus(data, cached, ROOT/extraction['artifact']['path'])
    runtime = ROOT/'data/stage_cvpr2027_experiments/optical_flow_runtime'
    assert file_digest(runtime/'cv2/cv2.abi3.so') == extraction['identity']['opencv_binary_sha256']
    sys.path.insert(0, str(runtime)); import cv2
    cv2.setNumThreads(1)
    records = json.loads(data.manifest_path.read_text())['records']; pair_count = 0
    for item in extraction['records']:
        assert file_digest(ROOT/item['path']) == item['sha256']
        rid = item['identity']['record']; folder = data.manifest_path.parent/records[rid]['recording']
        arrays = {}
        for name in ('rgb', 'coverage', 'image_boxes', 'source_flags'):
            path = folder/(name+'.npy')
            assert file_digest(path) == records[rid]['arrays'][name+'.npy']
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        with np.load(ROOT/item['path'], allow_pickle=False) as a:
            for row, pair in enumerate(a['pairs']):
                value = pair_features(arrays['rgb'][pair], arrays['coverage'][pair], arrays['image_boxes'][pair],
                    arrays['source_flags'][pair], np.array(item['identity']['scale_xy']), cv2)
                np.testing.assert_array_equal(a['features'][row], value); pair_count += 1
    assert pair_count == extraction['unique_pairs']
    report = json.loads((public/'report.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    control = json.loads((ROOT/reg['control_report']).read_text())
    assert len(report['trials']) == len(set(replay['exact_replays'])) == 24
    pieces = {(a,s):[] for a in reg['arms'] for s in base['seeds']}
    streams, normalizers, poison, roles = {}, {}, 0, 0
    for site in base['sites']:
        train, _, held, outer, scale = data.configure(site); norm = flow.configure(train)
        normalizers[site] = array_hash(*norm)
        prob, _ = episode_weights(train, ids, groups)
        for illegal in (held[:1], outer[:1], np.array([0])):
            for accessor in (lambda q: flow.inputs(q, 'motion', training=True), data.loss_targets):
                try: accessor(illegal)
                except ValueError: roles += 1
                else: raise AssertionError('Role guard failed')
        before = flow.inputs(held[:8], 'motion'); target = data.target.copy()
        try: data.target[:] = np.nan; after = flow.inputs(held[:8], 'motion')
        finally: data.target[:] = target
        assert all(torch.equal(a,b) for x,y in zip(before,after) for a,b in zip(x,y)); poison += 8
        for trial in [t for t in report['trials'] if t['site'] == site]:
            assert trial['identity']['normalizer_sha256'] == array_hash(*norm)
            assert file_digest(ROOT/trial['checkpoint_path']) == trial['checkpoint_sha256']
            assert file_digest(ROOT/trial['prediction_path']) == trial['prediction_sha256']
            saved = torch.load(ROOT/trial['checkpoint_path'], map_location='cpu', weights_only=False)
            assert saved['identity'] == trial['identity'] and saved['step'] == 10000
            assert saved['config'] == base['training'] and saved['scale'] == scale
            np.testing.assert_array_equal(saved['train_ids'], train)
            np.testing.assert_array_equal(saved['importance_factors'], uniform_risk_factors(prob))
            np.testing.assert_array_equal(saved['probabilities'], prob)
            rng = torch.Generator().manual_seed(trial['seed']+7919); counts = np.zeros(len(train), np.int64)
            for _ in range(10000):
                batch = torch.multinomial(torch.from_numpy(prob), 64, replacement=True, generator=rng).numpy()
                np.add.at(counts, batch, 1)
            np.testing.assert_array_equal(saved['draw_counts'], counts)
            assert torch.equal(saved['sampler_rng'], rng.get_state())
            old = next(t for t in control['trials'] if t['site'] == site and t['seed'] == trial['seed'] and t['arm'] == 'geometry')
            assert file_digest(ROOT/old['checkpoint_path']) == old['checkpoint_sha256']
            old_cp = torch.load(ROOT/old['checkpoint_path'], map_location='cpu', weights_only=False)
            np.testing.assert_array_equal(counts, old_cp['draw_counts'])
            key = (site, trial['seed'])
            if key in streams: np.testing.assert_array_equal(counts, streams[key])
            streams[key] = counts
            gain = training_readout_gain(data.radius[train-data.nmain], scale)
            model = BoxMotionDynamics(gain); model.load_state_dict(saved['model'])
            assert sum(v.numel() for v in model.parameters()) == 63960
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            with np.load(ROOT/trial['prediction_path'], allow_pickle=False) as a:
                for name, value in zip(('motion_mean','motion_std','motion_constant'), norm):
                    np.testing.assert_array_equal(a[name], value)
                for subset, admitted in [('train',train), ('held',held)]:
                    np.testing.assert_array_equal(a[subset+'_ids'], admitted)
                    pred = a[subset+'_prediction']; loc = admitted-data.nmain
                    assert np.isfinite(pred).all() and not pred[~data.support[loc]].any()
                    assert np.all(np.linalg.norm(pred.astype(float), axis=-1) <= data.radius[loc,None]*1.00001+1e-7)
                pieces[trial['arm'],trial['seed']].append(dict(ids=held, prediction=a['held_prediction'].copy(),
                                                              cost_scale=np.full(len(held), scale)))
    for item in report['oof_labels']:
        pred, scale = assemble_oof(ids, pieces[item['arm'],item['seed']])
        assert file_digest(ROOT/item['path']) == item['sha256']
        with np.load(ROOT/item['path'], allow_pickle=False) as a:
            for name, value in dict(ids=ids, prediction=pred, cost_scale=scale,
                                   **cost_labels(pred, data.target[ids-data.nmain], scale)).items():
                np.testing.assert_array_equal(a[name], value)
    paths = [p for p in private.rglob('*') if p.suffix in ('.json','.npz','.pt') and 'heartbeat' not in p.name]
    paths += [public/x for x in ('report.json','analysis.json','replay.json','input_checks.json','extraction.json')]
    hashes = {str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    child = subprocess.run([sys.executable, 'scripts/run_m3w_source_box_motion.py', '--registration', str(args.registration)],
                           cwd=ROOT, check=True, text=True, capture_output=True)
    event = [json.loads(line) for line in child.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'training_complete' and event['new_updates'] == 0
    assert hashes == {p:file_digest(ROOT/p) for p in hashes}
    result = dict(result_source='fresh_run_verification', registration_sha256=file_digest(args.registration),
        exact_motion_pair_replays=pair_count, exact_model_replays=24, regenerated_streams=24,
        control_stream_matches=24, paired_streams=len(streams), oof_archives_recomputed=6,
        normalizer_hashes=normalizers, future_label_poison_queries=poison, prohibited_role_checks=roles,
        unsupported_rows_retained=int((~data.support[ids-data.nmain]).sum()),
        immutable_artifacts=len(hashes), artifact_hashes=hashes, completed_resume=event,
        new_updates_on_completed_resume=0, actual_training_updates=240000,
        main_outer_rows_scored=0, new_deployment=False, stage5c_executed=False, smc_enabled=False)
    preserve_verification(public/'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifact_hashes'}, indent=2))


if __name__ == '__main__': main()
