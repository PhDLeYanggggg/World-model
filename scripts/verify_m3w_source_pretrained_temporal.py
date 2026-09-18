"""Verify real temporal experiment roles, sampler identity, OOF scores and resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import (
    load_config, context, FeatureCorpus, ARMS, normalized_embedding,
    resnet18, ResNet18_Weights,
)
from scripts.run_m3w_source_crossfit import array_hash
from scripts.verify_m3w_source_motion_quality import preserve_verification
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
import numpy as np
import torch


def replay_encoder(reg, data, private):
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(ROOT/reg['weight_path'], map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    receipts = sorted((private/'features').glob('*.json'))
    chosen = [receipts[i] for i in sorted({0, len(receipts)//2, len(receipts)-1})]
    verified = []
    for path in chosen:
        receipt = json.loads(path.read_text())
        with np.load(ROOT/receipt['path'], allow_pickle=False) as a:
            rows = a['source_rows'].copy()
            expected = a['features'].copy()
            coverage = a['coverage'].copy()
        images = data.images[receipt['identity']['record']]
        rgb = np.asarray(images['rgb'][rows]).copy()
        mask = np.asarray(images['coverage'][rows]).copy()
        rgb = np.where(mask[:, None] > 0, rgb, 0).astype(np.uint8)
        with torch.inference_mode():
            actual = normalized_embedding(model(transform(torch.from_numpy(rgb))),
                                          torch.from_numpy(coverage)).numpy()
        np.testing.assert_array_equal(actual, expected)
        verified.append(dict(chunk=path.stem, images=len(rows), exact=True))
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration); data = context(reg)
    private, public = ROOT/reg['output'], ROOT/reg['reports']
    prep = json.loads((public/'preparation.json').read_text())
    report = json.loads((public/'report.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    assert report['models'] == len(set(replay['exact_replays'])) == 36
    assert report['optimizer_updates'] == 360000 and replay['new_updates'] == 0
    cached = FeatureCorpus(data, private, prep); before = {}
    encoder_replays = replay_encoder(reg, data, private)
    for item in prep['artifacts']:
        assert file_digest(ROOT/item['path']) == item['sha256']
        before[item['path']] = item['sha256']
    offset = 0; loc = cached.ids-data.nmain
    for rid in np.unique(data.record_ids[data.sid[loc]]):
        positions = np.flatnonzero(data.record_ids[data.sid[loc]] == rid)
        store = data.images[int(rid)]
        original = store['image_rows'][data.local_ids[data.sid[loc[positions]]]]
        unique, inverse = np.unique(original, return_inverse=True)
        np.testing.assert_array_equal(cached.rows[positions], inverse.reshape(-1, 8)+offset)
        np.testing.assert_array_equal(cached.coverage[offset:offset+len(unique)],
            (store['coverage'][unique].mean((1, 2))/9).astype(np.float32))
        offset += len(unique)
    assert offset == prep['unique_past_images'] == 25300
    norm = np.linalg.norm(cached.embedding, axis=-1)
    np.testing.assert_allclose(norm[cached.coverage > 0], 1, atol=1e-6, rtol=0)
    assert not np.any(cached.embedding[cached.coverage == 0])
    control = json.loads((ROOT/reg['control_report']).read_text())
    keys = set(); pieces = {(arm, seed): [] for arm in ARMS for seed in reg['seeds']}
    for trial in report['trials']:
        key = (trial['site'], trial['seed'], trial['arm']); assert key not in keys; keys.add(key)
        train, _, held, outer, scale = data.configure(trial['site'])
        fold = trial['identity']['fold']
        assert array_hash(train) == fold['training_ids_sha256'] and array_hash(held) == fold['held_ids_sha256']
        assert array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant']) == fold['normalizer_sha256']
        old = next(t for t in control['trials'] if t['site'] == key[0] and t['seed'] == key[1])
        assert file_digest(ROOT/old['checkpoint_path']) == old['checkpoint_sha256']
        parent = torch.load(ROOT/old['checkpoint_path'], map_location='cpu', weights_only=False)
        for kind in ('checkpoint', 'prediction'):
            path = ROOT/trial[kind+'_path']; assert file_digest(path) == trial[kind+'_sha256']
            before[str(path.relative_to(ROOT))] = file_digest(path)
        saved = torch.load(ROOT/trial['checkpoint_path'], map_location='cpu', weights_only=False)
        assert saved['identity'] == trial['identity'] and saved['step'] == 10000
        assert saved['scale'] == scale and not saved['suppress_zero_targets']
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        np.testing.assert_array_equal(saved['train_ids'], train)
        np.testing.assert_array_equal(saved['draw_counts'], parent['draw_counts'])
        assert torch.equal(saved['sampler_rng'], parent['sampler_rng'])
        assert saved['draw_counts'].sum() == 640000
        assert not np.intersect1d(train, held).size and not np.intersect1d(train, outer).size
        with np.load(ROOT/trial['prediction_path'], allow_pickle=False) as a:
            np.testing.assert_array_equal(a['train_ids'], train); np.testing.assert_array_equal(a['held_ids'], held)
            prediction = a['held_prediction'].copy()
        assert np.isfinite(prediction).all()
        radius, support = data.radius[held-data.nmain], data.support[held-data.nmain]
        assert not prediction[~support].any()
        assert np.all(np.linalg.norm(prediction.astype(float), axis=-1) <= radius[:, None]*1.00001+1e-7)
        pieces[(key[2], key[1])].append(dict(ids=held, prediction=prediction, cost_scale=np.full(len(held), scale)))
    assert len(keys) == 36
    for item in report['oof_labels']:
        prediction, scale = assemble_oof(cached.ids, pieces[(item['arm'], item['seed'])])
        path = ROOT/item['path']; assert file_digest(path) == item['sha256']
        labels = cost_labels(prediction, data.target[loc], scale)
        with np.load(path, allow_pickle=False) as a:
            for name, value in dict(ids=cached.ids, prediction=prediction, cost_scale=scale, **labels).items():
                np.testing.assert_array_equal(a[name], value)
        before[item['path']] = item['sha256']
    for path in [*private.joinpath('trials').glob('*.json'), private/'identity.json',
                 *[public/(name+'.json') for name in ('input_checks', 'preparation', 'report', 'replay', 'analysis')]]:
        before[str(path.relative_to(ROOT))] = file_digest(path)
    run = subprocess.run([sys.executable, 'scripts/run_m3w_source_pretrained_temporal.py',
        '--registration', str(args.registration), '--phase', 'train'], cwd=ROOT,
        capture_output=True, text=True, check=True)
    event = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'training_complete' and event['new_updates'] == 0
    assert before == {name:file_digest(ROOT/name) for name in before}
    result = dict(result_source='fresh_run_role_sampler_feature_alignment_and_completed_resume_verification',
        registration_sha256=file_digest(args.registration), exact_replayed_heads=36,
        frozen_encoder_chunk_replays=encoder_replays,
        matched_sampling_streams=36, query_rows_aligned=15430, unique_images_aligned=25300,
        total_updates=360000, oof_archives_recomputed=9, immutable_artifacts=len(before),
        artifact_hashes=before, completed_resume=event, new_updates_on_resume=0,
        sensor_asof_certified=False, main_rows_scored=0, outer_rows_scored=0, new_deployment=False)
    preserve_verification(public/'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
