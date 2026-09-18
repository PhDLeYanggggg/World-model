"""Fixed, source-only comparison of geometry/current/temporal pretrained appearance."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_crossfit import (
    load_config as load_parent, load_data, CrossfitCorpus, metrics, save_arrays,
    immutable_json, array_hash,
)
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate
from src.world_model.m3w_source_cost_dynamics import forecast
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels, validate_query
from src.world_model.m3w_source_pretrained_temporal import (
    ARMS, TemporalSourceDynamics, normalized_embedding, embedding_lookup,
)
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights


def load_config(path):
    r = json.loads(path.read_text())
    if (str(path) != r['registration_path'] or r['role'] != 'training_only_frozen_visual_candidate'
            or r['arms'] != list(ARMS) or r['sites'] != ['coupa', 'deathCircle', 'gates', 'hyang']
            or r['seeds'] != [17, 29, 43] or r['models'] != 36 or r['updates'] != 360000
            or r['training']['updates'] != 10000 or r['selection'] or r['new_deployment']):
        raise ValueError('Fixed source-only temporal appearance protocol required')
    for name, digest in r['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Frozen dependency changed: '+name)
    if file_digest(ROOT/r['weight_path']) != r['weight_sha256']:
        raise ValueError('Official frozen visual weights changed')
    return r


def context(reg):
    parent = load_parent(Path(reg['parent_registration']))
    return CrossfitCorpus(load_data(Path(parent['data_registration'])))


def prepare(reg, data, private, public, beat, stop_chunks=None):
    identity = dict(registration_sha256=file_digest(ROOT/reg['registration_path']),
        source=data.identity, assignment=data.assignment_hash, weight_sha256=reg['weight_sha256'],
        torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__)
    completed = public/'preparation.json'
    if completed.exists():
        receipt = json.loads(completed.read_text())
        assert receipt['identity'] == identity
        for item in receipt['artifacts']:
            assert file_digest(ROOT/item['path']) == item['sha256']
        beat(state='cache_verified', rows=receipt['rows'], new_extractions=0)
        return receipt
    loc = np.flatnonzero(data.source_sites != 'bookstore'); ids = loc+data.nmain
    if len(ids) != 15430: raise ValueError('Frozen cohort changed')
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(ROOT/reg['weight_path'], map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity(); model.eval(); model.requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    mapping = np.full((len(ids), 8), -1, np.int64)
    embeddings, coverages, receipts = [], [], []
    offset = 0; new_chunks = 0
    for rid in np.unique(data.record_ids[data.sid[loc]]):
        positions = np.flatnonzero(data.record_ids[data.sid[loc]] == rid)
        store = data.images[int(rid)]
        original = store['image_rows'][data.local_ids[data.sid[loc[positions]]]]
        unique, inverse = np.unique(original, return_inverse=True)
        mapping[positions] = inverse.reshape(-1, 8)+offset
        for start in range(0, len(unique), reg['extraction_batch']):
            source_rows = unique[start:start+reg['extraction_batch']]
            name = f'record{rid}_chunk{start:06d}'
            path = private/'features'/f'{name}.npz'; rp = path.with_suffix('.json')
            chunk_id = dict(identity, record=int(rid), source_rows_sha256=array_hash(source_rows))
            if rp.exists():
                receipt = json.loads(rp.read_text())
                assert receipt['identity'] == chunk_id and file_digest(path) == receipt['sha256']
                with np.load(path, allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['source_rows'], source_rows)
                    features, cov = a['features'].copy(), a['coverage'].copy()
            else:
                begin = time.monotonic()
                rgb = np.asarray(store['rgb'][source_rows]).copy()
                coverage = np.asarray(store['coverage'][source_rows]).copy()
                if np.any(coverage > 9): raise ValueError('Invalid source coverage')
                rgb = np.where(coverage[:, None] > 0, rgb, 0).astype(np.uint8)
                cov = (coverage.mean((1, 2))/9).astype(np.float32)
                with torch.inference_mode():
                    features = normalized_embedding(model(transform(torch.from_numpy(rgb))),
                                                    torch.from_numpy(cov)).numpy()
                save_arrays(path, dict(source_rows=source_rows, features=features, coverage=cov))
                receipt = dict(identity=chunk_id, path=str(path.relative_to(ROOT)), sha256=file_digest(path),
                               images=len(source_rows), seconds=time.monotonic()-begin)
                immutable_json(rp, receipt); new_chunks += 1
                beat(state='feature_chunk_complete', chunk=name, images=len(source_rows), seconds=receipt['seconds'])
            assert features.shape == (len(source_rows), 512) and np.isfinite(features).all()
            embeddings.append(features); coverages.append(cov); receipts.append(receipt)
            if stop_chunks is not None and new_chunks >= stop_chunks:
                beat(state='extraction_pilot_complete_no_forecasts', new_chunks=new_chunks)
                return None
        offset += len(unique)
    if np.any(mapping < 0): raise ValueError('Every past token must align')
    path = private/'feature_store.npz'
    save_arrays(path, dict(ids=ids, query_rows=mapping, embeddings=np.concatenate(embeddings),
                           coverage=np.concatenate(coverages)))
    result = dict(identity=identity, rows=len(ids), unique_past_images=offset, records=len(set(data.source_records[loc])),
        artifacts=[dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path))] +
                  [dict(path=r['path'], sha256=r['sha256']) for r in receipts],
        extraction_seconds=sum(r['seconds'] for r in receipts), chunks=len(receipts),
        transform=str(transform), encoder_eval_mode=True, encoder_updated=False,
        no_target_feature_access=True, result_source='fresh_run_frozen_official_image_encoder',
        outer_rows_scored=0, main_rows_scored=0, sensor_asof_certified=False, new_deployment=False)
    immutable_json(completed, result)
    beat(state='cache_complete', rows=len(ids), images=offset, new_chunks=new_chunks)
    return result


class FeatureCorpus:
    def __init__(self, data, private, receipt):
        self.data = data
        path = private/'feature_store.npz'
        assert file_digest(path) == receipt['artifacts'][0]['sha256']
        with np.load(path, allow_pickle=False) as a:
            self.ids = a['ids'].copy(); self.rows = a['query_rows'].copy()
            self.embedding = a['embeddings'].copy(); self.coverage = a['coverage'].copy()
        np.testing.assert_array_equal(self.ids, np.flatnonzero(data.source_sites != 'bookstore')+data.nmain)

    def inputs(self, ids, *, training=False):
        d = self.data
        ids = validate_query(ids, d.nmain, len(d.y), d.allowed if training else d.predict_allowed)
        rows = embedding_lookup(ids, self.ids, self.rows); loc = ids-d.nmain
        features = (torch.from_numpy(d.z[ids].copy()), torch.from_numpy(self.embedding[rows].copy()),
                    torch.from_numpy(self.coverage[rows].copy()))
        frame = (torch.from_numpy(d.radius[loc].copy()), torch.from_numpy(d.rotation[loc].copy()),
                 torch.from_numpy(d.support[loc].copy()))
        return features, frame


def train(reg, data, cached, private, public, beat, trial_name=None, stop_at=None, replay=False):
    reference = json.loads((ROOT/reg['control_report']).read_text())
    identity = dict(registration_sha256=file_digest(ROOT/reg['registration_path']), data=data.identity,
        assignment=data.assignment_hash, feature_store_sha256=file_digest(private/'feature_store.npz'),
        torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
        torch_threads=4, num_workers=0, weight_sha256=reg['weight_sha256'])
    immutable_json(private/'identity.json', identity)
    names = {f'{site}_{arm}_seed{seed}' for site in reg['sites'] for arm in ARMS for seed in reg['seeds']}
    if trial_name is not None and trial_name not in names: raise ValueError('Unregistered trial')
    checks = []; trials = []; replayed = []; updates = 0
    for site in reg['sites']:
        train_ids, weights, held, outer, scale = data.configure(site)
        original = next(t for t in reference['trials'] if t['site'] == site)
        fold = original['identity']['fold']; hard = original['identity']['training_hard_cut']
        assert array_hash(train_ids) == fold['training_ids_sha256']
        assert array_hash(held) == fold['held_ids_sha256'] and scale == fold['cost_scale']
        assert array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant']) == fold['normalizer_sha256']
        for forbidden in [np.array([0]), outer[:1]]:
            try: cached.inputs(forbidden)
            except ValueError: pass
            else: raise AssertionError('Outer/main inference permitted')
        try: cached.inputs(held[:1], training=True)
        except ValueError: pass
        else: raise AssertionError('Held training features permitted')
        before = cached.inputs(held[:8]); labels = data.target.copy()
        try:
            data.target[:] = np.nan
            after = cached.inputs(held[:8])
        finally: data.target[:] = labels
        assert all(torch.equal(a, b) for left, right in zip(before, after) for a, b in zip(left, right))
        checks.append(dict(site=site, training_rows=len(train_ids), held_rows=len(held), fold=fold,
                           loaded_target_poison_checks=8))
        for seed in reg['seeds']:
            control = next(t for t in reference['trials'] if t['site'] == site and t['seed'] == seed)
            assert file_digest(ROOT/control['checkpoint_path']) == control['checkpoint_sha256']
            parent = torch.load(ROOT/control['checkpoint_path'], map_location='cpu', weights_only=False)
            for arm in ARMS:
                key = f'{site}_{arm}_seed{seed}'
                if trial_name and trial_name != key: continue
                ti = dict(identity, site=site, seed=seed, arm=arm, fold=fold, training_hard_cut=hard)
                cp, pp, rp = private/'checkpoints'/f'{key}.pt', private/'predictions'/f'{key}.npz', private/'trials'/f'{key}.json'
                old = json.loads(rp.read_text()) if rp.exists() else None
                if old:
                    assert old['identity'] == ti and file_digest(cp) == old['checkpoint_sha256']
                    assert file_digest(pp) == old['prediction_sha256']
                    if not replay: trials.append(old); continue
                elif replay: raise ValueError('Complete trial required for replay')
                torch.manual_seed(seed); model = TemporalSourceDynamics(arm)
                if not replay:
                    beat(state='fit_or_resume', trial=key)
                    result = fit_motion_candidate(model, lambda ids:cached.inputs(ids, training=True),
                        data.loss_targets, train_ids, weights, scale=scale, seed=seed, config=reg['training'],
                        identity=ti, checkpoint=cp, heartbeat=lambda **kw:beat(trial=key, **kw),
                        suppress_zero_targets=False, stop_at=stop_at)
                    updates += result['new_updates']
                    if not result['complete']:
                        beat(state='training_pilot_complete_no_held_forecast', trial=key, step=result['step']); return
                saved = torch.load(cp, map_location='cpu', weights_only=False)
                assert saved['identity'] == ti and saved['step'] == 10000 and not saved['suppress_zero_targets']
                model.load_state_dict(saved['model'])
                np.testing.assert_array_equal(saved['train_ids'], parent['train_ids'])
                np.testing.assert_array_equal(saved['draw_counts'], parent['draw_counts'])
                assert torch.equal(saved['sampler_rng'], parent['sampler_rng'])
                train_pred = forecast(model, cached.inputs, train_ids, 'mask_only')
                held_pred = forecast(model, cached.inputs, held, 'mask_only')
                arrays = dict(train_ids=train_ids, train_prediction=train_pred, held_ids=held, held_prediction=held_pred)
                if replay:
                    with np.load(pp, allow_pickle=False) as a:
                        for k, value in arrays.items(): np.testing.assert_array_equal(a[k], value)
                    replayed.append(key); beat(state='exact_replay', trial=key); continue
                save_arrays(pp, arrays)
                trial = dict(identity=ti, trial=key, site=site, arm=arm, seed=seed, result_source='fresh_run_torch_temporal_head',
                    parameters=sum(p.numel() for p in model.parameters()), fit=result,
                    training=metrics(data, train_ids, train_pred, scale, hard), held=metrics(data, held, held_pred, scale, hard),
                    checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
                    prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp), sampler_matches_control=True)
                immutable_json(rp, trial); trials.append(trial)
                beat(state='trial_complete', trial=key, held_gain=trial['held']['gain_percent'])
    immutable_json(public/'input_checks.json', dict(identity=identity, folds=checks, future_target_in_features=False,
        all_rows_retained=True, main_rows_scored=0, outer_rows_scored=0, sensor_asof_certified=False))
    if trial_name: return
    if replay:
        assert len(replayed) == 36
        immutable_json(public/'replay.json', dict(exact_replays=replayed, new_updates=0))
        beat(state='replay_complete', models=36); return
    assert len(trials) == 36 and sum(t['fit']['step'] for t in trials) == 360000
    ids = cached.ids; archives = []
    for arm in ARMS:
        for seed in reg['seeds']:
            pieces = []
            for t in trials:
                if t['arm'] != arm or t['seed'] != seed: continue
                with np.load(ROOT/t['prediction_path'], allow_pickle=False) as a:
                    pieces.append(dict(ids=a['held_ids'].copy(), prediction=a['held_prediction'].copy(),
                        cost_scale=np.full(len(a['held_ids']), t['identity']['fold']['cost_scale'])))
            prediction, scale = assemble_oof(ids, pieces)
            labels = cost_labels(prediction, data.target[ids-data.nmain], scale)
            path = private/'oof'/f'{arm}_seed{seed}.npz'
            save_arrays(path, dict(ids=ids, prediction=prediction, cost_scale=scale, **labels))
            archives.append(dict(arm=arm, seed=seed, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
    immutable_json(public/'report.json', dict(identity=identity, trials=trials, oof_labels=archives,
        models=36, optimizer_updates=360000, rows=len(ids), policy_trained=False,
        main_rows_scored=0, outer_rows_scored=0, new_deployment=False, stage5c_executed=False, smc_enabled=False))
    beat(state='training_complete', models=36, new_updates=updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'train', 'replay'], required=True)
    parser.add_argument('--trial'); parser.add_argument('--stop-at', type=int); parser.add_argument('--stop-chunks', type=int)
    args = parser.parse_args()
    if args.stop_at is not None and (not args.trial or args.phase != 'train'):
        raise ValueError('Training pilot needs a named trial')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration); data = context(reg)
    private, public = ROOT/reg['output'], ROOT/reg['reports']
    def beat(**kw):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **kw)
        json_write(private/'heartbeat.json', event); print(json.dumps(event), flush=True)
    beat(state='verified_inputs', phase=args.phase)
    if args.phase == 'prepare':
        prepare(reg, data, private, public, beat, args.stop_chunks); return
    receipt = json.loads((public/'preparation.json').read_text())
    assert receipt['identity']['registration_sha256'] == file_digest(args.registration)
    cached = FeatureCorpus(data, private, receipt)
    train(reg, data, cached, private, public, beat, args.trial, args.stop_at, args.phase == 'replay')


if __name__ == '__main__':
    main()
