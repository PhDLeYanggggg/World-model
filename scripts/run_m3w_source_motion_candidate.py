"""Fixed OOF loss intervention; retained full sampler/eval population, no policy fit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_crossfit import (load_config as load_control, load_data,
    CrossfitCorpus, SITES, metrics, save_arrays, immutable_json, array_hash)
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, forecast
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def load_config(path):
    reg = json.loads(path.read_text())
    if (str(path) != reg['registration_path'] or reg['role'] != 'training_only_matched_zero_gradient_intervention'
            or reg['sites'] != SITES or reg['seeds'] != [17,29,43] or reg['models'] != 12
            or reg['updates'] != 120000 or reg['training']['updates'] != 10000
            or reg['training']['start_step'] != 2000 or not reg['suppress_zero_target_gradients']
            or reg['model_selection'] or reg['outer_scoring'] or reg['new_deployment']
            or reg['bootstrap_resamples'] != 2000 or not reg['bindings']):
        raise ValueError('Fixed matched-training intervention required')
    for name, sha in reg['bindings'].items():
        if file_digest(ROOT/name) != sha:
            raise ValueError('Frozen dependency changed: '+name)
    return reg


def context(reg):
    old = load_control(Path(reg['control_registration']))
    report = json.loads((ROOT/reg['control_report']).read_text())
    assert report['models'] == 12 and report['optimizer_updates'] == 120000
    for key in ('batch_size','learning_rate','weight_decay','checkpoint_every','updates','start_step','minimum_lr_ratio'):
        assert reg['training'][key] == old['continuation'][key]
    return old, report, CrossfitCorpus(load_data(Path(old['data_registration'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--trial'); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot must name a fixed trial')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(out/'heartbeat.json', event); print(json.dumps(event), flush=True)
    beat(state='verifying_matched_training_roles')
    control_reg, controls, data = context(reg)
    identity = dict(registration_sha256=file_digest(args.registration), data_identity=data.identity,
        source_assignment_sha256=data.assignment_hash, torch=torch.__version__, numpy=np.__version__,
        torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(out/'identity.json', identity)
    fold_checks = []
    for site in SITES:
        train, weights, held, outer, scale = data.configure(site)
        control = next(t for t in controls['trials'] if t['site'] == site)
        fold = control['identity']['fold']
        assert array_hash(train) == fold['training_ids_sha256'] and array_hash(held) == fold['held_ids_sha256']
        assert array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant']) == fold['normalizer_sha256']
        assert scale == fold['cost_scale']
        for ids in (outer[:1], np.array([0])):
            try: data.dynamics_inputs(ids)
            except ValueError: pass
            else: raise AssertionError('Outer/main inference must fail')
        try: data.loss_targets(held[:1])
        except ValueError: pass
        else: raise AssertionError('Held target reached training')
        sample = held[:8]; before = data.dynamics_inputs(sample); target = data.target.copy()
        try:
            data.target[:] = np.nan
            after = data.dynamics_inputs(sample)
        finally:
            data.target[:] = target
        assert all(torch.equal(a,b) for left,right in zip(before,after) for a,b in zip(left,right))
        fold_checks.append(dict(site=site, training_rows=len(train), held_rows=len(held),
            positive_training_rows=int(np.any(data.target[train-data.nmain] != 0, axis=(1,2)).sum()),
            unchanged_fold=fold, target_poison_checks=len(sample)))
    immutable_json(public/'input_checks.json', dict(identity=identity, folds=fold_checks,
        full_sampler_and_eval_population_retained=True, future_labels_in_input=False,
        source_unit='offline_annotation_pixels_past_normalized_raw_frames',
        outer_rows_scored=0, main_rows_scored=0, new_deployment=False))
    if args.audit_only:
        beat(state='audit_complete', folds=4); return
    keys = {f'{site}_seed{seed}' for site in SITES for seed in reg['seeds']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    trials, replayed, updates = [], [], 0
    for site in SITES:
        train, weights, held, _, scale = data.configure(site)
        for seed in reg['seeds']:
            key = f'{site}_seed{seed}'
            if args.trial and key != args.trial: continue
            control = next(t for t in controls['trials'] if t['site'] == site and t['seed'] == seed)
            assert file_digest(ROOT/control['checkpoint_path']) == control['checkpoint_sha256']
            ti = dict(identity, fold=control['identity']['fold'], seed=seed,
                control_checkpoint_sha256=control['checkpoint_sha256'],
                training_hard_cut=control['identity']['training_hard_cut'])
            cp, pp, rp = out/'checkpoints'/f'{key}.pt', out/'predictions'/f'{key}.npz', out/'trials'/f'{key}.json'
            old = json.loads(rp.read_text()) if rp.exists() else None
            if old:
                assert old['identity'] == ti and file_digest(cp) == old['checkpoint_sha256']
                assert file_digest(pp) == old['prediction_sha256']
                if not args.replay: trials.append(old); continue
            elif args.replay: raise ValueError('Replay requires completed receipts')
            torch.manual_seed(seed); model = SourceDynamics()
            if args.replay:
                state = torch.load(cp, map_location='cpu', weights_only=False)
                assert state['identity'] == ti and state['step'] == reg['training']['updates']
                model.load_state_dict(state['model'])
            else:
                beat(state='fit_or_resume', trial=key)
                fit = fit_motion_candidate(model, lambda ids:data.dynamics_inputs(ids, training=True),
                    data.loss_targets, train, weights, scale=scale, seed=seed, config=reg['training'],
                    identity=ti, checkpoint=cp, heartbeat=lambda **v:beat(trial=key, **v), stop_at=args.stop_at)
                updates += fit['new_updates']
                if not fit['complete']:
                    beat(state='pilot_complete_no_held_scoring', trial=key, step=fit['step']); return
            state = torch.load(cp, map_location='cpu', weights_only=False)
            matched = torch.load(ROOT/control['checkpoint_path'], map_location='cpu', weights_only=False)
            np.testing.assert_array_equal(state['train_ids'], matched['train_ids'])
            np.testing.assert_array_equal(state['draw_counts'], matched['draw_counts'])
            assert torch.equal(state['sampler_rng'], matched['sampler_rng'])
            train_pred = forecast(model, data.dynamics_inputs, train, 'mask_only')
            held_pred = forecast(model, data.dynamics_inputs, held, 'mask_only')
            arrays = dict(train_ids=train, train_prediction=train_pred, held_ids=held, held_prediction=held_pred)
            if args.replay:
                with np.load(pp, allow_pickle=False) as cached:
                    for name, value in arrays.items(): np.testing.assert_array_equal(cached[name], value)
                replayed.append(key); beat(state='trial_replayed', trial=key); continue
            save_arrays(pp, arrays)
            result = dict(identity=ti, site=site, seed=seed, trial=key, result_source='fresh_run_cold_start_torch',
                parameters=sum(p.numel() for p in model.parameters()), fit=fit,
                training=metrics(data, train, train_pred, scale, ti['training_hard_cut']),
                held=metrics(data, held, held_pred, scale, ti['training_hard_cut']),
                sampler_matches_control=True, checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
                prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp))
            immutable_json(rp, result); trials.append(result)
            beat(state='trial_complete', trial=key, training_gain=result['training']['gain_percent'], held_gain=result['held']['gain_percent'])
    if args.replay:
        assert len(replayed) == 12
        immutable_json(public/'replay.json', dict(identity=identity, exact_replays=replayed, new_updates=0))
        beat(state='replay_complete', models=12); return
    if args.trial: return
    assert len(trials) == 12 and sum(t['fit']['step'] for t in trials) == 120000
    ids = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    archives = []
    for seed in reg['seeds']:
        pieces = []
        for trial in trials:
            if trial['seed'] != seed: continue
            with np.load(ROOT/trial['prediction_path'], allow_pickle=False) as a:
                pieces.append(dict(ids=a['held_ids'].copy(), prediction=a['held_prediction'].copy(),
                    cost_scale=np.full(len(a['held_ids']), trial['identity']['fold']['cost_scale'])))
        prediction, scale = assemble_oof(ids, pieces)
        labels = cost_labels(prediction, data.target[ids-data.nmain], scale)
        path = out/'oof'/f'seed{seed}.npz'
        save_arrays(path, dict(ids=ids, prediction=prediction, cost_scale=scale, **labels))
        archives.append(dict(seed=seed, path=str(path.relative_to(ROOT)), sha256=file_digest(path), rows=len(ids)))
    immutable_json(public/'report.json', dict(identity=identity, trials=trials, oof_labels=archives,
        models=12, optimizer_updates=120000, control_models_cached_verified=12,
        outer_rows_scored=0, main_rows_scored=0, policy_trained=False, new_deployment=False,
        main_primary_changed=False, stage5c_executed=False, smc_enabled=False))
    beat(state='motion_candidate_complete', models=12, new_updates=updates)


if __name__ == '__main__':
    main()
