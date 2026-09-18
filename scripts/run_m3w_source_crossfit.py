"""Cold-start candidate cross-fitting inside source training, never outer evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# This inherited import enforces native arm64 before importing Torch.
from scripts.run_m3w_source_cost_dynamics import DynamicsCorpus, load_config as load_data
from scripts.run_m3w_source_continuation import immutable_json
from scripts.run_m3w_source_start_probe import array_hash
import numpy as np
import torch
from src.world_model.m3w_source_crossfit import (
    nested_partition, fold_normalizer, validate_query, assemble_oof, cost_labels,
)
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, fit_dynamics, forecast
from src.world_model.m3w_source_modality_continuation import continue_modality
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import error_summary

SITES = ['coupa', 'deathCircle', 'gates', 'hyang']


class CrossfitCorpus(DynamicsCorpus):
    def configure(self, site):
        train, held, outer = nested_partition(self.source_sites, self.source_tracks,
            self.source_records, 'bookstore', site)
        train, held, outer = [a + self.nmain for a in (train, held, outer)]
        self.normalizer, self.z, weights = fold_normalizer(self.x, train)
        self.allowed[:] = False
        self.allowed[train] = True
        self.predict_allowed = self.allowed.copy()
        self.predict_allowed[held] = True
        scale = float(np.linalg.norm(self.target[train-self.nmain].astype(float), axis=-1).mean())
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError('Positive inner-training cost normalizer required')
        return train, weights, held, outer, scale

    def dynamics_inputs(self, ids, *, training=False):
        ids = validate_query(ids, self.nmain, len(self.y),
                             self.allowed if training else self.predict_allowed)
        return super().dynamics_inputs(ids, training=training)


def load_config(path):
    reg = json.loads(path.read_text())
    if (str(path) != reg['registration_path'] or reg['role'] != 'training_only_nested_source_crossfit'
            or reg['inner_sites'] != SITES or reg['outer_excluded_site'] != 'bookstore'
            or reg['seeds'] != [17, 29, 43] or reg['models'] != 12 or reg['updates'] != 120000
            or reg['arm'] != 'mask_only' or reg['initialization'] != 'random_no_fitted_parent'
            or reg['initial_training']['updates'] != 2000
            or reg['continuation']['start_step'] != 2000 or reg['continuation']['updates'] != 10000
            or reg['bootstrap_resamples'] != 2000 or reg['bootstrap_seed'] != 38113
            or reg['outer_scoring'] or reg['model_selection'] or reg['new_deployment']
            or not reg['bindings']):
        raise ValueError('Fixed cold-start training-side cross-fit contract required')
    for name, sha in reg['bindings'].items():
        if file_digest(ROOT/name) != sha:
            raise ValueError('Frozen dependency changed: '+name)
    return reg


def metrics(data, ids, prediction, scale, cutoff):
    loc = ids - data.nmain
    labels = cost_labels(prediction, data.target[loc], np.full(len(ids), scale))
    return error_summary(labels['ade'], labels['fde'], labels['cv_ade'], data.native_scale[loc],
                         np.any(prediction != 0, axis=(1, 2)), labels['cv_ade'] >= cutoff)


def save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            assert set(old.files) == set(arrays)
            for key, value in arrays.items():
                np.testing.assert_array_equal(old[key], value)
        return
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, **arrays)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--trial')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('A pilot must name one registered training trial')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(out/'heartbeat.json', event)
        print(json.dumps(event), flush=True)
    beat(state='verifying_source_and_nested_roles')
    data = CrossfitCorpus(load_data(Path(reg['data_registration'])))
    identity = dict(registration_sha256=file_digest(args.registration), data_identity=data.identity,
        source_assignment_sha256=data.assignment_hash, torch=torch.__version__, numpy=np.__version__,
        torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(out/'identity.json', identity)
    folds = []
    for site in SITES:
        train, weights, held, outer, scale = data.configure(site)
        for illegal in (outer[:1], np.array([0])):
            try:
                data.dynamics_inputs(illegal)
            except ValueError:
                pass
            else:
                raise AssertionError('Outer/main input access must fail')
        try:
            data.loss_targets(held[:1])
        except ValueError:
            pass
        else:
            raise AssertionError('Inner held labels reached trainer')
        folds.append(dict(site=site, training_rows=len(train), held_rows=len(held),
            training_sites=sorted(set(data.source_sites[train-data.nmain])),
            training_ids_sha256=array_hash(train), held_ids_sha256=array_hash(held),
            normalizer_sha256=array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant']),
            cost_scale=scale, outer_rows_excluded=len(outer), fitted_parent=None))
    immutable_json(public/'input_checks.json', dict(identity=identity, folds=folds,
        outer_rows_scored=0, main_rows_scored=0, future_labels_in_input=False,
        fitted_parent_reused=False, all_inference_outer_main_guards_checked=True,
        all_trainer_inner_held_guards_checked=True, independent_confirmation=False,
        units='past_normalized_annotation_pixels_raw_frames', stage5c_executed=False, smc_enabled=False))
    if args.audit_only:
        beat(state='audit_complete', folds=4, training_side_rows=sum(f['held_rows'] for f in folds))
        return
    keys = {f'{site}_seed{seed}' for site in SITES for seed in reg['seeds']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    trials, replayed = [], []
    new_updates = 0
    for site in SITES:
        train, weights, held, _, scale = data.configure(site)
        fold = next(f for f in folds if f['site'] == site)
        cutoff = float(np.quantile(np.linalg.norm(data.target[train-data.nmain].astype(float), axis=-1).mean(1), .9))
        for seed in reg['seeds']:
            key = f'{site}_seed{seed}'
            if args.trial and args.trial != key:
                continue
            ti = dict(identity, fold=fold, seed=seed, arm='mask_only', training_hard_cut=cutoff)
            parent_path = out/'parents'/f'{key}.pt'
            cp, pp, rp = out/'checkpoints'/f'{key}.pt', out/'predictions'/f'{key}.npz', out/'trials'/f'{key}.json'
            old = json.loads(rp.read_text()) if rp.exists() else None
            if old:
                assert old['identity'] == ti and file_digest(cp) == old['checkpoint_sha256']
                assert file_digest(pp) == old['prediction_sha256']
                assert file_digest(parent_path) == old['parent_sha256']
                if not args.replay:
                    trials.append(old)
                    continue
            elif args.replay:
                raise ValueError('Replay requires all completed trial receipts')
            torch.manual_seed(seed)
            model = SourceDynamics()
            if args.replay:
                state = torch.load(cp, map_location='cpu', weights_only=False)
                assert state['identity'] == ti and state['step'] == reg['continuation']['updates']
                model.load_state_dict(state['model'])
            else:
                beat(state='fit_or_resume_inner_fold', trial=key, training_rows=len(train), held_rows=len(held))
                parent_fit = fit_dynamics(model, lambda ids:data.dynamics_inputs(ids, training=True),
                    data.loss_targets, train, weights, arm='mask_only', objective='ade', normalizer=scale,
                    seed=seed, config=reg['initial_training'], identity=ti, checkpoint=parent_path,
                    heartbeat=lambda **v:beat(trial=key, phase='initial', **v), stop_at=args.stop_at)
                new_updates += parent_fit['new_updates']
                if not parent_fit['complete']:
                    beat(state='pilot_complete_no_inner_evaluation', trial=key, step=parent_fit['step'])
                    return
                parent = torch.load(parent_path, map_location='cpu', weights_only=False)
                np.testing.assert_array_equal(parent['train_ids'], train)
                continuation = continue_modality(model, lambda ids:data.dynamics_inputs(ids, training=True),
                    data.loss_targets, train, weights, parent=parent, arm='mask_only', schedule='cosine',
                    config=reg['continuation'], identity=ti, checkpoint=cp,
                    heartbeat=lambda **v:beat(trial=key, phase='continuation', **v),
                    snapshot=lambda _:None, stop_at=args.stop_at)
                new_updates += continuation['new_updates']
                if not continuation['complete']:
                    beat(state='pilot_complete_no_inner_evaluation', trial=key, step=continuation['step'])
                    return
            held_prediction = forecast(model, data.dynamics_inputs, held, 'mask_only')
            train_prediction = forecast(model, data.dynamics_inputs, train, 'mask_only')
            arrays = dict(held_ids=held, held_prediction=held_prediction, train_ids=train,
                          train_prediction=train_prediction)
            if args.replay:
                with np.load(pp, allow_pickle=False) as saved:
                    for name, values in arrays.items():
                        np.testing.assert_array_equal(saved[name], values)
                replayed.append(key)
                beat(state='trial_replayed', trial=key)
                continue
            save_arrays(pp, arrays)
            result = dict(identity=ti, site=site, seed=seed, trial=key,
                result_source='fresh_run_cold_start_torch', parameters=sum(v.numel() for v in model.parameters()),
                initial_fit=parent_fit, continuation_fit=continuation,
                training=metrics(data, train, train_prediction, scale, cutoff),
                inner_held=metrics(data, held, held_prediction, scale, cutoff),
                checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
                prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp),
                parent_path=str(parent_path.relative_to(ROOT)), parent_sha256=file_digest(parent_path))
            immutable_json(rp, result)
            trials.append(result)
            beat(state='trial_complete', trial=key, training_gain=result['training']['gain_percent'],
                 inner_held_gain=result['inner_held']['gain_percent'])
    if args.replay:
        assert len(replayed) == 12
        immutable_json(public/'replay.json', dict(identity=identity, exact_replays=replayed,
            new_training_updates=0, outer_rows_scored=0, main_rows_scored=0))
        beat(state='replay_complete', predictors=len(replayed))
        return
    if args.trial:
        return
    assert len(trials) == 12
    assert sum(t['initial_fit']['step'] + t['continuation_fit']['additional_updates'] for t in trials) == 120000
    outer_train = np.flatnonzero(data.source_sites != 'bookstore') + data.nmain
    labels_receipts = []
    for seed in reg['seeds']:
        pieces = []
        for result in trials:
            if result['seed'] != seed:
                continue
            with np.load(ROOT/result['prediction_path'], allow_pickle=False) as a:
                pieces.append(dict(ids=a['held_ids'].copy(), prediction=a['held_prediction'].copy(),
                    cost_scale=np.full(len(a['held_ids']), result['identity']['fold']['cost_scale'])))
        pred, scale = assemble_oof(outer_train, pieces)
        labels = cost_labels(pred, data.target[outer_train-data.nmain], scale)
        path = out/'oof'/f'seed{seed}.npz'
        save_arrays(path, dict(ids=outer_train, prediction=pred, cost_scale=scale, **labels))
        labels_receipts.append(dict(seed=seed, rows=len(outer_train), path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
    immutable_json(public/'report.json', dict(identity=identity, trials=trials, oof_labels=labels_receipts,
        models=12, optimizer_updates=120000, outer_rows_scored=0, main_rows_scored=0,
        no_model_or_threshold_selection=True, independent_confirmation=False, new_deployment=False,
        main_primary_changed=False, stage5c_executed=False, smc_enabled=False))
    beat(state='crossfit_complete', new_training_updates=new_updates, models=12)


if __name__ == '__main__':
    main()
