"""Fixed full-training learning curves; no held-source or main evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Parent entry point enforces arm64 and CPU thread settings before Torch import.
from scripts.run_m3w_source_cost_dynamics import DynamicsCorpus, load_config as load_parent, trajectory_metrics
import numpy as np
import torch
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, forecast
from src.world_model.m3w_source_continuation import SCHEDULES, continue_fit


def load_config(path):
    reg = json.loads(path.read_text())
    if (reg['registration_path'] != str(path) or reg['role'] != 'source_training_only_continuation'
            or reg['schedules'] != list(SCHEDULES) or reg['seeds'] != [17, 29, 43]
            or reg['excluded_site'] != 'bookstore' or reg['branches'] != 6
            or reg['additional_updates'] != 48000 or reg['held_evaluation']
            or reg['new_deployment'] or reg['main_primary_changed'] or not reg['bindings']):
        raise ValueError('Fixed training-only continuation registration required')
    if reg['training'] != dict(start_step=2000, updates=10000, batch_size=64,
            learning_rate=.0003, minimum_lr_ratio=.01, weight_decay=.0001,
            checkpoint_every=200, milestones=[2000, 4000, 6000, 10000]):
        raise ValueError('Continuation budget or schedules changed')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Registered dependency changed: ' + name)
    return reg


def build_data(reg):
    parent_reg = load_parent(Path(reg['parent_registration']))
    data = DynamicsCorpus(parent_reg)
    train, weights, excluded, _, scale = data.dynamics_design(reg['excluded_site'])
    if len(train) != 15430 or np.intersect1d(train, excluded).size:
        raise ValueError('Complete source training complement changed')
    report = json.loads((ROOT/reg['parent_report']).read_text())
    parents = {}
    for seed in reg['seeds']:
        key = f'ade_past_rgb_bookstore_seed{seed}'
        result = next(t for t in report['trials'] if t['trial'] == key)
        path = ROOT/result['checkpoint_path']
        if file_digest(path) != result['checkpoint_sha256']:
            raise ValueError('Frozen parent checkpoint changed')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if (cp['identity'] != result['identity'] or cp['config'] != parent_reg['training']
                or cp['normalizer'] != scale or cp['step'] != 2000
                or cp['identity']['source_assignment_sha256'] != data.assignment_hash
                or cp['identity']['training_rows_sha256'] != array_hash(train, weights, data.target[train-data.nmain])
                or cp['identity']['normalizer_sha256'] != array_hash(
                    data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant'])):
            raise ValueError('Parent training or input identity changed')
        np.testing.assert_array_equal(cp['train_ids'], train)
        parents[seed] = (result, cp)
    return data, train, weights, scale, parents


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Completed receipt changed: ' + str(path))
    else:
        json_write(path, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--trial')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']

    def beat(**values):
        receipt = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(out/'heartbeat.json', receipt)
        print(json.dumps(receipt), flush=True)

    beat(state='verifying_training_assets')
    data, train, weights, scale, parents = build_data(reg)
    loc = train-data.nmain
    target = data.target[loc]
    cv = np.linalg.norm(target.astype(float), axis=-1).mean(1)
    hard_cut = float(np.quantile(cv, .9))
    identity = dict(registration_sha256=file_digest(args.registration), parent_identity=data.identity,
        source_assignment_sha256=data.assignment_hash, training_rows_sha256=array_hash(train, weights, target),
        feature_sha256=array_hash(data.z[train]), frame_sha256=array_hash(
            data.radius[loc], data.rotation[loc], data.support[loc]), training_cost_scale=scale,
        parent_checkpoints={str(s):r['checkpoint_sha256'] for s, (r, _) in parents.items()},
        torch=torch.__version__, numpy=np.__version__, torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(out/'identity.json', identity)
    immutable_json(public/'input_checks.json', dict(identity=identity,
        result_source='fresh_run_training_audit_cached_verified_parent_assets', training_rows=len(train),
        zero_target_rows=int((cv == 0).sum()), nonzero_target_rows=int((cv > 0).sum()),
        training_sites={s:int((data.source_sites[loc] == s).sum()) for s in sorted(set(data.source_sites[loc]))},
        scoped_agents=len(set(data.source_tracks[loc])), source_videos=len(set(data.source_records[loc])),
        held_rows_scored=0, main_rows_scored=0, sealed_roles_opened=False, labels_in_input=False,
        observation='offline_supplied_annotations_not_sensor_asof', source='original_SDD_train40_only',
        task='8_to_12_at_stride12_plus144_raw_frames_no_seconds_equivalence', new_deployment=False))
    if args.audit_only:
        beat(state='audit_complete_no_training', rows=len(train))
        return
    keys = {f'{schedule}_seed{seed}' for seed in reg['seeds'] for schedule in reg['schedules']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered branch')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot names one training-only branch')
    trials, replays = [], []
    new_updates = new_branches = 0
    for seed in reg['seeds']:
        parent_result, parent = parents[seed]
        for schedule in reg['schedules']:
            key = f'{schedule}_seed{seed}'
            if args.trial and key != args.trial:
                continue
            ti = dict(identity, seed=seed, schedule=schedule,
                      parent_checkpoint_sha256=parent_result['checkpoint_sha256'])
            cp = out/'checkpoints'/f'{key}.pt'
            rp = out/'trials'/f'{key}.json'
            old = json.loads(rp.read_text()) if rp.exists() else None
            if old:
                if old['identity'] != ti or file_digest(cp) != old['checkpoint_sha256']:
                    raise ValueError('Completed branch changed')
                for snap in old['milestones']:
                    for kind in ('checkpoint', 'prediction', 'receipt'):
                        if file_digest(ROOT/snap[kind+'_path']) != snap[kind+'_sha256']:
                            raise ValueError('Completed milestone changed')
                if not args.replay:
                    trials.append(old)
                    continue
            torch.manual_seed(seed)
            model = SourceDynamics()

            def capture(state):
                step = state['step']
                sp = out/'milestones'/f'{key}_step{step}.pt'
                pp = sp.with_suffix('.npz')
                mp = sp.with_suffix('.json')
                if mp.exists():
                    receipt = json.loads(mp.read_text())
                    if receipt['identity'] != ti or receipt['step'] != step:
                        raise ValueError('Milestone identity changed')
                    for path, name in ((sp, 'checkpoint'), (pp, 'prediction')):
                        if file_digest(path) != receipt[name+'_sha256']:
                            raise ValueError('Milestone artifact changed')
                    saved = torch.load(sp, map_location='cpu', weights_only=False)
                    for name, tensor in state['model'].items():
                        torch.testing.assert_close(saved['model'][name], tensor, rtol=0, atol=0)
                    return
                prediction = forecast(model, lambda ids:data.dynamics_inputs(ids, training=True), train, 'past_rgb')
                metrics = trajectory_metrics(prediction, target, data.native_scale[loc],
                    np.any(prediction != 0, axis=(1, 2)), hard_cut)
                if step == 2000 and not np.isclose(metrics['ade'], parent_result['training_ade'], rtol=1e-12):
                    raise ValueError('Inherited complete-training score does not replay')
                sp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cp, sp)
                temporary = pp.with_suffix('.tmp.npz')
                np.savez(temporary, ids=train, prediction=prediction)
                os.replace(temporary, pp)
                json_write(mp, dict(identity=ti, step=step, metrics=metrics,
                    checkpoint_sha256=file_digest(sp), prediction_sha256=file_digest(pp)))
                beat(state='full_training_milestone', trial=key, step=step, **metrics)

            if args.replay:
                if old is None:
                    raise ValueError('Replay requires completed branch')
                for snap in old['milestones']:
                    state = torch.load(ROOT/snap['checkpoint_path'], map_location='cpu', weights_only=False)
                    if state['identity'] != ti or state['step'] != snap['step']:
                        raise ValueError('Replay checkpoint mismatch')
                    model.load_state_dict(state['model'])
                    prediction = forecast(model, lambda ids:data.dynamics_inputs(ids, training=True), train, 'past_rgb')
                    with np.load(ROOT/snap['prediction_path'], allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['ids'], train)
                        np.testing.assert_array_equal(a['prediction'], prediction)
                    replays.append(dict(trial=key, step=snap['step'], exact=True))
                    beat(state='milestone_replayed', trial=key, step=snap['step'])
                continue
            beat(state='continue_training', trial=key, parent_step=2000)
            fit = continue_fit(model, lambda ids:data.dynamics_inputs(ids, training=True), data.loss_targets,
                train, weights, parent=parent, schedule=schedule, config=reg['training'], identity=ti,
                checkpoint=cp, heartbeat=lambda **v:beat(trial=key, **v), snapshot=capture, stop_at=args.stop_at)
            new_updates += fit['new_updates']
            if not fit['complete']:
                beat(state='pilot_complete_training_only', trial=key, **fit)
                return
            milestones = []
            for step in reg['training']['milestones']:
                sp = out/'milestones'/f'{key}_step{step}.pt'
                paths = dict(checkpoint=sp, prediction=sp.with_suffix('.npz'), receipt=sp.with_suffix('.json'))
                receipt = json.loads(paths['receipt'].read_text())
                row = dict(step=step, metrics=receipt['metrics'])
                for kind, path in paths.items():
                    row[kind+'_path'], row[kind+'_sha256'] = str(path.relative_to(ROOT)), file_digest(path)
                milestones.append(row)
            result = dict(identity=ti, trial=key, seed=seed, schedule=schedule, fit=fit,
                result_source='fresh_run_continuation_cached_verified_parent_not_generalization',
                parameter_count=sum(p.numel() for p in model.parameters()), training_rows=len(train),
                checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp), milestones=milestones)
            json_write(rp, result)
            trials.append(result)
            new_branches += 1
            beat(state='branch_complete', trial=key, training_gain_percent=milestones[-1]['metrics']['gain_percent'])
    if args.replay:
        json_write(public/'replay.json', dict(identity=identity, exact_milestones=replays, all_exact=True, new_updates=0))
        beat(state='replay_complete', count=len(replays))
        return
    if args.trial:
        return
    assert len(trials) == reg['branches']
    report = dict(identity=identity, trials=trials, completed_branches=len(trials),
        unique_parent_models=3, inherited_unique_updates=6000,
        additional_updates=sum(t['fit']['additional_updates'] for t in trials),
        summed_continuation_seconds=sum(t['fit']['continuation_seconds'] for t in trials),
        held_rows_scored=0, main_rows_scored=0, main_primary_changed=False,
        sealed_roles_opened=False, new_deployment=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(public/'report.json', report)
    beat(state='continuation_complete' if new_branches else 'completed_resume_verified',
         new_branches=new_branches, new_updates=new_updates)


if __name__ == '__main__':
    main()
