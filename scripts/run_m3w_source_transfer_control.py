"""Matched source modality continuation, then a fixed exposed-site diagnostic."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Enforces native arm64 before importing Torch.
from scripts.run_m3w_source_continuation import load_config as load_rgb, build_data as build_rgb, immutable_json
from scripts.run_m3w_source_cost_dynamics import trajectory_metrics
import numpy as np
import torch
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, forecast
from src.world_model.m3w_source_modality_continuation import continue_modality


def load_config(path):
    reg = json.loads(path.read_text())
    if (reg['registration_path'] != str(path) or reg['role'] != 'exposed_source_modality_control'
            or reg['held_site'] != 'bookstore' or reg['seeds'] != [17, 29, 43]
            or reg['schedules'] != ['constant', 'cosine'] or reg['new_arm'] != 'mask_only'
            or reg['new_branches'] != 6 or reg['new_updates'] != 48000
            or reg['held_site_is_untouched'] or reg['model_selection'] or reg['new_deployment']
            or reg['fixed_probability_threshold'] != .9
            or reg['bootstrap_resamples'] != 2000 or reg['bootstrap_seed'] != 38113
            or not reg['bindings']):
        raise ValueError('Fixed exposed-site modality comparison required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Registered dependency changed: '+name)
    rgb = load_rgb(Path(reg['rgb_registration']))
    if reg['training'] != rgb['training']:
        raise ValueError('Modality arms require identical optimization budgets')
    return reg


def build_data(reg):
    rgb_reg = load_rgb(Path(reg['rgb_registration']))
    data, train, weights, scale, rgb_parents = build_rgb(rgb_reg)
    held = np.flatnonzero(data.source_sites == reg['held_site'])+data.nmain
    assert len(held) == 6944 and not np.intersect1d(train, held).size
    assert not set(data.source_tracks[train-data.nmain]) & set(data.source_tracks[held-data.nmain])
    previous = json.loads((ROOT/reg['parent_report']).read_text())
    rgb_report = json.loads((ROOT/reg['rgb_report']).read_text())
    assert rgb_report['completed_branches'] == 6 and rgb_report['additional_updates'] == 48000
    parents = {}
    for seed in reg['seeds']:
        for arm in ('mask_only', 'past_rgb'):
            result = next(t for t in previous['trials'] if t['trial'] == f'ade_{arm}_bookstore_seed{seed}')
            path = ROOT/result['checkpoint_path']
            if file_digest(path) != result['checkpoint_sha256']:
                raise ValueError('Frozen parent changed')
            cp = torch.load(path, map_location='cpu', weights_only=False)
            reference = rgb_parents[seed][1]
            if (cp['identity'] != result['identity'] or cp['step'] != 2000 or cp['arm'] != arm
                    or cp['objective'] != 'ade' or cp['normalizer'] != scale
                    or cp['config'] != reference['config']
                    or cp['identity']['training_rows_sha256'] != reference['identity']['training_rows_sha256']
                    or cp['identity']['normalizer_sha256'] != reference['identity']['normalizer_sha256']):
                raise ValueError('Parent arm, training schema or objective mismatch')
            np.testing.assert_array_equal(cp['train_ids'], train)
            np.testing.assert_array_equal(cp['draw_counts'], reference['draw_counts'])
            assert torch.equal(cp['sampler_rng'], reference['sampler_rng'])
            parents[arm, seed] = (result, cp)
    for trial in rgb_report['trials']:
        if file_digest(ROOT/trial['checkpoint_path']) != trial['checkpoint_sha256']:
            raise ValueError('Frozen RGB continuation changed')
    return data, train, weights, held, scale, parents, rgb_report


def train_controls(reg, identity, data, train, weights, scale, parents, out, public, beat, args):
    keys = {f'mask_only_{s}_seed{seed}' for seed in reg['seeds'] for s in reg['schedules']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered control branch')
    if args.stop_at is not None and not args.trial:
        raise ValueError('Pilot names a training branch')
    trials, new_updates, new_branches = [], 0, 0
    loc = train-data.nmain
    target = data.target[loc]
    hard_cut = float(np.quantile(np.linalg.norm(target.astype(float), axis=-1).mean(1), .9))
    for seed in reg['seeds']:
        parent_result, parent = parents['mask_only', seed]
        for schedule in reg['schedules']:
            key = f'mask_only_{schedule}_seed{seed}'
            if args.trial and key != args.trial:
                continue
            ti = dict(identity, arm='mask_only', seed=seed, schedule=schedule,
                      parent_checkpoint_sha256=parent_result['checkpoint_sha256'])
            cp, rp = out/'checkpoints'/f'{key}.pt', out/'trials'/f'{key}.json'
            if rp.exists():
                old = json.loads(rp.read_text())
                if old['identity'] != ti or file_digest(cp) != old['checkpoint_sha256']:
                    raise ValueError('Completed control changed')
                for milestone in old['milestones']:
                    for kind in ('checkpoint', 'prediction', 'receipt'):
                        if file_digest(ROOT/milestone[kind+'_path']) != milestone[kind+'_sha256']:
                            raise ValueError('Control milestone changed')
                trials.append(old)
                continue
            torch.manual_seed(seed)
            model = SourceDynamics()

            def capture(state):
                step = state['step']
                sp = out/'milestones'/f'{key}_step{step}.pt'
                pp, mp = sp.with_suffix('.npz'), sp.with_suffix('.json')
                if mp.exists():
                    receipt = json.loads(mp.read_text())
                    assert receipt['identity'] == ti and receipt['step'] == step
                    assert receipt['checkpoint_sha256'] == file_digest(sp)
                    assert receipt['prediction_sha256'] == file_digest(pp)
                    saved = torch.load(sp, map_location='cpu', weights_only=False)
                    for name in state['model']:
                        torch.testing.assert_close(saved['model'][name], state['model'][name], rtol=0, atol=0)
                    return
                prediction = forecast(model, lambda ids:data.dynamics_inputs(ids, training=True), train, 'mask_only')
                metrics = trajectory_metrics(prediction, target, data.native_scale[loc],
                    np.any(prediction != 0, axis=(1, 2)), hard_cut)
                if step == 2000:
                    np.testing.assert_allclose(metrics['ade'], parent_result['training_ade'], rtol=1e-12)
                sp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cp, sp)
                temporary = pp.with_suffix('.tmp.npz')
                np.savez(temporary, ids=train, prediction=prediction)
                os.replace(temporary, pp)
                json_write(mp, dict(identity=ti, step=step, metrics=metrics,
                    checkpoint_sha256=file_digest(sp), prediction_sha256=file_digest(pp)))
                beat(state='training_milestone', trial=key, step=step, gain_percent=metrics['gain_percent'])

            beat(state='continue_mask_control', trial=key)
            fit = continue_modality(model, lambda ids:data.dynamics_inputs(ids, training=True), data.loss_targets,
                train, weights, parent=parent, arm='mask_only', schedule=schedule, config=reg['training'],
                identity=ti, checkpoint=cp, heartbeat=lambda **v:beat(trial=key, **v),
                snapshot=capture, stop_at=args.stop_at)
            new_updates += fit['new_updates']
            if not fit['complete']:
                beat(state='pilot_complete_no_held_forecast', trial=key, **fit)
                return
            milestones = []
            for step in reg['training']['milestones']:
                sp = out/'milestones'/f'{key}_step{step}.pt'
                paths = dict(checkpoint=sp, prediction=sp.with_suffix('.npz'), receipt=sp.with_suffix('.json'))
                row = dict(step=step, metrics=json.loads(paths['receipt'].read_text())['metrics'])
                for kind, path in paths.items():
                    row[kind+'_path'], row[kind+'_sha256'] = str(path.relative_to(ROOT)), file_digest(path)
                milestones.append(row)
            trial = dict(identity=ti, trial=key, arm='mask_only', seed=seed, schedule=schedule,
                result_source='fresh_run_continuation_from_verified_parent', fit=fit, training_rows=len(train),
                checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp), milestones=milestones)
            json_write(rp, trial)
            trials.append(trial)
            new_branches += 1
            beat(state='branch_complete', trial=key, training_gain_percent=milestones[-1]['metrics']['gain_percent'])
    if args.trial:
        return
    assert len(trials) == 6
    immutable_json(public/'training_report.json', dict(identity=identity, trials=trials, new_branches=6,
        additional_updates=sum(t['fit']['additional_updates'] for t in trials),
        summed_continuation_seconds=sum(t['fit']['continuation_seconds'] for t in trials), held_rows_scored=0))
    beat(state='controls_complete' if new_branches else 'completed_resume_verified',
         new_branches=new_branches, new_updates=new_updates)


def evaluate(reg, identity, data, train, held, parents, rgb_report, out, public, beat, replay):
    controls = json.loads((public/'training_report.json').read_text())
    assert controls['identity'] == identity and controls['additional_updates'] == 48000
    frozen = []
    for (arm, seed), (result, _) in parents.items():
        frozen.append(dict(arm=arm, seed=seed, schedule='parent_2k', checkpoint_path=result['checkpoint_path'],
            checkpoint_sha256=result['checkpoint_sha256'], expected_identity=result['identity'],
            old_prediction_path=result['prediction_path'], old_prediction_sha256=result['prediction_sha256']))
    for arm, report in [('past_rgb', rgb_report), ('mask_only', controls)]:
        for trial in report['trials']:
            frozen.append(dict(arm=arm, seed=trial['seed'], schedule=trial['schedule'],
                checkpoint_path=trial['checkpoint_path'], checkpoint_sha256=trial['checkpoint_sha256'],
                expected_identity=trial['identity']))
    assert len(frozen) == 18
    immutable_json(public/'frozen_predictors.json', dict(identity=identity, predictors=frozen,
        held_site_previously_explored=True, winner_selection=False))
    probabilities = json.loads((ROOT/reg['frozen_probability_report']).read_text())
    producers = {(t['arm'], t['seed']):t for t in probabilities['trials'] if t['site']=='bookstore'}
    tr_loc, loc = train-data.nmain, held-data.nmain
    hard_cut = float(np.quantile(np.linalg.norm(data.target[tr_loc].astype(float), axis=-1).mean(1), .9))
    results, exact = [], []
    for item in frozen:
        arm, seed, schedule = item['arm'], item['seed'], item['schedule']
        key = f'{arm}_{schedule}_seed{seed}'
        cp = ROOT/item['checkpoint_path']
        if file_digest(cp) != item['checkpoint_sha256']:
            raise ValueError('Frozen predictor changed')
        state = torch.load(cp, map_location='cpu', weights_only=False)
        if state['identity'] != item['expected_identity'] or state['arm'] != arm:
            raise ValueError('Frozen predictor identity/arm mismatch')
        producer = producers[arm, seed]
        if file_digest(ROOT/producer['prediction_path']) != producer['prediction_sha256']:
            raise ValueError('Frozen diagnostic probability producer changed')
        with np.load(ROOT/producer['prediction_path'], allow_pickle=False) as a:
            np.testing.assert_array_equal(a['held_indices'], held)
            probability = a['probability'].copy()
        torch.manual_seed(seed)
        model = SourceDynamics()
        model.load_state_dict(state['model'])
        prediction = forecast(model, data.dynamics_inputs, held, arm)
        if schedule == 'parent_2k':
            assert file_digest(ROOT/item['old_prediction_path']) == item['old_prediction_sha256']
            with np.load(ROOT/item['old_prediction_path'], allow_pickle=False) as a:
                np.testing.assert_array_equal(a['held_indices'], held)
                np.testing.assert_array_equal(a['prediction'], prediction)
        pp = out/'held_predictions'/f'{key}.npz'
        if pp.exists():
            with np.load(pp, allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'], held)
                np.testing.assert_array_equal(a['prediction'], prediction)
                np.testing.assert_array_equal(a['probability'], probability)
        elif replay:
            raise ValueError('Replay requires saved predictions')
        else:
            pp.parent.mkdir(parents=True, exist_ok=True)
            temporary = pp.with_suffix('.tmp.npz')
            np.savez(temporary, ids=held, prediction=prediction, probability=probability)
            os.replace(temporary, pp)
        if replay:
            exact.append(key)
            beat(state='held_prediction_replayed', trial=key)
            continue
        modes = {}
        for mode, use in [('uncontrolled', np.ones(len(held), bool)),
                          ('fixed_probability_guard', probability >= .9)]:
            emitted = np.where(use[:, None, None], prediction, 0)
            modes[mode] = trajectory_metrics(emitted, data.target[loc], data.native_scale[loc], use, hard_cut)
            modes[mode]['actual_changed_rate'] = float(np.any(emitted != 0, axis=(1, 2)).mean())
        results.append(dict(trial=key, arm=arm, seed=seed, schedule=schedule,
            result_source='fresh_run_inference_from_cached_verified_checkpoint', metrics=modes,
            checkpoint_path=item['checkpoint_path'], checkpoint_sha256=item['checkpoint_sha256'],
            probability_producer_sha256=producer['prediction_sha256'],
            prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp)))
        beat(state='held_prediction_scored', trial=key, gain_percent=modes['uncontrolled']['gain_percent'])
    if replay:
        for trial in controls['trials']:
            for milestone in trial['milestones']:
                cp = ROOT/milestone['checkpoint_path']
                assert file_digest(cp) == milestone['checkpoint_sha256']
                assert file_digest(ROOT/milestone['prediction_path']) == milestone['prediction_sha256']
                state = torch.load(cp, map_location='cpu', weights_only=False)
                assert state['identity'] == trial['identity']
                model = SourceDynamics()
                model.load_state_dict(state['model'])
                prediction = forecast(model, lambda ids:data.dynamics_inputs(ids, training=True), train, 'mask_only')
                with np.load(ROOT/milestone['prediction_path'], allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['ids'], train)
                    np.testing.assert_array_equal(a['prediction'], prediction)
                exact.append(f"training:{trial['trial']}:{milestone['step']}")
                beat(state='training_milestone_replayed', trial=trial['trial'], step=milestone['step'])
        json_write(public/'replay.json', dict(identity=identity, exact=exact, all_exact=True, new_updates=0))
        beat(state='replay_complete', count=len(exact))
        return
    immutable_json(public/'evaluation.json', dict(identity=identity, results=results, held_rows=len(held),
        training_rows=len(train), training_hard_cut=hard_cut, held_rows_sha256=array_hash(held),
        held_recordings=len(set(data.source_records[loc])), held_scoped_agents=len(set(data.source_tracks[loc])),
        physical_sites=1, independent_confirmation=False, main_primary_changed=False,
        sealed_roles_opened=False, new_deployment=False, stage5c_executed=False, smc_enabled=False))
    beat(state='fixed_source_evaluation_complete', predictors=len(results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--trial')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    if args.evaluate and args.replay:
        raise ValueError('Choose scoring or exact replay, not both')
    if (args.evaluate or args.replay) and (args.trial or args.stop_at is not None or args.audit_only):
        raise ValueError('Evaluation cannot alter training or pilot budget')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(out/'heartbeat.json', event)
        print(json.dumps(event), flush=True)
    beat(state='verifying_assets')
    data, train, weights, held, scale, parents, rgb_report = build_data(reg)
    identity = dict(registration_sha256=file_digest(args.registration), data_identity=data.identity,
        source_assignment_sha256=data.assignment_hash, training_rows_sha256=array_hash(train, weights, data.target[train-data.nmain]),
        normalizer_sha256=array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant']),
        held_rows_sha256=array_hash(held), feature_width=480, training_cost_scale=scale,
        torch=torch.__version__, numpy=np.__version__, torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(out/'identity.json', identity)
    immutable_json(public/'input_checks.json', dict(identity=identity, training_rows=len(train), excluded_rows=len(held),
        no_recording_or_scoped_track_overlap=True, held_site_previously_explored=True,
        evaluation_is_independent_confirmation=False, input_role='offline_past_supplied_annotations',
        new_training_arm='mask_only', reused_arm='past_rgb', labels_in_input=False,
        original_source_role='SDD_train40_only', main_primary_changed=False, new_deployment=False))
    if args.audit_only:
        beat(state='audit_complete_no_training_or_forecast')
    elif args.evaluate or args.replay:
        evaluate(reg, identity, data, train, held, parents, rgb_report, out, public, beat, args.replay)
    else:
        train_controls(reg, identity, data, train, weights, scale, parents, out, public, beat, args)


if __name__ == '__main__':
    main()
