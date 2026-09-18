"""Training-side exact-baseline head comparison; held/main forecasting is prohibited."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The inherited entry enforces native arm64 before importing Torch.
from scripts.run_m3w_source_transfer_control import load_config as load_control, build_data as build_control
from scripts.run_m3w_source_continuation import immutable_json
from scripts.run_m3w_source_cost_dynamics import trajectory_metrics
import numpy as np
import torch
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_source_cost_deferral import CostDeferral, VARIANTS, fit_deferral, predict_deferral


def load_config(path):
    reg = json.loads(path.read_text())
    if (reg['registration_path'] != str(path) or reg['role'] != 'source_training_only_cost_deferral'
            or reg['variants'] != list(VARIANTS) or reg['seeds'] != [17,29,43]
            or reg['excluded_site'] != 'bookstore' or reg['arm'] != 'mask_only'
            or reg['new_branches'] != 6 or reg['additional_updates'] != 48000
            or reg['hard_score_threshold'] != 0 or reg['held_evaluation'] or reg['new_deployment']
            or reg['main_primary_changed'] or not reg['bindings']):
        raise ValueError('Fixed training-only deferral experiment required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Frozen dependency changed: '+name)
    control = load_control(Path(reg['control_registration']))
    if reg['training'] != control['training']:
        raise ValueError('Exact inherited cosine control budget required')
    return reg


def build_data(reg):
    control_reg = load_control(Path(reg['control_registration']))
    data, train, weights, held, scale, parents, _ = build_control(control_reg)
    control = json.loads((ROOT/reg['control_report']).read_text())
    controls = {t['seed']:t for t in control['trials'] if t['schedule'] == 'cosine'}
    assert set(controls) == set(reg['seeds'])
    for trial in controls.values():
        assert trial['arm'] == 'mask_only'
        assert file_digest(ROOT/trial['checkpoint_path']) == trial['checkpoint_sha256']
        for milestone in trial['milestones']:
            for kind in ('checkpoint','prediction','receipt'):
                assert file_digest(ROOT/milestone[kind+'_path']) == milestone[kind+'_sha256']
    assert not np.intersect1d(train, held).size
    return data, train, weights, scale, {s:parents['mask_only',s] for s in reg['seeds']}, controls


def training_metrics(proposal, score, target, native, hard_cut, scale, variant):
    proposal, target = np.asarray(proposal, float), np.asarray(target, float)
    if score.shape != (len(target),) or not np.isfinite(score).all():
        raise ValueError('Finite aligned observed-input scores required')
    use = score > 0
    hard = np.where(use[:,None,None], proposal, 0)
    metrics = {}
    for name, value, requested in [('proposal',proposal,np.ones(len(target),bool)), ('hard_action',hard,use)]:
        metrics[name] = trajectory_metrics(value, target, native, requested, hard_cut)
        metrics[name]['actual_changed_rate'] = float(np.any(value != 0, axis=(1,2)).mean())
    error = np.linalg.norm(proposal-target,axis=-1).mean(1)
    baseline = np.linalg.norm(target,axis=-1).mean(1)
    p = torch.from_numpy(score.copy()).sigmoid().numpy()
    gain = (baseline-error)/scale
    metrics['expected_action_risk'] = float(((1-p)*baseline+p*error).mean())
    metrics['expected_action_gain_percent'] = float(100*(1-metrics['expected_action_risk']/baseline.mean()))
    metrics['mean_soft_gate'] = float(p.mean())
    metrics['gate_target_mae'] = float(np.abs(score-gain).mean()) if variant == 'cost_supervised' else None
    metrics['gate_target_rmse'] = float(np.sqrt(np.mean((score-gain)**2))) if variant == 'cost_supervised' else None
    metrics['signed_gain_decision_accuracy'] = float(((score>0)==(gain>0)).mean())
    metrics['requested_gain_rows'] = int((use & (gain>0)).sum())
    metrics['requested_harm_rows'] = int((use & (gain<0)).sum())
    metrics['candidate_helpful_rows'] = int((gain>0).sum())
    metrics['zero_action_rows'] = int((~np.any(hard!=0,axis=(1,2))).sum())
    metrics['sigmoid_is_calibrated_probability'] = False
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--trial')
    p.add_argument('--stop-at', type=int)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot belongs to one registered training branch')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    def beat(**values):
        value = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(out/'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    beat(state='verifying_training_assets')
    data, train, weights, scale, parents, controls = build_data(reg)
    loc = train-data.nmain
    target = data.target[loc]
    cv = np.linalg.norm(target.astype(float),axis=-1).mean(1)
    hard_cut = float(np.quantile(cv,.9))
    identity = dict(registration_sha256=file_digest(args.registration), data_identity=data.identity,
        training_rows_sha256=array_hash(train,weights,target), source_assignment_sha256=data.assignment_hash,
        normalizer_sha256=array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant']),
        training_cost_scale=scale, parent_checkpoints={str(s):r['checkpoint_sha256'] for s,(r,_) in parents.items()},
        torch=torch.__version__, numpy=np.__version__, torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(out/'identity.json',identity)
    immutable_json(public/'input_checks.json',dict(identity=identity, training_rows=len(train),
        zero_target_rows=int((cv==0).sum()), nonzero_rows=int((cv>0).sum()),
        source_sites=sorted(set(data.source_sites[loc])), source_recordings=len(set(data.source_records[loc])),
        scoped_agents=len(set(data.source_tracks[loc])), excluded_source_site=reg['excluded_site'],
        held_rows_scored=0, main_rows_scored=0, labels_in_input=False, sealed_roles_opened=False,
        data_role='supervised_training_with_training_only_diagnostics',
        observation='offline_supplied_annotations_not_sensor_asof',
        units='past_normalized_and_annotation_pixels_raw_frame_only', new_deployment=False))
    if args.audit_only:
        beat(state='audit_complete_no_training_or_scoring')
        return
    keys = {f'{v}_seed{s}' for v in reg['variants'] for s in reg['seeds']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    trials, replayed = [], []
    new_updates = new_branches = 0
    for seed in reg['seeds']:
        parent_result,parent = parents[seed]
        for variant in reg['variants']:
            key = f'{variant}_seed{seed}'
            if args.trial and args.trial != key:
                continue
            ti = dict(identity,seed=seed,variant=variant,parent_checkpoint_sha256=parent_result['checkpoint_sha256'])
            cp, rp = out/'checkpoints'/f'{key}.pt',out/'trials'/f'{key}.json'
            old = json.loads(rp.read_text()) if rp.exists() else None
            if old:
                assert old['identity'] == ti and file_digest(cp) == old['checkpoint_sha256']
                for m in old['milestones']:
                    for kind in ('checkpoint','prediction','receipt'):
                        assert file_digest(ROOT/m[kind+'_path']) == m[kind+'_sha256']
                if not args.replay:
                    trials.append(old)
                    continue
            elif args.replay:
                raise ValueError('Replay requires completed training')
            torch.manual_seed(seed)
            model = CostDeferral()
            if args.replay:
                for m in old['milestones']:
                    state = torch.load(ROOT/m['checkpoint_path'],map_location='cpu',weights_only=False)
                    assert state['identity'] == ti
                    model.load_state_dict(state['model'])
                    proposal,score = predict_deferral(model,lambda ids:data.dynamics_inputs(ids,training=True),train)
                    with np.load(ROOT/m['prediction_path'],allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['ids'],train)
                        np.testing.assert_array_equal(a['proposal'],proposal)
                        np.testing.assert_array_equal(a['score'],score)
                    replayed.append(f"{key}:{m['step']}")
                    beat(state='milestone_replayed',trial=key,step=m['step'])
                continue
            def capture(state):
                step = state['step']
                sp = out/'milestones'/f'{key}_step{step}.pt'
                pp, mp = sp.with_suffix('.npz'),sp.with_suffix('.json')
                if mp.exists():
                    saved = json.loads(mp.read_text())
                    assert saved['identity'] == ti and saved['step'] == step
                    assert saved['checkpoint_sha256'] == file_digest(sp) and saved['prediction_sha256'] == file_digest(pp)
                    prior = torch.load(sp,map_location='cpu',weights_only=False)
                    for name in state['model']:
                        assert torch.equal(state['model'][name],prior['model'][name])
                    return
                proposal,score = predict_deferral(model,lambda ids:data.dynamics_inputs(ids,training=True),train)
                metrics = training_metrics(proposal,score,target,data.native_scale[loc],hard_cut,scale,variant)
                if step == 2000:
                    original = controls[seed]['milestones'][0]
                    with np.load(ROOT/original['prediction_path'],allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['ids'],train)
                        np.testing.assert_array_equal(a['prediction'],proposal)
                    assert not score.any() and metrics['hard_action']['gain_percent'] == 0
                sp.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(cp,sp)
                temp = pp.with_suffix('.tmp.npz')
                np.savez(temp,ids=train,proposal=proposal,score=score)
                os.replace(temp,pp)
                json_write(mp,dict(identity=ti,step=step,metrics=metrics,
                    checkpoint_sha256=file_digest(sp),prediction_sha256=file_digest(pp)))
                beat(state='training_milestone',trial=key,step=step,
                    hard_gain_percent=metrics['hard_action']['gain_percent'],
                    proposal_gain_percent=metrics['proposal']['gain_percent'],
                    actual_changed_rate=metrics['hard_action']['actual_changed_rate'])
            beat(state='continue_deferral',trial=key)
            fit = fit_deferral(model,lambda ids:data.dynamics_inputs(ids,training=True),data.loss_targets,
                train,weights,parent=parent,variant=variant,config=reg['training'],identity=ti,checkpoint=cp,
                heartbeat=lambda **values:beat(trial=key,**values),snapshot=capture,stop_at=args.stop_at)
            new_updates += fit['new_updates']
            if not fit['complete']:
                beat(state='pilot_complete_training_only',trial=key,**fit)
                return
            milestones=[]
            for step in reg['training']['milestones']:
                sp=out/'milestones'/f'{key}_step{step}.pt'
                paths=dict(checkpoint=sp,prediction=sp.with_suffix('.npz'),receipt=sp.with_suffix('.json'))
                row=dict(step=step,metrics=json.loads(paths['receipt'].read_text())['metrics'])
                for kind,path in paths.items():
                    row[kind+'_path'],row[kind+'_sha256']=str(path.relative_to(ROOT)),file_digest(path)
                milestones.append(row)
            trial=dict(identity=ti,trial=key,seed=seed,variant=variant,fit=fit,milestones=milestones,
                parameters=sum(v.numel() for v in model.parameters()),
                checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                result_source='fresh_run_continuation_cached_verified_parent')
            json_write(rp,trial)
            trials.append(trial)
            new_branches += 1
            beat(state='branch_complete',trial=key,hard_gain_percent=milestones[-1]['metrics']['hard_action']['gain_percent'])
    if args.replay:
        assert len(replayed) == 24
        json_write(public/'replay.json',dict(identity=identity,all_exact=True,replayed=replayed,new_updates=0))
        beat(state='replay_complete',count=len(replayed))
    elif not args.trial:
        assert len(trials)==6
        immutable_json(public/'report.json',dict(identity=identity,trials=trials,
            completed_branches=6,additional_updates=sum(t['fit']['additional_updates'] for t in trials),
            unique_parent_models=3,inherited_unique_updates=6000,cached_dense_controls=3,
            summed_continuation_seconds=sum(t['fit']['continuation_seconds'] for t in trials),
            held_rows_scored=0,main_rows_scored=0,independent_confirmation=False,new_deployment=False))
        beat(state='training_complete' if new_branches else 'completed_resume_verified',
             new_branches=new_branches,new_updates=new_updates)


if __name__=='__main__':
    main()
