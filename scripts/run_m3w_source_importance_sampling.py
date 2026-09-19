"""Fixed importance-corrected exposure experiment; no main/outer evaluation."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_episode_sampler import setup, ARMS
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays, array_hash, metrics
from src.world_model.m3w_source_importance_sampling import uniform_risk_factors, fit_importance_candidate, OBJECTIVE
from src.world_model.m3w_source_cost_dynamics import forecast
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
from src.world_model.m3w_source_temporal_centered import CenteredTemporalDynamics
from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def load_config(path):
    reg = json.loads(path.read_text())
    if (str(path) != reg['registration_path'] or reg['role'] != 'training_only_uniform_risk_importance_repair'
            or reg['arms'] != list(ARMS) or reg['seeds'] != [17,29,43]
            or reg['sites'] != ['coupa','deathCircle','gates','hyang']
            or reg['models'] != 24 or reg['updates'] != 240000
            or reg['objective'] != OBJECTIVE or reg['selection'] or reg['new_deployment']
            or reg['main_primary_changed'] or reg['stage5c_executed'] or reg['smc_enabled']):
        raise ValueError('Fixed source-only importance-correction contract required')
    for path,digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest: raise ValueError('Changed dependency: '+path)
    return reg


def context(reg):
    parent,data,cached,uniform = setup(reg)
    report = json.loads((ROOT/reg['episode_control_report']).read_text())
    verification = json.loads((ROOT/reg['episode_control_report']).with_name('verification.json').read_text())
    for path,digest in verification['artifact_hashes'].items():
        if file_digest(ROOT/path) != digest: raise ValueError('Changed episode control: '+path)
    audit = json.loads((ROOT/reg['event_audit']).read_text())
    ep = ROOT/audit['row_archive']['path']
    assert file_digest(ep) == audit['row_archive']['sha256']
    with np.load(ep,allow_pickle=False) as a: ids,groups = a['ids'].copy(),a['episodes'].copy()
    np.testing.assert_array_equal(ids,cached.ids)
    return parent,data,cached,uniform,report,ids,groups,file_digest(ep)


def objective_check(reg,data,ids,groups):
    folds = []
    for site in reg['sites']:
        train,_,_,_,scale = data.configure(site)
        p,support = episode_weights(train,ids,groups); factor = uniform_risk_factors(p)
        probability = torch.from_numpy(p/p.sum()); correction = torch.from_numpy(factor)
        target = torch.from_numpy(data.target[train-data.nmain].astype(np.float64))
        theta = torch.tensor([.013,-.021],dtype=torch.float64,requires_grad=True)
        per_row = torch.linalg.vector_norm(theta-target,dim=-1).mean(1)/scale
        reference = per_row.mean(); exact = (probability*correction*per_row).sum()
        ga = torch.autograd.grad(reference,theta,retain_graph=True)[0]
        gb = torch.autograd.grad(exact,theta)[0]
        torch.testing.assert_close(reference,exact,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(p/p.sum()*factor,np.full(len(train),1/len(train)),atol=1e-18,rtol=1e-12)
        folds.append(dict(site=site,**support,train_ids_sha256=array_hash(train),sampler_sha256=array_hash(train,p),
            importance_sha256=array_hash(train,factor),factor_min=float(factor.min()),factor_max=float(factor.max()),
            expected_factor=float(probability@correction),factor_second_moment=float(probability@correction.square()),
            asymptotic_importance_ess_fraction=float(1/(probability@correction.square())),
            reference_probe_loss=float(reference.detach()),expected_corrected_probe_loss=float(exact.detach()),
            loss_absolute_error=float(abs(reference-exact).detach()),gradient_max_error=float(abs(ga-gb).max()),
            probe_is_fixed_shared_offset_not_a_fitted_model=True))
    return dict(result_source='fresh_run_train_only_exact_expectation_checks',folds=folds,
        objective=OBJECTIVE,new_training_updates=0,held_labels_used=False,main_outer_rows_scored=0,
        new_deployment=False,unbiased_statement_applies_to_unclipped_gradient_not_Adam_update=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--check-objective',action='store_true'); p.add_argument('--replay',action='store_true')
    p.add_argument('--trial'); p.add_argument('--stop-at',type=int)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    if args.stop_at is not None and (not args.trial or args.replay or args.check_objective):
        raise ValueError('Named training pilot required')
    parent,data,cached,uniform,control,ids,groups,ep_sha = context(reg)
    private,public = ROOT/reg['output'],ROOT/reg['reports']
    preflight = objective_check(reg,data,ids,groups)
    preflight['registration_sha256'] = file_digest(args.registration)
    immutable_json(public/'objective_check.json',preflight)
    if args.check_objective:
        print(json.dumps(preflight,indent=2)); return
    identity = dict(registration_sha256=file_digest(args.registration),data=data.identity,assignment=data.assignment_hash,
        feature_store_sha256=file_digest(ROOT/parent['output']/'feature_store.npz'),episode_mapping_sha256=ep_sha,
        objective=OBJECTIVE,torch=torch.__version__,numpy=np.__version__,machine=platform.machine(),
        torch_threads=4,interop_threads=1,num_workers=0)
    immutable_json(private/'identity.json',identity)
    def beat(**kw):
        event = dict(pid=os.getpid(),timestamp_unix=time.time(),**kw)
        json_write(private/'heartbeat.json',event); print(json.dumps(event),flush=True)
    beat(state='verified_inputs_and_expected_objective')
    allowed = {f'{site}_{arm}_seed{seed}' for site in reg['sites'] for arm in ARMS for seed in reg['seeds']}
    if args.trial and args.trial not in allowed: raise ValueError('Unknown fixed trial')
    trials,checks,replays,updates = [],[],[],0
    for site in reg['sites']:
        train,_,held,outer,scale = data.configure(site)
        probs,support = episode_weights(train,ids,groups); factor = uniform_risk_factors(probs)
        sampler_sha = array_hash(train,probs); importance_sha = array_hash(train,factor)
        old = next(t for t in control['trials'] if t['site'] == site)
        fold,hard = old['identity']['fold'],old['identity']['training_hard_cut']
        assert array_hash(train) == fold['training_ids_sha256'] and array_hash(held) == fold['held_ids_sha256']
        assert array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant']) == fold['normalizer_sha256']
        assert scale == fold['cost_scale'] and sampler_sha == old['identity']['sampler_sha256']
        assert not set(groups[np.searchsorted(ids,train)]) & set(groups[np.searchsorted(ids,held)])
        for illegal in (np.array([0]),outer[:1]):
            try: cached.inputs(illegal)
            except ValueError: pass
            else: raise AssertionError('Prohibited role reached input')
        try: cached.inputs(held[:1],training=True)
        except ValueError: pass
        else: raise AssertionError('Held input reached trainer')
        before = cached.inputs(held[:8]); saved_labels = data.target.copy()
        try: data.target[:] = np.nan; after = cached.inputs(held[:8])
        finally: data.target[:] = saved_labels
        assert all(torch.equal(a,b) for x,y in zip(before,after) for a,b in zip(x,y))
        checks.append(dict(site=site,fold=fold,target_poison_rows=8,sampler_sha256=sampler_sha,
            importance_sha256=importance_sha,**support))
        for seed in reg['seeds']:
            rng = torch.Generator().manual_seed(seed+7919); draws = np.zeros(len(train),np.int64)
            for _ in range(reg['training']['updates']):
                local = torch.multinomial(torch.from_numpy(probs),reg['training']['batch_size'],replacement=True,generator=rng).numpy()
                np.add.at(draws,local,1)
            for arm in ARMS:
                key = f'{site}_{arm}_seed{seed}'
                if args.trial and key != args.trial: continue
                ti = dict(identity,site=site,seed=seed,arm=arm,fold=fold,training_hard_cut=hard,
                    sampler_sha256=sampler_sha,importance_sha256=importance_sha,**support)
                cp,pp,rp = private/'checkpoints'/f'{key}.pt',private/'predictions'/f'{key}.npz',private/'trials'/f'{key}.json'
                receipt = json.loads(rp.read_text()) if rp.exists() else None
                if receipt:
                    assert receipt['identity'] == ti and file_digest(cp) == receipt['checkpoint_sha256']
                    assert file_digest(pp) == receipt['prediction_sha256']
                    if not args.replay: trials.append(receipt); continue
                elif args.replay: raise ValueError('Missing completed trial')
                torch.manual_seed(seed)
                model = TemporalSourceDynamics('geometry') if arm == 'geometry' else CenteredTemporalDynamics('centered')
                if not args.replay:
                    beat(state='fit_or_resume',trial=key)
                    fit = fit_importance_candidate(model,lambda q:cached.inputs(q,training=True),data.loss_targets,
                        train,probs,scale=scale,seed=seed,config=reg['training'],identity=ti,checkpoint=cp,
                        heartbeat=lambda **kw:beat(trial=key,**kw),stop_at=args.stop_at)
                    updates += fit['new_updates']
                    if not fit['complete']: beat(state='pilot_complete_no_forecast',trial=key,step=fit['step']); return
                saved = torch.load(cp,map_location='cpu',weights_only=False)
                assert saved['identity'] == ti and saved['step'] == 10000 and saved['objective'] == OBJECTIVE
                assert torch.equal(saved['sampler_rng'],rng.get_state())
                np.testing.assert_array_equal(saved['draw_counts'],draws)
                np.testing.assert_array_equal(saved['train_ids'],train)
                np.testing.assert_array_equal(saved['importance_factors'],factor)
                old = next(t for t in control['trials'] if t['trial'] == key)
                previous = torch.load(ROOT/old['checkpoint_path'],map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(saved['draw_counts'],previous['draw_counts'])
                assert torch.equal(saved['sampler_rng'],previous['sampler_rng'])
                model.load_state_dict(saved['model'])
                arrays = dict(train_ids=train,held_ids=held,train_prediction=forecast(model,cached.inputs,train,'mask_only'),
                    held_prediction=forecast(model,cached.inputs,held,'mask_only'))
                if args.replay:
                    with np.load(pp,allow_pickle=False) as a:
                        for name,value in arrays.items(): np.testing.assert_array_equal(a[name],value)
                    replays.append(key); beat(state='exact_replay',trial=key); continue
                save_arrays(pp,arrays)
                trial = dict(identity=ti,trial=key,site=site,seed=seed,arm=arm,fit=fit,
                    result_source='fresh_run_torch_importance_corrected_exposure',parameters=sum(v.numel() for v in model.parameters()),
                    training=metrics(data,train,arrays['train_prediction'],scale,hard),
                    held=metrics(data,held,arrays['held_prediction'],scale,hard),
                    checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                    prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp),
                    matches_uncorrected_control_draws=True,sampled_draws_sha256=array_hash(saved['draw_counts']))
                immutable_json(rp,trial); trials.append(trial); beat(state='trial_complete',trial=key)
    if args.trial: return
    immutable_json(public/'input_checks.json',dict(identity=identity,folds=checks,main_outer_scored=0,
        sensor_asof_certified=False,target_in_features=False,all_rows_retained=True,
        sampling_counts_use_training_only=True,sampler_groups_use_past_only=True))
    if args.replay:
        assert len(replays) == 24
        immutable_json(public/'replay.json',dict(exact_replays=replays,new_updates=0))
        beat(state='replay_complete',models=24); return
    assert len(trials) == 24 and sum(t['fit']['step'] for t in trials) == 240000
    archives = []
    for arm in ARMS:
        for seed in reg['seeds']:
            pieces = []
            for trial in trials:
                if trial['arm'] != arm or trial['seed'] != seed: continue
                with np.load(ROOT/trial['prediction_path'],allow_pickle=False) as a:
                    pieces.append(dict(ids=a['held_ids'].copy(),prediction=a['held_prediction'].copy(),
                        cost_scale=np.full(len(a['held_ids']),trial['identity']['fold']['cost_scale'])))
            prediction,scale = assemble_oof(cached.ids,pieces)
            labels = cost_labels(prediction,data.target[cached.ids-data.nmain],scale)
            path = private/'oof'/f'{arm}_seed{seed}.npz'
            save_arrays(path,dict(ids=cached.ids,prediction=prediction,cost_scale=scale,**labels))
            archives.append(dict(arm=arm,seed=seed,path=str(path.relative_to(ROOT)),sha256=file_digest(path)))
    immutable_json(public/'report.json',dict(identity=identity,trials=trials,oof_labels=archives,models=24,
        optimizer_updates=240000,rows=len(cached.ids),new_policy=False,new_deployment=False,
        main_rows_scored=0,outer_rows_scored=0,stage5c_executed=False,smc_enabled=False))
    beat(state='training_complete',models=24,new_updates=updates)


if __name__ == '__main__': main()
