"""Fixed training-only microfit: isolate decoder scale before another benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_cost_dynamics import DynamicsCorpus, load_config as load_parent
import numpy as np
import torch
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_source_microfit import MicrofitDynamics, DECODERS, fit_micro, select_cohorts


def load_config(path):
    reg=json.loads(path.read_text())
    if (reg['registration_path']!=str(path) or reg['role']!='source_training_only_microfit'
            or reg['decoders']!=list(DECODERS) or reg['cohorts']!=['nonzero_only','mixed_zero']
            or reg['seeds']!=[17,29,43] or reg['excluded_site']!='bookstore'
            or reg['held_evaluation'] or reg['new_deployment'] or not reg['bindings']
            or reg['models']!=12 or reg['total_updates']!=24000 or reg['training']['updates']!=2000):
        raise ValueError('Fixed training-only microfit registration required')
    for path,digest in reg['bindings'].items():
        if file_digest(ROOT/path)!=digest:raise ValueError('Registered dependency changed: '+path)
    return reg


def build_data(reg):
    parent=load_parent(Path(reg['parent_registration']));data=DynamicsCorpus(parent)
    train,w,held,_,scale=data.dynamics_design(reg['excluded_site']);loc=train-data.nmain
    target=data.target[loc];cv=np.linalg.norm(target.astype(float),axis=-1).mean(1)
    sets=select_cohorts(train,data.source_tracks[loc],cv,np.linalg.norm(target,axis=-1).max(1),
        data.radius[loc],data.support[loc],count=reg['nonzero_rows'],seed=reg['selection_seed'])
    payloads={};audit=[]
    for name,ids in sets.items():
        assert np.isin(ids,train).all() and not np.intersect1d(ids,held).size
        features,frame=data.dynamics_inputs(ids,training=True);labels=data.loss_targets(ids)
        payloads[name]=(ids,features,frame,labels)
        local=ids-data.nmain;cost=torch.linalg.vector_norm(labels,dim=-1).mean(1)
        audit.append(dict(cohort=name,rows=len(ids),unique_scoped_tracks=len(set(data.source_tracks[local])),
            zero_rows=int((cost==0).sum()),nonzero_rows=int((cost>0).sum()),
            sites={s:int((data.source_sites[local]==s).sum()) for s in sorted(set(data.source_sites[local]))},
            rows_sha256=array_hash(ids),features_sha256=array_hash(*[v.numpy() for v in features]),
            frame_sha256=array_hash(*[v.numpy() for v in frame]),labels_sha256=array_hash(labels.numpy()),
            cv_ade=float(cost.mean()),context_radius_min=float(frame[0].min()),
            context_radius_median=float(frame[0].median()),context_radius_max=float(frame[0].max()),
            target_extent_over_radius_max=float((torch.linalg.vector_norm(labels,dim=-1).max(1).values/frame[0]).max()),
            training_only=True,held_rows_scored=0,selection_uses_training_labels=True))
    return data,scale,payloads,audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--trial');p.add_argument('--stop-at',type=int)
    p.add_argument('--replay',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);out=ROOT/reg['output'];reports=ROOT/reg['reports']
    def beat(**kwargs):
        row=dict(pid=os.getpid(),timestamp_unix=time.time(),**kwargs);json_write(out/'heartbeat.json',row)
        print(json.dumps(row),flush=True)
    beat(state='verifying_training_assets');data,scale,payloads,audit=build_data(reg)
    identity=dict(registration_sha256=file_digest(args.registration),source_assignment_sha256=data.assignment_hash,
        parent_identity=data.identity,training_cost_scale=scale,torch=torch.__version__,numpy=np.__version__,
        torch_threads=4,interop_threads=1,num_workers=0,cohorts=audit)
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Microfit input/runtime identity changed')
    json_write(ip,identity);json_write(reports/'input_checks.json',dict(result_source='fresh_run_training_only_audit',
        identity=identity,held_source_prediction=False,main_training=False,sealed_roles_opened=False,
        primary_changed=False,new_deployment=False))
    if args.audit_only:beat(state='audit_complete_no_training',rows=[a['rows'] for a in audit]);return
    keys={f'{cohort}_{decoder}_seed{seed}' for cohort in reg['cohorts'] for decoder in reg['decoders'] for seed in reg['seeds']}
    if args.trial and args.trial not in keys:raise ValueError('Unregistered microfit trial')
    if args.stop_at is not None and (not args.trial or args.replay):raise ValueError('Pilot names one training-only trial')
    trials=[];replays=[];new_updates=new_fits=0
    for cohort in reg['cohorts']:
        ids,features,frame,target=payloads[cohort]
        for seed in reg['seeds']:
            for decoder in reg['decoders']:
                key=f'{cohort}_{decoder}_seed{seed}'
                if args.trial and key!=args.trial:continue
                ti=dict(identity,cohort=cohort,seed=seed,decoder=decoder)
                cp=out/'checkpoints'/f'{key}.pt';pp=out/'predictions'/f'{key}.npz';rp=out/'trials'/f'{key}.json'
                old=json.loads(rp.read_text()) if rp.exists() else None
                if old:
                    if (old['identity']!=ti or file_digest(cp)!=old['checkpoint_sha256']
                            or file_digest(pp)!=old['prediction_sha256']):raise ValueError('Completed microfit changed')
                    if not args.replay:trials.append(old);continue
                beat(state='fit_or_replay',trial=key,rows=len(ids));torch.manual_seed(seed);model=MicrofitDynamics()
                if args.replay:
                    if old is None:raise ValueError('Replay requires completed trial')
                    state=torch.load(cp,map_location='cpu',weights_only=False)
                    assert state['identity']==ti and state['step']==reg['training']['updates']
                    model.load_state_dict(state['model']);model.eval()
                    with torch.no_grad():prediction=model.trajectory(features,frame,decoder,scale).numpy()
                    with np.load(pp,allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['prediction'],prediction)
                    replays.append(key);continue
                prediction,fit=fit_micro(model,features,frame,target,decoder=decoder,scale=scale,
                    config=reg['training'],identity=ti,checkpoint=cp,
                    heartbeat=lambda **v:beat(trial=key,**v),stop_at=args.stop_at)
                new_updates+=fit['new_updates']
                if not fit['complete']:
                    beat(state='pilot_complete_training_only',trial=key,step=fit['step'],fit_seconds=fit['fit_seconds']);return
                pp.parent.mkdir(parents=True,exist_ok=True);tmp=pp.with_suffix('.tmp.npz')
                np.savez(tmp,ids=ids,prediction=prediction);os.replace(tmp,pp)
                result=dict(trial=key,identity=ti,cohort=cohort,decoder=decoder,seed=seed,
                    result_source='fresh_run_training_only_not_generalization',fit=fit,
                    checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                    prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                json_write(rp,result);trials.append(result);new_fits+=1
                beat(state='trial_complete',trial=key,training_gain_percent=fit['final']['gain_percent'])
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,exact_trials=replays,all_exact=True,new_updates=0))
        beat(state='replay_complete',trials=len(replays));return
    if args.trial:return
    assert len(trials)==reg['models']
    rp=reports/'report.json'
    if rp.exists() and new_fits==0:
        prior=json.loads(rp.read_text());assert prior['identity']==identity and prior['trials']==trials
        beat(state='completed_resume_verified',new_fits=0,new_updates=0);return
    json_write(rp,dict(identity=identity,result_source='fresh_run_training_only_not_generalization',
        completed_models=len(trials),optimizer_updates=sum(t['fit']['step'] for t in trials),trials=trials,
        summed_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials),held_rows_scored=0,main_primary_changed=False,
        sealed_roles_opened=False,new_deployment=False,stage5c_executed=False,smc_enabled=False))
    beat(state='microfit_complete',new_fits=new_fits,new_updates=new_updates)


if __name__=='__main__':main()
