"""Fixed source-site trajectory objective comparison, not deployment selection."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_site_probe import SiteCorpus, SITES
import numpy as np
import torch
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_observed_unit_frame_v2 import observed_unit_frame
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_source_cost_dynamics import SourceDynamics, OBJECTIVES, forecast, fit_dynamics


class DynamicsCorpus(SiteCorpus):
    def __init__(self,reg):
        super().__init__(reg)
        x,r,q,s=observed_unit_frame(self.aux['geometry'][self.sid])
        np.testing.assert_array_equal(x,self.x[self.nmain:])
        self.radius,self.rotation,self.support=r,q,s
        self.target=self.aux['target'][self.sid].copy()
        if np.count_nonzero(self.aux['baseline'][self.sid]):
            raise ValueError('Stationary CV must equal zero in query coordinates')
        self.x=np.concatenate((self.x,np.zeros((len(self.x),4),np.float32)),axis=1)
        self.x[self.nmain:,476:]=q.reshape(-1,4)
        self.native_scale=np.empty(len(self.sid))
        receipts=json.loads(self.manifest_path.read_text())['records']
        for record in np.unique(self.record_ids[self.sid]):
            loc=np.flatnonzero(self.record_ids[self.sid]==record)
            receipt=receipts[int(record)];path=self.manifest_path.parent/receipt['recording']/'scale.npy'
            if file_digest(path)!=receipt['arrays']['scale.npy']:
                raise ValueError('Unverified coordinate scale')
            self.native_scale[loc]=np.load(path,mmap_mode='r')[self.local_ids[self.sid[loc]]]
        if not np.isfinite(self.target).all() or not np.all(self.native_scale>0):
            raise ValueError('Finite complete source supervision required')

    def dynamics_design(self,site):
        train,w,held,clipped=self.source_design(site)
        normalizer=float(w @ np.linalg.norm(self.target[train-self.nmain].astype(float),axis=-1).mean(1))
        if not np.isfinite(normalizer) or normalizer<=0:
            raise ValueError('Positive training-only mean CV ADE required')
        return train,w,held,clipped,normalizer

    def dynamics_inputs(self,ids,*,training=False):
        ids=np.asarray(ids,dtype=int)
        if ids.ndim!=1 or np.any(ids<self.nmain) or np.any(ids>=len(self.y)):
            raise ValueError('Source-only trajectory query ids required')
        loc=ids-self.nmain
        features=self.inputs(ids,training=training)
        frame=(torch.from_numpy(self.radius[loc].copy()),torch.from_numpy(self.rotation[loc].copy()),
               torch.from_numpy(self.support[loc].copy()))
        return features,frame

    def loss_targets(self,ids):
        ids=np.asarray(ids,dtype=int)
        if (ids.ndim!=1 or np.any(ids<self.nmain) or np.any(ids>=len(self.y))
                or not self.allowed[ids].all()):
            raise ValueError('Held/main target requested by trainer')
        return torch.from_numpy(self.target[ids-self.nmain].copy())


def trajectory_metrics(prediction,target,native_scale,gate,hard_cut):
    prediction,target=map(lambda a:np.asarray(a,dtype=float),(prediction,target))
    native_scale,gate=np.asarray(native_scale),np.asarray(gate)
    if (prediction.ndim!=3 or prediction.shape[1:]!=(12,2) or target.shape!=prediction.shape
            or native_scale.shape!=(len(target),) or gate.shape!=(len(target),)
            or not all(np.isfinite(a).all() for a in (prediction,target,native_scale))
            or np.any(native_scale<=0) or not np.isfinite(hard_cut) or hard_cut<0):
        raise ValueError('Finite aligned predictions, labels and legal metadata required')
    baseline=np.linalg.norm(target,axis=-1).mean(1)
    ade=np.linalg.norm(prediction-target,axis=-1).mean(1)
    fde=np.linalg.norm(prediction[:,-1]-target[:,-1],axis=-1)
    easy=baseline==0;hard=baseline>=hard_cut
    return dict(rows=len(ade),ade=float(ade.mean()),fde=float(fde.mean()),cv_ade=float(baseline.mean()),
        gain_percent=100*float((baseline.mean()-ade.mean())/baseline.mean()),
        native_pixel_ade=float((ade*native_scale).mean()),
        native_pixel_cv_ade=float((baseline*native_scale).mean()),
        hard_rows=int(hard.sum()),hard_ade=float(ade[hard].mean()) if hard.any() else None,
        hard_cv_ade=float(baseline[hard].mean()) if hard.any() else None,
        easy_rows=int(easy.sum()),easy_absolute_harm=float(ade[easy].mean()) if easy.any() else None,
        easy_native_pixel_harm=float((ade[easy]*native_scale[easy]).mean()) if easy.any() else None,
        easy_percentage_degradation=None,easy_percentage_reason='zero_error_CV_denominator',
        easy_nonzero_predictions=int((ade[easy]>0).sum()),
        intervention_rate=float(np.asarray(gate,dtype=bool).mean()),
        oracle_binary_ade=float(np.minimum(ade,baseline).mean()),
        harm_over_cv=float((ade-baseline).mean()))


def load_config(path):
    reg=json.loads(path.read_text())
    if (str(path)!=reg['registration_path'] or reg['role']!='source_fit_trajectory_diagnostic'
            or reg['sites']!=list(SITES) or reg['objectives']!=list(OBJECTIVES)
            or reg['arms']!=['mask_only','past_rgb'] or reg['seeds']!=[17,29,43]
            or reg['model_or_threshold_selection']):
        raise ValueError('Fixed source trajectory diagnostic required')
    for p,sha in reg['bindings'].items():
        if file_digest(ROOT/p)!=sha:
            raise ValueError('Registered dependency changed: '+p)
    return reg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--trial');parser.add_argument('--stop-at',type=int)
    parser.add_argument('--audit-only',action='store_true');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);out,reports=ROOT/reg['output'],ROOT/reg['reports']
    def heartbeat(**values):
        value=dict(pid=os.getpid(),timestamp_unix=time.time(),**values)
        json_write(out/'heartbeat.json',value);print(json.dumps(value),flush=True)
    heartbeat(state='verifying_assets');data=DynamicsCorpus(reg)
    previous=json.loads((ROOT/reg['frozen_probability_report']).read_text())
    if previous['identity']['source_assignment_sha256']!=data.assignment_hash:
        raise ValueError('Probability producer/source query mismatch')
    prior={(t['arm'],t['site'],t['seed']):t for t in previous['trials']}
    identity=dict(data.identity,source_assignment_sha256=data.assignment_hash,
        registration_sha256=file_digest(args.registration),torch=torch.__version__,numpy=np.__version__,
        torch_threads=4,interop_threads=1,num_workers=0)
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:
        raise ValueError('Runtime/data identity changed')
    json_write(ip,identity)
    checks=dict(data.audit(),identity=identity,feature_width=480,
        extra_features='four_observed_rotation_entries_for_image_vector_frame_alignment',
        supported_spatial_frames=int(data.support.sum()),unsupported_retained=int((~data.support).sum()),
        restoration_sha256=array_hash(data.radius,data.rotation,data.support),
        target_sha256=array_hash(data.target),source_scale_sha256=array_hash(data.native_scale),
        labels_in_input=False,primary_changed=False)
    json_write(reports/'input_checks.json',checks)
    if args.audit_only:
        heartbeat(state='inputs_verified_no_training',rows=len(data.sid),unsupported=checks['unsupported_retained']);return
    keys={f'{objective}_{arm}_{site}_seed{seed}' for objective in reg['objectives'] for arm in reg['arms']
          for site in reg['sites'] for seed in reg['seeds']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot must identify a training-only trial')
    trials,replays=[],[];new_updates,new_fits=0,0
    for site in reg['sites']:
        train,w,held,clipped,normalizer=data.dynamics_design(site)
        train_cv=np.linalg.norm(data.target[train-data.nmain].astype(float),axis=-1).mean(1)
        hard_cut=float(np.quantile(train_cv,.9))
        for seed in reg['seeds']:
            for arm in reg['arms']:
                producer=prior[arm,site,seed]
                if (file_digest(ROOT/producer['prediction_path'])!=producer['prediction_sha256']
                        or producer['identity']['site']!=site or producer['identity']['seed']!=seed):
                    raise ValueError('Frozen classifier producer changed')
                with np.load(ROOT/producer['prediction_path'],allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['held_indices'],held);probability=a['probability'].copy()
                gate=probability>=reg['fixed_probability_threshold']
                for objective in reg['objectives']:
                    key=f'{objective}_{arm}_{site}_seed{seed}'
                    if args.trial and key!=args.trial:
                        continue
                    ti=dict(identity,arm=arm,objective=objective,site=site,seed=seed,
                        training_rows_sha256=array_hash(train,w,data.target[train-data.nmain]),
                        normalizer_sha256=array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant']),
                        held_rows_sha256=array_hash(held),loss_normalizer=normalizer,training_hard_cut=hard_cut,
                        probability_producer_sha256=producer['prediction_sha256'])
                    cp,pp,rp=out/'checkpoints'/f'{key}.pt',out/'predictions'/f'{key}.npz',out/'trials'/f'{key}.json'
                    old=json.loads(rp.read_text()) if rp.exists() else None
                    if old:
                        if (old['identity']!=ti or file_digest(cp)!=old['checkpoint_sha256']
                                or file_digest(pp)!=old['prediction_sha256']):
                            raise ValueError('Completed trajectory trial changed')
                        with np.load(pp,allow_pickle=False) as a:
                            np.testing.assert_array_equal(a['held_indices'],held)
                            np.testing.assert_array_equal(a['gate'],gate);saved=a['prediction'].copy()
                        if not args.replay:
                            trials.append(old);continue
                    heartbeat(state='fit_or_replay',trial=key,training_rows=len(train))
                    torch.manual_seed(seed);model=SourceDynamics()
                    if args.replay:
                        if old is None:
                            raise ValueError('Replay requires completed trial')
                        state=torch.load(cp,map_location='cpu',weights_only=False)
                        if state['identity']!=ti or state['step']!=reg['training']['updates']:
                            raise ValueError('Checkpoint identity/budget changed')
                        model.load_state_dict(state['model'])
                    else:
                        fit=fit_dynamics(model,lambda ids:data.dynamics_inputs(ids,training=True),data.loss_targets,
                            train,w,arm=arm,objective=objective,normalizer=normalizer,seed=seed,
                            config=reg['training'],identity=ti,checkpoint=cp,
                            heartbeat=lambda **v:heartbeat(trial=key,**v),stop_at=args.stop_at)
                        new_updates+=fit['new_updates']
                        if not fit['complete']:
                            heartbeat(state='pilot_complete_no_forecast_evaluation',trial=key,**fit);return
                    prediction=forecast(model,data.dynamics_inputs,held,arm)
                    if args.replay:
                        np.testing.assert_array_equal(saved,prediction);replays.append(key);continue
                    train_prediction=forecast(model,data.dynamics_inputs,train,arm)
                    pp.parent.mkdir(parents=True,exist_ok=True);tmp=pp.with_suffix('.tmp.npz')
                    np.savez(tmp,held_indices=held,prediction=prediction,gate=gate);os.replace(tmp,pp)
                    target=data.target[held-data.nmain];native=data.native_scale[held-data.nmain]
                    train_ade=np.linalg.norm(train_prediction.astype(float)-data.target[train-data.nmain],axis=-1).mean(1)
                    evaluation={}
                    for mode,use in [('uncontrolled',np.ones(len(held),bool)),('fixed_probability_gate',gate)]:
                        evaluation[mode]=trajectory_metrics(np.where(use[:,None,None],prediction,0),target,native,use,hard_cut)
                    result=dict(identity=ti,trial=key,arm=arm,objective=objective,site=site,seed=seed,
                        result_source='fresh_run_torch',fit=fit,training_rows=len(train),held_rows=len(held),
                        parameter_count=sum(v.numel() for v in model.parameters()),
                        training_ade=float(w@train_ade),training_cv_ade=normalizer,
                        clipping_fraction=clipped,evaluation=evaluation,
                        checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                        prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                    json_write(rp,result);trials.append(result);new_fits+=1
                    heartbeat(state='trial_complete',trial=key,gain_percent=evaluation['uncontrolled']['gain_percent'])
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,exact_trials=replays,all_exact=True,new_updates=0))
        heartbeat(state='replay_complete',trials=len(replays));return
    if args.trial:
        return
    if len(trials)!=reg['fresh_models']:
        raise ValueError('Entire fixed comparison required')
    path=reports/'report.json'
    if path.exists() and new_fits==0:
        old=json.loads(path.read_text())
        if old['identity']!=identity or old['trials']!=trials:
            raise ValueError('Completed report changed')
        heartbeat(state='completed_resume_verified',new_fits=0,new_updates=0);return
    json_write(path,dict(identity=identity,result_source='fresh_run_torch',trials=trials,
        completed_models=len(trials),optimizer_updates=sum(t['fit']['step'] for t in trials),
        summed_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials),
        main_primary_changed=False,sealed_roles_opened=False,new_deployment=False))
    heartbeat(state='matrix_complete',new_fits=new_fits,new_updates=new_updates)


if __name__=='__main__':
    main()
