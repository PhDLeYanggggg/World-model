"""Fixed source-augmentation classification study on exposed fit roles only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression

from scripts.run_m3w_sdd_auxiliary import Corpus
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_motion_start_probe import paired_agent_interval
from src.world_model.m3w_observed_unit_frame_v2 import observed_unit_frame
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_auxiliary import load_registration
from src.world_model.m3w_source_start_probe import (
    stationary_membership,start_supervision,training_rows,fit_normalizer,normalize,
    metrics,fit_mlp,fit_trees,probabilities,
)


def array_hash(*arrays):
    h=hashlib.sha256()
    for x in arrays:
        x=np.ascontiguousarray(x); h.update(str((x.shape,x.dtype.str)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def save_model(path,bundle):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    joblib.dump(bundle,tmp); os.replace(tmp,path)


def prepare(parent):
    data=Corpus(parent)
    mid=np.flatnonzero(stationary_membership(data.main['geometry']))
    sid_all=np.flatnonzero(stationary_membership(data.aux['geometry']))
    sy,complete=start_supervision(data.aux['target'][sid_all],data.aux['valid'][sid_all])
    sid=sid_all[complete]; sy=sy[complete]
    my,valid=start_supervision(data.main['targets'][mid],np.ones((len(mid),12),bool))
    if not valid.all() or len(mid)!=365:
        raise ValueError('Frozen stationary main cohort changed')
    mx=observed_unit_frame(data.main['geometry'][mid])[0]
    sx=observed_unit_frame(data.aux['geometry'][sid])[0]
    manifest=json.loads(data.manifest_path.read_text())
    keys=np.concatenate([np.load(data.manifest_path.parent/r['recording']/'query_keys.npy') for r in manifest['records']])
    source_tracks=np.array([str(r)+':'+str(k) for r,k in zip(data.record_ids[sid],keys[sid,1])])
    receipt=dict(main_rows=len(mid),main_agents=len(np.unique(data.tracks[mid])),
        main_class_counts=np.bincount(my,minlength=2).tolist(),
        source_all_past_stationary_rows=len(sid_all),source_complete_supervised_rows=len(sid),
        source_incomplete_unscored_rows=int((~complete).sum()),source_class_counts=np.bincount(sy,minlength=2).tolist(),
        source_agents=len(np.unique(source_tracks)),source_recordings=len(np.unique(data.record_ids[sid])),
        source_sites=sorted({manifest['records'][i]['recording'].split('/')[0] for i in np.unique(data.record_ids[sid])}),
        source_supervision='complete_twelve_step_any_annotation_change_not_verified_intention',
        main_original_forecasting_rows=11966,source_original_past_rows=data.auxiliary_rows,
        primary_metric_changed=False,sealed_roles_opened=False,source_val_test_opened=False,
        future_inputs=False,ids_or_domains_as_features=False,
        main_feature_sha256=array_hash(mx),source_feature_sha256=array_hash(sx),
        main_supervision_sha256=array_hash(my),source_supervision_sha256=array_hash(sy))
    return dict(main_x=mx,main_y=my,source_x=sx,source_y=sy,
        main_ids=mid,source_ids=sid,folds=np.asarray(data.main['folds'][mid]),tracks=data.tracks[mid],
        receipt=receipt,identity=data.identity)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True); parser.add_argument('--replay',action='store_true')
    parser.add_argument('--trial'); parser.add_argument('--stop-at',type=int)
    args=parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg=json.loads(args.registration.read_text())
    if (str(args.registration)!=reg['registration_path'] or reg['role']!='fit_only_information_probe'
            or reg['schedules']!=['main_only','source_only','mixed'] or reg['seeds']!=[17,29,43]):
        raise ValueError('Fixed role, schedules and seeds required')
    for name,digest in reg['bindings'].items():
        if file_digest(ROOT/name)!=digest:
            raise ValueError('Registered dependency changed: '+name)
    parent=load_registration(ROOT,ROOT/reg['parent_registration'])
    out,reports=ROOT/reg['output'],ROOT/reg['reports']
    def heartbeat(**v):
        value=dict(pid=os.getpid(),timestamp_unix=time.time(),**v)
        json_write(out/'heartbeat.json',value); print(json.dumps(value),flush=True)
    heartbeat(state='verifying_training_assets')
    d=prepare(parent); json_write(reports/'input_checks.json',d['receipt'])
    identity=dict(registration_sha256=file_digest(args.registration),torch=torch.__version__,numpy=np.__version__,
        sklearn=sklearn.__version__,**d['identity'])
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:
        raise ValueError('Run runtime/population identity changed')
    json_write(ip,identity)
    keys={f'{family}_{schedule}_seed{seed}_fold{fold}' for family in reg['models'] for schedule in reg['schedules']
          for seed in reg['seeds'] for fold in ([-1] if schedule=='source_only' else [0,1])}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or not args.trial.startswith('mlp_') or args.replay):
        raise ValueError('Pilot requires one MLP training trial')
    trials=[]; new_fits=0; new_updates=0; replays=[]; saved_predictions={}
    for family in reg['models']:
        for schedule in reg['schedules']:
            for seed in reg['seeds']:
                for fold in ([-1] if schedule=='source_only' else [0,1]):
                    key=f'{family}_{schedule}_seed{seed}_fold{fold}'
                    if args.trial and key!=args.trial:
                        continue
                    train=np.flatnonzero(d['folds']!=fold) if fold>=0 else np.empty(0,int)
                    held=np.flatnonzero(d['folds']==fold) if fold>=0 else np.arange(len(d['main_y']))
                    x,y,w=training_rows(d['main_x'][train],d['main_y'][train],d['source_x'],d['source_y'],schedule)
                    normalizer=fit_normalizer(x,w); z,_=normalize(x,normalizer); zh,clip=normalize(d['main_x'][held],normalizer)
                    ti=dict(identity,family=family,schedule=schedule,seed=seed,fold=fold,
                        training_data_sha256=array_hash(x,y,w),main_training_ids_sha256=array_hash(d['main_ids'][train]),
                        source_training_ids_sha256=array_hash(d['source_ids']) if schedule!='main_only' else None)
                    mp=out/'models'/(key+'.joblib'); pp=out/'predictions'/(key+'.npz'); rp=out/'trials'/(key+'.json')
                    cp=out/'checkpoints'/(key+'.pt'); tcp=out/'tree_checkpoints'/(key+'.joblib')
                    saved=json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        if saved['identity']!=ti or file_digest(mp)!=saved['model_sha256'] or file_digest(pp)!=saved['prediction_sha256']:
                            raise ValueError('Completed classifier changed')
                        if family=='mlp' and file_digest(cp)!=saved['checkpoint_sha256']:
                            raise ValueError('Completed neural checkpoint changed')
                        if family=='extra_trees' and file_digest(tcp)!=saved['checkpoint_sha256']:
                            raise ValueError('Completed tree checkpoint changed')
                        with np.load(pp,allow_pickle=False) as a:
                            np.testing.assert_array_equal(a['held_indices'],held); saved_predictions[key]=(held,a['probability'].copy())
                        if not args.replay:
                            trials.append(saved); continue
                    heartbeat(state='fit_or_replay',trial=key,training_rows=len(y))
                    start=time.monotonic(); warn=[]
                    if args.replay:
                        if not saved:
                            raise ValueError('Replay requires complete receipts')
                        bundle=joblib.load(mp)
                        if bundle['identity']!=ti:
                            raise ValueError('Model identity mismatch')
                        model=bundle['model']
                    else:
                        if family=='mlp':
                            model,fit=fit_mlp(z,y,w,seed=seed,config=reg['models'][family],identity=ti,checkpoint=cp,
                                heartbeat=lambda **v:heartbeat(state='training',trial=key,**v),stop_at=args.stop_at)
                            new_updates+=fit['new_updates']
                            if not fit['complete']:
                                heartbeat(state='pilot_saved_without_held_evaluation',trial=key,**fit); return
                        elif family=='extra_trees':
                            model,fit=fit_trees(z,y,w,seed=seed,config=reg['models'][family],identity=ti,
                                checkpoint=tcp,
                                heartbeat=lambda **v:heartbeat(state='tree_checkpoint',trial=key,**v))
                        else:
                            model=LogisticRegression(**reg['models'][family],random_state=seed)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always'); model.fit(z,y,sample_weight=w)
                            warn=[str(w.message) for w in caught]
                            fit=dict(seconds=time.monotonic()-start,complete=True,iterations=model.n_iter_.tolist(),new_updates=0)
                        save_model(mp,dict(model=model,normalizer=normalizer,identity=ti,fit_seconds=fit['seconds']))
                    p=probabilities(model,zh)
                    if args.replay:
                        np.testing.assert_array_equal(p,saved_predictions[key][1]); replays.append(dict(trial=key,probability_exact=True)); continue
                    train_p=probabilities(model,z)
                    pp.parent.mkdir(parents=True,exist_ok=True); tmp=pp.with_suffix('.tmp.npz')
                    np.savez(tmp,held_indices=held,probability=p); os.replace(tmp,pp)
                    evaluation=[]
                    for f in ([0,1] if fold==-1 else [fold]):
                        mask=d['folds'][held]==f; target_train=d['folds']!=f
                        prior=float((d['main_y'][target_train].sum()+1)/(target_train.sum()+2))
                        eval_m=metrics(d['main_y'][held][mask],p[mask],prior)
                        evaluation.append(dict(fold=f,reference='opposite_main_site_training_prior',reference_prior=prior,
                            agents=len(np.unique(d['tracks'][held][mask])),**eval_m))
                    result=dict(identity=ti,trial=key,family=family,schedule=schedule,seed=seed,fold=fold,
                        result_source='fresh_run_torch' if family=='mlp' else 'fresh_run_sklearn',fit=fit,
                        training_rows=len(y),training_weighted_brier=float(np.sum(w*(train_p-y)**2)),
                        warnings=warn,held_feature_clipping_fraction=clip,evaluation=evaluation,
                        model_path=str(mp.relative_to(ROOT)),model_sha256=file_digest(mp),
                        prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                    if family=='mlp':
                        result.update(checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp))
                    elif family=='extra_trees':
                        result.update(checkpoint_path=str(tcp.relative_to(ROOT)),checkpoint_sha256=file_digest(tcp))
                    json_write(rp,result); trials.append(result); saved_predictions[key]=(held,p); new_fits+=1
                    heartbeat(state='classifier_complete',trial=key,brier_lifts=[e['brier_lift'] for e in evaluation])
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,trials=replays,all_exact=True,new_fits=0,new_updates=0))
        heartbeat(state='replay_complete',trials=len(replays)); return
    if args.trial:
        return
    if len(trials)!=45:
        raise ValueError('All 45 classifiers required')
    report_path=reports/'report.json'
    if report_path.exists() and new_fits==0:
        old=json.loads(report_path.read_text())
        if old['identity']!=identity or old['trials']!=trials:
            raise ValueError('Completed report changed')
        heartbeat(state='completed_resume_verified',new_fits=0,new_updates=0); return
    summary={}
    lookup={(t['family'],t['schedule'],t['seed'],t['fold']):t for t in trials}
    for family in reg['models']:
        for schedule in reg['schedules']:
            per_fold=[]
            for fold in (0,1):
                ps=[]; refs=[]; main_probs=[]; es=[]
                for seed in reg['seeds']:
                    t=lookup[family,schedule,seed,-1 if schedule=='source_only' else fold]
                    es.append(next(e for e in t['evaluation'] if e['fold']==fold))
                    held,p=saved_predictions[t['trial']]; ps.append(p[d['folds'][held]==fold])
                    refs.append(np.full((d['folds']==fold).sum(),es[-1]['reference_prior']))
                    main=saved_predictions[lookup[family,'main_only',seed,fold]['trial']][1]; main_probs.append(main)
                mask=d['folds']==fold; y=d['main_y'][mask]; tracks=d['tracks'][mask]
                per_fold.append(dict(fold=fold,mean_brier_lift=float(np.mean([e['brier_lift'] for e in es])),
                    mean_brier=float(np.mean([e['brier'] for e in es])),mean_auroc=float(np.mean([e['auroc'] for e in es])),
                    mean_auprc=float(np.mean([e['auprc'] for e in es])),mean_ece=float(np.mean([e['ece'] for e in es])),
                    mean_lift_vs_main_only=float(np.mean((np.asarray(main_probs)-y)**2-(np.asarray(ps)-y)**2)),
                    positive_seeds=sum(e['brier_lift']>0 for e in es),
                    vs_prior_agent_interval=paired_agent_interval(y,np.asarray(ps),np.asarray(refs),tracks),
                    vs_main_agent_interval=paired_agent_interval(y,np.asarray(ps),np.asarray(main_probs),tracks)))
            summary[family+'_'+schedule]=per_fold
    result=dict(identity=identity,complete=True,trials=trials,summary=summary,fresh_models=45,
        fresh_sklearn_models=30,fresh_torch_models=15,total_torch_updates=15000,
        fresh_evaluation_cells=54,summed_fit_seconds=sum(t['fit']['seconds'] for t in trials),
        main_primary_changed=False,sealed_roles_opened=False,predictive_trajectory_lift_evaluated=False,
        independent_confirmation=False,no_threshold_search=True,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    json_write(report_path,result)
    lines=['# Source-Supported Start Information','',
        '45 fresh classifiers, 54 prediction cells; fixed exposed-fit sites only. Not a forecasting or deployment result.','',
        '| Model / source | Held site | Brier lift vs main prior | Brier lift vs main-only | AUROC | Positive seeds |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for key,items in summary.items():
        for e in items:
            lines.append(f"| {key} | {['ETH','Hotel'][e['fold']]} | {e['mean_brier_lift']:.6f} | {e['mean_lift_vs_main_only']:.6f} | {e['mean_auroc']:.6f} | {e['positive_seeds']}/3 |")
    lines+=['','Zara has no exactly-stationary rows: not_run for this information probe, not another passing domain.',
        'No forecast is switched. Held-agent intervals are descriptive, not new-scene confirmation or calibration.',
        'Source-only model inputs, normalization and fit use no main data; the reported comparator prior uses the opposite main training site.',
        'Offline/silver, dataset-local/raw annotation steps only. No metric, seconds, foundation, Stage5C or SMC.','']
    (reports/'results.md').write_text('\n'.join(lines))
    heartbeat(state='complete',fresh_models=new_fits,new_torch_updates=new_updates)


if __name__=='__main__':
    main()
