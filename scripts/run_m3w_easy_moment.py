"""Frozen-predictor conditional easy-moment training and separate readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch

from scripts import run_m3w_protected_eqmotion_controls as parent
from scripts.run_m3w_bounded_cost import training_data, features, read_arrays
from scripts.run_m3w_native_forecast import array_hash, file_digest, assert_current, immutable_json, json_write
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_easy_moment import targets, decisions
from src.evaluation.m3w_protected_motion_controls import causal_candidate, supported_costs, match_strict_count
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.training.m3w_easy_moment import fit, predict
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_bounded_cost_head import build, predict as predict_cost

CONFIG='configs/m3w_easy_moment_v1.json'
CODE=('scripts/run_m3w_easy_moment.py','src/evaluation/m3w_easy_moment.py',
      'src/training/m3w_easy_moment.py','tests/test_m3w_easy_moment.py')


def load():
    cfg=json.loads((ROOT/CONFIG).read_text())
    pc,data,ev,ep,mv,neural,refs,ma,ea,cuts,pi=parent.load()
    ap=Path(pc['reports'])/'analysis.json'
    assert file_digest(ROOT/ap)==cfg['parent_analysis_sha256']
    pa=json.loads((ROOT/ap).read_text())
    assert pa['identity']==pi
    assert cfg['sites']==pc['sites'] and cfg['seeds']==pc['seeds'] and cfg['forest']==pc['forest']
    assert cfg['actions']==['damped_velocity_005','transformer','eqmotion'] and cfg['easy_rho']==.02
    assert cfg['policies']==['net_stop','strict_stop','joint_easy_moment','product_easy_marginals']
    assert not any(cfg[k] for k in ('threshold_search','model_selection','new_forecast_training',
        'external_readout','risk_calibration','independent_confirmation','deployment','stage5c_executed','smc_enabled'))
    bindings=dict(pi['source_bindings'])
    for path in (str(ap),CONFIG,cfg['registration'],*CODE):
        bindings[path]=file_digest(ROOT/path)
    for a in pa['decision_archives']:
        assert file_digest(ROOT/a['path'])==a['sha256']
        bindings[a['path']]=a['sha256']
    identity=dict(config=cfg,source_bindings=bindings,parent_identity_sha256=file_digest(ROOT/pc['output']/'identity.json'),
                  runtime=dict(torch_threads=4,interop_threads=1,num_workers=0),
                  torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__)
    assert_current(identity)
    frozen={}
    for key in ev:
        for action in cfg['actions']:
            frozen[key,action]=(neural[key] if action=='eqmotion' else
                next(r for r in ma['fits'] if r['view']==key and r['action']==action and r['head']=='neural'))
    return cfg,data,ev,mv,ep,cuts,identity,frozen,ma,pa


def old_head(ref):
    assert file_digest(ROOT/ref['checkpoint'])==ref['checkpoint_sha256']
    return torch.load(ROOT/ref['checkpoint'],map_location='cpu',weights_only=False)


def prepare(data,meta,action,cp,cutoff):
    ids,x,y,d,pr=training_data(meta,data)
    with np.load(ROOT/meta['targets_path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(ids,z['ids']); cv=z['baseline_ade'].copy()
    cv[~pr['known']]=np.nan
    if action=='damped_velocity_005':
        g,s=data['geometry'][ids],data['scale'][ids]
        p=causal_candidate(g,action)
        x,d,_=features(g,p,s)
        valid=read_arrays(data,ids,'valid'); truth=read_arrays(data,ids,'target')
        cv,_=native_errors(g[:,332:356].reshape(-1,12,2),truth,valid,s)
        err,_=native_errors(p,truth,valid,s)
        y,cv=supported_costs(err,cv,valid.all(1))
        pr=preprocess(x,y,cv,data['sites'][ids],meta['outer_site'])
    for field in ('mean','std','known','weights','constant'):
        np.testing.assert_array_equal(pr[field],cp['preprocess'][field])
    assert pr['positive_easy_cut']==cp['preprocess']['positive_easy_cut']
    assert cp['draws'].sum()==768000 and not cp['draws'][~pr['known']].any()
    assert meta['outer_site'] not in pr['training_sites']
    return ids,x,d,pr,targets(y,cv,d,pr['known'],cutoff)


def trial(cfg,key,action):
    return ROOT/cfg['output']/'trials'/key/action


def train(pack,args,beat):
    cfg,data,ev,mv,ep,cuts,identity,frozen,ma,pa=pack
    ish=file_digest(ROOT/cfg['output']/'identity.json')
    for key in ev:
        if args.view and key!=args.view:continue
        for action in cfg['actions']:
            if args.action and action!=args.action:continue
            meta=ev[key] if action=='eqmotion' else mv[key]
            cp=old_head(frozen[key,action])
            cutoff=cuts[key]['positive_easy_cut']
            ids,x,d,pr,q=prepare(data,meta,action,cp,cutoff)
            ti=dict(experiment_sha256=ish,view=key,action=action,
                inputs_sha256=array_hash(ids,x,d),targets_sha256=array_hash(q),
                draws_sha256=array_hash(cp['draws']),known_sha256=array_hash(pr['known']),
                training_sites=pr['training_sites'],easy_cut=cutoff,
                frozen_head_sha256=frozen[key,action]['checkpoint_sha256'])
            folder=trial(cfg,key,action); rp=folder/'complete.json'
            if rp.exists():
                r=json.loads(rp.read_text())
                assert r['identity']==ti and r['fit']['complete'] and file_digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
                beat(state='cached_verified_fit',view=key,action=action);continue
            result=fit(x,q,d,pr,cp['draws'],settings=cfg['forest'],seed=meta['seed'],
                identity=ti,directory=folder,resume=args.resume,stop_at=args.stop_at,
                heartbeat=lambda **v:beat(view=key,action=action,**v))
            if not result['complete']:
                beat(state='pilot_checkpoint_not_complete_matrix',view=key,action=action);return
            path=folder/'checkpoint.joblib'
            immutable_json(rp,dict(identity=ti,fit=result,checkpoint=str(path.relative_to(ROOT)),
                                  checkpoint_sha256=file_digest(path),result_source='fresh_run'))
    assert_current(identity)
    beat(state='all_requested_fits_complete')


def evaluate(pack,beat,verify=False):
    cfg,data,ev,mv,ep,cuts,identity,frozen,ma,pa=pack
    root,public=ROOT/cfg['output'],ROOT/cfg['reports']
    ish=file_digest(root/'identity.json'); receipts={}
    for key in ev:
        for action in cfg['actions']:
            r=json.loads((trial(cfg,key,action)/'complete.json').read_text())
            assert r['identity']['experiment_sha256']==ish and r['fit']['complete'] and r['fit']['trees']==128
            assert file_digest(ROOT/r['checkpoint'])==r['checkpoint_sha256']
            receipts[key,action]=r
    cache={}; archives=[]
    for key,meta in ev.items():
        ids=np.flatnonzero(data['sites']==meta['outer_site'])
        g,s=data['geometry'][ids],data['scale'][ids]; arr=dict(ids=ids)
        for action in cfg['actions']:
            if action=='damped_velocity_005':p=causal_candidate(g,action)
            else:
                path=ep[key]['path'] if action=='eqmotion' else mv[key]['outer_prediction']['prediction']['path']
                with np.load(ROOT/path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(ids,z['ids']);p=z['prediction'].copy()
            x,d,_=features(g,p,s)
            old=old_head(frozen[key,action])
            head=build(x.shape[1],old['settings']['width'],meta['seed']);head.load_state_dict(old['model'])
            score=predict_cost(head,x,d,old['preprocess'],'bounded_fraction')
            previous=next(r for r in (pa['decision_archives'] if action=='eqmotion' else ma['decision_archives']) if r['view']==key)
            with np.load(ROOT/previous['path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(ids,z['ids'])
                np.testing.assert_array_equal(score,z['neural_score' if action=='eqmotion' else action+'__neural__score'])
            cp=joblib.load(ROOT/receipts[key,action]['checkpoint'])
            assert cp['identity']==receipts[key,action]['identity'] and meta['outer_site'] not in cp['preprocess']['training_sites']
            moment=predict(cp['model'],x,cp['preprocess'])
            chosen=decisions(moment,d,cuts[key]['positive_easy_cut'],score,g[:,:16].reshape(-1,8,2),cfg['easy_rho'])
            left,right=match_strict_count(score,score,chosen['joint_easy_moment'],chosen['product_easy_marginals'],ids)
            chosen.update(matched_joint=left,matched_product=right)
            for policy,bits in chosen.items():arr[action+'__'+policy]=bits
            arr[action+'__moments']=moment; arr[action+'__score']=score;arr[action+'__distance']=d
            cache[key,action]=(ids,p,chosen,moment,d)
        path=root/'decisions'/f'{key}.npz'
        if verify and not path.exists():raise ValueError('Missing original decisions')
        write_arrays(path,arr)
        archives.append(dict(view=key,path=str(path.relative_to(ROOT)),sha256=file_digest(path)))
        beat(state='past_only_decisions_saved',view=key)
    immutable_json(root/'decisions_complete.json',dict(experiment_sha256=ish,archives=archives,
        all_fits_complete=True,target_arrays_loaded_for_decisions=False))
    n=len(data['sites']); ids=np.arange(n)
    truth,valid=read_arrays(data,ids,'target'),read_arrays(data,ids,'valid')
    baseline=data['geometry'][:,332:356].reshape(-1,12,2)
    cv,cf=native_errors(baseline,truth,valid,data['scale'])
    masks=dict(complete=valid.all(1),zero_CV=valid.all(1)&(cv==0),
               positive_easy=np.zeros(n,bool),hard=np.zeros(n,bool))
    for key,meta in ev.items():
        use=data['sites']==meta['outer_site']
        masks['positive_easy'][use]=(cv[use]>0)&(cv[use]<=cuts[key]['positive_easy_cut'])
        masks['hard'][use]=cv[use]>=cuts[key]['hard_cut']
    def metric(error,reference=cv,mask=None,bootstrap=False):
        use=np.ones(n,bool) if mask is None else mask
        return paired_scene_metrics(error[use],reference[use],data['sites'][use],expected_scenes=cfg['sites'],
            dataset='sdd',coordinate_unit='annotation_pixel',bootstrap_resamples=cfg['bootstrap_resamples'] if bootstrap else 0)
    policies=cfg['policies']+['matched_joint','matched_product']
    summary={}; error_bank={}; quality={}; outcome_files=[]
    for action in cfg['actions']:
        errors={p:[] for p in policies}; ends={p:[] for p in policies}; records={p:{} for p in policies}
        for seed in cfg['seeds']:
            err=np.full(n,np.nan);end=err.copy();lower=np.zeros(n);upper=lower.copy()
            selected={p:np.zeros(n,bool) for p in policies}
            for key,meta in ev.items():
                if meta['seed']!=seed:continue
                row,p,chosen,moment,d=cache[key,action]
                err[row],end[row]=native_errors(p,truth[row],valid[row],data['scale'][row])
                bounds=partial_gain_bounds(p,baseline[row],truth[row],valid[row],data['scale'][row])
                lower[row],upper[row]=bounds['lower'],bounds['upper']
                costs,_=supported_costs(err[row],cv[row],valid[row].all(1))
                q=targets(costs,cv[row],d,valid[row].all(1),cuts[key]['positive_easy_cut'])
                known=valid[row].all(1)
                quality[key+'__'+action]=dict(complete_count=int(known.sum()),
                    complete_fraction_mse=np.mean((moment[known]-q[known])**2,axis=0).tolist(),
                    selected={pol:dict(count=int((bits&known).sum()),
                        predicted_easy_harm_sum=float((moment[:,0]*d)[bits&known].sum()),
                        actual_easy_harm_sum=float((q[:,0]*d)[bits&known].sum()),
                        predicted_easy_denominator_sum=float((moment[:,1]*cuts[key]['positive_easy_cut'])[bits&known].sum()),
                        actual_easy_denominator_sum=float((q[:,1]*cuts[key]['positive_easy_cut'])[bits&known].sum()))
                        for pol,bits in chosen.items()})
                for pol in policies:selected[pol][row]=chosen[pol]
            outcome=root/'outcomes'/f'{action}_seed{seed}.npz'
            write_arrays(outcome,dict(cv=cv,cf=cf,candidate_ade=err,candidate_fde=end,
                lower=lower,upper=upper,**selected,**masks))
            outcome_files.append(dict(path=str(outcome.relative_to(ROOT)),sha256=file_digest(outcome)))
            for pol in policies:
                bits=selected[pol];e=np.where(bits,err,cv);f=np.where(bits,end,cf)
                errors[pol].append(e);ends[pol].append(f)
                records[pol][str(seed)]=dict(ADE=metric(e),FDE=metric(f,cf),
                    subsets={k:metric(e,mask=m) for k,m in masks.items()},selected=int(bits.sum()),
                    selected_unknown=int((bits&~valid.any(1)).sum()),selected_incomplete=int((bits&~valid.all(1)).sum()),
                    zero_CV_harmed=int((e[masks['zero_CV']]>0).sum()),
                    full_grid_gain_bounds={site:[float(np.where(bits,b,0)[data['sites']==site].mean())
                        for b in (lower,upper)] for site in cfg['sites']})
        for pol in policies:
            name=action+'__'+pol;mean=np.mean(errors[pol],axis=0);error_bank[name]=mean
            summary[name]=dict(ADE=metric(mean,bootstrap=True),FDE=metric(np.mean(ends[pol],axis=0),cf,bootstrap=True),
                subsets={k:metric(mean,mask=m,bootstrap=True) for k,m in masks.items()},seeds=records[pol])
    contrasts={}
    for action in cfg['actions']:
        for left,right in [('joint_easy_moment','product_easy_marginals'),('joint_easy_moment','strict_stop'),('matched_joint','matched_product')]:
            key=action+'__'+left+'_minus_'+right;contrasts[key]={}
            for subset in ('all','hard','positive_easy'):
                mask=None if subset=='all' else masks[subset]
                a=metric(error_bank[action+'__'+left],mask=mask)['by_scene']
                b=metric(error_bank[action+'__'+right],mask=mask)['by_scene']
                contrasts[key][subset]=paired_scene_contrast([a[s]['gain_percent'] for s in cfg['sites']],
                    [b[s]['gain_percent'] for s in cfg['sites']],resamples=cfg['bootstrap_resamples'])
    result=dict(result_source='fresh_run_complete_development_only',experiment_sha256=ish,
        rows=n,complete_rows=int(valid.all(1).sum()),no_future_rows=int((~valid.any(1)).sum()),
        sites=cfg['sites'],seeds=cfg['seeds'],summary=summary,contrasts=contrasts,
        conditional_quality=quality,decision_archives=archives,outcome_archives=outcome_files,
        fits=[dict(view=k,action=a,**r) for (k,a),r in receipts.items()],
        independent_calibration=False,independent_confirmation=False,deployment=False,
        stage5c_executed=False,smc_enabled=False)
    assert_current(identity);immutable_json(public/'analysis.json',result)
    if verify:
        json_write(public/'verification.json',dict(all_checks_passed=True,decision_and_aggregate_replay_exact=True,
            fits=36,views=12,analysis_sha256=file_digest(public/'analysis.json')))
    beat(state='verified_replay_complete' if verify else 'evaluation_complete',rows=n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['preflight','train','evaluate','verify'],default='preflight')
    p.add_argument('--resume',action='store_true');p.add_argument('--view');p.add_argument('--action')
    p.add_argument('--stop-at',type=int);args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();cfg=pack[0];root=ROOT/cfg['output'];root.mkdir(parents=True,exist_ok=True)
    if args.view and args.view not in pack[2]:raise ValueError('Unknown view')
    if args.action and args.action not in cfg['actions']:raise ValueError('Unknown action')
    immutable_json(root/'identity.json',pack[6])
    def beat(**v):
        r=dict(pid=os.getpid(),updated_unix=time.time(),**v)
        json_write(root/'heartbeat.json',r);print(json.dumps(r),flush=True)
    with (root/'runner.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='preflight':beat(state='preflight_pass',rows=len(pack[1]['sites']),bindings=len(pack[6]['source_bindings']))
        elif args.phase=='train':train(pack,args,beat)
        else:evaluate(pack,beat,args.phase=='verify')


if __name__=='__main__':main()
