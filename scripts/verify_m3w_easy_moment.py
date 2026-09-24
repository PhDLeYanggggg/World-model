"""Separate policy arithmetic, fit support and actual-label reductions."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from scripts.run_m3w_easy_moment import load, trial, old_head
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import file_digest, immutable_json, assert_current
from scripts.verify_m3w_native_joint_controls import distances
from scripts.verify_m3w_protected_motion_controls import check_metrics
from scripts.report_m3w_protected_eqmotion_controls import check_contrast
from src.data_unification.m3w_causal_recordings import BASELINES


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg,data,ev,mv,ep,cuts,identity,frozen,ma,pa=load()
    root,public=ROOT/cfg['output'],ROOT/cfg['reports']
    a=json.loads((public/'analysis.json').read_text())
    v=json.loads((public/'verification.json').read_text())
    assert v['all_checks_passed'] and v['analysis_sha256']==file_digest(public/'analysis.json')
    assert a['experiment_sha256']==file_digest(root/'identity.json')
    n=len(data['sites']);all_ids=np.arange(n)
    truth,mask=read_arrays(data,all_ids,'target'),read_arrays(data,all_ids,'valid')
    base=data['geometry'][:,332:356].reshape(-1,12,2)
    cv,cf=distances(base,truth,mask,data['scale'])
    policies=cfg['policies']+['matched_joint','matched_product']
    choices={(s,act,pol):np.zeros(n,bool) for s in cfg['seeds'] for act in cfg['actions'] for pol in policies}
    costs={(s,act):np.full(n,np.nan) for s in cfg['seeds'] for act in cfg['actions']}
    ends={key:val.copy() for key,val in costs.items()}
    lower={key:np.zeros(n) for key in costs};upper={key:np.zeros(n) for key in costs}
    groups=dict(complete=mask.all(1),zero_CV=mask.all(1)&(cv==0),positive_easy=np.zeros(n,bool),hard=np.zeros(n,bool))
    counts=dict(fit_budgets=0,decision_arrays=0,matched_pairs=0,scene_reductions=0,contrasts=0)
    for key,meta in ev.items():
        ids=np.flatnonzero(data['sites']==meta['outer_site']); train=np.flatnonzero(data['sites']!=meta['outer_site'])
        cutoff=cuts[key]['positive_easy_cut'];seed=meta['seed']
        groups['positive_easy'][ids]=(cv[ids]>0)&(cv[ids]<=cutoff)
        groups['hard'][ids]=cv[ids]>=cuts[key]['hard_cut']
        rec=next(r for r in a['decision_archives'] if r['view']==key)
        assert file_digest(ROOT/rec['path'])==rec['sha256']
        with np.load(ROOT/rec['path'],allow_pickle=False) as z: q={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(q['ids'],ids)
        for act in cfg['actions']:
            receipt=json.loads((trial(cfg,key,act)/'complete.json').read_text())
            cp=joblib.load(ROOT/receipt['checkpoint']);old=old_head(frozen[key,act])
            assert file_digest(ROOT/receipt['checkpoint'])==receipt['checkpoint_sha256']
            assert cp['identity']==receipt['identity'] and cp['identity']['easy_cut']==cutoff
            assert cp['preprocess']['training_sites']==sorted(set(cfg['sites'])-{meta['outer_site']})
            np.testing.assert_array_equal(cp['draws'],old['draws'])
            np.testing.assert_array_equal(cp['preprocess']['known'],mask[train].all(1))
            assert cp['draws'].sum()==768000 and not cp['draws'][~mask[train].all(1)].any()
            assert len(cp['model'].estimators_)==128
            assert [r['trees'] for r in cp['trace']]==list(range(16,129,16))
            for field in ('mean','std','known'):
                np.testing.assert_array_equal(cp['preprocess'][field],old['preprocess'][field])
            if act=='damped_velocity_005':
                k=BASELINES.index(act);prediction=data['geometry'][ids,308+24*k:332+24*k].reshape(-1,12,2)
                train_prediction=data['geometry'][train,308+24*k:332+24*k].reshape(-1,12,2)
            else:
                path=ep[key]['path'] if act=='eqmotion' else mv[key]['outer_prediction']['prediction']['path']
                with np.load(ROOT/path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ids);prediction=z['prediction'].copy()
                training_meta=ev[key] if act=='eqmotion' else mv[key]
                with np.load(ROOT/training_meta['inputs_path'],allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],train);train_prediction=z['prediction'].copy()
            td=np.sqrt(np.sum((train_prediction.astype(float)-base[train].astype(float))**2,axis=2)).mean(1)*data['scale'][train]
            np.testing.assert_array_equal(cp['sample_weight'],cp['draws']*(td>0))
            counts['fit_budgets']+=1
            dist=np.sqrt(np.sum((prediction.astype(float)-base[ids].astype(float))**2,axis=2)).mean(1)*data['scale'][ids]
            np.testing.assert_allclose(dist,q[act+'__distance'],rtol=1e-10,atol=1e-10)
            score=q[act+'__score'];f=q[act+'__moments'];past=data['geometry'][ids,:16].reshape(-1,8,2)
            support=np.any(past[:,-1]!=past[:,-2],axis=1)&(dist>0)
            net=support&(score[:,0]>score[:,1])
            chosen=dict(net_stop=net,strict_stop=net&(score[:,1]<=.1*score[:,0]),
                joint_easy_moment=net&(f[:,1]>0)&(f[:,0]*dist<=.02*f[:,1]*cutoff),
                product_easy_marginals=net&(f[:,1]>0)&(f[:,2]*f[:,3]*dist<=.02*f[:,1]*cutoff))
            count=min(chosen['joint_easy_moment'].sum(),chosen['product_easy_marginals'].sum())
            for side,pol in [('joint','joint_easy_moment'),('product','product_easy_marginals')]:
                ranked=sorted(np.flatnonzero(chosen[pol]),key=lambda i:(float(score[i,1]-score[i,0]),int(ids[i])))
                bits=np.zeros(len(ids),bool);bits[ranked[:count]]=True;chosen['matched_'+side]=bits
            counts['matched_pairs']+=1
            for pol,bits in chosen.items():
                np.testing.assert_array_equal(bits,q[act+'__'+pol]);choices[seed,act,pol][ids]=bits
                counts['decision_arrays']+=1
            costs[seed,act][ids],ends[seed,act][ids]=distances(prediction,truth[ids],mask[ids],data['scale'][ids])
            point_distance=np.sqrt(np.sum((prediction.astype(float)-base[ids].astype(float))**2,axis=2))*data['scale'][ids,None]
            point_gain=np.sqrt(np.sum((base[ids].astype(float)-truth[ids].astype(float))**2,axis=2))
            point_gain-=np.sqrt(np.sum((prediction.astype(float)-truth[ids].astype(float))**2,axis=2))
            observed=np.where(mask[ids],point_gain*data['scale'][ids,None],0).sum(1)/12
            radius=np.where(mask[ids],0,point_distance).sum(1)/12
            lower[seed,act][ids],upper[seed,act][ids]=observed-radius,observed+radius
        print(json.dumps(dict(view=key,state='independent_view_verified',**counts)),flush=True)
    for act in cfg['actions']:
        for pol in policies:
            report=a['summary'][act+'__'+pol];aa=[];ff=[]
            for seed in cfg['seeds']:
                bits=choices[seed,act,pol];ade=np.where(bits,costs[seed,act],cv);fde=np.where(bits,ends[seed,act],cf)
                aa.append(ade);ff.append(fde);r=report['seeds'][str(seed)]
                assert r['selected']==bits.sum() and r['selected_unknown']==(bits&~mask.any(1)).sum()
                assert r['selected_incomplete']==(bits&~mask.all(1)).sum()
                assert r['zero_CV_harmed']==(ade[groups['zero_CV']]>0).sum()
                for site in cfg['sites']:
                    bounds=[np.where(bits,b[seed,act],0)[data['sites']==site].mean() for b in (lower,upper)]
                    np.testing.assert_allclose(bounds,r['full_grid_gain_bounds'][site],rtol=1e-10,atol=1e-10)
                counts['scene_reductions']+=check_metrics(ade,cv,data['sites'],cfg['sites'],r['ADE'])
                counts['scene_reductions']+=check_metrics(fde,cf,data['sites'],cfg['sites'],r['FDE'])
                for name,m in groups.items():
                    counts['scene_reductions']+=check_metrics(ade[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][name])
            mean=np.mean(aa,axis=0)
            counts['scene_reductions']+=check_metrics(mean,cv,data['sites'],cfg['sites'],report['ADE'])
            counts['scene_reductions']+=check_metrics(np.mean(ff,0),cf,data['sites'],cfg['sites'],report['FDE'])
            for name,m in groups.items():
                counts['scene_reductions']+=check_metrics(mean[m],cv[m],data['sites'][m],cfg['sites'],report['subsets'][name])
        for left,right in [('joint_easy_moment','product_easy_marginals'),('joint_easy_moment','strict_stop'),('matched_joint','matched_product')]:
            r=a['contrasts'][act+'__'+left+'_minus_'+right]
            l,rr=a['summary'][act+'__'+left],a['summary'][act+'__'+right]
            for name in ('all','hard','positive_easy'):
                check_contrast(l['ADE'] if name=='all' else l['subsets'][name],rr['ADE'] if name=='all' else rr['subsets'][name],r[name])
                counts['contrasts']+=1
    result=dict(all_checks_passed=True,analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)),**counts,same_agent_independent_arithmetic=True,
        independent_research_confirmation=False,scope='raw_future_error_recount_policy_arithmetic_sampling_and_scene_reductions',
        deployment=False)
    assert_current(identity);immutable_json(public/'independent_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
