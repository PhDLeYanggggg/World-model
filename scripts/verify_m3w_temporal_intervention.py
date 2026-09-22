"""Separate transform, label, choice, reduction and bound verification."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_temporal_intervention import load, load_states, training_arrays, check_state
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from scripts.verify_m3w_tempered_cost import check_contrast
from src.world_model.m3w_temporal_intervention import ARMS, POLICIES, policy_arm
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def manual_candidates(b, p):
    b, p = b.astype(float), p.astype(float)
    w = np.linspace(0., 1., 12)
    dd = np.sqrt(np.square(p-b).sum(-1))
    den = dd.sum(1)
    alpha = np.divide((dd*w).sum(1), den, out=np.zeros(len(b)), where=den>0)
    out = dict(ramp=w[None,:,None]*p+(1-w)[None,:,None]*b,
               uniform=alpha[:,None,None]*p+(1-alpha[:,None,None])*b)
    # Preserve exact coincidence; convex-form roundoff must not create eligibility.
    for name in out:
        out[name] = np.where(p==b, b, out[name])
    out['ramp'][:, 0] = b[:, 0]; out['ramp'][:, -1] = p[:, -1]
    return out, alpha


def verify_contrasts(report, scalar, sites):
    for name, c in report['contrasts'].items():
        ref = scalar['summaries']['strict_stop'] if name=='scalar_log_strict' else report['summaries'][name]
        check_contrast(report['summaries']['ramp_strict']['ADE']['by_scene'], ref['ADE']['by_scene'], sites, c)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, scalar, refs, identity = load()
    records, states = load_states(cfg, views, identity); public = ROOT/cfg['reports']
    report = json.loads((public/'analysis.json').read_text()); replay = json.loads((public/'replay.json').read_text())
    assert report['identity']==identity and replay['all_checks_passed']
    assert replay['analysis_sha256']==file_digest(public/'analysis.json')
    assert report['decision_manifest_sha256']==file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n = len(data['sites']); chosen = {s:{p:np.zeros(n,bool) for p in POLICIES} for s in cfg['seeds']}
    forecasts = {s:{a:np.empty((n,12,2),float) for a in ARMS} for s in cfg['seeds']}
    fit_rows = choices = 0
    for key, meta in views.items():
        for arm in ARMS:
            ids,x,labels,d,pr,bits,w,q = training_arrays(meta,data,refs,key,cfg,arm)
            with np.load(ROOT/meta['inputs_path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(ids,z['ids']); p=z['prediction'].copy()
            b=data['geometry'][ids,332:356].reshape(-1,12,2)
            expected,_=manual_candidates(b,p)
            np.testing.assert_allclose(q,expected[arm],rtol=1e-12,atol=1e-12)
            y,v=read_arrays(data,ids,'target'),read_arrays(data,ids,'valid')
            bc,_=distances(b,y,v,data['scale'][ids]); nc,_=distances(expected[arm],y,v,data['scale'][ids])
            target=np.column_stack((np.maximum(bc-nc,0),np.maximum(nc-bc,0)));target[~v.all(1)]=np.nan
            np.testing.assert_allclose(target,labels,rtol=1e-10,atol=1e-8,equal_nan=True)
            cp=states[key,arm];check_state(cp,refs[key,'frozen_region'],pr,w,cfg)
            assert cp['identity']['inputs_sha256']==array_hash(ids,x,d,q)
            assert cp['identity']['labels_sha256']==array_hash(labels)
            assert cp['identity']['region_sha256']==array_hash(ids,bits,w)
            fit_rows+=len(ids)
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            hi,p=z['ids'].copy(),z['prediction'].copy()
        b=data['geometry'][hi,332:356].reshape(-1,12,2); q,alpha=manual_candidates(b,p)
        d=np.sqrt(np.square(q['ramp']-b).sum(-1)).mean(1)*data['scale'][hi]
        du=np.sqrt(np.square(q['uniform']-b).sum(-1)).mean(1)*data['scale'][hi]
        np.testing.assert_allclose(d,du,rtol=1e-10,atol=1e-8)
        archive=next(r for r in report['archives'] if r['view']==key)
        assert file_digest(ROOT/archive['path'])==archive['sha256']
        with np.load(ROOT/archive['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(hi,z['ids']);np.testing.assert_allclose(alpha,z['uniform_alpha'],atol=1e-12)
            np.testing.assert_allclose(d,z['distance'],rtol=1e-10,atol=1e-8)
            past=data['geometry'][hi,:16].reshape(-1,8,2)
            support=(d>0)&np.any(past[:,-1]!=past[:,-2],axis=1)
            def strict(score):
                return support&(score[:,0]>score[:,1])&(score[:,1]<=.1*score[:,0])
            r,u=strict(z['ramp_score']),strict(z['uniform_score'])
            score=z['uniform_score'];pool=np.flatnonzero(support)
            order=sorted(pool,key=lambda i: (float(score[i,1]-score[i,0]),int(hi[i])))
            matched=np.zeros(len(hi),bool);matched[order[:int(r.sum())]]=True
            masks=dict(ramp_strict=r,uniform_strict=u,uniform_matched=matched,ramp_at_uniform=u,
                       uniform_at_ramp=r,ramp_uncontrolled=support,uniform_uncontrolled=support)
            for name,use in masks.items():
                np.testing.assert_array_equal(use,z[name]);chosen[meta['seed']][name][hi]=use;choices+=1
        for arm in ARMS:
            forecasts[meta['seed']][arm][hi]=q[arm]
    y,v=read_arrays(data,np.arange(n),'target'),read_arrays(data,np.arange(n),'valid')
    b=data['geometry'][:,332:356].reshape(n,12,2);cv,cf=distances(b,y,v,data['scale']);full=v.all(1)
    masks=dict(complete=full,zero_CV=full&(cv==0),hard=np.zeros(n,bool),positive_easy=np.zeros(n,bool))
    for key,meta in views.items():
        ix=data['sites']==meta['outer_site'];pr=states[key,'ramp']['preprocess']
        masks['hard'][ix]=cv[ix]>=pr['hard_cut'];masks['positive_easy'][ix]=(cv[ix]>0)&(cv[ix]<=pr['positive_easy_cut'])
    errors,bounds={},{}
    safe=np.where(v[...,None],y,0).astype(float)
    be=np.sqrt(np.square(b.astype(float)-safe).sum(-1))*data['scale'][:,None]
    for seed in cfg['seeds']:
        for arm in ARMS:
            p=forecasts[seed][arm];errors[seed,arm]=distances(p,y,v,data['scale'])
            pe=np.sqrt(np.square(p-safe).sum(-1))*data['scale'][:,None]
            dd=np.sqrt(np.square(p-b).sum(-1))*data['scale'][:,None]
            observed=np.where(v,be-pe,0).sum(1)/12;unknown=np.where(v,0,dd).sum(1)/12
            bounds[seed,arm]=observed-unknown,observed+unknown
    reductions=0
    for name in POLICIES:
        arm=policy_arm(name);ades,fdes=[],[]
        for seed in cfg['seeds']:
            use=chosen[seed][name];ade,fde=np.where(use,errors[seed,arm][0],cv),np.where(use,errors[seed,arm][1],cf)
            ades.append(ade);fdes.append(fde);r=report['summaries'][name]['seeds'][str(seed)]
            assert r['selected']==int(use.sum()) and r['selected_unknown']==int((use&~v.any(1)).sum())
            assert r['selected_incomplete']==int((use&~full).sum())
            assert r['zero_CV_harmed']==int((ade[masks['zero_CV']]>0).sum())
            reductions+=check_reduction(ade,cv,data['sites'],cfg['sites'],r['ADE'])
            reductions+=check_reduction(fde,cf,data['sites'],cfg['sites'],r['FDE'])
            for g,m in masks.items():
                reductions+=check_reduction(ade[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][g])
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use,q,0)[data['sites']==site].mean() for q in bounds[seed,arm]],
                                           r['full_grid_absolute_gain_bounds'][site],rtol=1e-10,atol=1e-8)
        r=report['summaries'][name]
        reductions+=check_reduction(np.mean(ades,0),cv,data['sites'],cfg['sites'],r['ADE'])
        reductions+=check_reduction(np.mean(fdes,0),cf,data['sites'],cfg['sites'],r['FDE'])
        for g,m in masks.items():
            reductions+=check_reduction(np.mean(ades,0)[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][g])
    verify_contrasts(report,scalar,cfg['sites']);assert_current(identity)
    immutable_json(public/'separate_verification.json',dict(analysis_sha256=file_digest(public/'analysis.json'),
                   all_checks_passed=True,fitting_arm_row_instances=fit_rows,policy_choices_checked=choices,
                   scene_reductions_checked=reductions,checkpoints_draw_matched=24,
                   bounds_checked=True,same_agent_shared_source_arrays=True,
                   prefix_and_interaction_diagnostics_checkpoint_replayed_only=True,
                   independent_research_confirmation=False))
    print(json.dumps(dict(verified=True,fit_rows=fit_rows,choices=choices,scene_reductions=reductions)))


if __name__=='__main__':
    main()
