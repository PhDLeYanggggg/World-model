"""Separate label, simplex, weight, policy and metric checks for log-cost fitting."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_log_cost import load,load_states
from scripts.run_m3w_bounded_cost import read_arrays,training_data
from scripts.run_m3w_native_forecast import assert_current,immutable_json,array_hash
from scripts.verify_m3w_native_joint_controls import distances,check_reduction
from scripts.verify_m3w_tempered_cost import expected_choices,check_contrast
from src.world_model.m3w_bounded_cost_head import build,predict
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg,data,views,predictions,prior,refs,identity=load()
    records,states=load_states(cfg,views,identity);public=ROOT/cfg['reports']
    report=json.loads((public/'analysis.json').read_text());replay=json.loads((public/'replay.json').read_text())
    assert report['identity']==identity and replay['all_checks_passed']
    assert replay['analysis_sha256']==file_digest(public/'analysis.json')
    assert report['decision_manifest_sha256']==file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n=len(data['sites']);policies=('net_stop','strict_stop','matched_count')
    chosen={s:{p:np.zeros(n,bool) for p in policies} for s in cfg['seeds']}
    forecasts={s:np.empty((n,12,2),np.float32) for s in cfg['seeds']}
    fit_rows=choices=0
    for key,meta in views.items():
        ids,x,labels,d,pr=training_data(meta,data)
        with np.load(ROOT/meta['inputs_path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);trainp=z['prediction'].copy()
        ty,tv=read_arrays(data,ids,'target'),read_arrays(data,ids,'valid')
        tb=data['geometry'][ids,332:356].reshape(-1,12,2)
        bc,_=distances(tb,ty,tv,data['scale'][ids]);ec,_=distances(trainp,ty,tv,data['scale'][ids])
        expected=np.column_stack((np.maximum(bc-ec,0),np.maximum(ec-bc,0)))
        expected[~tv.all(1)]=np.nan
        np.testing.assert_allclose(expected,labels,rtol=1e-12,atol=1e-9,equal_nan=True)
        np.testing.assert_array_equal(pr['known'],tv.all(1))
        reference=refs[key,'tempered'];model=build(x.shape[1],cfg['training']['width'],meta['seed'])
        model.load_state_dict(reference['model']);score=predict(model,x,d,pr,'bounded_native')
        past=data['geometry'][ids,:16].reshape(-1,8,2)
        bits=(d>0)&np.any(past[:,-1]!=past[:,-2],axis=1)&(score[:,0]>score[:,1])&(score[:,1]<=.1*score[:,0])
        weight=np.ones(len(ids));weight[bits]=4.
        weight/=np.dot(pr['weights'],weight);weight[~pr['known']]=0;weight=weight.astype(np.float32)
        cp=states[key]
        assert cp['objective']=='distance_weighted_compositional_log'
        nonzero=pr['known']&(d>0)
        fractions=labels[nonzero]/d[nonzero,None]
        assert np.all(fractions>=0) and np.all(fractions.sum(1)<=1+1e-9)
        assert np.all(labels[pr['known']&(d==0)]==0)
        np.testing.assert_array_equal(cp['loss_weights'],weight)
        assert cp['identity']['inputs_sha256']==array_hash(ids,x,d)
        assert cp['identity']['region_sha256']==array_hash(ids,bits,weight)
        assert cp['identity']['labels_sha256']==array_hash(labels)
        assert cp['identity']['complete_mask_sha256']==array_hash(pr['known'])
        assert cp['step']==12000 and cp['settings']==cfg['training'] and cp['seed']==meta['seed']
        np.testing.assert_array_equal(cp['draws'],reference['draws'])
        assert cp['draws'].sum()==3072000 and not cp['draws'][~pr['known']].any()
        assert records[key]['emphasized_complete_rows']==int((bits&pr['known']).sum())
        fit_rows+=len(ids)
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            hi,pred=z['ids'].copy(),z['prediction'].copy()
        np.testing.assert_array_equal(hi,np.flatnonzero(data['sites']==meta['outer_site']))
        archive=next(r for r in report['archives'] if r['view']==key)
        assert file_digest(ROOT/archive['path'])==archive['sha256']
        hp=data['geometry'][hi,:16].reshape(-1,8,2)
        baseline=data['geometry'][hi,332:356].reshape(-1,12,2)
        delta=pred.astype(float)-baseline.astype(float)
        distance=np.sqrt(np.sum(delta**2,axis=-1)).mean(1)*data['scale'][hi]
        supported=(distance>0)&np.any(hp[:,-1]!=hp[:,-2],axis=1)
        old=next(r for r in prior['archives'] if r['view']==key)
        with np.load(ROOT/old['path'],allow_pickle=False) as z:anchor=z['anchor'].copy()
        with np.load(ROOT/archive['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],hi);np.testing.assert_array_equal(z['anchor'],anchor)
            np.testing.assert_allclose(z['distance'],distance,rtol=1e-12,atol=1e-10)
            for policy,use in expected_choices(z['score'],supported,hi,anchor).items():
                np.testing.assert_array_equal(use,z[policy]);chosen[meta['seed']][policy][hi]=use;choices+=1
        forecasts[meta['seed']][hi]=pred
    y,valid=read_arrays(data,np.arange(n),'target'),read_arrays(data,np.arange(n),'valid')
    baseline=data['geometry'][:,332:356].reshape(n,12,2)
    cv,cf=distances(baseline,y,valid,data['scale']);full=valid.all(1)
    masks=dict(complete=full,zero_CV=full&(cv==0),hard=np.zeros(n,bool),positive_easy=np.zeros(n,bool))
    for key,meta in views.items():
        ids=np.flatnonzero(data['sites']==meta['outer_site']);pr=states[key]['preprocess']
        masks['hard'][ids]=cv[ids]>=pr['hard_cut'];masks['positive_easy'][ids]=(cv[ids]>0)&(cv[ids]<=pr['positive_easy_cut'])
    errors,bounds={},{}
    for seed,p in forecasts.items():
        errors[seed]=distances(p,y,valid,data['scale'])
        safe=np.where(valid[...,None],y,0).astype(float)
        dd=np.sqrt(np.sum((p.astype(float)-baseline.astype(float))**2,axis=-1))*data['scale'][:,None]
        pe=np.sqrt(np.sum((p.astype(float)-safe)**2,axis=-1))*data['scale'][:,None]
        be=np.sqrt(np.sum((baseline.astype(float)-safe)**2,axis=-1))*data['scale'][:,None]
        observed=np.where(valid,be-pe,0).sum(1)/12;unknown=np.where(valid,0,dd).sum(1)/12
        bounds[seed]=observed-unknown,observed+unknown
    reductions=0
    for policy in policies:
        ades,fdes=[],[]
        for seed in cfg['seeds']:
            use=chosen[seed][policy];ade,fde=np.where(use,errors[seed][0],cv),np.where(use,errors[seed][1],cf)
            ades.append(ade);fdes.append(fde);r=report['summaries'][policy]['seeds'][str(seed)]
            assert r['selected']==int(use.sum()) and r['selected_unknown']==int((use&~valid.any(1)).sum())
            assert r['selected_incomplete']==int((use&~full).sum())
            assert r['zero_CV_harmed']==int((ade[masks['zero_CV']]>0).sum())
            reductions+=check_reduction(ade,cv,data['sites'],cfg['sites'],r['ADE'])
            reductions+=check_reduction(fde,cf,data['sites'],cfg['sites'],r['FDE'])
            for g,m in masks.items():reductions+=check_reduction(ade[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][g])
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use,v,0)[data['sites']==site].mean() for v in bounds[seed]],
                    r['full_grid_absolute_gain_bounds'][site],rtol=1e-10,atol=1e-10)
        r=report['summaries'][policy]
        reductions+=check_reduction(np.mean(ades,0),cv,data['sites'],cfg['sites'],r['ADE'])
        reductions+=check_reduction(np.mean(fdes,0),cf,data['sites'],cfg['sites'],r['FDE'])
        for g,m in masks.items():reductions+=check_reduction(np.mean(ades,0)[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][g])
        for a in ('frozen_region','tempered','native','fraction'):
            check_contrast(r['ADE']['by_scene'],prior['summaries'][a][policy]['ADE']['by_scene'],cfg['sites'],report['contrasts'][policy][a])
    p=report['summaries']['strict_stop'];c=report['contrasts']['strict_stop']['frozen_region']
    expected_gate=dict(positive_primary_ci=c['ci95_pp'][0]>0,
        each_seed_positive_cv=all(r['ADE']['equal_scene_gain_percent']>0 for r in p['seeds'].values()),
        aggregate_easy=all(-r['subsets']['positive_easy']['equal_scene_gain_percent']<=2 for r in p['seeds'].values()),
        each_scene_seed_easy=all(-v['gain_percent']<=2 for r in p['seeds'].values() for v in r['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=all(r['zero_CV_harmed']==0 for r in p['seeds'].values()))
    assert expected_gate==report['primary_gates'] and all(expected_gate.values())==report['primary_joint_empirical_pass']
    assert report['reference_summaries']==prior['summaries']
    assert sum(r['fit']['step'] for r in records.values())==report['new_updates']==144000
    assert sum(r['fit']['total_draws'] for r in records.values())==report['new_draws']==36864000
    out=dict(analysis_sha256=file_digest(public/'analysis.json'),verifier_sha256=file_digest(Path(__file__)),
        all_checks_passed=True,policy_choices_checked=choices,scene_reductions=reductions,
        supervision_rows_checked=fit_rows,weight_vectors_checked=12,
        preprocessing_and_model_forward_shared=True,conditional_diagnostics_replayed_not_separately_implemented=True,
        independent_implementation_same_agent=True,independent_research_confirmation=False,new_training=False)
    assert_current(identity);immutable_json(public/'independent_verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
