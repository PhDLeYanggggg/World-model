"""Separate policy, matched-count, outcome and uncertainty arithmetic."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_cross_objective_review import load
from scripts.run_m3w_native_forecast import assert_current,immutable_json
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.verify_m3w_native_joint_controls import distances,check_reduction
from scripts.verify_m3w_tempered_cost import check_contrast
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg,_,data,views,predictions,_,prior,_,nomination,identity=load()
    public=ROOT/cfg['reports'];r=json.loads((public/'analysis.json').read_text())
    replay=json.loads((public/'replay.json').read_text())
    assert r['identity']==identity and replay['all_checks_passed']
    assert replay['analysis_sha256']==file_digest(public/'analysis.json')
    assert r['decision_manifest_sha256']==file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n=len(data['sites']);policies=('nomination','reviewed','matched_nomination')
    chosen={s:{p:np.zeros(n,bool) for p in policies} for s in cfg['seeds']}
    forecasts={s:np.empty((n,12,2),np.float32) for s in cfg['seeds']}
    checked=0
    for key,meta in views.items():
        item=next(a for a in r['archives'] if a['view']==key)
        assert file_digest(ROOT/item['path'])==item['sha256']
        with np.load(ROOT/item['path'],allow_pickle=False) as z:
            ids=z['ids'];a,b,c=z['nomination_score'],z['native_score'],z['fraction_score']
            past=data['geometry'][ids,:16].reshape(-1,8,2)
            nom=(z['distance']>0)&np.any(past[:,-1]!=past[:,-2],axis=1)
            nom&=(a[:,0]>a[:,1])&(a[:,1]<=a[:,0]/10)
            reviewed=nom&(b[:,1]<=.1*a[:,0])&(c[:,1]<=.1*a[:,0])
            count=int(reviewed.sum());pool=np.flatnonzero(nom)
            ranked=sorted(pool,key=lambda j:(-(a[j,0]-a[j,1]),int(ids[j])))
            matched=np.zeros(len(ids),bool);matched[ranked[:count]]=True
            for p,bits in dict(nomination=nom,reviewed=reviewed,matched_nomination=matched).items():
                np.testing.assert_array_equal(bits,z[p]);chosen[meta['seed']][p][ids]=bits;checked+=1
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);forecasts[meta['seed']][ids]=z['prediction']
    y,valid=read_arrays(data,np.arange(n),'target'),read_arrays(data,np.arange(n),'valid')
    baseline=data['geometry'][:,332:356].reshape(n,12,2)
    cv,cf=distances(baseline,y,valid,data['scale']);full=valid.all(1)
    masks=dict(complete=full,zero_CV=full&(cv==0),hard=np.zeros(n,bool),positive_easy=np.zeros(n,bool))
    for key,meta in views.items():
        ids=np.flatnonzero(data['sites']==meta['outer_site']);pr=nomination[key]['preprocess']
        masks['hard'][ids]=cv[ids]>=pr['hard_cut']
        masks['positive_easy'][ids]=(cv[ids]>0)&(cv[ids]<=pr['positive_easy_cut'])
    errors,bounds={},{}
    for seed,p in forecasts.items():
        errors[seed]=distances(p,y,valid,data['scale'])
        safe=np.where(valid[...,None],y,0).astype(float)
        dd=np.sqrt(((p.astype(float)-baseline.astype(float))**2).sum(-1))*data['scale'][:,None]
        pe=np.sqrt(((p.astype(float)-safe)**2).sum(-1))*data['scale'][:,None]
        be=np.sqrt(((baseline.astype(float)-safe)**2).sum(-1))*data['scale'][:,None]
        observed=np.where(valid,be-pe,0).sum(1)/12;unknown=np.where(valid,0,dd).sum(1)/12
        bounds[seed]=observed-unknown,observed+unknown
    reductions=0
    for policy in policies:
        ades,fdes=[],[]
        for seed in cfg['seeds']:
            use=chosen[seed][policy];ade=np.where(use,errors[seed][0],cv);fde=np.where(use,errors[seed][1],cf)
            ades.append(ade);fdes.append(fde);v=r['summaries'][policy]['seeds'][str(seed)]
            assert v['selected']==int(use.sum()) and v['selected_unknown']==int((use&~valid.any(1)).sum())
            assert v['selected_incomplete']==int((use&~full).sum())
            assert v['zero_CV_harmed']==int((ade[masks['zero_CV']]>0).sum())
            reductions+=check_reduction(ade,cv,data['sites'],cfg['sites'],v['ADE'])
            reductions+=check_reduction(fde,cf,data['sites'],cfg['sites'],v['FDE'])
            for g,m in masks.items():reductions+=check_reduction(ade[m],cv[m],data['sites'][m],cfg['sites'],v['subsets'][g])
            for site in cfg['sites']:
                np.testing.assert_allclose(v['full_grid_absolute_gain_bounds'][site],
                    [np.where(use,q,0)[data['sites']==site].mean() for q in bounds[seed]],rtol=1e-10,atol=1e-10)
        v=r['summaries'][policy]
        reductions+=check_reduction(np.mean(ades,0),cv,data['sites'],cfg['sites'],v['ADE'])
        reductions+=check_reduction(np.mean(fdes,0),cf,data['sites'],cfg['sites'],v['FDE'])
        for g,m in masks.items():reductions+=check_reduction(np.mean(ades,0)[m],cv[m],data['sites'][m],cfg['sites'],v['subsets'][g])
    for policy in ('nomination','matched_nomination'):
        check_contrast(r['summaries']['reviewed']['ADE']['by_scene'],r['summaries'][policy]['ADE']['by_scene'],
            cfg['sites'],r['contrasts'][policy])
    assert r['summaries']['nomination']==prior['summaries']['strict_stop']
    v=r['summaries']['reviewed'];ci=r['contrasts']['matched_nomination']['ci95_pp']
    gates=dict(positive_matched_count_ci=ci[0]>0,positive_vs_cv_ci=v['ADE']['scene_bootstrap_ci95'][0]>0,
        each_seed_positive_cv=all(z['ADE']['equal_scene_gain_percent']>0 for z in v['seeds'].values()),
        aggregate_easy=all(-z['subsets']['positive_easy']['equal_scene_gain_percent']<=2 for z in v['seeds'].values()),
        each_scene_seed_easy=all(-w['gain_percent']<=2 for z in v['seeds'].values()
            for w in z['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=all(z['zero_CV_harmed']==0 for z in v['seeds'].values()))
    assert gates==r['primary_gates'] and all(gates.values())==r['primary_joint_empirical_pass']
    out=dict(analysis_sha256=file_digest(public/'analysis.json'),verifier_sha256=file_digest(Path(__file__)),
        all_checks_passed=True,choices_checked=checked,scene_reductions=reductions,
        matched_counts_by_view=12,conditional_diagnostics_replayed_not_separately_implemented=True,
        shared_preprocessing_model_forward=True,independent_research_replication=False,new_training=False)
    assert_current(identity);immutable_json(public/'independent_verification.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
