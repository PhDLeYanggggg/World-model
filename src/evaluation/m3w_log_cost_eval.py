"""Fixed compositional-log-loss readout against frozen-region square loss."""
from pathlib import Path
import numpy as np
from scripts import run_m3w_bounded_cost as bounded
from scripts.run_m3w_log_cost import weighted_data, check_state
from src.evaluation.m3w_conditional_cost_eval import gate, cost_means
from scripts.run_m3w_tempered_cost import causal_scores
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_bounded_cost_head import build,predict
from src.evaluation.m3w_conditional_cost_audit import strict_bits
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds

ROOT=Path(__file__).resolve().parents[2]
POLICIES=('net_stop','strict_stop','matched_count')


def evaluate(cfg,data,views,predictions,prior,refs,identity,records,states,beat,verify=False):
    root,public=ROOT/cfg['output'],ROOT/cfg['reports'];n=len(data['sites'])
    chosen={s:{p:np.zeros(n,bool) for p in POLICIES} for s in cfg['seeds']}
    candidates={s:np.empty((n,12,2),np.float32) for s in cfg['seeds']}
    old={r['view']:r for r in prior['archives']}
    archives,scores,old_scores,old_bits={}, {}, {}, {}
    for key,meta in views.items():
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            ids,prediction=z['ids'].copy(),z['prediction'].copy()
        np.testing.assert_array_equal(ids,np.flatnonzero(data['sites']==meta['outer_site']))
        with np.load(ROOT/old[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            anchor=z['anchor'].copy();old_scores[key]=z['score'].copy()
            old_bits[key]=z['strict_stop'].copy()
        score,d=causal_scores(data,ids,prediction,states[key],cfg)
        past=data['geometry'][ids,:16].reshape(-1,8,2)
        bits=bounded.selections(score,past,d,anchor,ids)
        arrays=dict(ids=ids,score=score,distance=d,anchor=anchor,**bits)
        for p,use in bits.items():chosen[meta['seed']][p][ids]=use
        candidates[meta['seed']][ids]=prediction;scores[key]=score
        path=root/'decisions'/f'{key}.npz'
        if verify and not path.exists():raise ValueError('Cannot verify missing decisions')
        write_arrays(path,arrays)
        archives[key]=dict(view=key,path=str(path.relative_to(ROOT)),sha256=file_digest(path))
        beat(state='replayed' if verify else 'decisions_frozen',view=key)
    immutable_json(root/'decisions_complete.json',dict(identity=identity,archives=list(archives.values()),
        future_targets_used_in_decisions=False,score_rows=n*len(cfg['seeds'])))
    y,valid=bounded.read_arrays(data,np.arange(n),'target'),bounded.read_arrays(data,np.arange(n),'valid')
    baseline=data['geometry'][:,332:356].reshape(n,12,2)
    cv,cf=native_errors(baseline,y,valid,data['scale']);full=valid.all(1)
    masks=dict(complete=full,zero_CV=full & (cv==0),hard=np.zeros(n,bool),positive_easy=np.zeros(n,bool))
    errors={s:native_errors(p,y,valid,data['scale']) for s,p in candidates.items()}
    bounds={s:partial_gain_bounds(p,baseline,y,valid,data['scale']) for s,p in candidates.items()}
    quality=[]
    for key,meta in views.items():
        ids=np.flatnonzero(data['sites']==meta['outer_site']);cp=states[key];pr=cp['preprocess']
        masks['hard'][ids]=cv[ids]>=pr['hard_cut']
        masks['positive_easy'][ids]=(cv[ids]>0)&(cv[ids]<=pr['positive_easy_cut'])
        delta=cv[ids]-errors[meta['seed']][0][ids]
        costs=np.column_stack((np.maximum(delta,0),np.maximum(-delta,0)))
        ti,x,ty,td,tpr,tbits,w,ts=weighted_data(meta,data,refs[key,'tempered'],cfg)
        check_state(cp,refs[key,'tempered'],w,cfg)
        model=build(x.shape[1],cfg['training']['width'],meta['seed']);model.load_state_dict(cp['model'])
        train_score=predict(model,x,td,tpr,'bounded_native')
        new_train_bits=strict_bits(train_score,data['geometry'][ti,:16].reshape(-1,8,2),td)
        model.load_state_dict(refs[key,'frozen_region']['model'])
        ts=predict(model,x,td,tpr,'bounded_native')
        tbits=strict_bits(ts,data['geometry'][ti,:16].reshape(-1,8,2),td)
        for pop,oldscore,newscore,target,complete,original,new in (
            ('fitting',ts,train_score,ty,tpr['known'],tbits,new_train_bits),
            ('held_source',old_scores[key],scores[key],costs,full[ids],old_bits[key],chosen[meta['seed']]['strict_stop'][ids])):
            for name,bits in dict(all=np.ones(len(complete),bool),old_region=original,new_region=new).items():
                use=bits&complete
                quality.append(dict(view=key,population=pop,group=name,rows=int(bits.sum()),
                    complete=int(use.sum()),incomplete=int((bits&~complete).sum()),
                    costs=cost_means(oldscore,newscore,target,use)))
        beat(state='fitting_and_held_diagnostics_complete',view=key)
    def metric(a,r,mask=None):
        if mask is None:mask=np.ones(n,bool)
        return paired_scene_metrics(a[mask],r[mask],data['sites'][mask],expected_scenes=cfg['sites'],
            dataset='sdd',coordinate_unit='annotation_pixel',bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries={}
    for policy in POLICIES:
        seeds,ades,fdes={},[],[]
        for seed in cfg['seeds']:
            use=chosen[seed][policy];ade,fde=np.where(use,errors[seed][0],cv),np.where(use,errors[seed][1],cf)
            ades.append(ade);fdes.append(fde)
            seeds[str(seed)]=dict(ADE=metric(ade,cv),FDE=metric(fde,cf),selected=int(use.sum()),
                selected_unknown=int((use&~valid.any(1)).sum()),selected_incomplete=int((use&~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']]>0).sum()),
                full_grid_absolute_gain_bounds={site:[float(np.where(use,bounds[seed][k],0)[data['sites']==site].mean())
                    for k in ('lower','upper')] for site in cfg['sites']},
                subsets={g:metric(ade,cv,mask) for g,mask in masks.items()})
        summaries[policy]=dict(ADE=metric(np.mean(ades,0),cv),FDE=metric(np.mean(fdes,0),cf),seeds=seeds,
            subsets={g:metric(np.mean(ades,0),cv,mask) for g,mask in masks.items()})
    contrasts={p:{a:paired_scene_contrast(
        [summaries[p]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
        [prior['summaries'][a][p]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
        for a in ('frozen_region','tempered','native','fraction')} for p in POLICIES}
    checks=gate(summaries['strict_stop'],contrasts['strict_stop']['frozen_region'])
    result=dict(identity=identity,result_source='fresh_12_compositional_log_heads_and_readout',
        reference_source='cached_verified_frozen_region_and_equal_budget_objective_controls',summaries=summaries,
        reference_summaries=prior['summaries'],contrasts=contrasts,archives=list(archives.values()),
        training=[dict(view=k,**{f:r[f] for f in ('fit','rows','supported_rows','emphasized_complete_rows',
            'mean_weight_under_sampler','checkpoint','checkpoint_sha256')}) for k,r in records.items()],
        new_updates=sum(r['fit']['step'] for r in records.values()),new_draws=sum(r['fit']['total_draws'] for r in records.values()),
        conditional_quality=quality,primary_gates=checks,primary_joint_empirical_pass=all(checks.values()),
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),independent_confirmation=False,
        risk_calibrated=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    assert result['new_updates']==144000 and result['new_draws']==36864000 and len(quality)==72
    assert_current(identity);immutable_json(public/'analysis.json',result)
    if verify:immutable_json(public/'replay.json',dict(analysis_sha256=file_digest(public/'analysis.json'),
        checkpoint_endpoints_replayed=12,score_rows=n*len(cfg['seeds']),all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated',primary_gates=checks)
