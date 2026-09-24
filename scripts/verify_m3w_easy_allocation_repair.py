"""Separate query budgets, small-query optima and raw-label metric recount."""
import itertools
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from scripts.run_m3w_easy_allocation import causal_view, ARMS
from scripts.run_m3w_easy_allocation_repair import load
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import file_digest, immutable_json, assert_current
from scripts.verify_m3w_native_joint_controls import distances
from scripts.verify_m3w_protected_motion_controls import check_metrics
from scripts.report_m3w_protected_eqmotion_controls import check_contrast
from src.evaluation.m3w_native_scene_alignment import scene_index


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();cfg,data,context,parent,tr,eq,identity=pack
    root,public=ROOT/cfg['output'],ROOT/cfg['reports']
    a=json.loads((public/'analysis.json').read_text())
    v=json.loads((public/'aggregate_replay.json').read_text())
    d=json.loads((public/'decision_replay.json').read_text())
    assert v['all_checks_passed'] and v['analysis_sha256']==file_digest(public/'analysis.json')
    assert d['all_checks_passed'] and d['decision_manifest_sha256']==file_digest(root/'decisions_complete.json')
    assert a['experiment_sha256']==file_digest(root/'identity.json')
    completed=json.loads((root/'decisions_complete.json').read_text())
    provenance=verify_repair_provenance(completed,identity,data)
    n=len(data['sites']);index=scene_index(data['recordings'],data['frames'],data['tracks']);group=index['group']
    ng=len(index['frames']);assert ng==20932
    choices={(s,act):np.zeros((n,len(ARMS)),bool) for s in cfg['seeds'] for act in cfg['actions']}
    seen={k:np.zeros(n,bool) for k in choices};flags={}
    for k in choices:flags[k]={t:np.zeros(ng,bool) for t in ('selected_optimal','population_optimal','unary_optimal','joint_optimal','matched')}
    for r in completed['receipts']:
        assert file_digest(ROOT/r['path'])==r['sha256'];record=json.loads((ROOT/r['path']).read_text())
        assert file_digest(ROOT/record['path'])==record['sha256']
        key=(int(record['view'].rsplit('seed',1)[1]),record['action'])
        with np.load(ROOT/record['path'],allow_pickle=False) as z:
            ids=z['ids'];assert not seen[key][ids].any()
            assert np.all(data['recordings'][ids]==record['recording'])
            assert set(data['frames'][ids])==set(record['frames'])
            seen[key][ids]=True;choices[key][ids]=z['choices']
        for q in record['queries']:
            loc=ids[data['frames'][ids]==q['frame']];g=int(group[loc[0]])
            assert np.all(group[loc]==g)
            for name in flags[key]:flags[key][name][g]=q[name]
    assert all(v.all() for v in seen.values())
    truth,valid=read_arrays(data,np.arange(n),'target'),read_arrays(data,np.arange(n),'valid')
    base=data['geometry'][:,332:356].reshape(-1,12,2)
    cv,cf=distances(base,truth,valid,data['scale'])
    cuts=json.loads((ROOT/'data/stage_cvpr2027_experiments/protected_motion_controls_v1/decisions_complete.json').read_text())['cuts']
    masks=dict(complete=valid.all(1),zero_CV=valid.all(1)&(cv==0),positive_easy=np.zeros(n,bool),hard=np.zeros(n,bool))
    error={k:np.full(n,np.nan) for k in choices};end={k:np.full(n,np.nan) for k in choices}
    lower={k:np.zeros(n) for k in choices};upper={k:np.zeros(n) for k in choices}
    counts=dict(query_action_seed_instances=0,decision_rows=0,small_query_optima=0,small_query_checks_skipped_large=0,
                scene_reductions=0,paired_contrasts=0)
    for key in tr:
        seed=int(key.rsplit('seed',1)[1])
        for action in cfg['actions']:
            values,pred,scale,cut=causal_view(pack,key,action)
            ids=values['ids'];gg=group[ids];active=np.unique(gg);b=choices[seed,action][ids]
            score=values[action+'__score'];moment=values[action+'__moments'];past=data['geometry'][ids,:16].reshape(-1,8,2)
            dist=np.sqrt(np.sum((pred.astype(float)-base[ids])**2,axis=2)).mean(1)*data['scale'][ids]
            np.testing.assert_allclose(dist,values[action+'__distance'],rtol=1e-10,atol=1e-10)
            support=np.any(past[:,-1]!=past[:,-2],axis=1)&(dist>0)&(score[:,0]>score[:,1])
            q=moment[:,0]*dist;r=moment[:,1]*cut;gain=score[:,0]-score[:,1]
            for name,actual in [('floor',np.zeros(len(ids),bool)),('net_stop',support),
                ('strict_stop',support&(score[:,1]<=.1*score[:,0])),
                ('pointwise',support&(r>0)&(q<=.02*r))]:
                np.testing.assert_array_equal(b[:,ARMS.index(name)],actual)
            assert not np.any(b&~support[:,None])
            total_r=np.bincount(gg,weights=r,minlength=ng)
            def total(v):return np.bincount(gg,weights=v,minlength=ng)
            for name in ('aggregate_selected','aggregate_population','aggregate_unary','aggregate_joint','scene_uniform'):
                bits=b[:,ARMS.index(name)]
                limit=.02*(total(r*bits) if name=='aggregate_selected' else total_r)
                assert np.all(total(q*bits)<=limit+1e-8)
            expected_uniform=(total(support)==total(np.ones(len(ids))))&(total(q)<=.02*total_r)
            np.testing.assert_array_equal(b[:,ARMS.index('scene_uniform')],expected_uniform[gg])
            pop_count=total(b[:,ARMS.index('aggregate_population')])
            matched=flags[seed,action]['matched'][active]
            for name in ('aggregate_unary','aggregate_joint'):
                np.testing.assert_array_equal(total(b[:,ARMS.index(name)])[active][matched],pop_count[active][matched])
            for g in active[np.unique(np.linspace(0,len(active)-1,min(17,len(active)),dtype=int))]:
                loc=np.flatnonzero(gg==g);pool=loc[support[loc]]
                if len(pool)>10:
                    counts['small_query_checks_skipped_large']+=1;continue
                xx=np.array(list(itertools.product((False,True),repeat=len(pool))),bool).reshape(-1,len(pool)) if len(pool) else np.zeros((1,0),bool)
                benefit=xx@gain[pool]
                for name,feasible in [('aggregate_selected',xx@(q[pool]-.02*r[pool])<=1e-10),
                    ('aggregate_population',xx@q[pool]<=.02*r[loc].sum()+1e-10)]:
                    flag='selected_optimal' if name=='aggregate_selected' else 'population_optimal'
                    if not flags[seed,action][flag][g]:continue
                    observed=float(gain[loc]@b[loc,ARMS.index(name)])
                    np.testing.assert_allclose(observed,benefit[feasible].max(),rtol=1e-7,atol=1e-8)
                    counts['small_query_optima']+=1
            error[seed,action][ids],end[seed,action][ids]=distances(pred,truth[ids],valid[ids],data['scale'][ids])
            point_distance=np.sqrt(np.sum((pred.astype(float)-base[ids])**2,axis=2))*data['scale'][ids,None]
            point_gain=np.sqrt(np.sum((base[ids].astype(float)-truth[ids])**2,axis=2))
            point_gain-=np.sqrt(np.sum((pred.astype(float)-truth[ids])**2,axis=2))
            observed=np.where(valid[ids],point_gain*data['scale'][ids,None],0).sum(1)/12
            radius=np.where(valid[ids],0,point_distance).sum(1)/12
            lower[seed,action][ids],upper[seed,action][ids]=observed-radius,observed+radius
            masks['positive_easy'][ids]=(cv[ids]>0)&(cv[ids]<=cut)
            masks['hard'][ids]=cv[ids]>=cuts[key]['hard_cut']
            counts['query_action_seed_instances']+=len(active);counts['decision_rows']+=len(ids)*len(ARMS)
        print(json.dumps(dict(view=key,state='independent_query_risks_verified',**counts)),flush=True)
    for action in cfg['actions']:
        for j,arm in enumerate(ARMS):
            report=a['summary'][action+'__'+arm];aa=[];ff=[]
            for seed in cfg['seeds']:
                bits=choices[seed,action][:,j];ade=np.where(bits,error[seed,action],cv);fde=np.where(bits,end[seed,action],cf)
                r=report['seeds'][str(seed)];aa.append(ade);ff.append(fde)
                assert r['selected']==bits.sum() and r['selected_unknown']==(bits&~valid.any(1)).sum()
                assert r['selected_incomplete']==(bits&~valid.all(1)).sum()
                assert r['zero_CV_harmed']==(ade[masks['zero_CV']]>0).sum()
                for site in cfg['sites']:
                    bounds=[np.where(bits,x[seed,action],0)[data['sites']==site].mean() for x in (lower,upper)]
                    np.testing.assert_allclose(bounds,r['full_grid_gain_bounds'][site],rtol=1e-10,atol=1e-10)
                counts['scene_reductions']+=check_metrics(ade,cv,data['sites'],cfg['sites'],r['ADE'])
                counts['scene_reductions']+=check_metrics(fde,cf,data['sites'],cfg['sites'],r['FDE'])
                for name,m in masks.items():counts['scene_reductions']+=check_metrics(ade[m],cv[m],data['sites'][m],cfg['sites'],r['subsets'][name])
            mean=np.mean(aa,axis=0)
            counts['scene_reductions']+=check_metrics(mean,cv,data['sites'],cfg['sites'],report['ADE'])
            counts['scene_reductions']+=check_metrics(np.mean(ff,axis=0),cf,data['sites'],cfg['sites'],report['FDE'])
            for name,m in masks.items():counts['scene_reductions']+=check_metrics(mean[m],cv[m],data['sites'][m],cfg['sites'],report['subsets'][name])
        for key,r in a['contrasts'][action].items():
            left,right=key.split('_minus_');l=a['summary'][action+'__'+left];rr=a['summary'][action+'__'+right]
            for subset in ('all','hard','positive_easy'):
                check_contrast(l['ADE'] if subset=='all' else l['subsets'][subset],rr['ADE'] if subset=='all' else rr['subsets'][subset],r[subset])
                counts['paired_contrasts']+=1
    assert counts['query_action_seed_instances']==188388
    assert_current(identity)
    result=dict(all_checks_passed=True,analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)),**counts,repair_provenance=provenance,
        scope='raw_label_errors_query_risks_small_query_optima_full_grid_bounds_and_site_metrics',
        verification_by_same_agent=True,independent_research_confirmation=False,deployment=False)
    immutable_json(public/'independent_verification.json',result);print(json.dumps(result,indent=2))


def verify_repair_provenance(completed,identity,data):
    original=json.loads((ROOT/identity['parent_decision_manifest']).read_text())
    assert len(original['receipts'])==len(completed['receipts'])
    preserved=repaired=changed=0
    for before,after in zip(original['receipts'],completed['receipts']):
        old=json.loads((ROOT/before['path']).read_text())
        new=json.loads((ROOT/after['path']).read_text())
        assert old['frames']==new['frames'] and old['view']==new['view'] and old['action']==new['action']
        with np.load(ROOT/old['path'],allow_pickle=False) as z:ids=z['ids'].copy();bits=z['choices'].copy()
        with np.load(ROOT/new['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids']);new_bits=z['choices'].copy()
        for old_q,new_q in zip(old['queries'],new['queries']):
            assert old_q['frame']==new_q['frame']
            if old_q['matched']:
                assert old_q==new_q
                loc=data['frames'][ids]==old_q['frame']
                np.testing.assert_array_equal(bits[loc],new_bits[loc]);preserved+=1
            else:
                assert new_q['matched'];repaired+=1
        if before==after:
            np.testing.assert_array_equal(bits,new_bits)
        else:
            np.testing.assert_array_equal(bits[:,[0,1,2,3,4,6]],new_bits[:,[0,1,2,3,4,6]])
            changed+=int(np.count_nonzero(bits!=new_bits))
    assert repaired==127 and preserved==188388-127 and changed==877
    def function(path,name):
        node=next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(node,include_attributes=False)
    for path,name in [('src/world_model/m3w_easy_allocation.py','allocate'),
                      ('scripts/run_m3w_easy_allocation.py','query')]:
        assert function(path,name)==function('src/world_model/m3w_easy_allocation_repaired.py',name)
    return dict(preserved_queries=preserved,repaired_queries=repaired,
        changed_agent_arm_bits=changed,allocation_and_query_function_AST_unchanged=True,
        risk_threshold_changed=False,policy_objective_changed=False)


if __name__=='__main__':main()
