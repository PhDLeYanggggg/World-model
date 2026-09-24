"""Frozen causal query allocation, then separate cached-label evaluation."""
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
import scipy
import torch

from scripts.build_m3w_native_scene_context import load_past_queries
from scripts.run_m3w_native_forecast import file_digest, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_native_scene_alignment import restore
from src.evaluation.m3w_native_metrics import paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_protected_motion_controls import causal_candidate
from src.world_model.m3w_native_joint_controls import make_problem
from src.world_model.m3w_easy_allocation import allocate, ARMS

CONFIG='configs/m3w_easy_allocation_v1.json'
CODE=('scripts/run_m3w_easy_allocation.py','src/world_model/m3w_easy_allocation.py',
      'tests/test_m3w_easy_allocation.py','src/world_model/m3w_interaction_controls.py',
      'src/world_model/m3w_native_joint_controls.py','src/evaluation/m3w_native_scene_alignment.py')
CONTEXT_KEYS=('context_target_rows','context_frame_ids','context_agent_ids',
              'context_cv_rollout','context_cv_valid','context_xy')


def load():
    cfg=json.loads((ROOT/CONFIG).read_text())
    base=ROOT/'outputs/publication_readiness_2026_09'
    path=base/'easy_moment_v1/analysis.json';a=json.loads(path.read_text())
    assert file_digest(path)==cfg['parent_analysis_sha256']
    parent_path=ROOT/'data/stage_cvpr2027_experiments/easy_moment_v1/identity.json'
    assert file_digest(parent_path)==a['experiment_sha256']
    parent=json.loads(parent_path.read_text());bindings=dict(parent['source_bindings'])
    assert cfg['sites']==a['sites'] and cfg['seeds']==a['seeds']
    assert cfg['actions']==parent['config']['actions'] and cfg['easy_rho']==.02
    assert cfg['geometry']==dict(past_radius='median_target_past_scale',threshold_fraction=.1,pair_weight=1.)
    assert cfg['solver_seconds']==5. and cfg['checkpoint_queries']==256 and cfg['bootstrap_resamples']==3000
    assert not any(cfg[k] for k in ('new_training','threshold_search','model_selection','external_readout',
        'independent_calibration','independent_confirmation','deployment','stage5c_executed','smc_enabled'))
    def bind(path,expected=None):
        path=Path(path);key=str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        sha=file_digest(ROOT/key)
        if (expected is not None and sha!=expected) or key in bindings and bindings[key]!=sha:
            raise ValueError('Changed dependency: '+key)
        bindings[key]=sha
    for p in (path,parent_path,CONFIG,cfg['registration'],*CODE):bind(p)
    for name in ('verification.json','independent_verification.json'):
        p=path.with_name(name);v=json.loads(p.read_text());bind(p)
        assert v['all_checks_passed'] and v['analysis_sha256']==cfg['parent_analysis_sha256']
    for r in a['fits']:
        assert r['fit']['complete'] and r['fit']['trees']==128
        assert r['view'].rsplit('_seed',1)[0] not in r['identity']['training_sites']
        bind(r['checkpoint'],r['checkpoint_sha256'])
    for r in a['decision_archives']+a['outcome_archives']:bind(r['path'],r['sha256'])
    cp=base/'native_scene_context_v2/analysis.json';context=json.loads(cp.read_text());bind(cp)
    vp=cp.with_name('verification.json');v=json.loads(vp.read_text());bind(vp)
    assert v['all_checks_passed'] and v['analysis_sha256']==file_digest(cp)
    for p,sha in context['source_bindings'].items():bind(p,sha)
    for r in context['records']:bind(r['cache']['path'],r['cache']['sha256'])
    ap=base/'native_scene_alignment_v1/analysis.json';align=json.loads(ap.read_text());bind(ap)
    bind(align['cache']['path'],align['cache']['sha256'])
    data=load_past_queries(bindings)
    with np.load(ROOT/align['cache']['path'],allow_pickle=False) as z:
        for k in ('origin','rotation','stored_metric_scale'):data[k]=z[k].copy()
    data['scale']=data.pop('stored_metric_scale')
    data['sites']=np.array([v.split('/')[0] for v in data['recordings']])
    mp=base/'native_nested_v1/materialized_views.json';m=json.loads(mp.read_text());bind(mp)
    assert m['all_checks_passed'] and m['outer_rows_in_training']==0
    transformer={}
    for r in m['views']:
        bind(r['receipt_path'],r['receipt_sha256']);meta=json.loads((ROOT/r['receipt_path']).read_text())
        transformer[f"{r['outer_site']}_seed{r['seed']}"]=meta['outer_prediction']['prediction']
    eqp=ROOT/'data/stage_cvpr2027_experiments/eqmotion_nested_v1/cost_views.json'
    eq=json.loads(eqp.read_text());bind(eqp)
    eqmotion={f"{r['outer_site']}_seed{r['seed']}":r['outer_producer']['prediction'] for r in eq['views']}
    for r in list(transformer.values())+list(eqmotion.values()):bind(r['path'],r['sha256'])
    assert len(data['sites'])==175756 and len(context['records'])==33
    assert len(transformer)==len(eqmotion)==12
    identity=dict(config=cfg,source_bindings=bindings,numpy=np.__version__,scipy=scipy.__version__,
        torch=torch.__version__,architecture=platform.machine(),arms=list(ARMS),
        runtime=dict(torch_threads=4,interop_threads=1,num_workers=0),
        source_role='design_exposed_source_only',outcome_arrays_used_for_decisions=False)
    assert_current(identity)
    return cfg,data,context,a,transformer,eqmotion,identity


def causal_view(pack,key,action):
    cfg,data,context,a,tr,eq,identity=pack
    r=next(v for v in a['decision_archives'] if v['view']==key)
    names=['ids']+[action+'__'+s for s in ('moments','distance','score','net_stop','strict_stop','joint_easy_moment')]
    with np.load(ROOT/r['path'],allow_pickle=False) as z:values={k:z[k].copy() for k in names}
    ids=values['ids'];np.testing.assert_array_equal(ids,np.flatnonzero(data['sites']==key.rsplit('_seed',1)[0]))
    if action=='damped_velocity_005':prediction=causal_candidate(data['geometry'][ids],action)
    else:
        with np.load(ROOT/(eq if action=='eqmotion' else tr)[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids']);prediction=z['prediction'].copy()
    fit=next(r for r in a['fits'] if r['view']==key and r['action']==action)
    cp=joblib.load(ROOT/fit['checkpoint'])
    assert cp['identity']==fit['identity'] and key.rsplit('_seed',1)[0] not in cp['preprocess']['training_sites']
    return values,prediction,float(cp['preprocess']['cost_scale']),float(fit['identity']['easy_cut'])


def query(rows,c,ids_to_view,values,prediction,data,scale,cut,action,cfg):
    target=c['context_target_rows'][rows]>=0
    tid=c['context_target_rows'][rows][target];loc=ids_to_view[tid]
    assert (loc>=0).all()
    n=len(rows);valid=c['context_cv_valid'][rows].all(1)
    b=c['context_cv_rollout'][rows].copy();candidate=b.copy()
    b[target]=restore(data['geometry'][tid,332:356].reshape(-1,12,2),data['origin'][tid],data['rotation'][tid],data['scale'][tid])
    candidate[target]=restore(prediction[loc],data['origin'][tid],data['rotation'][tid],data['scale'][tid])
    valid[target]=True
    scores=values[action+'__score'][loc];moment=values[action+'__moments'][loc]
    support=np.zeros(n,bool);support[target]=values[action+'__net_stop'][loc]
    benefit=np.zeros(n);harm=np.zeros(n);benefit[target],harm[target]=scores[:,0],scores[:,1]
    p,geo=make_problem(baseline=b,candidate=candidate,current=c['context_xy'][rows],forecast_valid=valid,
        target_mask=target,eligible=support,benefit=benefit,harm=harm,scale=scale,past_target_scales=data['scale'][tid])
    q=np.zeros(n);r=np.zeros(n);q[target]=moment[:,0]*values[action+'__distance'][loc];r[target]=moment[:,1]*cut
    point=np.zeros(n,bool);strict=point.copy()
    point[target]=values[action+'__joint_easy_moment'][loc];strict[target]=values[action+'__strict_stop'][loc]
    choices,report=allocate(p,q,r,target,cost_scale=scale,pointwise=point,strict=strict,
                            rho=cfg['easy_rho'],seconds=cfg['solver_seconds'])
    report['geometry']=geo
    return tid,np.column_stack([choices[k][target] for k in ARMS]),report


def decide(pack,args,beat):
    cfg,data,context,a,tr,eq,identity=pack
    root=ROOT/cfg['output'];ish=file_digest(root/'identity.json');receipts=[];count=0;new=0
    started=time.monotonic()
    for key in tr:
        for action in cfg['actions']:
            values,pred,scale,cut=causal_view(pack,key,action)
            index=np.full(len(data['sites']),-1,np.int64);index[values['ids']]=np.arange(len(values['ids']))
            for rec in context['records']:
                name=rec['recording']
                if name.split('/')[0]!=key.rsplit('_seed',1)[0]:continue
                with np.load(ROOT/rec['cache']['path'],allow_pickle=False) as z:c={k:z[k].copy() for k in CONTEXT_KEYS}
                frames=np.unique(c['context_frame_ids'])
                for start in range(0,len(frames),cfg['checkpoint_queries']):
                    part=frames[start:start+cfg['checkpoint_queries']]
                    path=root/'decisions'/key/action/name/f'{start:06d}.npz';rp=path.with_suffix('.json')
                    if rp.exists():
                        old=json.loads(rp.read_text())
                        assert old['identity_sha256']==ish and old['frames']==part.tolist() and file_digest(path)==old['sha256']
                        if not args.verify:
                            receipts.append(dict(path=str(rp.relative_to(ROOT)),sha256=file_digest(rp)))
                            count+=len(part);continue
                    elif args.verify:raise ValueError('Replay cannot create new decisions')
                    ids=[];bits=[];reports=[]
                    for frame in part:
                        rows=np.flatnonzero(c['context_frame_ids']==frame)
                        tid,b,info=query(rows,c,index,values,pred,data,scale,cut,action,cfg)
                        ids.append(tid);bits.append(b);reports.append(dict(frame=int(frame),**info))
                    write_arrays(path,dict(ids=np.concatenate(ids),choices=np.concatenate(bits)))
                    result=dict(identity_sha256=ish,view=key,action=action,recording=name,frames=part.tolist(),
                        path=str(path.relative_to(ROOT)),sha256=file_digest(path),queries=reports)
                    immutable_json(rp,result);receipts.append(dict(path=str(rp.relative_to(ROOT)),sha256=file_digest(rp)))
                    new+=len(part);count+=len(part)
                    beat(state='replaying_decisions' if args.verify else 'causal_allocation',view=key,action=action,
                         recording=name,completed_queries=count,new_queries=new,seconds=time.monotonic()-started)
                    if args.pilot and new>=256:
                        beat(state='pilot_complete_not_full_matrix',queries=count,seconds=time.monotonic()-started);return
    assert count==20932*3*3
    assert_current(identity)
    immutable_json(root/'decisions_complete.json',dict(identity_sha256=ish,receipts=receipts,
        query_action_seed_instances=count,future_outcome_arrays_loaded=False))
    if args.verify:
        immutable_json(ROOT/cfg['reports']/'decision_replay.json',dict(all_checks_passed=True,
            decision_manifest_sha256=file_digest(root/'decisions_complete.json'),queries_replayed=new))
    beat(state='decision_replay_complete' if args.verify else 'all_decisions_complete',queries=count)


def evaluate(pack,beat,verify=False):
    cfg,data,context,a,tr,eq,identity=pack;root=ROOT/cfg['output'];public=ROOT/cfg['reports']
    completed=json.loads((root/'decisions_complete.json').read_text())
    assert completed['identity_sha256']==file_digest(root/'identity.json')
    n=len(data['sites']);choices={(s,act):np.zeros((n,len(ARMS)),bool) for s in cfg['seeds'] for act in cfg['actions']}
    seen={k:np.zeros(n,bool) for k in choices};queries={act:[] for act in cfg['actions']}
    for r in completed['receipts']:
        assert file_digest(ROOT/r['path'])==r['sha256'];v=json.loads((ROOT/r['path']).read_text())
        assert file_digest(ROOT/v['path'])==v['sha256'];key=(int(v['view'].rsplit('seed',1)[1]),v['action'])
        with np.load(ROOT/v['path'],allow_pickle=False) as z:
            ids=z['ids'];assert not seen[key][ids].any();seen[key][ids]=True;choices[key][ids]=z['choices']
        queries[v['action']].extend(v['queries'])
    assert all(s.all() for s in seen.values())
    def metric(e,cv,mask=None,ci=False):
        use=np.ones(n,bool) if mask is None else mask
        return paired_scene_metrics(e[use],cv[use],data['sites'][use],expected_scenes=cfg['sites'],dataset='sdd',
            coordinate_unit='annotation_pixel',bootstrap_resamples=cfg['bootstrap_resamples'] if ci else 0)
    summary={};bank={};contrasts={};allocation={};archives=[]
    for act in cfg['actions']:
        arrays={p:[] for p in ARMS};ends={p:[] for p in ARMS};details={p:{} for p in ARMS}
        for seed in cfg['seeds']:
            path=ROOT/'data/stage_cvpr2027_experiments/easy_moment_v1/outcomes'/f'{act}_seed{seed}.npz'
            record=next(r for r in a['outcome_archives'] if ROOT/r['path']==path)
            assert file_digest(path)==record['sha256']
            with np.load(path,allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
            cv,cf=o['cv'],o['cf'];groups={k:o[k] for k in ('complete','zero_CV','positive_easy','hard')}
            for p,old in [('net_stop','net_stop'),('strict_stop','strict_stop'),('pointwise','joint_easy_moment')]:
                np.testing.assert_array_equal(choices[seed,act][:,ARMS.index(p)],o[old])
            for j,p in enumerate(ARMS):
                bits=choices[seed,act][:,j];e=np.where(bits,o['candidate_ade'],cv);f=np.where(bits,o['candidate_fde'],cf)
                arrays[p].append(e);ends[p].append(f)
                details[p][str(seed)]=dict(ADE=metric(e,cv),FDE=metric(f,cf),
                    subsets={k:metric(e,cv,m) for k,m in groups.items()},selected=int(bits.sum()),
                    selected_unknown=int((bits&np.isnan(cv)).sum()),selected_incomplete=int((bits&~groups['complete']).sum()),
                    zero_CV_harmed=int((e[groups['zero_CV']]>0).sum()),
                    full_grid_gain_bounds={site:[float(np.where(bits,o[b],0)[data['sites']==site].mean())
                        for b in ('lower','upper')] for site in cfg['sites']})
            op=root/'outcomes'/f'{act}_seed{seed}.npz';write_arrays(op,dict(choices=choices[seed,act]))
            archives.append(dict(path=str(op.relative_to(ROOT)),sha256=file_digest(op)))
        for p in ARMS:
            mean=np.mean(arrays[p],axis=0);bank[act,p]=mean
            summary[act+'__'+p]=dict(ADE=metric(mean,cv,ci=True),FDE=metric(np.mean(ends[p],0),cf,ci=True),
                subsets={k:metric(mean,cv,m,True) for k,m in groups.items()},seeds=details[p])
        contrasts[act]={}
        for left,right in [('aggregate_selected','pointwise'),('aggregate_population','aggregate_selected'),
                ('aggregate_population','strict_stop'),('aggregate_joint','aggregate_unary'),('aggregate_joint','aggregate_population')]:
            contrasts[act][left+'_minus_'+right]={}
            for subset in ('all','hard','positive_easy'):
                mask=None if subset=='all' else groups[subset]
                l=metric(bank[act,left],cv,mask)['by_scene'];r=metric(bank[act,right],cv,mask)['by_scene']
                contrasts[act][left+'_minus_'+right][subset]=paired_scene_contrast(
                    [l[s]['gain_percent'] for s in cfg['sites']],[r[s]['gain_percent'] for s in cfg['sites']],resamples=3000)
        qq=queries[act]
        allocation[act]=dict(query_seed_instances=len(qq),
            failed_selected=sum(not r['selected_optimal'] for r in qq),
            failed_population=sum(not r['population_optimal'] for r in qq),
            failed_unary=sum(not r['unary_optimal'] for r in qq),failed_joint=sum(not r['joint_optimal'] for r in qq),
            unmatched=sum(not r['matched'] for r in qq),
            nonadditive_queries=sum(r['nonadditive_edges']>0 for r in qq),
            changed_joint_unary_queries=sum(r['joint_unary_changed_agents']>0 for r in qq),
            changed_joint_unary_agents=sum(r['joint_unary_changed_agents'] for r in qq),
            unsupported_forecast_edges=sum(r['geometry']['unknown_forecast_edges'] for r in qq),
            policies={p:dict(selected=sum(r['policies'][p]['selected'] for r in qq),
                mean_query_pair_proxy=float(np.mean([r['policies'][p]['pair_proxy'] for r in qq])),
                predicted_population_budget_violations=sum(not r['policies'][p]['population_budget_satisfied'] for r in qq)) for p in ARMS})
    result=dict(result_source='fresh_allocation_and_readout_on_cached_verified_predictions_and_costs',
        experiment_sha256=file_digest(root/'identity.json'),rows=n,sites=cfg['sites'],seeds=cfg['seeds'],
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        summary=summary,contrasts=contrasts,allocation=allocation,outcome_archives=archives,
        independent_calibration=False,independent_confirmation=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    assert_current(identity);immutable_json(public/'analysis.json',result)
    if verify:
        immutable_json(public/'aggregate_replay.json',dict(all_checks_passed=True,analysis_sha256=file_digest(public/'analysis.json')))
    beat(state='aggregate_replay_complete' if verify else 'evaluation_complete',rows=n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['preflight','decide','evaluate'],default='preflight')
    p.add_argument('--pilot',action='store_true');p.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.verify and args.pilot:raise ValueError('Pilot and replay are distinct')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();cfg=pack[0];root=ROOT/cfg['output'];root.mkdir(parents=True,exist_ok=True)
    immutable_json(root/'identity.json',pack[-1])
    def beat(**v):
        e=dict(pid=os.getpid(),updated_unix=time.time(),**v);json_write(root/'heartbeat.json',e)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
        print(json.dumps(e),flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='preflight':beat(state='preflight_pass',bindings=len(pack[-1]['source_bindings']))
        elif args.phase=='decide':decide(pack,args,beat)
        else:evaluate(pack,beat,args.verify)


if __name__=='__main__':main()
