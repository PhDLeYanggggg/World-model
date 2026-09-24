"""Registered nested gain/harm learning on opened European source sites only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json,json_write,array_hash
from scripts import run_m3w_european_source_forecast as parent
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_european_source_intervention import (
    nested_producers,causal_cost_features,query_subset,controls,paired_cost_labels)
from src.world_model.m3w_native_gain_harm import (
    preprocess,fit_ridge,predict_ridge,fit_neural,predict_neural)
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_source_intervention_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_source_intervention_v1'
CONFIG = 'configs/m3w_european_source_intervention_v1.json'
FILES = (CONFIG,'scripts/run_m3w_european_source_intervention.py',
    'src/world_model/m3w_european_source_intervention.py','tests/test_m3w_european_source_intervention.py',
    'src/world_model/m3w_native_gain_harm.py','src/world_model/m3w_scaled_risk_controls.py',
    'src/world_model/m3w_interaction_controls.py','src/world_model/m3w_joint_intervention.py',
    'outputs/publication_readiness_2026_09/european_source_intervention_v1/registration.md')
CONTROL_ARMS = ('independent','scene_uniform','joint','unary_exact','joint_exact')


def beat(state,**values):
    row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**values)
    json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    if reg['seeds']!=[17,29,43] or reg['independent_reserved_readout'] or reg['deployment_promotion']:
        raise ValueError('Fixed source-only experiment required')
    preg,manifest,receipts,pidentity = parent.load_source()
    data = parent.prepare(manifest,receipts,pidentity)
    designs = parent.trials(preg,data,pidentity)
    for key,_,ti in designs:
        parent.complete(key,ti,preg)
        path = parent.PRIVATE/'predictions'/(key+'.json')
        if not path.exists():
            raise ValueError('All parent source-excluded predictions required')
        receipt = json.loads(path.read_text())
        if receipt['identity']!=ti or digest(ROOT/receipt['path'])!=receipt['sha256']:
            raise ValueError('Parent prediction identity differs')
    if not (parent.PUBLIC/'analysis.json').exists():
        raise ValueError('Parent experiment has not completed')
    qmask = query_subset(data['sites'],data['recordings'],data['frames'],
                        reg['joint_queries_per_locality'],reg['query_salt'])
    identity = dict(bindings={p:digest(ROOT/p) for p in FILES},parent_identity=pidentity,
        parent_analysis_sha256=digest(parent.PUBLIC/'analysis.json'),query_mask_sha256=array_hash(qmask),
        query_rows=int(qmask.sum()),folds=pidentity['folds'],numpy=np.__version__,torch=torch.__version__)
    immutable_json(PRIVATE/'identity.json',identity)
    return reg,data,identity,designs,qmask


def assert_identity(identity):
    for path,sha in identity['bindings'].items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Frozen cost experiment code changed: '+path)
    parent.assert_identity(identity['parent_identity'])


def artifacts_ok(receipt):
    return all(digest(ROOT/v['path'])==v['sha256'] for v in receipt['artifacts'].values())


def prediction(key,ids):
    path = parent.PRIVATE/'predictions'/(key+'.npz')
    r = json.loads(path.with_suffix('.json').read_text())
    if digest(path)!=r['sha256']:
        raise ValueError('Parent prediction bytes changed')
    with np.load(path,allow_pickle=False) as z:
        pos = np.searchsorted(z['ids'],ids)
        np.testing.assert_array_equal(z['ids'][pos],ids)
        return z['prediction'][pos].copy()


def assemble(data,identity,key,design,ti):
    n = len(data['sites'])
    assignment = np.array([identity['folds'][s] for s in data['sites']])
    fit,held = design['train_ids'],design['held_ids']
    outer,seed = ti['fold'],ti['seed']
    if set(data['sites'][fit])&set(data['sites'][held]) or np.any(assignment[held]!=outer):
        raise ValueError('Outer source leakage')
    p = np.empty((n,12,2),np.float32)
    p[held] = prediction(key,held)
    producers = {}
    for predicted,trained in nested_producers(outer).items():
        ids = np.flatnonzero(assignment==predicted)
        source = f'single{trained}_seed{seed}'
        receipt = json.loads((parent.PRIVATE/'predictions'/(source+'.json')).read_text())
        training_sites = set(receipt['identity']['fit_sites'])
        if training_sites & (set(data['sites'][ids])|set(data['sites'][held])):
            raise ValueError('Cost-label producer fitted predicted or outer site')
        p[ids] = prediction(source,ids)
        producers[source] = dict(predicted_sites=sorted(set(data['sites'][ids])),
            training_sites=sorted(training_sites),prediction_sha256=receipt['sha256'])
    b = baseline_numpy(data['history'],design['baseline_index'])-data['origin'][:,None]
    x,scale = causal_cost_features(data['geometry'],b,p)
    same = np.all(b==p,axis=(1,2))
    ade,_ = native_errors(p[fit].astype(float)+data['origin'][fit,None],data['target_eval'][fit],
                          data['valid'][fit],np.ones(len(fit)))
    reference = np.asarray(data['baseline_ade'][:,design['baseline_index']])
    y = paired_cost_labels(reference[fit],ade)
    pr = preprocess(x[fit],y,reference[fit],data['sites'][fit],sorted(set(data['sites'][held]))[0])
    if set(pr['training_sites'])!=set(data['sites'][fit]):
        raise ValueError('Cost-head preprocessing site mismatch')
    if not np.all(y[pr['known']&same[fit]]==0):
        raise ValueError('Identical predictions have nonzero cost labels')
    lineage = dict(outer_fold=outer,seed=seed,training_sites=sorted(set(data['sites'][fit])),
        held_sites=sorted(set(data['sites'][held])),nested_producers=producers,
        final_producer=key,baseline_index=design['baseline_index'],
        train_ids_sha256=array_hash(fit),held_ids_sha256=array_hash(held),
        feature_train_sha256=array_hash(x[fit]),labels_train_sha256=array_hash(y),
        cost_scale=pr['cost_scale'],easy_cut=design['easy_cut'],hard_cut=design['hard_cut'],
        supported_training_rows=int(pr['known'].sum()),unknown_training_rows=int((~pr['known']).sum()))
    return dict(p=p,b=b,x=x,scale=scale,same=same,y=y,pr=pr,lineage=lineage)


def save_state(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def train(reg,data,identity,designs,resume):
    for key,design,ti in designs:
        if ti['kind']!='complement':
            continue
        beat('assembling_nested_cost_training',trial=key)
        a = assemble(data,identity,key,design,ti)
        fit,held = design['train_ids'],design['held_ids']
        for arm in reg['arms']:
            name = key+'_'+arm
            directory = PRIVATE/'heads'/name
            receipt_path = directory/'complete.json'
            hid = dict(identity=identity,lineage=a['lineage'],arm=arm)
            if receipt_path.exists():
                r = json.loads(receipt_path.read_text())
                if r['identity']!=hid or not artifacts_ok(r):
                    raise ValueError('Completed cost head differs')
                beat('verified_completed_cost_head',trial=name)
                continue
            started = time.monotonic()
            beat('cost_head_training',trial=name,training_rows=len(fit),held_rows=len(held))
            if arm=='ridge':
                head = fit_ridge(a['x'][fit],a['y'],a['pr'],alpha=reg['ridge_alpha'])
                costs = predict_ridge(head,a['x'][held],a['same'][held],a['pr'])
                fit_report = dict(method='weighted_ridge_closed_form',gradient_updates=0,
                    supported_rows=int(a['pr']['known'].sum()))
                cp = directory/'ridge.pt'
                save_state(cp,dict(identity=hid,head=head,preprocess=a['pr']))
            elif arm=='neural_underharm4':
                model,fit_report = fit_neural(a['x'][fit],a['y'],data['sites'][fit],a['same'][fit],a['pr'],
                    seed=ti['seed'],arm='underharm4',settings=reg['training'],identity=hid,
                    directory=directory,resume=resume,heartbeat=lambda **kw:beat(trial=name,**kw))
                costs = predict_neural(model,a['x'][held],a['same'][held],a['pr'])
                cp = directory/'checkpoint.pt'
            else:
                raise ValueError('Unregistered cost model')
            costs = np.maximum(costs,0)
            score_path = directory/'scores.npz'
            np.savez(score_path,ids=held,costs=costs)
            r = dict(identity=hid,fit=fit_report,wall_seconds=time.monotonic()-started,
                artifacts={k:dict(path=str(v.relative_to(ROOT)),sha256=digest(v))
                           for k,v in [('checkpoint',cp),('predicted_costs',score_path)]})
            immutable_json(receipt_path,r)
            assert_identity(identity)
            beat('cost_head_complete',trial=name,seconds=r['wall_seconds'])
        del a


def decisions(reg,data,a,held,costs,qmask,scale):
    ids = held[qmask[held]]
    pointwise = (costs[:,0]>costs[:,1])
    moving = np.linalg.norm(np.diff(data['history'][held],axis=1),axis=2).sum(1)>0
    pointwise &= moving
    bits = {arm:np.zeros(len(ids),bool) for arm in CONTROL_ARMS}
    matched,nonzero = np.zeros(len(ids),bool),np.zeros(len(ids),bool)
    summaries = []
    local_scores = costs[qmask[held]]/scale
    local_moving = moving[qmask[held]]
    keys = np.column_stack((data['recordings'][ids],data['frames'][ids]))
    unique,inverse = np.unique(keys,axis=0,return_inverse=True)
    for qi,key in enumerate(unique):
        loc = np.flatnonzero(inverse==qi)
        ix = ids[loc]
        result = controls(local_scores[loc],local_moving[loc],data['origin'][ix],data['width'][ix],
            a['b'][ix]+data['origin'][ix,None],a['p'][ix].astype(float)+data['origin'][ix,None],
            budget=reg['predicted_positive_harm_budget'],pair_weight=reg['pair_weight'],
            radius_widths=reg['edge_radius_bbox_widths'],threshold_widths=reg['proximity_threshold_bbox_widths'],
            seconds=reg['solver_seconds'])
        matched[loc],nonzero[loc] = result['matched'],result['matched_nonzero']
        for arm in CONTROL_ARMS:
            bits[arm][loc] = result[arm]['switch']
        summary = dict(recording=int(key[0]),frame=int(key[1]),site=str(data['sites'][ix[0]]),
            agents=len(ix),edges=result['edges'],matched=result['matched'],matched_nonzero=result['matched_nonzero'],
            reference_count=result['reference_count'],arms={})
        for arm in CONTROL_ARMS:
            r = result[arm]
            summary['arms'][arm] = {k:r[k] for k in ('reason','solver_optimal','predicted_constraints_satisfied',
                'mean_pair_proxy','mean_predicted_gain','mean_predicted_harm','switch_rate')}
        summaries.append(summary)
        if qi%96==0:
            beat('joint_control_inference',query=qi+1,total=len(unique))
    return dict(ids=ids,pointwise_ids=held,pointwise=pointwise,matched=matched,matched_nonzero=nonzero,**bits),summaries


def evaluate(reg,data,identity,designs,qmask,verify):
    reports = {}
    for key,_,ti in designs:
        if ti['kind']!='complement':
            continue
        for arm in reg['arms']:
            path = PRIVATE/'heads'/(key+'_'+arm)/'complete.json'
            r = json.loads(path.read_text())
            if r['identity']['identity']!=identity or not artifacts_ok(r):
                raise ValueError('Missing or changed fixed-endpoint cost head')
            reports[key+'_'+arm] = r
    n = len(data['sites'])
    outputs,control_receipts = {},[]
    cv = np.asarray(data['baseline_ade'][:,1])
    for key,design,ti in designs:
        if ti['kind']!='complement':
            continue
        held = design['held_ids']
        beat('cost_and_control_readout',trial=key)
        a = assemble(data,identity,key,design,ti)
        ade,fde = native_errors(a['p'][held].astype(float)+data['origin'][held,None],
            data['target_eval'][held],data['valid'][held],np.ones(len(held)))
        for arm in reg['arms']:
            name = key+'_'+arm
            r = reports[name]
            if r['identity']['lineage']!=a['lineage']:
                raise ValueError('Recomputed nested lineage differs')
            with np.load(ROOT/r['artifacts']['predicted_costs']['path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],held)
                costs = z['costs'].copy()
            directory = PRIVATE/'decisions'
            directory.mkdir(parents=True,exist_ok=True)
            path = directory/(name+'.npz')
            receipt_path = path.with_suffix('.json')
            if receipt_path.exists():
                dr = json.loads(receipt_path.read_text())
                if dr['identity']!=r['identity'] or digest(path)!=dr['sha256']:
                    raise ValueError('Control decision bytes changed')
                with np.load(path,allow_pickle=False) as z:
                    choice = {k:z[k].copy() for k in z.files}
            else:
                if verify:
                    raise ValueError('Cannot verify missing control decisions')
                choice,queries = decisions(reg,data,a,held,costs,qmask,a['pr']['cost_scale'])
                np.savez(path,**choice)
                dr = dict(identity=r['identity'],path=str(path.relative_to(ROOT)),sha256=digest(path),queries=queries)
                immutable_json(receipt_path,dr)
            control_receipts.append(dict(path=str(receipt_path.relative_to(ROOT)),sha256=digest(receipt_path)))
            s = outputs.setdefault((ti['seed'],arm),dict(
                floor_ade=np.full(n,np.nan),floor_fde=np.full(n,np.nan),neural_ade=np.full(n,np.nan),
                neural_fde=np.full(n,np.nan),easy=np.zeros(n,bool),hard=np.zeros(n,bool),
                matched=np.zeros(n,bool),nonzero=np.zeros(n,bool),seen=np.zeros(n,bool),
                bits={k:np.zeros(n,bool) for k in ('pointwise',*CONTROL_ARMS)},queries=[],cost_diagnostics=[]))
            if s['seen'][held].any():
                raise ValueError('Repeated outer cost readout')
            s['seen'][held] = True
            s['neural_ade'][held],s['neural_fde'][held] = ade,fde
            s['floor_ade'][held] = data['baseline_ade'][held,design['baseline_index']]
            s['floor_fde'][held] = data['baseline_fde'][held,design['baseline_index']]
            s['easy'][held] = (cv[held]>0)&(cv[held]<=design['easy_cut'])
            s['hard'][held] = cv[held]>=design['hard_cut']
            np.testing.assert_array_equal(choice['pointwise_ids'],held)
            np.testing.assert_array_equal(choice['ids'],held[qmask[held]])
            s['bits']['pointwise'][held] = choice['pointwise']
            for control in CONTROL_ARMS:
                s['bits'][control][choice['ids']] = choice[control]
            s['matched'][choice['ids']],s['nonzero'][choice['ids']] = choice['matched'],choice['matched_nonzero']
            s['queries'].extend(dr['queries'])
            actual = paired_cost_labels(s['floor_ade'][held],ade)
            known = np.isfinite(actual).all(1)
            for site in sorted(set(data['sites'][held])):
                use = known & (data['sites'][held]==site)
                signed = actual[use,0]-actual[use,1]
                psigned = costs[use,0]-costs[use,1]
                corr = float(np.corrcoef(signed,psigned)[0,1]) if np.std(signed)>0 and np.std(psigned)>0 else None
                s['cost_diagnostics'].append(dict(site=site,rows=int(use.sum()),
                    benefit_MAE=float(np.abs(costs[use,0]-actual[use,0]).mean()),
                    harm_MAE=float(np.abs(costs[use,1]-actual[use,1]).mean()),
                    signed_gain_correlation=corr,underestimated_harm_fraction=float((costs[use,1]<actual[use,1]).mean())))
        del a
    roster = sorted(identity['folds'])
    def metric(model,reference,mask):
        return paired_scene_metrics(model[mask],reference[mask],data['sites'][mask],expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
    def describe(s,bits,mask):
        ade = np.where(bits,s['neural_ade'],s['floor_ade'])
        fde = np.where(bits,s['neural_fde'],s['floor_fde'])
        zero = mask&np.isfinite(cv)&(cv==0)
        return dict(ADE_vs_floor=metric(ade,s['floor_ade'],mask),ADE_vs_CV=metric(ade,cv,mask),
            FDE_vs_floor=metric(fde,s['floor_fde'],mask),hard_ADE_vs_CV=metric(ade,cv,mask&s['hard']),
            positive_easy_ADE_vs_CV=metric(ade,cv,mask&s['easy']),
            positive_easy_ADE_vs_floor=metric(ade,s['floor_ade'],mask&s['easy']),
            complete_ADE_vs_floor=metric(ade,s['floor_ade'],mask&data['valid'].all(1)),
            zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((ade[zero]>0).sum()),
                total_added_error=float(ade[zero].sum()),floor_harmed_rows=int((s['floor_ade'][zero]>0).sum())),
            switch_rate=float(bits[mask].mean()) if mask.any() else None,indexed_rows=int(mask.sum()))
    summaries = {}
    allmask = np.ones(n,bool)
    for (seed,arm),s in outputs.items():
        if not s['seen'].all():
            raise ValueError('Missing outer source costs')
        summaries[f'{seed}_{arm}'] = dict(
            full=dict(floor=describe(s,np.zeros(n,bool),allmask),neural=describe(s,np.ones(n,bool),allmask),
                      pointwise=describe(s,s['bits']['pointwise'],allmask)),
            joint_population={k:describe(s,b,qmask) for k,b in dict(floor=np.zeros(n,bool),neural=np.ones(n,bool),**s['bits']).items()},
            matched_nonzero={k:describe(s,s['bits'][k],qmask&s['nonzero']) for k in ('independent','unary_exact','joint_exact')},
            queries=dict(total=len(s['queries']),matched=sum(v['matched'] for v in s['queries']),
                matched_nonzero=sum(v['matched_nonzero'] for v in s['queries']),
                with_edges=sum(v['edges']>0 for v in s['queries']),
                solver_failures={k:sum(not v['arms'][k]['solver_optimal'] for v in s['queries']) for k in CONTROL_ARMS},
                mean_pair_proxy={k:float(np.mean([v['arms'][k]['mean_pair_proxy'] for v in s['queries']])) for k in CONTROL_ARMS}),
            cost_diagnostics=s['cost_diagnostics'])
    result = dict(result_source='fresh_nested_cost_fit_and_source_excluded_control_readout',identity=identity,
        training=reports,controls=control_receipts,seeds=summaries,total_cost_heads=len(reports),
        source_rows=n,joint_query_rows=int(qmask.sum()),positive_easy_limit_percent=2,zero_reference_allowed_added_harm=0,
        risk_budget_is_predicted_not_certified=True,independent_reserved_readout=False,deployment_changed=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:
        immutable_json(PUBLIC/'verification.json',dict(result_source='cached_verified',
            analysis_sha256=digest(PUBLIC/'analysis.json'),metrics_recomputed=True,
            new_training=False,new_forecast_inference=False,new_control_solver=False))
    beat('source_intervention_readout_complete',heads=len(reports),source_rows=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train','resume','evaluate','verify'):
        parser.add_argument('--'+name,action='store_true')
    args = parser.parse_args()
    if not (args.train or args.evaluate or args.verify):
        parser.error('Explicit phase required')
    for directory in (PRIVATE,PUBLIC):
        if any(p.is_symlink() for p in (directory,*directory.parents)):
            raise ValueError('Symlinked experiment destination')
        directory.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        reg,data,identity,designs,qmask = load()
        beat('verified_nested_source_population',rows=len(data['sites']),joint_rows=int(qmask.sum()))
        if args.train:
            train(reg,data,identity,designs,args.resume)
        if args.evaluate or args.verify:
            evaluate(reg,data,identity,designs,qmask,args.verify)
        beat('phase_complete')


if __name__=='__main__':
    main()
