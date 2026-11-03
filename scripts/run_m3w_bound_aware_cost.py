"""Matched SOURCE cost targets; every evaluation site excluded from both fits."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before importing Torch')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_controller as prior
from scripts import run_m3w_european_eqmotion as forecaster
from scripts.run_m3w_european_eqmotion_controls import aligned_rows
from src.world_model import m3w_bound_aware_cost as api
from src.evaluation.m3w_eqmotion_amplitude import locality_interval
from src.evaluation.m3w_selection_cost_fit_objective import score as fit_score

NAME='european_bound_aware_cost_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_bound_aware_cost_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_bound_aware_cost.py',
       ROOT/'tests/test_m3w_bound_aware_cost.py',PUBLIC/'protocol.md')
sha,once,write,artifact,checked=prior.sha,prior.once,prior.write,prior.artifact,prior.checked
MATCHED=('initial_model','parameters','budget','sampler','draw_hash','fit_ids','preprocess','torch_rng')


def job_designs(producers,src,whitelist,cfg):
    folds={j['fold']:set(j['fit_sites']) for j in producers}
    if (set(folds)!={0,1,2} or any(len(s)!=4 for s in folds.values())
            or any(folds[a]&folds[b] for a in folds for b in folds if a!=b)
            or set.union(*folds.values())!=set(src['sites'][whitelist])):
        raise ValueError('The three original disjoint four-site SOURCE groups are required')
    scopes={}
    for group,sites in folds.items():
        ids=whitelist[np.isin(src['sites'][whitelist],list(sites))]
        queries=api.base.query_groups(src['recordings'][ids],src['frames'][ids],ids)
        names=sorted(sites);qsites=np.array([src['sites'][ids[ix[0]]] for ix in queries])
        source_groups=[np.flatnonzero(qsites==s) for s in names]
        budget=api.base.steps_for(dict(source_groups=source_groups),cfg['training'])
        scopes[group]=dict(sites=names,rows=len(ids),ids_sha256=prior.frozen.fingerprint(ids),
            queries=len(queries),source_query_counts=list(map(len,source_groups)),budget=budget)
    jobs=[]
    for p in producers:
        for group in sorted(set(folds)-{p['fold']}):
            evaluation=next(iter(set(folds)-{p['fold'],group}))
            jobs.append(dict(key=p['trial']+'_critic'+str(group),outer_trial=p['trial'],seed=p['seed'],
                producer_fold=p['fold'],critic_fold=group,evaluation_fold=evaluation,
                producer_sites=sorted(p['fit_sites']),critic_sites=scopes[group]['sites'],
                evaluation_sites=scopes[evaluation]['sites'],floor_index=p['baseline_index'],
                easy_cut=p['easy_cut'],hard_cut=p['hard_cut'],fit_scope=scopes[group],
                evaluation_scope=scopes[evaluation]))
    if len(jobs)!=18:raise ValueError('All eighteen nested producer/critic jobs required')
    return jobs


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());_,producers,ids=prior.admission()
    _,_,_,eqjobs,_,_=forecaster.admission()
    src=prior.source_inputs(json.loads(prior.frozen.REG.read_text()))
    current={j['trial']:j for j in producers}
    for key,design,ti in eqjobs:
        p=current[key]
        if (set(p['fit_sites'])!=set(ti['fit_sites']) or p['seed']!=ti['seed']
                or p['baseline_index']!=design['baseline_index']
                or p['easy_cut']!=design['easy_cut'] or p['hard_cut']!=design['hard_cut']):
            raise ValueError('Fixed EqMotion and original producer floor/role metadata must agree')
    support=ROOT/'outputs/publication_readiness_2026_09/european_source_label_support_v1'
    for root,key in ((forecaster.PUBLIC,'all72_SOURCE_groups_exact'),(support,'all144_groups_exact')):
        v=json.loads((root/'verification.json').read_text())
        if not v[key] or v['summary_sha256']!=sha(root/'summary.json'):raise ValueError('Completed precursor replay required')
    parent_settings=json.loads((ROOT/'configs/m3w_selection_controller_v1.json').read_text())['training']
    if (cfg['training']!=parent_settings or cfg['arms']!=list(api.ARMS) or cfg['new_models']!=36
            or cfg['source_whitelist_rows']!=len(ids) or cfg['seeds']!=[17,29,43]
            or cfg['positive_harm_budget']!=.02 or cfg['easy_net_degradation_budget']!=.02
            or any(cfg[k] for k in ('new_trajectory_training','threshold_selection','development_arrays_opened',
                'calibration_arrays_opened','confirmation_opened','deployment_changed','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed matched model/budget/roles/risk are required')
    jobs=job_designs(producers,src,ids,cfg)
    predictions=json.loads((forecaster.PUBLIC/'prediction_freeze.json').read_text())
    refs={json.loads(checked(ref).read_text())['trial']:ref for ref in predictions['predictions']}
    for ref in refs.values():
        r=json.loads(checked(ref).read_text())
        if sha(ROOT/r['prediction'])!=r['sha256']:raise ValueError('Frozen EqMotion array changed')
    deps=(ROOT/'src/world_model/m3w_supported_bound_targets.py',ROOT/'src/world_model/m3w_selection_controller.py',
        ROOT/'src/world_model/m3w_multiaction_cost.py',ROOT/'src/world_model/m3w_reference_cost_refit.py',
        ROOT/'src/evaluation/m3w_selection_forecast_readout.py',ROOT/'src/evaluation/m3w_selection_cost_fit_objective.py',
        ROOT/'src/evaluation/m3w_eqmotion_amplitude.py')
    definition=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in (*FILES,*deps)},jobs=jobs,
        forecast_freeze=artifact(forecaster.PUBLIC/'prediction_freeze.json'),forecast_verification=artifact(forecaster.PUBLIC/'verification.json'),
        source_roles=artifact(prior.frozen.inputs.ROLES),source_packed=artifact(prior.SOURCE/'receipt.json'),
        source_controller=artifact(prior.REG),support_probe=artifact(support/'summary.json'),support_verification=artifact(support/'verification.json'),
        source_whitelist_rows=len(ids),source_localities=sorted(set(src['sites'][ids])),source_recordings=len(set(src['recordings'][ids])),
        models=36,primary=cfg['primary'],eval_excluded_from_both_fits=True,source_is_exposed_development=True,
        independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    if register:once(REG,definition)
    if json.loads(REG.read_text())!=definition:raise ValueError('Registered bound-aware comparison changed')
    if not register:
        for p in (*FILES,REG):
            blob=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha(p):raise ValueError('Commit the entire contract before execution')
    return cfg,jobs,src,ids,refs


def runtime(cfg,phase):
    torch.set_num_threads(cfg['torch_threads']);torch.set_num_interop_threads(cfg['interop_threads']);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged atomic checkpoints')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB free; no budget reduction')
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h phase boundary; resume same budget')
    def beat(**kw):
        guard()
        if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>cfg['private_cache_cap_bytes']:
            raise OSError('Registered private cache cap1GiB')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**kw)
        write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def receipt():return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state='runtime_ready',architecture=platform.machine(),torch=torch.__version__,device='cpu',
         threads=cfg['torch_threads'],interop=cfg['interop_threads'],workers=0,resource_probing=False)
    return beat,guard,receipt


def bank(job,src,whitelist,refs,*,training=False):
    sites=job['critic_sites'] if training else job['evaluation_sites']
    ids=whitelist[np.isin(src['sites'][whitelist],sites)]
    scope=job['fit_scope'] if training else job['evaluation_scope']
    if len(ids)!=scope['rows'] or prior.frozen.fingerprint(ids)!=scope['ids_sha256']:
        raise ValueError('Fixed fitting/evaluation row alignment required')
    meta=json.loads(checked(refs[job['outer_trial']]).read_text())
    if sha(ROOT/meta['prediction'])!=meta['sha256']:raise ValueError('EqMotion prediction changed')
    with np.load(ROOT/meta['prediction'],allow_pickle=False) as z:
        ix=aligned_rows(z['ids'],ids);neural=z['prediction'][ix].astype(float)
    origin=src['origin'][ids]
    physical=np.stack([forecaster.source.baseline_numpy(src['history'][ids],k)-origin[:,None] for k in range(6)],1)
    paths=np.concatenate((physical,neural[:,None]),1)
    causal=api.base.features(geometry=src['geometry'][ids],detector_history=src['detector_history'][ids],rollouts=paths,
        floor_index=job['floor_index'],recordings=src['recordings'][ids],frames=src['frames'][ids],ids=ids,sites=src['sites'][ids])
    if set(causal['sites'])&set(job['producer_sites']):raise ValueError('Critic/evaluation cannot use producer-fitting rows')
    data=dict(ids=ids,paths=paths,causal=causal,recordings=src['recordings'][ids],sites=src['sites'][ids])
    if training:
        target=np.load(prior.SOURCE/'target_eval.npy',mmap_mode='r',allow_pickle=False)[ids]-origin[:,None]
        valid=np.load(prior.SOURCE/'valid.npy',mmap_mode='r',allow_pickle=False)[ids]
        data['learning']=api.training_data(causal,paths,target,valid,job['floor_index'],job['easy_cut'])
    return data


def path_for(job,arm,family='fits'):return PRIVATE/family/(job['key']+'_'+arm+'.pt.gz')


def one(data,job,arm,cfg,refs,beat,guard,*,family='fits',resume=False,stop_at=None):
    return api.fit(data['learning'],arm=arm,seed=job['seed'],settings=cfg['training'],
        identity=dict(registration_sha256=sha(REG),job=job,forecast=refs[job['outer_trial']]),path=path_for(job,arm,family),
        heartbeat=lambda **v:beat(job=job['key'],**v),guard=guard,resume=resume,stop_at=stop_at)


def recovered(a,b):
    for k in a:
        if k!='seconds':api.base.exact(a[k],b[k])


def pilot(cfg,jobs,src,ids,refs,beat,guard,receipt,resume):
    if (PUBLIC/'pilot.json').exists():raise ValueError('Complete pilot immutable')
    cells=[];updates=0
    for group in cfg['pilot_critic_groups']:
        job=next(j for j in jobs if j['critic_fold']==group);data=bank(job,src,ids,refs,training=True);states=[]
        for arm in cfg['arms']:
            rp=path_for(job,arm,'pilot_resume');dp=path_for(job,arm,'pilot_direct')
            if rp.exists() and not resume:raise ValueError('Partial pilot requires --resume')
            step=api.base.read_checkpoint(rp)['step'] if rp.exists() else 0
            if step<cfg['pilot_stop']:
                one(data,job,arm,cfg,refs,beat,guard,family='pilot_resume',resume=rp.exists(),stop_at=cfg['pilot_stop'])
            a=one(data,job,arm,cfg,refs,beat,guard,family='pilot_resume',resume=True,stop_at=cfg['pilot_target'])
            b=one(data,job,arm,cfg,refs,beat,guard,family='pilot_direct',resume=resume and dp.exists(),stop_at=cfg['pilot_target'])
            recovered(a,b);states.append(b);updates+=2*b['step']
            if arm=='known_only' and b['unknown_bound_supervised_draws']!=0:raise ValueError('Control cannot supervise unknown bounds')
            if arm=='bound_aware' and b['unknown_bound_supervised_draws']<=0:raise ValueError('Real unknown-bound exposure required')
            cells.append(dict(job=job['key'],critic_group=group,arm=arm,updates=b['step'],budget=b['budget'],
                seconds=b['seconds'],parameters=b['parameters'],exact_resume=True,
                observed_supervised_draws=b['observed_supervised_draws'],unknown_bound_supervised_draws=b['unknown_bound_supervised_draws']))
        for k in MATCHED:api.base.exact(states[0][k],states[1][k])
        del data,states;gc.collect()
    projected=6*sum(c['seconds']/c['updates']*c['budget'] for c in cells)
    first_group=jobs[0]['critic_fold'];refits=sum(c['seconds']/c['updates']*c['budget'] for c in cells if c['critic_group']==first_group)
    once(PUBLIC/'pilot.json',dict(**receipt(),cells=cells,actual_optimizer_updates=updates,engineering_pass=True,
        all_three_training_groups_recovery_exact=True,paired_initialization_preprocessing_draws_exact=True,
        projected36_primary_fit_seconds=projected,projected_first_full_refits_seconds=refits,
        projection_excludes_preparation_inference_evaluation=True,held_outcomes_scored=False))


def objectives(pred,data,counts):
    c=data['causal'];out={}
    for arm in api.ARMS:
        defined=data['defined_by_arm'][arm]
        value=fit_score(pred,data['truth_by_arm'][arm].numpy(),c['allowed'],defined,c['scales'],
            data['preprocess']['rms'],c['groups'],counts,c['sites'])
        value.pop('unknown_supervised_rows')
        value['defined_unknown_bound_rows']=int((defined&~data['observed_future']).sum())
        value['observed_future_rows']=int(data['observed_future'].sum())
        for v in value['by_site'].values():v['queries_without_defined_targets']=v.pop('unlabeled_queries')
        out[arm]=value
    return out


def train(cfg,jobs,src,ids,refs,beat,guard,receipt,resume):
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Complete fits immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Real three-group pilot required')
    fits=[];updates=refit_updates=0
    for job in jobs:
        data=bank(job,src,ids,refs,training=True);states=[]
        for arm in cfg['arms']:
            state=one(data,job,arm,cfg,refs,beat,guard,resume=resume and path_for(job,arm).exists())
            if state['step']!=job['fit_scope']['budget'] or state['sampler']['counts'].min()<4:
                raise ValueError('Every full registered query budget must finish')
            if job==jobs[0]:
                other=one(data,job,arm,cfg,refs,beat,guard,family='full_refit',
                    resume=resume and path_for(job,arm,'full_refit').exists())
                recovered(state,other);refit_updates+=state['step']
                once(PUBLIC/('first_full_refit_'+arm+'.json'),dict(scientific_state_exact=True,updates=state['step'],
                    checkpoint=artifact(path_for(job,arm)),refit=artifact(path_for(job,arm,'full_refit'))))
                del other
            pred=api.base.predict(state,data['causal'])
            scores=objectives(pred,data['learning'],state['sampler']['counts'])
            value=dict(job=job,arm=arm,checkpoint=artifact(path_for(job,arm)),step=state['step'],budget=state['budget'],
                parameters=state['parameters'],seconds=state['seconds'],minimum_query_visits=int(state['sampler']['counts'].min()),
                draw_hash=state['draw_hash'],fitting_objectives=scores,observed_supervised_draws=state['observed_supervised_draws'],
                unknown_bound_supervised_draws=state['unknown_bound_supervised_draws'],unknown_rows_mislabeled_as_observed=0,
                initial_trace=state['trace'][0],final_trace=state['trace'][-1],registration_sha256=sha(REG))
            p=PUBLIC/'fits'/(job['key']+'_'+arm+'.json');once(p,value);fits.append(artifact(p));updates+=state['step'];states.append(state)
            beat(state='fit_complete',job=job['key'],arm=arm,fits=len(fits),total=36)
            del pred
        for k in MATCHED:api.base.exact(states[0][k],states[1][k])
        del data,states,state;gc.collect()
    once(PUBLIC/'training_freeze.json',dict(**receipt(),fits=fits,primary_optimizer_updates=updates,
        first_full_refit_updates=refit_updates,all36complete=True,paired_common_fields_exact=True,
        eval_excluded_from_both_fits=True,held_outcomes_scored=False))


def fitted():
    out={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        r=json.loads(checked(ref).read_text());checked(r['checkpoint'])
        if r['step']!=r['budget'] or r['minimum_query_visits']<4:raise ValueError('Complete fixed endpoints required')
        out[r['job']['key'],r['arm']]=r
    if len(out)!=36:raise ValueError('All thirty-six fits required')
    return out


def infer(cfg,jobs,src,ids,refs,beat,guard,receipt,resume,verify=False):
    end=PUBLIC/('inference_verification.json' if verify else 'policy_freeze.json')
    if end.exists():raise ValueError('Complete inference phase immutable')
    models=fitted();policies=[]
    for job in jobs:
        data=bank(job,src,ids,refs);c=data['causal']
        for arm in cfg['arms']:
            meta=PUBLIC/'policies'/(job['key']+'_'+arm+'.json')
            if resume and meta.exists() and not verify:
                saved=json.loads(meta.read_text());checked(saved['costs']);checked(saved['amplitudes'])
                if saved['registration_sha256']!=sha(REG):raise ValueError('Inference resume binding changed')
                policies.append(artifact(meta));continue
            state=api.base.read_checkpoint(checked(models[job['key'],arm]['checkpoint']));p=api.base.predict(state,c)
            alpha=np.stack([api.amplitudes(p,c['allowed'],binary=binary) for binary in (False,True)],1)
            home=PRIVATE/'policies'/(job['key']+'_'+arm);home.mkdir(parents=True,exist_ok=True)
            if verify:
                saved=json.loads(checked(artifact(meta)).read_text())
                np.testing.assert_array_equal(np.load(checked(saved['costs']),allow_pickle=False),p)
                np.testing.assert_array_equal(np.load(checked(saved['amplitudes']),allow_pickle=False),alpha)
            else:
                prior.frozen.inputs.store_array(home/'costs.npy',p);prior.frozen.inputs.store_array(home/'amplitudes.npy',alpha)
            value=dict(job=job['key'],arm=arm,checkpoint=models[job['key'],arm]['checkpoint'],rows=len(c['ids']),
                ids_sha256=prior.frozen.fingerprint(c['ids']),costs=artifact(home/'costs.npy'),amplitudes=artifact(home/'amplitudes.npy'),
                feature_sha256=prior.frozen.fingerprint(c['x']),future_labels_used_for_inference=False,registration_sha256=sha(REG))
            once(meta,value);policies.append(artifact(meta))
            beat(state='policy_replayed' if verify else 'policy_frozen',job=job['key'],arm=arm,policies=len(policies),total=36)
            del state,p,alpha
        del data,c;gc.collect()
    freeze=dict(policies=policies,training_freeze=artifact(PUBLIC/'training_freeze.json'),registration_sha256=sha(REG),
        held_outcomes_used_for_decisions=False,threshold_selection=False)
    once(PUBLIC/'policy_freeze.json',freeze)
    if verify:once(end,dict(**receipt(),all36_cost_and_amplitude_arrays_exact=True,policy_freeze=artifact(PUBLIC/'policy_freeze.json')))


def summarize(rows,cfg,roster):
    if len(rows)!=72 or len({(r['trial'],r['site']) for r in rows})!=72:raise ValueError('All72fixed nested SOURCE groups required')
    ci=lambda fn:locality_interval(rows,fn,roster,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    names=tuple(rows[0]['policies']);accuracy={};safety={}
    for name in names:
        accuracy[name]=dict(known=ci(lambda r:r['policies'][name]['known_ADE_gain_floor_percent']),
            completion_lower=ci(lambda r:r['policies'][name]['completion_lower_gain_floor_percent']),
            easy=ci(lambda r:r['policies'][name]['subsets']['easy']['gain_percent']),
            hard=ci(lambda r:r['policies'][name]['subsets']['hard']['gain_percent']),
            switch_rate=ci(lambda r:r['policies'][name]['switch_rate']))
        views=[v for r in rows for v in r['policies'][name]['recordings'].values()]
        easy=[v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[name]=dict(required_record_views=len(views),supported_record_views=sum(v['supported'] for v in views),
            known_all_violations=sum(v['known_all_risk'] is not None and v['known_all_risk']>.02 for v in views),
            known_easy_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in views),
            completion_all_violations=sum(v['completion_all_risk_upper'] is not None and v['completion_all_risk_upper']>.02 for v in views),
            completion_easy_violations=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper']>.02 for v in views),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
            worst_signed_easy_net_percent=100*max(easy) if easy else None,
            full_contract_pass=bool(views and all(v['supported'] for v in views)))
    paired={field:ci(lambda r:r['primary'][field]) for field in ('known_pp','completion_lower_pp','completion_upper_pp')}
    lower=paired['completion_lower_pp']['CI95']
    return dict(result_source='fresh_nested_SOURCE_cost_fits_and_held_policy_evaluation_cached_verified_EqMotion',
        groups=72,source_rows=149510,source_recordings=74,source_localities=roster,models=36,
        primary=cfg['primary'],primary_effects=paired,accuracy=accuracy,safety=safety,
        primary_supported=bool(lower is not None and lower[0]>0),
        bound_policy_original_risk_pass=safety['bound_aware_bounded']['full_contract_pass'],
        eval_excluded_from_both_fits=True,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(cfg,jobs,src,ids,refs,beat,guard,receipt,resume,verify=False):
    end=PUBLIC/('verification.json' if verify else 'complete.json')
    if end.exists():raise ValueError('Complete metric phase immutable')
    freeze=json.loads((PUBLIC/'policy_freeze.json').read_text());saved={}
    if len(freeze['policies'])!=36 or freeze['training_freeze']!=artifact(PUBLIC/'training_freeze.json'):
        raise ValueError('All models and decisions must freeze before held outcomes')
    for ref in freeze['policies']:
        r=json.loads(checked(ref).read_text());checked(r['costs']);checked(r['amplitudes']);saved[r['job'],r['arm']]=r
    groups=[]
    for job in jobs:
        data=bank(job,src,ids,refs);paths=data['paths'];b=paths[:,job['floor_index']];neural=paths[:,6]
        origin=src['origin'][data['ids']]
        y=np.load(prior.SOURCE/'target_eval.npy',mmap_mode='r',allow_pickle=False)[data['ids']]-origin[:,None]
        valid=np.load(prior.SOURCE/'valid.npy',mmap_mode='r',allow_pickle=False)[data['ids']]
        policies=dict(full_neural=neural,floor=b);amps={};predicted={}
        for arm in cfg['arms']:
            r=saved[job['key'],arm];p=np.load(checked(r['costs']),allow_pickle=False);a=np.load(checked(r['amplitudes']),allow_pickle=False)
            for i,mode in enumerate(('bounded','binary')):
                name=arm+'_'+mode;policies[name]=api.prediction(b,neural,a[:,i]);amps[name]=a[:,i];predicted[name]=p[:,6]
        for site in job['evaluation_sites']:
            use=data['sites']==site;rec=data['recordings'][use]
            results={name:prior.metrics.policy(p[use],b[use],paths[use,1],y[use],valid[use],rec,
                easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in policies.items()}
            amplitude_stats={}
            for name,a in amps.items():
                value=a[use];p=predicted[name][use];active=value>0
                amplitude_stats[name]=dict(mean=float(value.mean()),maximum=float(value.max()),
                    quantiles=np.quantile(value,[0,.25,.5,.75,.9,.99,1]).tolist(),positive_rows=int(active.sum()),
                    mean_if_positive=float(value[active].mean()) if active.any() else None,
                    predicted_record_costs={str(k):dict(
                        harm=float((value[(rec==k)&active]*p[(rec==k)&active,1]).sum()),
                        reference=float(p[(rec==k)&active,2].sum()),
                        easy_harm=float((value[(rec==k)&active]*p[(rec==k)&active,4]).sum()),
                        easy_reference=float(p[(rec==k)&active,3].sum())) for k in sorted(set(rec))})
            row=dict(trial=job['key'],outer_trial=job['outer_trial'],seed=job['seed'],site=site,
                evaluation_excluded_from_both_fits=site not in job['producer_sites']+job['critic_sites'],
                policies=results,amplitudes=amplitude_stats,
                primary=prior.metrics.paired(policies['known_only_bounded'][use],policies['bound_aware_bounded'][use],b[use],y[use],valid[use]))
            path=PUBLIC/'groups'/(job['key']+'_'+site+'.json')
            if verify and not path.exists():raise ValueError('Missing primary group')
            once(path,row);groups.append(row)
        beat(state='held_groups_replayed' if verify else 'held_groups_complete',job=job['key'],groups=len(groups),total=72)
        del data,policies,paths;gc.collect()
    roster=sorted(set(src['sites'][ids]));s=summarize(groups,cfg,roster)
    if verify and not (PUBLIC/'summary.json').exists():raise ValueError('Cannot verify absent summary')
    once(PUBLIC/'summary.json',s)
    if verify and not json.loads((PUBLIC/'inference_verification.json').read_text())['all36_cost_and_amplitude_arrays_exact']:
        raise ValueError('Complete fresh cost/amplitude inference replay required')
    once(end,dict(**receipt(),all72_metric_groups_exact=verify,summary_sha256=sha(PUBLIC/'summary.json'),
        policy_freeze=artifact(PUBLIC/'policy_freeze.json'),independent_confirmation=False))
    print(json.dumps(s),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','train','infer','eval','verify'))
    p.add_argument('--resume',action='store_true');args=p.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,jobs,src,ids,refs=admission(args.phase=='register')
        if args.phase=='register':return
        live=runtime(cfg,args.phase);values=(cfg,jobs,src,ids,refs,*live,args.resume)
        if args.phase=='pilot':pilot(*values)
        elif args.phase=='train':train(*values)
        elif args.phase=='infer':infer(*values)
        elif args.phase=='eval':evaluate(*values)
        elif args.phase=='verify':infer(*values,verify=True);evaluate(*values,verify=True)


if __name__=='__main__':main()
