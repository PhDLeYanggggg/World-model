"""Matched observation-support cost training on doubly excluded SOURCE roles."""
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

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_bound_aware_cost as parent
from src.world_model import m3w_observation_cost as api
from src.evaluation.m3w_source_label_support import score as probability_score

NAME = 'european_observation_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_observation_cost_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_observation_cost.py',
    ROOT/'tests/test_m3w_observation_cost.py', PUBLIC/'protocol.md')
sha, once, write, artifact, checked = parent.sha, parent.once, parent.write, parent.artifact, parent.checked


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); old, jobs, src, ids, refs = parent.admission()
    same = set(old)-{'name', 'arms', 'primary'}
    if any(cfg.get(k) != old[k] for k in same) or cfg['arms'] != list(api.ARMS) or cfg['observation_prior'] != .95:
        raise ValueError('Original roles, data, full budgets, runtime and thresholds required')
    for name, key in (('european_bound_aware_cost_v1', 'all72_metric_groups_exact'),
            ('european_bound_cost_decomposition_v1', 'all36_job_role_summaries_exact')):
        home = PUBLIC.parent/name; v = json.loads((home/'verification.json').read_text())
        if not v[key] or v['summary_sha256'] != sha(home/'summary.json'): raise ValueError('Complete precursor replay required')
    definition = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES}, jobs=jobs,
        parent_registration=artifact(parent.REG), parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_policies=artifact(parent.PUBLIC/'policy_freeze.json'), parent_verification=artifact(parent.PUBLIC/'verification.json'),
        support_metric_implementation=artifact(ROOT/'src/evaluation/m3w_source_label_support.py'),
        selected_cost_diagnosis=artifact(PUBLIC.parent/'european_bound_cost_decomposition_v1/verification.json'),
        source_rows=len(ids), source_localities=sorted(set(src['sites'][ids])), source_recordings=len(set(src['recordings'][ids])),
        matched_new_models=36, primary=cfg['primary'],
        new_parameter_count=sum(p.numel() for p in api.ObservationCostHead(444,64,17,np.zeros(5,np.float32),.95).parameters()),
        old_parameter_count=sum(p.numel() for p in api.base.CostHead(444,64,17,np.zeros(5,np.float32),'context').parameters()),
        architecture_matched_within_new_arms_only=True, independent_confirmation=False,
        thresholds_changed=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register: once(REG, definition)
    if json.loads(REG.read_text()) != definition: raise ValueError('Observation-cost registration changed')
    if not register:
        for p in (*FILES, REG):
            blob = subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(p): raise ValueError('Commit complete contract before execution')
    return cfg, jobs, src, ids, refs


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoints unchanged')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Keep10GiB disk reserve; no budget downgrade')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h phase boundary; same-budget resume')
    def beat(**kw):
        guard()
        if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['private_cache_cap_bytes']:
            raise OSError('Keep registered private1GiBcap')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**kw)
        write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def receipt(): return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state='runtime_ready',architecture=platform.machine(),torch=torch.__version__,device='cpu',threads=4,interop=1,workers=0,resource_probing=False)
    return beat,guard,receipt


def bank(job,src,ids,refs,training=False):
    data=parent.bank(job,src,ids,refs,training=training)
    data['radius']=np.linalg.norm(data['paths']-data['paths'][:,job['floor_index'],None],axis=-1).max(-1)
    if training: data['learning']=api.prepare(data['learning'],data['paths'],job['floor_index'])
    return data


def path_for(job,arm,family='fits'): return PRIVATE/family/(job['key']+'_'+arm+'.pt.gz')


def one(data,job,arm,cfg,refs,beat,guard,*,family='fits',resume=False,stop_at=None):
    return api.fit(data['learning'],arm=arm,seed=job['seed'],settings=cfg['training'],observation_prior=cfg['observation_prior'],
        identity=dict(registration_sha256=sha(REG),job=job,forecast=refs[job['outer_trial']]),path=path_for(job,arm,family),
        heartbeat=lambda **v:beat(job=job['key'],**v),guard=guard,resume=resume,stop_at=stop_at)


def pilot(cfg,jobs,src,ids,refs,beat,guard,receipt,resume):
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Complete pilot immutable')
    cells=[];updates=0
    for group in cfg['pilot_critic_groups']:
        job=next(j for j in jobs if j['critic_fold']==group);data=bank(job,src,ids,refs,training=True);states=[]
        for arm in cfg['arms']:
            rp=path_for(job,arm,'pilot_resume');dp=path_for(job,arm,'pilot_direct')
            if rp.exists() and not resume: raise ValueError('Incomplete pilot requires explicit resume')
            step=api.base.read_checkpoint(rp)['step'] if rp.exists() else 0
            if step<cfg['pilot_stop']:
                one(data,job,arm,cfg,refs,beat,guard,family='pilot_resume',resume=rp.exists(),stop_at=cfg['pilot_stop'])
            a=one(data,job,arm,cfg,refs,beat,guard,family='pilot_resume',resume=True,stop_at=cfg['pilot_target'])
            b=one(data,job,arm,cfg,refs,beat,guard,family='pilot_direct',resume=resume and dp.exists(),stop_at=cfg['pilot_target'])
            parent.recovered(a,b);states.append(b);updates+=2*b['step']
            cells.append(dict(job=job['key'],critic_group=group,arm=arm,updates=b['step'],budget=b['budget'],
                seconds=b['seconds'],parameters=b['parameters'],exact_resume=True,
                unknown_bound_supervised_draws=b['unknown_bound_supervised_draws']))
        for k in parent.MATCHED: api.base.exact(states[0][k],states[1][k])
        del data,states;gc.collect()
    projected=6*sum(c['seconds']/c['updates']*c['budget'] for c in cells)
    projected_refit=sum(c['seconds']/c['updates']*c['budget'] for c in cells if c['critic_group']==jobs[0]['critic_fold'])
    once(PUBLIC/'pilot.json',dict(**receipt(),cells=cells,actual_optimizer_updates=updates,engineering_pass=True,
        all_three_group_recoveries_exact=True,paired_initialization_preprocessing_draws_exact=True,
        projected36fit_optimizer_seconds=projected,projected_first_full_refits_seconds=projected_refit,
        projection_excludes_preparation_and_inference=True,held_outcomes_scored=False))


def train(cfg,jobs,src,ids,refs,beat,guard,receipt,resume):
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Complete fits immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real complete pilot required')
    fits=[];updates=refit_updates=0
    for job in jobs:
        data=bank(job,src,ids,refs,training=True);states=[]
        for arm in cfg['arms']:
            state=one(data,job,arm,cfg,refs,beat,guard,resume=resume and path_for(job,arm).exists())
            if state['step']!=job['fit_scope']['budget'] or state['sampler']['counts'].min()<4: raise ValueError('Complete query budgets required')
            if job==jobs[0]:
                other=one(data,job,arm,cfg,refs,beat,guard,family='full_refit',resume=resume and path_for(job,arm,'full_refit').exists())
                parent.recovered(state,other);refit_updates+=state['step']
                once(PUBLIC/('first_full_refit_'+arm+'.json'),dict(scientific_state_exact=True,updates=state['step'],
                    checkpoint=artifact(path_for(job,arm)),refit=artifact(path_for(job,arm,'full_refit'))));del other
            pred=api.predict(state,data['causal'],data['radius'])
            scores=parent.objectives(pred['costs'],data['learning'],state['sampler']['counts'])
            fit_probability=probability_score(~data['learning']['observed_future'],1-pred['observation_probability'])
            value=dict(job=job,arm=arm,checkpoint=artifact(path_for(job,arm)),step=state['step'],budget=state['budget'],
                parameters=state['parameters'],seconds=state['seconds'],minimum_query_visits=int(state['sampler']['counts'].min()),
                draw_hash=state['draw_hash'],fitting_objectives=scores,fitting_support_diagnostic=fit_probability,
                initial_trace=state['trace'][0],final_trace=state['trace'][-1],registration_sha256=sha(REG))
            p=PUBLIC/'fits'/(job['key']+'_'+arm+'.json');once(p,value);fits.append(artifact(p));states.append(state);updates+=state['step']
            beat(state='fit_complete',job=job['key'],arm=arm,fits=len(fits),total=36);del pred
        for k in parent.MATCHED: api.base.exact(states[0][k],states[1][k])
        del data,states;gc.collect()
    once(PUBLIC/'training_freeze.json',dict(**receipt(),fits=fits,primary_optimizer_updates=updates,
        first_full_refit_updates=refit_updates,all36complete=True,paired_common_fields_exact=True,held_outcomes_scored=False))


def fitted():
    out={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        r=json.loads(checked(ref).read_text());checked(r['checkpoint'])
        if r['step']!=r['budget'] or r['minimum_query_visits']<4: raise ValueError('Complete fitting endpoints required')
        out[r['job']['key'],r['arm']]=r
    if len(out)!=36: raise ValueError('All36fitsrequired')
    return out


def infer(cfg,jobs,src,ids,refs,beat,guard,receipt,resume,verify=False):
    end=PUBLIC/('inference_verification.json' if verify else 'policy_freeze.json')
    if end.exists(): raise ValueError('Complete inference phase immutable')
    models=fitted();policies=[]
    for job in jobs:
        data=bank(job,src,ids,refs);c=data['causal']
        for arm in cfg['arms']:
            meta=PUBLIC/'policies'/(job['key']+'_'+arm+'.json')
            if resume and meta.exists() and not verify:
                saved=json.loads(meta.read_text())
                for key in ('costs','observation_probability','amplitudes'): checked(saved[key])
                if saved['registration_sha256']!=sha(REG): raise ValueError('Prediction resume identity changed')
                policies.append(artifact(meta));continue
            state=api.base.read_checkpoint(checked(models[job['key'],arm]['checkpoint']));pred=api.predict(state,c,data['radius'])
            alpha=np.stack([parent.api.amplitudes(pred['costs'],c['allowed'],binary=b) for b in (False,True)],1)
            arrays=dict(costs=pred['costs'],observation_probability=pred['observation_probability'],amplitudes=alpha)
            home=PRIVATE/'policies'/(job['key']+'_'+arm);home.mkdir(parents=True,exist_ok=True)
            if verify:
                saved=json.loads(meta.read_text())
                for key,value in arrays.items(): np.testing.assert_array_equal(np.load(checked(saved[key]),allow_pickle=False),value)
            else:
                for key,value in arrays.items(): parent.prior.frozen.inputs.store_array(home/(key+'.npy'),value)
            value=dict(job=job['key'],arm=arm,checkpoint=models[job['key'],arm]['checkpoint'],rows=len(c['ids']),
                ids_sha256=parent.prior.frozen.fingerprint(c['ids']),feature_sha256=parent.prior.frozen.fingerprint(c['x']),
                radius_sha256=parent.prior.frozen.fingerprint(data['radius']),
                **{k:artifact(home/(k+'.npy')) for k in arrays},future_labels_used_for_inference=False,registration_sha256=sha(REG))
            once(meta,value);policies.append(artifact(meta));beat(state='policy_replayed' if verify else 'policy_frozen',job=job['key'],arm=arm,policies=len(policies),total=36)
            del state,pred,alpha,arrays
        del data,c;gc.collect()
    freeze=dict(policies=policies,training_freeze=artifact(PUBLIC/'training_freeze.json'),registration_sha256=sha(REG),held_outcomes_used_for_decisions=False)
    once(PUBLIC/'policy_freeze.json',freeze)
    if verify: once(end,dict(**receipt(),all36_cost_support_amplitude_arrays_exact=True,policy_freeze=artifact(PUBLIC/'policy_freeze.json')))


def summarize(groups,cfg,roster):
    if len(groups)!=72 or len({(r['trial'],r['site']) for r in groups})!=72: raise ValueError('Every original nested SOURCE group required')
    ci=lambda fn:parent.locality_interval(groups,fn,roster,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    accuracy={};safety={}
    for name in groups[0]['policies']:
        accuracy[name]=dict(known=ci(lambda r:r['policies'][name]['known_ADE_gain_floor_percent']),
            completion_lower=ci(lambda r:r['policies'][name]['completion_lower_gain_floor_percent']),
            easy=ci(lambda r:r['policies'][name]['subsets']['easy']['gain_percent']),hard=ci(lambda r:r['policies'][name]['subsets']['hard']['gain_percent']),
            switch_rate=ci(lambda r:r['policies'][name]['switch_rate']))
        views=[v for r in groups for v in r['policies'][name]['recordings'].values()]
        easy=[v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[name]=dict(required_record_views=len(views),supported_record_views=sum(v['supported'] for v in views),
            known_all_violations=sum(v['known_all_risk'] is not None and v['known_all_risk']>.02 for v in views),
            known_easy_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in views),
            completion_all_violations=sum(v['completion_all_risk_upper'] is not None and v['completion_all_risk_upper']>.02 for v in views),
            completion_easy_violations=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper']>.02 for v in views),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
            worst_signed_easy_net_percent=100*max(easy) if easy else None,full_contract_pass=all(v['supported'] for v in views))
    contrasts={name:{field:ci(lambda r,n=name,f=field:r[n][f]) for field in ('known_pp','completion_lower_pp','completion_upper_pp')}
        for name in ('primary','identified_minus_marginal','marginal_minus_direct')}
    support={a:{m:ci(lambda r,a=a,m=m:r['support'][a][m]) for m in ('AUROC','AUPRC','Brier','ECE')} for a in cfg['arms']}
    low=contrasts['primary']['completion_lower_pp']['CI95']
    return dict(result_source='fresh36observation_cost_fits_and_held_SOURCE_evaluation_cached_verified_direct_and_EqMotion',
        groups=72,models=36,source_rows=149510,source_recordings=74,source_localities=roster,
        primary=cfg['primary'],contrasts=contrasts,accuracy=accuracy,safety=safety,support=support,
        primary_supported=bool(low is not None and low[0]>0),
        original_risk_pass=safety['identified_support_bounded']['full_contract_pass'],
        independent_confirmation=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(cfg,jobs,src,ids,refs,beat,guard,receipt,resume,verify=False):
    end=PUBLIC/('verification.json' if verify else 'complete.json')
    if end.exists(): raise ValueError('Complete metric phase immutable')
    freeze=json.loads((PUBLIC/'policy_freeze.json').read_text());saved={};old={}
    if len(freeze['policies'])!=36 or freeze['training_freeze']!=artifact(PUBLIC/'training_freeze.json'):
        raise ValueError('All new models and predictions must freeze before outcomes')
    for ref in freeze['policies']:
        r=json.loads(checked(ref).read_text())
        for key in ('costs','observation_probability','amplitudes'): checked(r[key])
        saved[r['job'],r['arm']]=r
    for ref in json.loads((parent.PUBLIC/'policy_freeze.json').read_text())['policies']:
        r=json.loads(checked(ref).read_text())
        if r['arm']=='bound_aware': old[r['job']]=r
    groups=[]
    for job in jobs:
        data=bank(job,src,ids,refs);paths=data['paths'];b=paths[:,job['floor_index']];neural=paths[:,6];c=data['causal']
        y=np.load(parent.prior.SOURCE/'target_eval.npy',mmap_mode='r',allow_pickle=False)[data['ids']]-src['origin'][data['ids'],None]
        valid=np.load(parent.prior.SOURCE/'valid.npy',mmap_mode='r',allow_pickle=False)[data['ids']]
        olda=np.load(checked(old[job['key']]['amplitudes']),allow_pickle=False)[:,0]
        policies=dict(floor=b,full_neural=neural,direct_bound_bounded=parent.api.prediction(b,neural,olda));amps={};predicted={};prob={}
        for arm in cfg['arms']:
            r=saved[job['key'],arm];p=np.load(checked(r['costs']),allow_pickle=False);a=np.load(checked(r['amplitudes']),allow_pickle=False)
            prob[arm]=np.load(checked(r['observation_probability']),allow_pickle=False)
            for j,mode in enumerate(('bounded','binary')):
                key=arm+'_'+mode;policies[key]=parent.api.prediction(b,neural,a[:,j]);amps[key]=a[:,j];predicted[key]=p[:,6]
        for site in job['evaluation_sites']:
            use=data['sites']==site;rec=data['recordings'][use]
            results={name:parent.prior.metrics.policy(p[use],b[use],paths[use,1],y[use],valid[use],rec,
                easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in policies.items()}
            original=json.loads((parent.PUBLIC/'groups'/(job['key']+'_'+site+'.json')).read_text())['policies']['bound_aware_bounded']
            if results['direct_bound_bounded']!=original: raise ValueError('Frozen direct control must reproduce exactly')
            stats={}
            for name,a in amps.items():
                v=a[use];p=predicted[name][use];take=v>0
                stats[name]=dict(mean=float(v.mean()),quantiles=np.quantile(v,[0,.25,.5,.75,.9,.99,1]).tolist(),positive_rows=int(take.sum()),
                    predicted_record_costs={str(k):dict(harm=float((v[(rec==k)&take]*p[(rec==k)&take,1]).sum()),
                        reference=float(p[(rec==k)&take,2].sum()),easy_harm=float((v[(rec==k)&take]*p[(rec==k)&take,4]).sum()),
                        easy_reference=float(p[(rec==k)&take,3].sum())) for k in sorted(set(rec))})
            pair=lambda a,z:parent.prior.metrics.paired(policies[a][use],policies[z][use],b[use],y[use],valid[use])
            row=dict(trial=job['key'],seed=job['seed'],site=site,policies=results,amplitudes=stats,
                support={a:probability_score(~valid[use].any(1),1-prob[a][use]) for a in cfg['arms']},
                primary=pair('direct_bound_bounded','identified_support_bounded'),
                identified_minus_marginal=pair('marginal_support_bounded','identified_support_bounded'),
                marginal_minus_direct=pair('direct_bound_bounded','marginal_support_bounded'))
            path=PUBLIC/'groups'/(job['key']+'_'+site+'.json')
            if verify and not path.exists(): raise ValueError('Missing original metric group')
            once(path,row);groups.append(row)
        beat(state='metric_groups_replayed' if verify else 'metric_groups_complete',job=job['key'],groups=len(groups),total=72)
        del data,paths,policies,amps,predicted,prob;gc.collect()
    summary=summarize(groups,cfg,sorted(set(src['sites'][ids])))
    if verify and not json.loads((PUBLIC/'inference_verification.json').read_text())['all36_cost_support_amplitude_arrays_exact']:
        raise ValueError('Separate complete prediction replay required')
    once(PUBLIC/'summary.json',summary)
    once(end,dict(**receipt(),all72_metric_groups_exact=verify,summary_sha256=sha(PUBLIC/'summary.json'),policy_freeze=artifact(PUBLIC/'policy_freeze.json')))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','train','infer','eval','verify'));p.add_argument('--resume',action='store_true')
    args=p.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,jobs,src,ids,refs=admission(args.phase=='register')
        if args.phase=='register': return
        values=(cfg,jobs,src,ids,refs,*runtime(cfg,args.phase),args.resume)
        if args.phase=='pilot': pilot(*values)
        elif args.phase=='train': train(*values)
        elif args.phase=='infer': infer(*values)
        elif args.phase=='eval': evaluate(*values)
        else: infer(*values,verify=True);evaluate(*values,verify=True)


if __name__=='__main__':main()
