"""Matched frozen-interaction versus zero-feature SOURCE intervention critics."""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_bound_aware_cost as parent
from scripts import run_m3w_eqmotion_context_cache_v2 as cache
from src.world_model import m3w_interaction_feature_cost as api

NAME = 'european_interaction_feature_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_interaction_feature_cost_v1.json'
REG = PUBLIC/'registration_v2.json'
INITIAL_REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_interaction_feature_cost.py',
         ROOT/'tests/test_m3w_interaction_feature_cost.py', ROOT/'tests/test_m3w_interaction_feature_summary.py',
         ROOT/'tests/test_m3w_interaction_feature_roles.py',
         PUBLIC/'protocol.md')
sha, once, write, artifact, checked = parent.sha, parent.once, parent.write, parent.artifact, parent.checked


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    previous, jobs, source, ids, refs = parent.admission()
    same = set(previous)-{'name', 'arms', 'primary'}
    if any(cfg.get(k) != previous[k] for k in same) or cfg['arms'] != list(api.ARMS):
        raise ValueError('Only the fixed matched feature treatment changes')
    verified = json.loads((cache.PUBLIC/'verification.json').read_text())
    if not verified['all9latent_arrays_exact'] or not verified['all_parent_forecasts_exact']:
        raise ValueError('Every frozen latent and original forecast must replay before learning')
    if verified['cache_freeze'] != artifact(cache.PUBLIC/'cache_freeze.json'):
        raise ValueError('Verified cache freeze changed')
    caches = {}
    for ref in json.loads((cache.PUBLIC/'cache_freeze.json').read_text())['caches']:
        row = json.loads(checked(ref).read_text())
        checked(row['latent'])
        checked(row['ids'])
        caches[row['identity']['trial']] = row
    if set(caches) != set(refs):
        raise ValueError('All nine aligned producer caches required')
    initial = json.loads(INITIAL_REG.read_text())
    if initial['jobs'] != jobs or initial['primary'] != cfg['primary'] or initial['cache_freeze'] != artifact(cache.PUBLIC/'cache_freeze.json'):
        raise ValueError('The input-role correction must not change jobs,primary or cache')
    definition = dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES}, jobs=jobs,
                      initial_registration=artifact(INITIAL_REG), initial_failed_pilot=artifact(PRIVATE/'pilot.log'),
                      pre_fit_amendment='forward_training_role_to_parent_bank_before_preprocessing_no_optimizer_update_or_policy_score_in_failed_attempt',
                      prior_failed_H_label_preprocessing_computed_and_discarded=True,
                      parent_registration=artifact(parent.REG), parent_verification=artifact(parent.PUBLIC/'verification.json'),
                      cache_registration=artifact(cache.REG), cache_verification=artifact(cache.PUBLIC/'verification.json'),
                      cache_freeze=artifact(cache.PUBLIC/'cache_freeze.json'), parent_objective=artifact(ROOT/'src/world_model/m3w_bound_aware_cost.py'),
                      new_models=36, input_dimension=508, parameter_count=sum(p.numel() for p in api.base.base.CostHead(508,64,17,np.zeros(5,np.float32),'context').parameters()),
                      primary=cfg['primary'], normalizer_fit_on_critic_sites_only=True,
                      source_rows=len(ids), source_recordings=len(set(source['recordings'][ids])),
                      source_localities=sorted(set(source['sites'][ids])), independent_confirmation=False,
                      thresholds_changed=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register:
        once(REG, definition)
    if json.loads(REG.read_text()) != definition:
        raise ValueError('Frozen matched-feature study changed')
    if not register:
        for p in (*FILES, REG):
            blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(p):
                raise ValueError('Commit all experiment definitions before execution')
    return cfg, jobs, source, ids, refs, caches


def runtime(cfg, phase):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep10GiB free; no data deletion or budget downgrade')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h phase boundary; same-budget resume')
    def beat(**kw):
        guard()
        if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['private_cache_cap_bytes']:
            raise OSError('Keep1GiB private cap')
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phase=phase, **kw)
        write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    def receipt():
        return dict(pid=os.getpid(), seconds=time.monotonic()-start,
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, registration_sha256=sha(REG))
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return beat, guard, receipt


def bank(job, source, ids, refs, caches, training=False):
    data = parent.bank(job, source, ids, refs, training=training)
    expected = job['critic_sites'] if training else job['evaluation_sites']
    if set(data['causal']['sites']) != set(expected):
        raise ValueError('Requested fitting or evaluation role must match before feature preparation')
    ref = caches[job['outer_trial']]
    cached_ids = np.load(checked(ref['ids']), allow_pickle=False)
    latent = np.load(checked(ref['latent']), mmap_mode='r', allow_pickle=False)
    positions = parent.aligned_rows(cached_ids, data['ids'])
    causal = api.attach(data['causal'], latent[positions])
    if causal['x'].shape[-1] != 508:
        raise ValueError('Fixed444plus64feature schema required')
    data['causal'] = causal
    data['cache_ref'] = ref
    if training:
        origin = source['origin'][data['ids']]
        target = np.load(parent.prior.SOURCE/'target_eval.npy', mmap_mode='r', allow_pickle=False)[data['ids']]-origin[:,None]
        valid = np.load(parent.prior.SOURCE/'valid.npy', mmap_mode='r', allow_pickle=False)[data['ids']]
        data['learning'] = api.base.training_data(causal, data['paths'], target, valid, job['floor_index'], job['easy_cut'])
        if set(data['learning']['preprocess']['source_names']) != set(job['critic_sites']):
            raise ValueError('Feature normalization is critic-fitting only')
    return data


def path_for(job, arm, family='fits'):
    return PRIVATE/family/(job['key']+'_'+arm+'.pt.gz')


def one(data, job, arm, cfg, refs, beat, guard, family='fits', resume=False, stop_at=None):
    return api.fit(data['learning'], arm=arm, seed=job['seed'], settings=cfg['training'],
                   identity=dict(registration_sha256=sha(REG), job=job, forecast=refs[job['outer_trial']],
                                 latent_cache=data['cache_ref']['latent']),
                   path=path_for(job, arm, family), heartbeat=lambda **v: beat(job=job['key'], **v),
                   guard=guard, resume=resume, stop_at=stop_at)


def pilot(cfg, jobs, source, ids, refs, caches, beat, guard, receipt, resume):
    if (PUBLIC/'pilot.json').exists():
        raise ValueError('Complete pilot immutable')
    cells, updates = [], 0
    for group in cfg['pilot_critic_groups']:
        job = next(j for j in jobs if j['critic_fold'] == group)
        data = bank(job, source, ids, refs, caches, training=True)
        states = []
        for arm in cfg['arms']:
            rp = path_for(job, arm, 'pilot_resume')
            dp = path_for(job, arm, 'pilot_direct')
            if rp.exists() and not resume:
                raise ValueError('Partial pilot requires explicit resume')
            step = api.base.base.read_checkpoint(rp)['step'] if rp.exists() else 0
            if step < cfg['pilot_stop']:
                one(data, job, arm, cfg, refs, beat, guard, family='pilot_resume', resume=rp.exists(), stop_at=cfg['pilot_stop'])
            a = one(data, job, arm, cfg, refs, beat, guard, family='pilot_resume', resume=True, stop_at=cfg['pilot_target'])
            b = one(data, job, arm, cfg, refs, beat, guard, family='pilot_direct', resume=resume and dp.exists(), stop_at=cfg['pilot_target'])
            parent.recovered(a, b)
            states.append(b)
            updates += 2*b['step']
            cells.append(dict(job=job['key'], critic_group=group, arm=arm, updates=b['step'], budget=b['budget'],
                              seconds=b['seconds'], parameters=b['parameters'], exact_resume=True))
        for key in parent.MATCHED:
            api.base.base.exact(states[0][key], states[1][key])
        del data, states
        gc.collect()
    projected = 6*sum(c['seconds']/c['updates']*c['budget'] for c in cells)
    once(PUBLIC/'pilot.json', dict(**receipt(), cells=cells, actual_optimizer_updates=updates, engineering_pass=True,
                                 all_three_group_recoveries_exact=True, paired_capacity_initialization_normalizer_draws_exact=True,
                                 projected36fit_optimizer_seconds=projected, projection_excludes_preparation_and_inference=True,
                                 held_outcomes_scored=False))


def train(cfg, jobs, source, ids, refs, caches, beat, guard, receipt, resume):
    if (PUBLIC/'training_freeze.json').exists():
        raise ValueError('Complete training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Complete three-group real recovery pilot required')
    fits, updates, refits = [], 0, 0
    for job in jobs:
        data = bank(job, source, ids, refs, caches, training=True)
        states = []
        for arm in cfg['arms']:
            state = one(data, job, arm, cfg, refs, beat, guard, resume=resume and path_for(job, arm).exists())
            if state['step'] != job['fit_scope']['budget'] or state['sampler']['counts'].min() < 4:
                raise ValueError('Every original full coverage budget must finish')
            if job == jobs[0]:
                other = one(data, job, arm, cfg, refs, beat, guard, family='full_refit',
                            resume=resume and path_for(job, arm, 'full_refit').exists())
                parent.recovered(state, other)
                refits += state['step']
                once(PUBLIC/('first_full_refit_'+arm+'.json'), dict(scientific_state_exact=True, updates=state['step']))
                del other
            pred = api.predict(state, data['causal'])
            score = parent.objectives(pred, data['learning'], state['sampler']['counts'])
            value = dict(job=job, arm=arm, checkpoint=artifact(path_for(job,arm)), step=state['step'], budget=state['budget'],
                         parameters=state['parameters'], seconds=state['seconds'], minimum_query_visits=int(state['sampler']['counts'].min()),
                         draw_hash=state['draw_hash'], fitting_objectives=score, initial_trace=state['trace'][0],
                         final_trace=state['trace'][-1], registration_sha256=sha(REG))
            path = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            once(path, value)
            fits.append(artifact(path))
            updates += state['step']
            states.append(state)
            beat(state='fit_complete', job=job['key'], arm=arm, fits=len(fits), total=36)
            del pred
        for key in parent.MATCHED:
            api.base.base.exact(states[0][key], states[1][key])
        del data, states, state
        gc.collect()
    once(PUBLIC/'training_freeze.json', dict(**receipt(), fits=fits, primary_optimizer_updates=updates,
                                          first_full_refit_updates=refits, all36complete=True,
                                          paired_common_fields_exact=True, held_outcomes_scored=False))


def fitted():
    out = {}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        row = json.loads(checked(ref).read_text())
        checked(row['checkpoint'])
        if row['step'] != row['budget'] or row['minimum_query_visits'] < 4:
            raise ValueError('Every fixed endpoint must finish')
        out[row['job']['key'], row['arm']] = row
    if len(out) != 36:
        raise ValueError('All36fits required')
    return out


def infer(cfg, jobs, source, ids, refs, caches, beat, guard, receipt, resume, verify=False):
    end = PUBLIC/('inference_verification.json' if verify else 'policy_freeze.json')
    if end.exists():
        raise ValueError('Complete inference phase immutable')
    models, policies = fitted(), []
    for job in jobs:
        data = bank(job, source, ids, refs, caches)
        c = data['causal']
        for arm in cfg['arms']:
            meta = PUBLIC/'policies'/(job['key']+'_'+arm+'.json')
            if resume and meta.exists() and not verify:
                saved = json.loads(meta.read_text())
                checked(saved['costs'])
                checked(saved['amplitudes'])
                if saved['registration_sha256'] != sha(REG):
                    raise ValueError('Inference resume definition changed')
                policies.append(artifact(meta))
                continue
            state = api.base.base.read_checkpoint(checked(models[job['key'],arm]['checkpoint']))
            p = api.predict(state, c)
            a = np.stack([api.base.amplitudes(p,c['allowed'],binary=b) for b in (False,True)],1)
            home = PRIVATE/'policies'/(job['key']+'_'+arm)
            home.mkdir(parents=True, exist_ok=True)
            if verify:
                saved = json.loads(meta.read_text())
                np.testing.assert_array_equal(np.load(checked(saved['costs']),allow_pickle=False),p)
                np.testing.assert_array_equal(np.load(checked(saved['amplitudes']),allow_pickle=False),a)
            else:
                parent.prior.frozen.inputs.store_array(home/'costs.npy',p)
                parent.prior.frozen.inputs.store_array(home/'amplitudes.npy',a)
            once(meta,dict(job=job['key'],arm=arm,checkpoint=models[job['key'],arm]['checkpoint'],rows=len(c['ids']),
                           ids_sha256=parent.prior.frozen.fingerprint(c['ids']),feature_sha256=parent.prior.frozen.fingerprint(c['x']),
                           costs=artifact(home/'costs.npy'),amplitudes=artifact(home/'amplitudes.npy'),
                           zero64_standardized_block_exact_zero=arm=='zero64',future_labels_used_for_inference=False,
                           registration_sha256=sha(REG)))
            policies.append(artifact(meta))
            beat(state='policy_replayed' if verify else 'policy_frozen',job=job['key'],arm=arm,policies=len(policies),total=36)
            del state,p,a
        del data,c
        gc.collect()
    freeze = dict(policies=policies,training_freeze=artifact(PUBLIC/'training_freeze.json'),registration_sha256=sha(REG),
                  held_outcomes_used_for_decisions=False,threshold_selection=False)
    once(PUBLIC/'policy_freeze.json',freeze)
    if verify:
        once(end,dict(**receipt(),all36_cost_amplitude_arrays_exact=True,policy_freeze=artifact(PUBLIC/'policy_freeze.json')))


def summarize(groups,cfg,roster):
    if len(groups)!=72 or len({(r['trial'],r['site']) for r in groups})!=72:
        raise ValueError('All72 original nested SOURCE groups required')
    ci = lambda fn: parent.locality_interval(groups,fn,roster,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    accuracy,safety = {},{}
    for name in groups[0]['policies']:
        accuracy[name] = dict(known=ci(lambda r:r['policies'][name]['known_ADE_gain_floor_percent']),
                              completion_lower=ci(lambda r:r['policies'][name]['completion_lower_gain_floor_percent']),
                              easy=ci(lambda r:r['policies'][name]['subsets']['easy']['gain_percent']),
                              hard=ci(lambda r:r['policies'][name]['subsets']['hard']['gain_percent']),
                              switch_rate=ci(lambda r:r['policies'][name]['switch_rate']),
                              known_ADE_pixel=ci(lambda r:r['policies'][name]['known_ADE_pixel']),
                              available_endpoint_FDE_diagnostic_pixel=ci(lambda r:r['policies'][name]['available_endpoint_FDE_diagnostic_pixel']),
                              mean_role_specific_p95_ADE_pixel=ci(lambda r:r['policies'][name]['p95_known_ADE_pixel']))
        def cv_gain(row):
            cv = row['policies']['constant_velocity_causal_fd']['known_ADE_pixel']
            model = row['policies'][name]['known_ADE_pixel']
            return None if cv is None or cv<=0 or model is None else 100*(cv-model)/cv
        accuracy[name]['known_gain_CV_percent'] = ci(cv_gain)
        views = [v for r in groups for v in r['policies'][name]['recordings'].values()]
        easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[name] = dict(required_record_views=len(views),supported_record_views=sum(v['supported'] for v in views),
                            known_all_violations=sum(v['known_all_risk'] is not None and v['known_all_risk']>.02 for v in views),
                            known_easy_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in views),
                            completion_all_violations=sum(v['completion_all_risk_upper'] is not None and v['completion_all_risk_upper']>.02 for v in views),
                            completion_easy_violations=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper']>.02 for v in views),
                            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
                            worst_signed_easy_net_percent=100*max(easy) if easy else None,
                            full_contract_pass=bool(views and all(v['supported'] for v in views)))
    contrasts = {name:{field:ci(lambda r,n=name,f=field:r[n][f]) for field in ('known_pp','completion_lower_pp','completion_upper_pp')}
                 for name in ('primary','interaction_minus_direct','zero_minus_direct')}
    low = contrasts['primary']['completion_lower_pp']['CI95']
    return dict(result_source='fresh36matched_interaction_feature_cost_fits_and_SOURCE_readout_cached_verified_forecaster',
                models=36,groups=72,source_rows=149510,source_recordings=74,source_localities=roster,
                primary=cfg['primary'],contrasts=contrasts,accuracy=accuracy,safety=safety,
                primary_supported=bool(low is not None and low[0]>0),original_risk_pass=safety['interaction64_bounded']['full_contract_pass'],
                independent_confirmation=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(cfg,jobs,source,ids,refs,caches,beat,guard,receipt,resume,verify=False):
    end = PUBLIC/('verification.json' if verify else 'complete.json')
    if end.exists():
        raise ValueError('Complete metric phase immutable')
    freeze = json.loads((PUBLIC/'policy_freeze.json').read_text())
    if len(freeze['policies'])!=36 or freeze['training_freeze']!=artifact(PUBLIC/'training_freeze.json'):
        raise ValueError('All models and decisions must freeze before outcomes')
    saved,old = {},{}
    for ref in freeze['policies']:
        row = json.loads(checked(ref).read_text())
        checked(row['costs'])
        checked(row['amplitudes'])
        saved[row['job'],row['arm']] = row
    for ref in json.loads((parent.PUBLIC/'policy_freeze.json').read_text())['policies']:
        row = json.loads(checked(ref).read_text())
        if row['arm']=='bound_aware':
            old[row['job']] = row
    groups = []
    for job in jobs:
        data = bank(job,source,ids,refs,caches)
        paths,c = data['paths'],data['causal']
        b,neural = paths[:,job['floor_index']],paths[:,6]
        y = np.load(parent.prior.SOURCE/'target_eval.npy',mmap_mode='r',allow_pickle=False)[data['ids']]-source['origin'][data['ids'],None]
        valid = np.load(parent.prior.SOURCE/'valid.npy',mmap_mode='r',allow_pickle=False)[data['ids']]
        olda = np.load(checked(old[job['key']]['amplitudes']),allow_pickle=False)[:,0]
        policies = dict(floor=b,constant_velocity_causal_fd=paths[:,1],full_neural=neural,
                        direct_bound_bounded=api.base.prediction(b,neural,olda))
        amps = {}
        for arm in cfg['arms']:
            row = saved[job['key'],arm]
            a = np.load(checked(row['amplitudes']),allow_pickle=False)
            for j,mode in enumerate(('bounded','binary')):
                name = arm+'_'+mode
                policies[name] = api.base.prediction(b,neural,a[:,j])
                amps[name] = a[:,j]
        for site in job['evaluation_sites']:
            use = data['sites']==site
            rec = data['recordings'][use]
            results = {name:parent.prior.metrics.policy(p[use],b[use],paths[use,1],y[use],valid[use],rec,
                        easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in policies.items()}
            original = json.loads((parent.PUBLIC/'groups'/(job['key']+'_'+site+'.json')).read_text())['policies']['bound_aware_bounded']
            if results['direct_bound_bounded']!=original:
                raise ValueError('Frozen direct control must remain exactly reproducible')
            pair = lambda left,right:parent.prior.metrics.paired(policies[left][use],policies[right][use],b[use],y[use],valid[use])
            row = dict(trial=job['key'],seed=job['seed'],site=site,policies=results,
                       evaluation_excluded_from_both_fits=site not in job['producer_sites']+job['critic_sites'],
                       amplitudes={name:dict(mean=float(a[use].mean()),positive_rows=int((a[use]>0).sum()),
                                  quantiles=np.quantile(a[use],[0,.25,.5,.75,.9,.99,1]).tolist()) for name,a in amps.items()},
                       primary=pair('zero64_bounded','interaction64_bounded'),
                       interaction_minus_direct=pair('direct_bound_bounded','interaction64_bounded'),
                       zero_minus_direct=pair('direct_bound_bounded','zero64_bounded'))
            path = PUBLIC/'groups'/(job['key']+'_'+site+'.json')
            if verify and not path.exists():
                raise ValueError('Missing original metric group')
            once(path,row)
            groups.append(row)
        beat(state='metric_groups_replayed' if verify else 'metric_groups_complete',job=job['key'],groups=len(groups),total=72)
        del data,paths,c,policies,amps
        gc.collect()
    summary = summarize(groups,cfg,sorted(set(source['sites'][ids])))
    if verify and not json.loads((PUBLIC/'inference_verification.json').read_text())['all36_cost_amplitude_arrays_exact']:
        raise ValueError('Complete prediction replay required')
    once(PUBLIC/'summary.json',summary)
    once(end,dict(**receipt(),all72_metric_groups_exact=verify,summary_sha256=sha(PUBLIC/'summary.json'),
                  policy_freeze=artifact(PUBLIC/'policy_freeze.json')))
    print(json.dumps(summary),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','infer','eval','verify'))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        values = admission(args.phase=='register')
        if args.phase=='register':
            return
        args0 = (*values,*runtime(values[0],args.phase),args.resume)
        if args.phase=='pilot':
            pilot(*args0)
        elif args.phase=='train':
            train(*args0)
        elif args.phase=='infer':
            infer(*args0)
        elif args.phase=='eval':
            evaluate(*args0)
        else:
            infer(*args0,verify=True)
            evaluate(*args0,verify=True)


if __name__=='__main__':
    main()
