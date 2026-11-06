"""Full exposed-own-B readout of frozen risk heads; never a new held test."""
import argparse
import fcntl
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
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_donor_seeds as seeds
from scripts import run_m3w_moving_donor_augmentation as donor
from scripts import run_m3w_moving_neural_transport as neural
from scripts import run_m3w_moving_producer_crossfit as source
from src.data_unification import m3w_producer_crossfit as packing
from src.data_unification import m3w_complete_policy_packet as features
from src.evaluation import m3w_complete_donor_policy as method
from src.evaluation.m3w_risk_float_parity import compare as float_parity
from src.evaluation.m3w_seed_consensus import scores
from src.evaluation.m3w_peer_path_probe import predict as forest_predict
from src.evaluation.m3w_matched_intervention_utility import matched_utility
from src.world_model import m3w_proposal_conditioned_risk as proposer

np,torch = seeds.np,seeds.torch
read,stable,sha,once,checked,artifact = seeds.read,seeds.stable,seeds.sha,seeds.once,seeds.checked,seeds.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_complete_donor_policy_v1'
CONFIG = ROOT/'configs/m3w_moving_complete_donor_policy_v1.json'
REG = PUBLIC/'registration_v2.json'
BASE = ROOT/'outputs/publication_readiness_2026_09'
FILES = (Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_complete_donor_policy.py',
         ROOT/'src/data_unification/m3w_complete_policy_packet.py',ROOT/'tests/test_m3w_complete_donor_policy.py',
         ROOT/'src/evaluation/m3w_risk_float_parity.py',ROOT/'tests/test_m3w_risk_float_parity.py',
         PUBLIC/'numeric_verification_amendment.md')


def register():
    cfg = read(CONFIG)
    seeds.register()
    registry = read(BASE/'moving_ego_frame_v1/registration_v2.json')
    models = read(BASE/'moving_proposal_context_v1/registration.json')
    bank = read(BASE/'moving_shared_motion_cost_v1/registration.json')
    forest = read(BASE/'moving_channel_specialist_v1/analysis.json')
    if (cfg['expected_rows']!=62466 or cfg['risk_budget']!=.02 or cfg['risk_threshold']!=0
            or cfg['prediction_rtol']!=1e-6 or cfg['prediction_atol']!=1e-9
            or any(cfg[k] for k in ('new_training','threshold_search','new_outer_C_H_readout',
                'identity_certification','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed complete-population diagnostic and original rules required')
    jobs = []
    refs = []
    for job in registry['jobs']:
        part = registry['partitions'][job['key']][0]
        names = set(part['names'])
        if names & (set(job['producer_sites'])|set(job['evaluation_sites'])):
            raise ValueError('Only original exposed own-B, not producer or outer H')
        ctx = next(c for c in forest['contexts'] if c['job']==job['key'])
        fr = {f['source']:f['arms']['shared64']['checkpoint'] for f in ctx['folds']}
        if set(fr)!=names:
            raise ValueError('Complete source-excluded forest control required')
        jobs.append(dict(job=job,part=part,proposer=models['proposer_checkpoints'][job['key']],
                         forecast=bank['forecasts'][job['forecast_key']],forest=fr))
        refs.extend([models['proposer_checkpoints'][job['key']],bank['forecasts'][job['forecast_key']],*fr.values()])
    if sum(j['part']['routing_job']['evaluation_scope']['rows'] for j in jobs)!=cfg['expected_rows']:
        raise ValueError('Do not change old full own-B scopes')
    for ref in refs:
        checked(ref)
    value = dict(config=cfg,jobs=jobs,files={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        original_definition=artifact(PUBLIC/'registration.json'),definition_version=2,
        numeric_verification='training_scaled_64float32epsilon_all_channel_and_ensemble_choices_exact',
        dependencies={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__)) for m in
            (packing,features.shared,features.amplitude,features.relation,features.neighbors,method.original,proposer,proposer.router)},
        frozen_models=artifact(seeds.PUBLIC/'training_freeze.json'),
        cached_known_result=artifact(seeds.PUBLIC/'analysis.json'),
        cached_consensus=artifact(seeds.PUBLIC/'consensus/analysis.json'),
        admission=artifact(BASE/'moving_development_admission_v1/admission.json'),
        role='complete_exposed_own_B_development_not_independent_confirmation',
        new_training=False,new_outer_C_H_readout=False,original_rules_unchanged=True,
        identity_certification=False)
    once(REG,stable(value))
    return cfg,value


def committed():
    for p in (*FILES,REG):
        b = subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(b).hexdigest()!=sha(p):
            raise ValueError('Commit exact readout before real data work')


def runtime(cfg,phase):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume frozen context cache')))
    start,last = time.monotonic(),[0.]
    def guard():
        now = time.monotonic()
        if now-start>cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Resume same diagnostic after12h')
        if now-last[0]>1:
            last[0]=now
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:
                raise OSError('Keep5GiB reserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>cfg['artifact_cap_bytes']:
                raise OSError('Private artifact cap exceeded')
    def beat(**values):
        guard()
        event=dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
        source.atomic(PRIVATE/'heartbeat.json',event)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    beat(state='ready',machine=platform.machine(),device='cpu',torch_threads=4,interop_threads=1,workers=0)
    return start,guard,beat


def save_arrays(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            for k,v in payload.items():np.testing.assert_array_equal(old[k],v)
    else:
        pending=path.with_suffix('.pending')
        with pending.open('wb') as f:np.savez(f,**payload)
        pending.replace(path)


def prepare(reg,beat):
    allowed = set(s for j in reg['jobs'] for s in j['part']['names'])
    d,population = packing.pack(ROOT,'outputs/publication_readiness_2026_09/moving_development_admission_v1/admission.json',allowed,beat)
    if len(d['sites'])!=62466:
        raise ValueError('Complete original exposed own-B population required')
    path=PRIVATE/'data.npz'
    save_arrays(path,d)
    once(PUBLIC/'population.json',stable(dict(population,cache=artifact(path))))
    beat(state='complete_own_B_packed',rows=len(d['sites']))


def load_data():
    with np.load(checked(read(PUBLIC/'population.json')['cache']),allow_pickle=False) as cache:
        return {k:cache[k].copy() for k in cache.files}


def risk_checkpoint(job,site,seed):
    if seed==17:
        return read(donor.PUBLIC/'fits'/f'{job}_{site}_augmented.json')['checkpoint']
    return read(seeds.PUBLIC/'fits'/f'{job}_{site}_augmented_seed{seed}.pt.json')['checkpoint']


def infer_one(data,node,cfg,beat):
    job,part = node['job'],node['part']
    ix = np.flatnonzero(np.isin(data['sites'],part['names']))
    ids = data['global_ids'][ix]
    scope = part['routing_job']['evaluation_scope']
    if len(ids)!=scope['rows'] or packing.array_hash(ids)!=scope['ids_sha256']:
        raise ValueError('Global row scope differs from the frozen own-B partition')
    with np.load(checked(node['forecast']),allow_pickle=False) as saved:
        where=np.searchsorted(saved['ids'],ids)
        np.testing.assert_array_equal(saved['ids'][where],ids)
        neural_path=saved['relative_prediction'][where].copy()
    p = features.packet(geometry=data['geometry'][ix],shared_context=data['shared_context'][ix],
        history=data['history'][ix],recordings=data['recordings'][ix],frames=data['frames'][ix],
        ids=ids,sites=data['sites'][ix],relative_forecast=neural_path,floor_index=job['floor_index'])
    c=p['causal']
    router=neural.rel.base.read_checkpoint(checked(node['proposer']))
    proposer.assert_excluded(router,c,part['names'])
    own,utility=proposer.router.choose(proposer.router.predict(router,c),c['allowed'])
    active,x=features.summary(c,own)
    risk=np.zeros((3,len(ids),2),float)
    training_scales={}
    forest_score=np.zeros((len(ids),2),float)
    for site in part['names']:
        sub=np.flatnonzero(c['sites'][active]==site)
        positions=active[sub]
        forest=neural.rel.base.read_checkpoint(checked(node['forest'][site]))
        forest_score[positions]=forest_predict(forest['state'],x[sub])
        for si,seed in enumerate((17,29,43)):
            state=neural.rel.base.read_checkpoint(checked(risk_checkpoint(job['key'],site,seed)))
            if site in state['preprocess']['train_sources'] or state['step']!=4096 or not state['complete']:
                raise ValueError('Frozen complete, source-excluded risk head required')
            risk[si,positions]=neural.method.predict(state,x[sub])
            training_scales[seed,site]=state['preprocess']['target_scale']
    aggregated=scores(risk)
    choices=dict(physical_floor=np.full(len(ids),-1,np.int64),proposal_self=own,
        forest64=method.veto(own,forest_score),seed17=method.veto(own,risk[0]),
        mean=method.veto(own,aggregated['mean']),unanimity=method.veto(own,aggregated['unanimity']))
    for name in ('mean','unanimity'):
        choices[name+'_matched']=method.matched(own,utility,choices[name],c['groups'],ids)
    # All choices above are frozen without looking at future validity or targets.
    context=next(v for v in read(neural.PUBLIC/'analysis.json')['contexts'] if v['job']==job['key'])
    old,_=neural.load(context)
    lookup=np.searchsorted(ids,old['ids'])
    np.testing.assert_array_equal(ids[lookup],old['ids'])
    apos=np.searchsorted(active,lookup)
    np.testing.assert_array_equal(active[apos],lookup)
    np.testing.assert_array_equal(x[apos],old['x'])
    np.testing.assert_array_equal(forest_score[lookup],old['old_prediction'])
    frozen=read(seeds.PUBLIC/'analysis.json')
    comparisons=[];old_risk=np.zeros((3,len(old['ids']),2),float)
    for si,seed in enumerate((17,29,43)):
        for site in part['names']:
            wanted=old['sites']==site
            cache=seeds.PRIVATE/'predictions'/f'{job["key"]}_{site}_augmented_seed{seed}.npy'
            before=np.load(cache,allow_pickle=False)
            row=next(r for r in frozen['folds'] if r['seed']==seed and r['source']==site
                     and r['job']==job['key'] and r['arm']=='augmented')
            if neural.method.digest(before)!=row['prediction_hash']:
                raise ValueError('Changed old prediction cache')
            after=risk[si,lookup[wanted]]
            comparisons.append(dict(seed=seed,source=site,**float_parity(after,before,training_scales[seed,site])))
            old_risk[si,wanted]=before
    old_ensemble=scores(old_risk)
    for name in ('mean','unanimity'):
        expected=method.veto(own[lookup],old_ensemble[name])
        np.testing.assert_array_equal(choices[name][lookup],expected)
    active_known=data['valid'][ix].any(1)&(own>=0)
    np.testing.assert_array_equal(ids[active_known],old['ids'])
    payload=dict(ids=ids,local_ids=ix,sites=c['sites'],recordings=data['recordings'][ix],
        frames=data['frames'][ix],paths=p['paths'],floor=p['floor'],cv=p['cv'],
        active=active,summary120=x,risk=risk,forest_score=forest_score,utility=utility,**choices)
    receipt=dict(job=job['key'],rows=len(ids),proposals=int((own>=0).sum()),
        known_proposals=len(old['ids']),unknown_proposals=int(((own>=0)&~data['valid'][ix].any(1)).sum()),
        input_features_exact=True,forest_known_prediction_exact=True,
        known_decisions_exact=True,known_ensemble_choices_exact=True,prediction_comparisons=comparisons,
        causal_choices_before_outcome_filter=True)
    beat(state='complete_causal_inference_verified',**{k:receipt[k] for k in ('job','rows','proposals','unknown_proposals')})
    return payload,receipt


def infer(reg,cfg,beat,replay=False,resume=False):
    data=load_data();refs=[];receipts=[]
    for node in reg['jobs']:
        key=node['job']['key'];path=PRIVATE/'inference_v2'/f'{key}.npz';out=PUBLIC/'inference_v2'/f'{key}.json'
        if resume and out.exists() and not replay:
            record=read(out);checked(record['cache']);refs.append(artifact(out));receipts.append(record)
            beat(state='completed_context_cache_verified',job=key);continue
        payload,receipt=infer_one(data,node,cfg,beat)
        if replay:
            checked(read(out)['cache'])
            with np.load(path,allow_pickle=False) as cached:
                for k,v in payload.items():np.testing.assert_array_equal(cached[k],v)
        else:
            save_arrays(path,payload)
            once(out,stable(dict(receipt,cache=artifact(path),
                array_hashes={k:neural.method.digest(v) for k,v in payload.items()},result_source='fresh_run')))
        refs.append(artifact(out));receipts.append(read(out))
    if sum(r['known_proposals'] for r in receipts)!=cfg['expected_known_proposals'] or sum(r['unknown_proposals'] for r in receipts)!=cfg['expected_unknown_proposals']:
        raise ValueError('Do not drop or add old known/unknown proposals')
    if not replay:once(PUBLIC/'inference_freeze_v2.json',dict(contexts=refs,all_choices_before_readout=True,new_training=False))


def evaluate(reg,cfg,beat,replay=False):
    freeze=read(PUBLIC/'inference_freeze_v2.json')
    if not freeze['all_choices_before_readout']:raise ValueError('Freeze all contexts first')
    data=load_data();rows=[]
    for node,ref in zip(reg['jobs'],freeze['contexts']):
        receipt=read(checked(ref));job=node['job']
        with np.load(checked(receipt['cache']),allow_pickle=False) as cache:
            d={k:cache[k].copy() for k in cache.files}
        for k,v in d.items():
            if neural.method.digest(v)!=receipt['array_hashes'][k]:raise ValueError('Changed frozen full choices')
        ix=d['local_ids'];target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        paths={name:method.paths_for(d['paths'],d['floor'],d[name]) for name in cfg['policies']}
        for site in node['part']['names']:
            at=d['sites']==site
            policies={name:method.original.policy(path[at],d['floor'][at],d['cv'][at],target[at],valid[at],
                d['recordings'][at],easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,path in paths.items()}
            contrasts={name:method.original.paired(paths['forest64'][at],paths[name][at],d['floor'][at],target[at],valid[at]) for name in ('mean','unanimity')}
            groups=features.shared.base.base.query_groups(d['recordings'][at],d['frames'][at],d['ids'][at])
            controls={name:matched_utility(paths[name+'_matched'][at],paths[name][at],d['floor'][at],
                target[at],valid[at],d[name+'_matched'][at],d[name][at],groups) for name in ('mean','unanimity')}
            rows.append(dict(job=job['key'],source=site,policies=policies,paired_vs_forest=contrasts,matched_utility=controls))
        beat(state='full_policy_metrics_complete',job=job['key'])
    policies={name:dict(safety=method.safety([r['policies'][name] for r in rows]),
        source_macro_known_ADE_gain=float(np.mean([r['policies'][name]['known_ADE_gain_floor_percent'] for r in rows])),
        source_macro_completion_lower_gain=float(np.mean([r['policies'][name]['completion_lower_gain_floor_percent'] for r in rows])),
        selected=sum(round(r['policies'][name]['switch_rate']*r['policies'][name]['rows']) for r in rows),
        unknown_selected=sum(v['unknown_selected'] for r in rows for v in r['policies'][name]['recordings'].values()))
        for name in cfg['policies']}
    rng=np.random.default_rng(cfg['bootstrap_seed']);pick=rng.integers(0,len(rows),(cfg['bootstrap_resamples'],len(rows)))
    contrasts={}
    for name in ('mean','unanimity'):
        contrasts[name]={}
        for key in ('known_pp','completion_lower_pp','completion_upper_pp'):
            values=[r['paired_vs_forest'][name][key] for r in rows]
            contrasts[name][key]=dict(point=float(np.mean(values)),interval95=np.quantile(np.asarray(values)[pick].mean(1),[.025,.975]).tolist(),
                unit='exposed_source_component_not_window',independent_confirmation=False) if all(v is not None for v in values) else dict(point=None,interval95=None,status='not_estimable')
    result=stable(dict(rows=rows,policies=policies,paired_vs_forest=contrasts,
        result_source='fresh_run',complete_own_B_rows=62466,known_proposals=48061,unknown_proposals=1274,
        original_rules_unchanged=True,identity_certification=False,new_outer_C_H_readout=False,
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay:neural.rel.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)
    beat(state='full_policy_replay_exact' if replay else 'full_policy_readout_complete')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['register','prepare','infer','evaluate','replay'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,reg=register()
        if args.phase=='register':print('Complete own-B policy diagnostic registered.');return
        committed();start,guard,beat=runtime(cfg,args.phase)
        before={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase=='prepare':prepare(reg,beat)
        elif args.phase=='infer':infer(reg,cfg,beat,resume=args.resume)
        elif args.phase=='evaluate':evaluate(reg,cfg,beat)
        else:
            infer(reg,cfg,beat,replay=True);evaluate(reg,cfg,beat,replay=True)
            neural.rel.base.exact(before,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        once((PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json'),
            dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        beat(state='phase_complete')


if __name__=='__main__':main()
