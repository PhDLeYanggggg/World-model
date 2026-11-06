"""A matched causal-context intervention on frozen cost reference branches."""
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
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as parent
from src.world_model import m3w_expert_reference as method

np,torch=parent.np,parent.torch
read,sha,stable,once,checked,artifact=parent.read,parent.sha,parent.stable,parent.once,parent.checked,parent.artifact
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_expert_reference_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_expert_reference_v1'
CONFIG=ROOT/'configs/m3w_moving_expert_reference_v1.json'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_expert_reference.py',
       ROOT/'tests/test_m3w_expert_reference.py')
POLICIES=parent.POLICIES


def register():
    cfg=read(CONFIG);_,old=parent.register()
    if (cfg['arms']!=list(method.ARMS) or cfg['risk_head_seeds']!=[17,29,43]
            or cfg['fresh_models']!=48 or cfg['steps_per_model']!=4096 or cfg['risk_budget']!=.02
            or cfg['policies']!=list(POLICIES) or cfg['primary_policy']!='joint_free'
            or cfg['expected_evaluation_rows']!=62466
            or any(cfg[k] for k in ('threshold_search','new_outer_C_H_readout','identity_certification',
                                  'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Original contract, all sources and fixed matched budget required')
    result=stable(dict(config=cfg,settings=method.SETTINGS,jobs=old['jobs'],
        files={str(p.relative_to(ROOT)):sha(p) for p in FILES},parent_registration=artifact(parent.REG),
        parent_training=artifact(parent.PUBLIC/'training_freeze.json'),parent_inference=artifact(parent.PUBLIC/'inference_freeze.json'),
        parent_result=artifact(parent.PUBLIC/'analysis.json'),
        parent_code={str(p.relative_to(ROOT)):sha(p) for p in (Path(parent.allocation.__file__),
            Path(parent.full.method.original.__file__),Path(parent.method.__file__))},
        role='already_exposed_own_B_source_held_model_development',original_rules_unchanged=True,
        new_outer_C_H_readout=False,independent_confirmation=False))
    once(REG,result);return cfg,result


def committed():
    for p in (*FILES,REG):
        b=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(b).hexdigest()!=sha(p):raise ValueError('Commit fixed study before actual execution')


def runtime(cfg,phase):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume same saved experiment')))
    start,last=time.monotonic(),[0.]
    def guard():
        now=time.monotonic()
        if now-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h phase bound; resume same checkpoint')
        if now-last[0]>1:
            last[0]=now
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep5GiB free')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>cfg['artifact_cap_bytes']:
                raise OSError('New private artifact cap exceeded')
    def beat(**value):
        guard();event=dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**value)
        parent.source.atomic(PRIVATE/'heartbeat.json',event)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    beat(state='ready',machine=platform.machine(),device='cpu',torch_threads=4,interop_threads=1,workers=0)
    return start,guard,beat


def table_path(name):return PRIVATE/'context'/f'{name}.npz'


def prepare(reg,guard,beat):
    refs=[]
    banks=[(n['job']['key'],lambda n=n:parent.own_and_floor(n)[0]) for n in reg['jobs']]
    banks.extend((f'A{fold}',lambda fold=fold:parent.added_data(fold)) for fold in (0,1))
    for name,load in banks:
        guard();d=load();table=method.context_table(d['x'],d['ids'])
        path=table_path(name);parent.full.save_arrays(path,table)
        reconstructed=method.lookup(table,d['ids'])
        order=np.argsort(d['ids']);at=order[np.searchsorted(d['ids'][order],(d['ids']//7)*7+6)]
        np.testing.assert_array_equal(reconstructed,d['x'][at,50:])
        refs.append(dict(bank=name,cache=artifact(path),candidate_rows=len(d['ids']),agents=len(table['ids']),
            shared_context_dim=70,exact_full_expert_anchor_join=True,first50_action_invariant=True,
            inputs='raw_causal120_and_original_candidate_ids_only',source='fresh_run_from_cached_verified_features'))
        beat(state='shared_expert_context_built',bank=name,agents=len(table['ids']))
    once(PUBLIC/'context.json',dict(banks=refs,features_do_not_accept_outcome_labels=True))


def load_context(name):
    ref=next(r['cache'] for r in read(PUBLIC/'context.json')['banks'] if r['bank']==name)
    return parent.load_cache(ref)


def entries(reg):
    for node in reg['jobs']:
        key=node['job']['key'];own,_=parent.own_and_floor(node);added=parent.added_data(node['job']['producer_fold'])
        context=np.concatenate((method.lookup(load_context(key),own['ids']),
                               method.lookup(load_context(f'A{node["job"]["producer_fold"]}'),added['ids'])))
        for site in node['part']['names']:
            data=dict(parent.mixed(own,added,site),context=context)
            for seed in (17,29,43):
                ref=read(parent.PUBLIC/'fits'/f'{key}_{site}_seed{seed}_role_separated.json')['checkpoint']
                origin=method.base.read_checkpoint(checked(ref))
                for arm in method.ARMS:
                    identity=dict(registration_sha256=sha(REG),job=key,source=site,seed=seed,arm=arm,parent_checkpoint=ref)
                    yield node,site,seed,arm,data,origin,identity


def fit_path(key,site,seed,arm,kind='fits'):
    return PRIVATE/kind/f'{key}_{site}_seed{seed}_{arm}.pt.gz'


def pilot(reg,guard,beat,resume):
    node,site,seed,_,data,origin,identity=next(entries(reg));identity=dict(identity,arm='expert_context',pilot=True)
    a=PRIVATE/'pilot/resume.pt.gz';b=PRIVATE/'pilot/direct.pt.gz'
    if not a.exists():method.fit(data,site,origin,identity,a,arm='expert_context',seed=seed,stop_at=128,guard=guard,heartbeat=beat)
    resumed=method.fit(data,site,origin,identity,a,arm='expert_context',seed=seed,stop_at=256,resume=True,guard=guard,heartbeat=beat)
    direct=method.fit(data,site,origin,identity,b,arm='expert_context',seed=seed,stop_at=256,resume=resume,guard=guard,heartbeat=beat)
    method.exact(resumed,direct)
    once(PUBLIC/'pilot.json',stable(dict(real128plus128_vs256_exact=True,parameters=resumed['parameters'],
        pilot_updates=512,direct256_seconds=direct['seconds'],pure_fit48_projection_seconds=direct['seconds']*16*48,
        projection_not_actual_completion=True)))


def train(reg,guard,beat,*,resume,verify=False):
    if not read(PUBLIC/'pilot.json')['real128plus128_vs256_exact']:raise ValueError('Actual pilot first')
    count=0;refs=[];pairs={}
    for node,site,seed,arm,data,origin,identity in entries(reg):
        key=node['job']['key'];path=fit_path(key,site,seed,arm,'verify' if verify else 'fits')
        state=method.fit(data,site,origin,identity,path,arm=arm,seed=seed,resume=resume,guard=guard,
            heartbeat=lambda **kw:beat(job=key,source=site,seed=seed,**kw))
        if not state['complete'] or site in state['preprocess']['parent_preprocess']['train_sources']:
            raise ValueError('Complete source-excluded reference training required')
        if verify:method.exact(method.base.read_checkpoint(fit_path(key,site,seed,arm)),state)
        else:
            dest=PUBLIC/'fits'/f'{key}_{site}_seed{seed}_{arm}.json'
            once(dest,stable(dict(job=key,source=site,seed=seed,arm=arm,checkpoint=artifact(path),
                parameters=state['parameters'],steps=state['step'],seconds=state['seconds'],trace=state['trace'],result_source='fresh_run')))
            refs.append(artifact(dest))
        pair=key,site,seed
        if arm=='history_refit':pairs[pair]=state
        else:
            old=pairs.pop(pair)
            for field in ('preprocess','initial_model','generator','visits','parameters','step'):method.base.exact(old[field],state[field])
        count+=1;beat(state='refit_exact' if verify else 'fit_complete',complete=count,total=48)
    if verify:once(PUBLIC/'verification.json',dict(all48_full_refits_exact=True,repeat_updates=196608,matched_draws_and_initialization=True))
    else:once(PUBLIC/'training_freeze.json',dict(fits=refs,models=48,primary_updates=196608,final_checkpoint_only=True))


def predict_one(node,arm,seed):
    key=node['job']['key'];own,old=parent.own_and_floor(node)
    frozen=parent.load_cache(read(parent.PUBLIC/'inference'/f'{key}_role_separated_seed{seed}.json')['cache'])
    table=load_context(key);refs=np.zeros((len(old['ids']),2))
    for site in node['part']['names']:
        at=np.flatnonzero(own['sites']==site);rows,first=np.unique(own['row'][at],return_index=True);ix=at[first]
        state=method.base.read_checkpoint(checked(read(PUBLIC/'fits'/f'{key}_{site}_seed{seed}_{arm}.json')['checkpoint']))
        origin=method.base.read_checkpoint(checked(state['identity']['parent_checkpoint']))
        rr=method.predict(state,origin,own['x'][ix],method.lookup(table,own['ids'][ix]))
        refs[rows]=rr*own['scales'][ix,None]
    allowed=np.zeros((len(old['ids']),7),bool);allowed[own['row'],own['action']]=True
    references=np.broadcast_to(refs[:,None,:],(*allowed.shape,2)).copy()
    prediction=method.replace_reference(frozen,references,allowed)
    return prediction,allowed,old


def infer(reg,cfg,beat,*,resume=False,replay=False):
    if not read(PUBLIC/'verification.json')['all48_full_refits_exact']:raise ValueError('All refits before choices')
    refs=[];(PRIVATE/'allocation').mkdir(parents=True,exist_ok=True)
    for node in reg['jobs']:
        for seed in (17,29,43):
            for arm in method.ARMS:
                key=f'{node["job"]["key"]}_{arm}_seed{seed}';out=PUBLIC/'inference'/f'{key}.json'
                if resume and out.exists() and not replay:checked(read(out)['cache']);refs.append(artifact(out));continue
                prediction,allowed,old=predict_one(node,arm,seed);p=prediction['moments']
                proposal=np.zeros_like(allowed);at=old['active'];proposal[at,old['proposal_self'][at]]=True
                proposal_only,_=parent.allocation.select_actions(p,proposal,arm='physical_neural')
                if replay:
                    cached=parent.load_cache(read(out)['cache'])
                    for k,v in prediction.items():np.testing.assert_array_equal(v,cached[k])
                    np.testing.assert_array_equal(proposal_only,cached['proposal_only'])
                    independent,_=parent.allocation.select_actions(p,allowed,arm='physical_neural')
                    np.testing.assert_array_equal(independent,cached['independent'])
                    for ix in parent.allocation.query_groups(old['recordings'],old['frames'],old['ids']):
                        for name in parent.allocation.ARMS:
                            if not parent.allocation.feasible(p[ix],allowed[ix],cached[name][ix]):raise ValueError('Infeasible cached choice')
                        if (cached['joint_matched'][ix]>=0).sum()!=(independent[ix]>=0).sum():raise ValueError('Matched count changed')
                else:
                    checkpoint=PRIVATE/'allocation'/f'{key}.pt.gz'
                    saved=method.base.read_checkpoint(checkpoint) if resume and checkpoint.exists() else None
                    choices,_,info=parent.allocation.allocate(p,allowed,old['recordings'],old['frames'],old['ids'],
                        seconds=cfg['solver_seconds_per_query'],heartbeat=lambda **kw:beat(job=key,state='allocation',**kw),
                        checkpoint=lambda value:method.base.save_checkpoint(checkpoint,value),resume_state=saved)
                    path=PRIVATE/'inference'/f'{key}.npz';parent.full.save_arrays(path,dict(**prediction,proposal_only=proposal_only,**choices))
                    once(out,stable(dict(job=node['job']['key'],arm=arm,seed=seed,cache=artifact(path),allocation=info,
                        frozen_harm_support_exact=True,references_broadcast_per_agent=True,result_source='fresh_run')))
                refs.append(artifact(out));beat(state='choices_replayed' if replay else 'choices_frozen',context=key)
    if not replay:once(PUBLIC/'inference_freeze.json',dict(contexts=refs,all18_new_choices_before_readout=True,
        frozen_control_inference=artifact(parent.PUBLIC/'inference_freeze.json')))


def evaluate(reg,cfg,beat,replay=False):
    if not read(PUBLIC/'inference_freeze.json')['all18_new_choices_before_readout']:raise ValueError('Freeze first')
    data=parent.full.load_data();rows=[];primary=[];contrasts={a:[] for a in method.ARMS}
    old_analysis=read(parent.PUBLIC/'analysis.json')
    for node in reg['jobs']:
        own,old=parent.own_and_floor(node);key=node['job']['key'];ix=old['local_ids'];job=node['job']
        target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        for seed in (17,29,43):
            bank={a:parent.load_cache(read(PUBLIC/'inference'/f'{key}_{a}_seed{seed}.json')['cache']) for a in method.ARMS}
            bank['frozen']=parent.load_cache(read(parent.PUBLIC/'inference'/f'{key}_role_separated_seed{seed}.json')['cache'])
            paths={a:{name:parent.full.method.paths_for(old['paths'],old['floor'],b[name]) for name in POLICIES} for a,b in bank.items()}
            for site in node['part']['names']:
                take=old['sites']==site
                for arm in (*method.ARMS,'frozen'):
                    policies={name:parent.full.method.original.policy(p[take],old['floor'][take],old['cv'][take],target[take],valid[take],
                        old['recordings'][take],easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in paths[arm].items()}
                    if arm=='frozen':
                        prior=next(r for r in old_analysis['rows'] if r['job']==key and r['source']==site and r['seed']==seed and r['arm']=='role_separated')
                        method.base.exact(policies,prior['policies'])
                    rows.append(dict(job=key,source=site,seed=seed,arm=arm,policies=policies))
                primary.append(dict(source=site,seed=seed,**parent.full.method.original.paired(paths['history_refit']['joint_free'][take],
                    paths['expert_context']['joint_free'][take],old['floor'][take],target[take],valid[take])))
                for arm in method.ARMS:
                    contrasts[arm].append(dict(source=site,seed=seed,**parent.full.method.original.paired(paths['frozen']['joint_free'][take],
                        paths[arm]['joint_free'][take],old['floor'][take],target[take],valid[take])))
        beat(state='original_policy_metrics_complete',job=key)
    policies={}
    for arm in (*method.ARMS,'frozen'):
        subset=[r for r in rows if r['arm']==arm];policies[arm]={}
        for name in POLICIES:
            pp=[r['policies'][name] for r in subset]
            policies[arm][name]=dict(safety=parent.full.method.safety(pp),
                source_seed_macro_known_gain=float(np.mean([p['known_ADE_gain_floor_percent'] for p in pp])),
                source_seed_macro_completion_lower_gain=float(np.mean([p['completion_lower_gain_floor_percent'] for p in pp])),
                safety_by_seed={str(s):parent.full.method.safety([r['policies'][name] for r in subset if r['seed']==s]) for s in (17,29,43)})
    keys=('known_pp','completion_lower_pp','completion_upper_pp')
    result=stable(dict(rows=rows,policies=policies,primary={k:parent.interval(primary,k,cfg) for k in keys},
        primary_source_seed_rows=primary,versus_frozen={a:{k:parent.interval(v,k,cfg) for k in keys} for a,v in contrasts.items()},
        fresh_models=48,source_blocks=8,unique_recordings=24,required_repeated_views_per_policy=72,
        risk_seeds=[17,29,43],forecaster_seeds=[17],cached_frozen_control_exact=True,
        original_rules_unchanged=True,independent_confirmation=False,new_outer_C_H_readout=False,
        original24group_main='not_estimable_not_passed',deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False,result_source='fresh_run_with_cached_verified_control'))
    if replay:method.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('register','prepare','pilot','train','verify','infer','evaluate','replay'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,reg=register()
        if args.phase=='register':print('Matched expert-reference experiment registered.');return
        committed();start,guard,beat=runtime(cfg,args.phase)
        snapshot={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase=='prepare':prepare(reg,guard,beat)
        elif args.phase=='pilot':pilot(reg,guard,beat,args.resume)
        elif args.phase in ('train','verify'):train(reg,guard,beat,resume=args.resume,verify=args.phase=='verify')
        elif args.phase=='infer':infer(reg,cfg,beat,resume=args.resume)
        elif args.phase=='evaluate':evaluate(reg,cfg,beat)
        else:
            infer(reg,cfg,beat,replay=True);evaluate(reg,cfg,beat,replay=True)
            method.base.exact(snapshot,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        where=PRIVATE/'replays' if args.phase=='replay' else PUBLIC/'executions'
        once(where/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json',dict(phase=args.phase,pid=os.getpid(),
            seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            predictions_recomputed=args.phase=='replay',timed_solver_recomputed=False if args.phase=='replay' else None))
        beat(state='phase_complete')


if __name__=='__main__':main()
