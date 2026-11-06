"""Four-cell matched support/context training with original complete readout."""
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

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as parent
from scripts import run_m3w_moving_simple_controls as controls
from src.world_model import m3w_supported_global_motion as method

np, torch = parent.np, parent.torch
read, sha, stable, once, checked, artifact = parent.read, parent.sha, parent.stable, parent.once, parent.checked, parent.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_supported_global_motion_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_supported_global_motion_v1'
CONFIG = ROOT/'configs/m3w_moving_supported_global_motion_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', Path(method.__file__),
    ROOT/'tests/test_m3w_supported_global_motion.py')
POLICIES = parent.POLICIES


def register():
    cfg = read(CONFIG); _, old = parent.register()
    if (cfg['arms'] != list(method.ARMS) or cfg['risk_head_seeds'] != [17,29,43]
            or cfg['fresh_models'] != 96 or cfg['full_exact_refits'] != 96 or cfg['steps_per_model'] != 4096
            or cfg['expected_evaluation_rows'] != 62466 or cfg['augmentation_mass'] != .5
            or cfg['risk_budget'] != .02 or cfg['policies'] != list(POLICIES)
            or cfg['primary_policy'] != 'joint_free'
            or any(cfg[k] for k in ('threshold_search','new_outer_C_H_readout','identity_certification',
                'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Original complete contract and fixed four cells required')
    value = stable(dict(config=cfg, settings=method.SETTINGS, jobs=old['jobs'],
        files={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration=artifact(parent.REG), parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_features=artifact(parent.PUBLIC/'own.json'), parent_A=artifact(parent.PUBLIC/'A_moments.json'),
        static_controls=artifact(controls.PUBLIC/'analysis.json'),
        prior_negative=artifact(ROOT/'outputs/publication_readiness_2026_09/moving_shared_motion_cost_v1/readout.md'),
        parent_code={str(p.relative_to(ROOT)):sha(p) for p in (Path(parent.__file__),Path(parent.method.__file__),
            Path(parent.allocation.__file__),Path(parent.full.method.original.__file__))},
        role='already_exposed_own_B_source_held_model_development', independent_confirmation=False,
        original_rules_unchanged=True, new_outer_C_H_readout=False))
    once(REG, value); return cfg, value


def committed():
    for p in (*FILES, REG):
        b = subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(b).hexdigest() != sha(p): raise ValueError('Commit fixed definition before execution')


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same checkpoint')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h phase bound; resume same definition')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Keep5GiB reserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']:
                raise OSError('Registered new artifact cap exceeded')
    def beat(**value):
        guard(); event = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**value)
        parent.source.atomic(PRIVATE/'heartbeat.json',event)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    beat(state='ready',machine=platform.machine(),device='cpu',torch_threads=4,interop_threads=1,workers=0)
    return start,guard,beat


def prepare(reg, guard, beat):
    refs=[]; data=parent.full.load_data()
    for node in reg['jobs']:
        guard(); own,old=parent.own_and_floor(node); ix=old['local_ids']
        np.testing.assert_array_equal(data['global_ids'][ix],old['ids'])
        context=method.lookup(own['ids'],data['global_ids'][ix],data['shared_context'][ix])
        path=PRIVATE/'context'/f'{node["job"]["key"]}.npz'
        parent.full.save_arrays(path,dict(ids=own['ids'],context=context))
        refs.append(dict(bank=node['job']['key'],cache=artifact(path),candidate_rows=len(context),
            agent_rows=len(old['ids']),feature_api_outcome_arguments=False))
        beat(state='own_context_exact',bank=node['job']['key'],candidates=len(context))
    del data
    data=parent.source.load_data()
    for fold in (0,1):
        guard(); added=parent.added_data(fold)
        context=method.lookup(added['ids'],data['global_ids'],data['shared_context'])
        path=PRIVATE/'context'/f'A{fold}.npz';parent.full.save_arrays(path,dict(ids=added['ids'],context=context))
        refs.append(dict(bank=f'A{fold}',cache=artifact(path),candidate_rows=len(context),
            agent_rows=len(np.unique(added['ids']//7)),feature_api_outcome_arguments=False))
        beat(state='donor_context_exact',bank=f'A{fold}',candidates=len(context))
    once(PUBLIC/'context.json',dict(banks=refs,features='already_admitted_global_past48',images_admitted=False))


def context(name, ids):
    ref=next(r['cache'] for r in read(PUBLIC/'context.json')['banks'] if r['bank']==name)
    table=parent.load_cache(ref); np.testing.assert_array_equal(table['ids'],ids)
    return table['context']


def entries(reg):
    for node in reg['jobs']:
        key=node['job']['key'];own,_=parent.own_and_floor(node);added=parent.added_data(node['job']['producer_fold'])
        own_context=context(key,own['ids']);extra=context(f'A{node["job"]["producer_fold"]}',added['ids'])
        for site in node['part']['names']:
            banks=dict(own=dict(own,context=own_context),
                augmented=dict(parent.mixed(own,added,site),context=np.concatenate((own_context,extra))))
            for seed in (17,29,43):
                for arm in method.ARMS:
                    d=banks[arm.split('_')[0]]
                    identity=dict(registration_sha256=sha(REG),job=key,source=site,seed=seed,arm=arm)
                    yield node,site,seed,arm,d,identity


def fit_path(key,site,seed,arm,kind='fits'):
    return PRIVATE/kind/f'{key}_{site}_seed{seed}_{arm}.pt.gz'


def pilot(reg,guard,beat,resume):
    node,site,seed,_,data,identity=next(entries(reg));identity=dict(identity,arm='own_global',pilot=True)
    p,q=PRIVATE/'pilot/resume.pt.gz',PRIVATE/'pilot/direct.pt.gz'
    if not p.exists():method.fit(data,site,identity,p,arm='own_global',seed=seed,stop_at=128,guard=guard,heartbeat=beat)
    a=method.fit(data,site,identity,p,arm='own_global',seed=seed,stop_at=256,resume=True,guard=guard,heartbeat=beat)
    b=method.fit(data,site,identity,q,arm='own_global',seed=seed,stop_at=256,resume=resume,guard=guard,heartbeat=beat)
    method.exact(a,b)
    once(PUBLIC/'pilot.json',stable(dict(actual128plus128_vs256_exact=True,pilot_updates=512,
        parameters=a['parameters'],fit_rows=a['fit_rows'],direct256_seconds=b['seconds'],
        pure_fit96_projection_seconds=b['seconds']*16*96,projection_not_actual_completion=True)))


def train(reg,guard,beat,resume,verify=False):
    if not read(PUBLIC/'pilot.json')['actual128plus128_vs256_exact']:raise ValueError('Actual recovery pilot first')
    refs=[];pairs={};count=0
    for node,site,seed,arm,data,identity in entries(reg):
        key=node['job']['key'];path=fit_path(key,site,seed,arm,'verify' if verify else 'fits')
        state=method.fit(data,site,identity,path,arm=arm,seed=seed,resume=resume,guard=guard,
            heartbeat=lambda **kw:beat(job=key,source=site,seed=seed,**kw))
        if not state['complete'] or state['step']!=4096 or site in state['preprocess']['train_sources']:
            raise ValueError('Complete source-excluded final fit required')
        if verify:method.exact(method.base.read_checkpoint(fit_path(key,site,seed,arm)),state)
        else:
            p=PUBLIC/'fits'/f'{key}_{site}_seed{seed}_{arm}.json'
            once(p,stable(dict(job=key,source=site,seed=seed,arm=arm,checkpoint=artifact(path),
                parameters=state['parameters'],steps=state['step'],fit_rows=state['fit_rows'],seconds=state['seconds'],
                trace=state['trace'],fitting_sources=state['preprocess']['train_sources'],result_source='fresh_run')))
            refs.append(artifact(p))
        pair=key,site,seed,arm.split('_')[0]
        if arm.endswith('_local'):pairs[pair]=state
        else:
            first=pairs.pop(pair)
            for field in ('preprocess','initial_model','generator','visits','parameters','step'):
                method.base.exact(first[field],state[field])
            if first['trace'][0]['total']!=state['trace'][0]['total']:raise ValueError('Identical starting function required')
        count+=1;beat(state='full_refit_exact' if verify else 'fit_complete',complete=count,total=96,job=key,source=site,seed=seed,arm=arm)
    if verify:once(PUBLIC/'verification.json',dict(all96_full_refits_exact=True,paired_draws_and_initialization=True,repeat_updates=393216))
    else:once(PUBLIC/'training_freeze.json',dict(fits=refs,models=96,updates=393216,all_before_readout=True,final_checkpoint_only=True))


def predict_one(node,arm,seed):
    own,old=parent.own_and_floor(node);c=context(node['job']['key'],own['ids']);n=len(old['ids'])
    moments=np.zeros((n,7,5));conditional=np.zeros_like(moments);prob=np.zeros((n,7))
    for site in node['part']['names']:
        at=np.flatnonzero(own['sites']==site)
        ref=read(PUBLIC/'fits'/f'{node["job"]["key"]}_{site}_seed{seed}_{arm}.json')['checkpoint']
        state=method.base.read_checkpoint(checked(ref))
        if site in state['preprocess']['train_sources']:raise ValueError('Query entered fitting')
        out=method.predict(state,own['x'][at],own['envelope'][at],c[at]);row,action=own['row'][at],own['action'][at]
        moments[row,action]=out['moments']*own['scales'][at,None]
        conditional[row,action]=out['conditional']*own['scales'][at,None]
        prob[row,action]=out['observation_probability']
    return dict(moments=moments,conditional=conditional,probability=prob),own,old


def infer(reg,cfg,beat,resume=False,replay=False):
    if not read(PUBLIC/'verification.json')['all96_full_refits_exact']:raise ValueError('All complete refits first')
    refs=[]
    for node in reg['jobs']:
        for seed in (17,29,43):
            for arm in method.ARMS:
                key=f'{node["job"]["key"]}_{arm}_seed{seed}';out=PUBLIC/'inference'/f'{key}.json'
                if resume and out.exists() and not replay:checked(read(out)['cache']);refs.append(artifact(out));continue
                prediction,own,old=predict_one(node,arm,seed);p=prediction['moments']
                allowed=np.zeros(p.shape[:2],bool);allowed[own['row'],own['action']]=True
                proposal=np.zeros_like(allowed);at=old['active'];proposal[at,old['proposal_self'][at]]=True
                proposal_only,_=parent.allocation.select_actions(p,proposal,arm='physical_neural')
                path=PRIVATE/'inference'/f'{key}.npz'
                if replay:
                    cached=parent.load_cache(read(out)['cache'])
                    for k,v in prediction.items():np.testing.assert_array_equal(v,cached[k])
                    np.testing.assert_array_equal(proposal_only,cached['proposal_only'])
                    independent,_=parent.allocation.select_actions(p,allowed,arm='physical_neural')
                    np.testing.assert_array_equal(independent,cached['independent'])
                    for group in parent.allocation.query_groups(old['recordings'],old['frames'],old['ids']):
                        for name in parent.allocation.ARMS:
                            if not parent.allocation.feasible(p[group],allowed[group],cached[name][group]):raise ValueError('Frozen choice violates predicted budgets')
                        if (cached['joint_matched'][group]>=0).sum()!=(independent[group]>=0).sum():raise ValueError('Count changed')
                else:
                    cp=PRIVATE/'allocation'/f'{key}.pt.gz'
                    restored=method.base.read_checkpoint(cp) if resume and cp.exists() else None
                    choices,_,info=parent.allocation.allocate(p,allowed,old['recordings'],old['frames'],old['ids'],
                        seconds=cfg['solver_seconds_per_query'],heartbeat=lambda **kw:beat(job=key,state='allocation',**kw),
                        checkpoint=lambda state:method.base.save_checkpoint(cp,state),resume_state=restored)
                    parent.full.save_arrays(path,dict(**prediction,proposal_only=proposal_only,**choices))
                    once(out,stable(dict(job=node['job']['key'],seed=seed,arm=arm,cache=artifact(path),allocation=info,
                        all_choices_before_labels=True,result_source='fresh_run')))
                refs.append(artifact(out));beat(state='prediction_replayed' if replay else 'choices_frozen',job=key)
    if not replay:once(PUBLIC/'inference_freeze.json',dict(contexts=refs,all36_choices_before_readout=True))


def evaluate(reg,cfg,beat,replay=False):
    if not read(PUBLIC/'inference_freeze.json')['all36_choices_before_readout']:raise ValueError('Freeze all choices first')
    data=parent.full.load_data();rows=[];contrasts={k:[] for k in ('primary','own_context','interaction','global_support','fixed_control')}
    for node in reg['jobs']:
        own,old=parent.own_and_floor(node);job=node['job'];ix=old['local_ids']
        target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        fixed=controls.method.choices(data['history'][ix],old['paths'],old['floor'],1,'last_velocity_nonzero')
        fixed_paths=parent.full.method.paths_for(old['paths'],old['floor'],fixed)
        for seed in (17,29,43):
            banks={arm:parent.load_cache(read(PUBLIC/'inference'/f'{job["key"]}_{arm}_seed{seed}.json')['cache']) for arm in method.ARMS}
            paths={arm:{name:parent.full.method.paths_for(old['paths'],old['floor'],banks[arm][name]) for name in POLICIES} for arm in method.ARMS}
            for site in node['part']['names']:
                at=old['sites']==site
                for arm in method.ARMS:
                    policies={name:parent.full.method.original.policy(p[at],old['floor'][at],old['cv'][at],target[at],valid[at],old['recordings'][at],
                        easy_cut=job['easy_cut'],hard_cut=job['hard_cut']) for name,p in paths[arm].items()}
                    rows.append(dict(job=job['key'],source=site,seed=seed,arm=arm,policies=policies))
                def pair(a,b):return parent.full.method.original.paired(a[at],b[at],old['floor'][at],target[at],valid[at])
                p=pair(paths['augmented_local']['joint_free'],paths['augmented_global']['joint_free'])
                o=pair(paths['own_local']['joint_free'],paths['own_global']['joint_free'])
                contrasts['primary'].append(dict(source=site,seed=seed,**p))
                contrasts['own_context'].append(dict(source=site,seed=seed,**o))
                contrasts['global_support'].append(dict(source=site,seed=seed,**pair(paths['own_global']['joint_free'],paths['augmented_global']['joint_free'])))
                contrasts['fixed_control'].append(dict(source=site,seed=seed,**pair(fixed_paths,paths['augmented_global']['joint_free'])))
                contrasts['interaction'].append(dict(source=site,seed=seed,known_pp=p['known_pp']-o['known_pp'],
                    completion_lower_pp=p['completion_lower_pp']-o['completion_upper_pp'],
                    completion_upper_pp=p['completion_upper_pp']-o['completion_lower_pp']))
        beat(state='original_complete_metrics',job=job['key'])
    summary={}
    for arm in method.ARMS:
        rr=[r for r in rows if r['arm']==arm];summary[arm]={}
        for name in POLICIES:
            pp=[r['policies'][name] for r in rr]
            utility={k:parent.interval([dict(source=r['source'],value=r['policies'][name][key]) for r in rr],'value',cfg)
                for k,key in (('known','known_ADE_gain_floor_percent'),('lower','completion_lower_gain_floor_percent'))}
            summary[arm][name]=dict(safety=parent.full.method.safety(pp),utility=utility,
                hard_gain_percent=float(np.mean([p['subsets']['hard']['gain_percent'] for p in pp])),
                easy_gain_percent=float(np.mean([p['subsets']['easy']['gain_percent'] for p in pp])),
                switch_rate=float(np.mean([p['switch_rate'] for p in pp])),
                safety_by_seed={str(s):parent.full.method.safety([r['policies'][name] for r in rr if r['seed']==s]) for s in (17,29,43)})
    intervals={name:{k:parent.interval(v,k,cfg) for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for name,v in contrasts.items()}
    result=stable(dict(rows=rows,policies=summary,contrasts=intervals,contrast_rows=contrasts,
        source_components=8,unique_recordings=24,required_record_views_per_policy=72,risk_head_seeds=[17,29,43],
        forecaster_seeds=[17],fresh_models=96,result_source='fresh_run',independent_confirmation=False,
        original24group_main='not_estimable_not_passed',original_rules_unchanged=True,new_outer_C_H_readout=False,
        identity_certification=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay:method.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('register','prepare','pilot','train','verify','infer','evaluate','replay'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,reg=register()
        if args.phase=='register':print('Four-cell support/global-motion study registered.');return
        committed();start,guard,beat=runtime(cfg,args.phase)
        before={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase=='prepare':prepare(reg,guard,beat)
        elif args.phase=='pilot':pilot(reg,guard,beat,args.resume)
        elif args.phase in ('train','verify'):train(reg,guard,beat,args.resume,verify=args.phase=='verify')
        elif args.phase=='infer':infer(reg,cfg,beat,args.resume)
        elif args.phase=='evaluate':evaluate(reg,cfg,beat)
        else:
            infer(reg,cfg,beat,replay=True);evaluate(reg,cfg,beat,replay=True)
            method.base.exact(before,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        where=PRIVATE/'replays' if args.phase=='replay' else PUBLIC/'executions'
        once(where/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json',dict(pid=os.getpid(),phase=args.phase,
            seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            predictions_recomputed=args.phase=='replay',timed_solver_search_recomputed=False if args.phase=='replay' else None))
        beat(state='phase_complete')


if __name__=='__main__':main()
