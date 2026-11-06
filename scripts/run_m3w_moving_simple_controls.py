"""Exact-cohort, fixed causal controls; no new model fitting or threshold search."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_expert_reference as previous
from src.evaluation import m3w_fixed_amplitude_controls as method

parent=previous.parent
np,torch=previous.np,previous.torch
read,sha,once,stable,checked,artifact=previous.read,previous.sha,previous.once,previous.stable,previous.checked,previous.artifact
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_simple_controls_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_simple_controls_v1'
CONFIG=ROOT/'configs/m3w_moving_simple_controls_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_fixed_amplitude_controls.py',
       ROOT/'tests/test_m3w_fixed_amplitude_controls.py')


def register():
    cfg=read(CONFIG);_,old=previous.register()
    if (cfg['fractions']!=list(method.FRACTIONS) or cfg['guards']!=list(method.GUARDS)
            or cfg['expected_evaluation_rows']!=62466 or cfg['risk_budget']!=.02
            or cfg['primary_baseline']!='last_velocity_nonzero_a1'
            or cfg['primary_learned_policy']!='expert_context_joint_free'
            or any(cfg[k] for k in ('coefficient_selection_from_readout','new_outer_C_H_readout','identity_certification',
                                  'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed whole matrix, original rules and no readout selection required')
    value=stable(dict(config=cfg,jobs=old['jobs'],files={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration=artifact(previous.REG),parent_inference=artifact(previous.PUBLIC/'inference_freeze.json'),
        parent_results=artifact(previous.PUBLIC/'analysis.json'),
        dependencies={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__)) for m in (parent.full.method.original,parent.full.method)},
        role='already_exposed_own_B_exact_cohort_control',new_training=False,independent_confirmation=False))
    once(REG,value);return cfg,value


def committed():
    for p in (*FILES,REG):
        raw=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(raw).hexdigest()!=sha(p):raise ValueError('Commit definition before actual causal preparation')


def context(node):
    _,old=parent.own_and_floor(node)
    np.testing.assert_array_equal(old['floor'],old['cv'])
    return old


def build_choices(data,old):
    ix=old['local_ids'];np.testing.assert_array_equal(data['global_ids'][ix],old['ids'])
    h=data['history'][ix];out=dict(ids=old['ids'],floor=np.full(len(ix),-1,np.int64))
    for name,action,guard in method.variants():out[name]=method.choices(h,old['paths'],old['floor'],action,guard)
    return out


def prepare(reg,beat,resume=False,replay=False):
    data=parent.full.load_data();refs=[]
    for node in reg['jobs']:
        key=node['job']['key'];receipt=PUBLIC/'choices'/f'{key}.json'
        if resume and receipt.exists() and not replay:checked(read(receipt)['cache']);refs.append(artifact(receipt));continue
        old=context(node);choices=build_choices(data,old)
        if replay:
            cached=parent.load_cache(read(receipt)['cache'])
            for name,v in choices.items():np.testing.assert_array_equal(v,cached[name])
        else:
            path=PRIVATE/'choices'/f'{key}.npz';parent.full.save_arrays(path,choices)
            once(receipt,stable(dict(job=key,cache=artifact(path),rows=len(old['ids']),
                policies=13,inputs='eight_past_positions_frozen_forecasts_original_floor',
                no_future_mask_or_target_input=True,result_source='fresh_run_from_cached_verified_paths')))
        refs.append(artifact(receipt));beat(state='causal_choices_replayed' if replay else 'causal_choices_frozen',job=key)
    if not replay:once(PUBLIC/'choices_freeze.json',dict(contexts=refs,all_static_choices_before_scoring=True))


def bank(node,arm,seed):
    key=node['job']['key']
    where=parent.PUBLIC if arm=='frozen' else previous.PUBLIC
    name='role_separated' if arm=='frozen' else arm
    return parent.load_cache(read(where/'inference'/f'{key}_{name}_seed{seed}.json')['cache'])


def evaluate(reg,cfg,beat,resume=False,replay=False):
    if not read(PUBLIC/'choices_freeze.json')['all_static_choices_before_scoring']:raise ValueError('Freeze every choice first')
    data=parent.full.load_data();prior=read(previous.PUBLIC/'analysis.json');chunks=[]
    for node in reg['jobs']:
        key=node['job']['key'];out=PUBLIC/'metrics'/f'{key}.json'
        if resume and out.exists() and not replay:chunks.append(read(out));continue
        old=context(node);job=node['job'];ix=old['local_ids'];target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        choices=parent.load_cache(read(PUBLIC/'choices'/f'{key}.json')['cache'])
        static_paths={n:parent.full.method.paths_for(old['paths'],old['floor'],v) for n,v in choices.items() if n!='ids'}
        static=[];learned=[];paired=[]
        def score(p,take):
            return parent.full.method.original.policy(p[take],old['floor'][take],old['cv'][take],target[take],valid[take],
                old['recordings'][take],easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])
        for site in node['part']['names']:
            take=old['sites']==site
            static.append(dict(job=key,source=site,policies={n:score(p,take) for n,p in static_paths.items()}))
        for seed in (17,29,43):
            for arm in ('frozen',*previous.method.ARMS):
                b=bank(node,arm,seed);paths={n:parent.full.method.paths_for(old['paths'],old['floor'],b[n]) for n in previous.POLICIES}
                for site in node['part']['names']:
                    take=old['sites']==site;row=dict(job=key,source=site,seed=seed,arm=arm,
                        policies={n:score(p,take) for n,p in paths.items()})
                    reference=next(r for r in prior['rows'] if all(r[k]==row[k] for k in ('job','source','seed','arm')))
                    previous.method.base.exact(row,reference);learned.append(row)
                    if arm=='expert_context':
                        for control,baseline in static_paths.items():
                            paired.append(dict(source=site,seed=seed,control=control,**parent.full.method.original.paired(
                                baseline[take],paths['joint_free'][take],old['floor'][take],target[take],valid[take])))
        chunk=stable(dict(job=key,static=static,learned=learned,paired=paired,cached_learned_controls_exact=True))
        if replay:previous.method.base.exact(read(out),chunk)
        else:once(out,chunk)
        chunks.append(chunk);beat(state='original_metrics_replayed' if replay else 'original_metrics_computed',job=key)
    static=[r for c in chunks for r in c['static']];learned=[r for c in chunks for r in c['learned']];paired=[r for c in chunks for r in c['paired']]
    sources=sorted(r['source'] for r in static)
    if len(static)!=8 or len(learned)!=72 or sum(r['policies']['floor']['rows'] for r in static)!=62466:
        raise ValueError('Complete registered cohort required')
    def ci(rows,k,seeds=None):return method.interval(rows,k,sources,seeds=seeds,resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
    def summary(rr,name,seeds=None):
        pp=[r['policies'][name] for r in rr]
        values=[dict(source=r['source'],**({'seed':r['seed']} if seeds else {}),
            known=r['policies'][name]['known_ADE_gain_floor_percent'],lower=r['policies'][name]['completion_lower_gain_floor_percent'],
            hard=r['policies'][name]['subsets']['hard']['gain_percent'],easy=r['policies'][name]['subsets']['easy']['gain_percent']) for r in rr]
        views=[v for p in pp for v in p['recordings'].values()]
        def maximum(k):
            vv=[v[k] for v in views if v[k] is not None]
            return dict(maximum_defined=max(vv) if vv else None,undefined_required_views=len(views)-len(vv))
        return dict(utility={k:ci(values,k,seeds) for k in ('known','lower','hard','easy')},safety=parent.full.method.safety(pp),
            switch_rate=float(np.mean([p['switch_rate'] for p in pp])),
            unknown_selected=sum(v['unknown_selected'] for v in views),
            tails={k:maximum(k) for k in ('known_all_risk','known_easy_risk','known_easy_net_degradation')})
    simple={n:summary(static,n) for n in static[0]['policies']}
    fitted={a:{n:summary([r for r in learned if r['arm']==a],n,[17,29,43]) for n in previous.POLICIES} for a in ('frozen',*previous.method.ARMS)}
    comparisons={n:{k:ci([r for r in paired if r['control']==n],k,[17,29,43]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
        for n in static[0]['policies']}
    result=stable(dict(static_policies=simple,learned_policies=fitted,
        primary=comparisons[cfg['primary_baseline']],all_fixed_comparisons=comparisons,
        sources=sources,unique_recordings=24,static_forecaster_seeds=[17],static_risk_seed_replicates=0,
        learned_risk_head_seeds=[17,29,43],frozen_learned_metrics_exact=True,
        fresh_run='static_causal_choices_all_metrics_and_paired_source_bootstrap',
        cached_verified='forecasts_parent_roles_and_learned_choices',not_run='new_training_independent_confirmation_or_deployment',
        original_rules_unchanged=True,original24group_main='not_estimable_not_passed',
        independent_confirmation=False,coefficient_selected_from_readout=False,deployment_changed=False,
        submission_ready=False,identity_certification=False,stage5c_executed=False,smc_enabled=False))
    if replay:previous.method.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('register','prepare','evaluate','replay'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,reg=register()
        if args.phase=='register':print('Exact-cohort fixed control matrix registered.');return
        committed();torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
        def beat(**value):
            elapsed=time.monotonic()-start
            if elapsed>cfg['hard_runtime_limit_seconds']:raise TimeoutError('Resume original source-context receipts after12h')
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep5GiB free')
            event=dict(pid=os.getpid(),phase=args.phase,seconds=elapsed,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**value)
            parent.source.atomic(PRIVATE/'heartbeat.json',event)
            with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        beat(state='ready',machine=platform.machine(),device='cpu',torch_threads=4,interop_threads=1,workers=0)
        before={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase=='prepare':prepare(reg,beat,args.resume)
        elif args.phase=='evaluate':evaluate(reg,cfg,beat,args.resume)
        else:
            prepare(reg,beat,replay=True);evaluate(reg,cfg,beat,replay=True)
            previous.method.base.exact(before,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        where=PRIVATE/'replays' if args.phase=='replay' else PUBLIC/'executions'
        once(where/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json',dict(pid=os.getpid(),phase=args.phase,
            seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            static_choices_recomputed=True,learned_forecasts_recomputed=False,training=False))
        beat(state='phase_complete')


if __name__=='__main__':main()
