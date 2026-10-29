"""Matched TRAIN-only test of uniform recording constraint weights."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

from scripts import run_m3w_cost_sensitive_policy as previous
from src.world_model import m3w_record_balanced_policy as api
from src.evaluation import m3w_record_balanced_readout as readout

ROOT,sha,once,base = previous.ROOT,previous.sha,previous.once,api.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_record_balanced_policy_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_record_balanced_policy_v1'
CONFIG = ROOT/'configs/m3w_record_balanced_policy_v1.json'
REG = PUBLIC/'registration.json'
committed = previous.diag.direct.previous.pooled.previous.previous.committed


def register():
    reg,old_cfg,ancestor=previous.admission();cfg=json.loads(CONFIG.read_text())
    if cfg['training']!=old_cfg['training'] or cfg['head_seeds']!=old_cfg['head_seeds']:
        raise ValueError('Only constraint weights may change')
    if cfg['risk_budget']!=.02 or cfg['decision_threshold']!=0.:
        raise ValueError('Fixed scientific policy budget and threshold required')
    done=json.loads((previous.PUBLIC/'complete.json').read_text())
    if done['summary_sha256']!=sha(previous.PUBLIC/'summary.json'):
        raise ValueError('Prior verified fitting experiment required')
    paths=[Path(__file__).resolve(),CONFIG,PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_record_balanced_policy.py',ROOT/'src/evaluation/m3w_record_balanced_readout.py',
        ROOT/'tests/test_m3w_record_balanced_policy.py',ROOT/'tests/test_m3w_record_balanced_readout.py',
        previous.REG,previous.PUBLIC/'training_freeze.json',previous.PUBLIC/'complete.json',
        previous.PUBLIC/'summary.json',previous.PUBLIC/'constraint_diagnosis.json']
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},contexts=sorted(reg['contexts']),
        seeds=reg['seeds'],fits=18,steps_per_fit=2000,comparator='cost_sensitive_cached_verified',
        constraint_weights='q_j=1/J; utility_weights unchanged; coefficient multiplier q_j/w_j',
        parent_pilot_sha256=sha(previous.PRIVATE/'pilot/direct.pt.gz'),
        risk_budget=.02,threshold=0.,inner_held_scored=False,independent_roles_read=False,
        no_seed_or_checkpoint_selection=True,scope='fitting_constraint_optimization_only',deployment_changed=False))
    return reg,cfg,ancestor


def admission():
    value=register()
    for rel in [*json.loads(REG.read_text())['bindings'],str(REG.relative_to(ROOT))]:committed(ROOT/rel)
    return value


def runtime(cfg):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume latest atomic checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['checkpoint_cap_bytes']:
            raise OSError('128MiB checkpoint cap')
    def beat(**kw):
        row=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(row)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('dual','exact_record_risk')}),flush=True)
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume same experiment')
    guard();beat(state='admitted');return started,beat,guard


def prepared(context,seed,packets,origins,exact,parents):
    data,_,oref,_=previous.prepared(context,seed,packets,origins,exact)
    pref=parents[context,seed]['checkpoint'];parent=base.core.read_checkpoint(ROOT/pref['path'])
    for key in ('input_hashes','source_preprocess','record_keys','record_scales'):
        base.core.exact(data[key],parent[key])
    base.core.exact(data['sizes'],parent['record_sizes'])
    return data,parent,oref,pref


def train(pilot=False,resume=False):
    reg,cfg,ancestor=admission();parents=previous.models()
    _,origins=previous.diag.direct.ORIGINS.models(reg);exact=previous.diag.whole.models()
    finish=PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real exact-resume and neutral-control pilot required')
    start,beat,guard=runtime(cfg);receipts=[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context,refs in (sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())):
            packets=previous.diag.fit_packets(refs,ancestor,beat)
            for seed in (reg['seeds'][:1] if pilot else reg['seeds']):
                data,parent,oref,pref=prepared(context,seed,packets,origins,exact,parents)
                path=PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}.pt.gz'
                kw=dict(data=data,settings=cfg['training'],seed=seed,path=path,
                    identity=dict(context=context,seed=seed,registration_sha256=sha(REG),
                        origin_sha256=oref['sha256'],parent_sha256=pref['sha256']),heartbeat=beat,guard=guard)
                if pilot and not path.exists():api.fit(**kw,stop_at=10)
                state=api.fit(**kw,stop_at=20 if pilot else None,resume=path.exists() and (resume or pilot))
                base.core.exact(state['initial_model'],parent['initial_model'])
                if not any(not api.torch.equal(v,state['initial_model'][k]) for k,v in state['model'].items()):
                    raise ValueError('No real parameter update')
                if pilot:
                    direct_path=PRIVATE/'pilot/direct.pt.gz'
                    direct=api.fit(**{**kw,'path':direct_path},stop_at=20,resume=resume and direct_path.exists())
                    for k in state:
                        if k!='seconds':base.core.exact(state[k],direct[k])
                    neutral_path=PRIVATE/'pilot/neutral.pt.gz'
                    neutral=api.fit(**{**kw,'path':neutral_path},balanced=False,stop_at=20,
                        resume=resume and neutral_path.exists())
                    old=base.core.read_checkpoint(previous.PRIVATE/'pilot/direct.pt.gz')
                    for k in ('model','optimizer','dual','torch_rng','trace','initial_model','input_hashes'):
                        base.core.exact(old[k],neutral[k])
                    once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,actual_updates=60,
                        exact_resume=True,neutral_replay_matches_frozen_parent=True,
                        estimated_full_seconds=state['seconds']*100*18,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        engineering_pass=True,registration_sha256=sha(REG)))
                else:
                    factor=state['constraint_factor']
                    receipt=PUBLIC/'fits'/f'{context}_seed{seed}.json'
                    once(receipt,dict(context=context,seed=seed,step=state['step'],seconds=state['seconds'],
                        parameters=state['parameters'],full_gradient_rows=state['full_gradient_rows'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                        constraint_factor_min=float(factor.min()),constraint_factor_max=float(factor.max()),
                        first_trace=state['trace'][0],final_trace=state['trace'][-1],
                        registration_sha256=sha(REG),result_source='fresh_run_neural_policy_TRAIN_only',new_forecaster=False))
                    receipts.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)))
                    beat(state='fit_complete',fits=len(receipts),total=18)
        if not pilot:
            once(finish,dict(fits=receipts,count=18,optimizer_steps=36000,pid=os.getpid(),
                seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                registration_sha256=sha(REG),inner_held_scored=False,independent_roles_read=False))
        beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    path=PUBLIC/'training_freeze.json';committed(path)
    freeze=json.loads(path.read_text());out={}
    if freeze['count']!=18 or freeze['registration_sha256']!=sha(REG):raise ValueError('18 frozen fits required')
    for ref in freeze['fits']:
        p=ROOT/ref['path'];r=json.loads(p.read_text());committed(p)
        if sha(p)!=ref['sha256'] or sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:
            raise ValueError('Frozen model changed')
        if r['step']!=2000:raise ValueError('Incomplete fit')
        out[r['context'],r['seed']]=r
    if len(out)!=18:raise ValueError('Distinct complete fits required')
    return out


def evaluate(check=False,resume=False):
    reg,cfg,ancestor=admission();learned=models();parents=previous.models()
    _,origins=previous.diag.direct.ORIGINS.models(reg)
    if (PUBLIC/'complete.json').exists() and not check:raise ValueError('Completed phase immutable')
    done=json.loads((previous.PUBLIC/'complete.json').read_text())
    oldrefs={Path(r['path']).name:r for r in done['groups']}
    start,beat,guard=runtime(cfg);rows,refs=[],[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context,items in sorted(reg['contexts'].items()):
            packets=previous.diag.fit_packets(items,ancestor,beat)
            for seed in reg['seeds']:
                origin=base.core.read_checkpoint(ROOT/origins[context,seed,'self']['checkpoint']['path'])
                predictions=previous.diag.direct.origin_predictions(origin,packets)
                states={name:base.core.read_checkpoint(ROOT/refs[context,seed]['checkpoint']['path'])
                        for name,refs in (('cost_sensitive',parents),('record_balanced',learned))}
                for source,args in sorted(packets.items()):
                    path=PUBLIC/'groups'/f'{context}_{source}_seed{seed}_fit.json'
                    if path.exists() and not (check or resume):raise ValueError('Explicit resume required')
                    eligible=base.shared.eligible(args)
                    masks=dict(origin=base.actions(predictions[source],eligible))
                    ranks=dict(origin=predictions[source][:,0]-predictions[source][:,1])
                    for name,state in states.items():
                        ranks[name],masks[name]=base.predict(state,source,args[0],args[1],predictions[source],eligible)
                    result=readout.evaluate(args,masks,ranks)
                    oldref=oldrefs[path.name];oldpath=ROOT/oldref['path']
                    if sha(oldpath)!=oldref['sha256']:raise ValueError('Frozen control result changed')
                    old=json.loads(oldpath.read_text())['result']
                    for arm in ('origin','cost_sensitive'):
                        if result['policies'][arm]!=old['policies'][arm]:raise ValueError('Whole-view control changed')
                        for rec,item in result['recordings'].items():
                            if item[arm]!=old['recordings'][rec][arm]:raise ValueError('Recording control changed')
                    row=dict(context=context,source=source,seed=seed,result=result,
                        action_sha256={k:base.fingerprint(v) for k,v in masks.items()},
                        rank_sha256={k:base.fingerprint(v) for k,v in ranks.items()},
                        registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        parent_fit_sha256=sha(oldpath),result_source='fresh_run_fitting_only_controls_cached_verified',inner_held_scored=False)
                    if check:
                        if row!=json.loads(path.read_text()):raise ValueError('Readout replay differs')
                    else:once(path,row)
                    rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='fit_readout_progress',views=len(rows),total=72);guard()
        summary=readout.summarize(rows)
        if check:
            end=json.loads((PUBLIC/'complete.json').read_text())
            if summary!=json.loads((PUBLIC/'summary.json').read_text()) or refs!=end['groups']:
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(PUBLIC/'complete.json',dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),
                registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                pid=os.getpid(),seconds=time.monotonic()-start,inner_held_scored=False,independent_roles_read=False))
        beat(state='exact_replay_complete' if check else 'fit_readout_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','pilot','train','evaluate','check'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='register':register()
    elif args.action in ('pilot','train'):train(args.action=='pilot',args.resume)
    else:evaluate(args.action=='check',args.resume)
