"""A single preregistered primal-loss repair, fitted and read out on TRAIN only."""
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

from scripts import run_m3w_train_feasibility as diag
from src.world_model import m3w_cost_sensitive_policy as api
from src.evaluation import m3w_cost_sensitive_readout as readout

ROOT,sha,once,base = diag.ROOT,diag.sha,diag.once,api.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cost_sensitive_policy_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cost_sensitive_policy_v1'
CONFIG = ROOT/'configs/m3w_cost_sensitive_policy_v1.json'
REG = PUBLIC/'registration.json'


def register():
    reg,ancestor = diag.register(); cfg = json.loads(CONFIG.read_text())
    prior_cfg = json.loads(diag.whole.CONFIG.read_text())
    if cfg['training']!=prior_cfg['training'] or cfg['head_seeds']!=reg['seeds'] or cfg['risk_budget']!=.02:
        raise ValueError('Only the primal loss may change')
    done = json.loads((diag.PUBLIC/'complete.json').read_text())
    if done['summary_sha256']!=sha(diag.PUBLIC/'summary.json'): raise ValueError('Verified diagnosis required')
    paths = [Path(__file__).resolve(),CONFIG,PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_cost_sensitive_policy.py',ROOT/'src/evaluation/m3w_cost_sensitive_readout.py',
        ROOT/'tests/test_m3w_cost_sensitive_policy.py',ROOT/'tests/test_m3w_cost_sensitive_readout.py',
        diag.REG,diag.PUBLIC/'complete.json',diag.PUBLIC/'summary.json',diag.whole.PUBLIC/'training_freeze.json']
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},contexts=sorted(reg['contexts']),
        seeds=reg['seeds'],fits=18,steps_per_fit=2000,comparator='exact_gradient_cached_verified',
        risk_budget=.02,threshold=0.,inner_held_scored=False,independent_roles_read=False,
        no_seed_or_checkpoint_selection=True,scope='fitting_objective_repair_only',deployment_changed=False))
    return reg,cfg,ancestor


def admission():
    out = register()
    for rel in [*json.loads(REG.read_text())['bindings'],str(REG.relative_to(ROOT))]:
        diag.direct.previous.pooled.previous.previous.committed(ROOT/rel)
    return out


def runtime(cfg):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);began = time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['checkpoint_cap_bytes']:
            raise OSError('128MiB model cap exceeded')
    def beat(**kw):
        row = dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp = PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(row)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('dual','exact_record_risk')}),flush=True)
        if time.monotonic()-began>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume same roster')
    guard();beat(state='admitted');return began,beat,guard


def prepared(context,seed,packets,origins,parents):
    ref = origins[context,seed,'self']['checkpoint'];oref = parents[context,seed,'exact_gradient']['checkpoint']
    origin = base.core.read_checkpoint(ROOT/ref['path'])
    predictions = diag.direct.origin_predictions(origin,packets)
    data = base.prepare(packets,predictions);parent = base.core.read_checkpoint(ROOT/oref['path'])
    for k in ('input_hashes','source_preprocess','record_keys','record_scales'):
        base.core.exact(data[k],parent[k])
    base.core.exact(data['sizes'],parent['record_sizes'])
    return data,parent,ref,oref


def train(pilot=False,resume=False):
    reg,cfg,ancestor = admission();_,origins = diag.direct.ORIGINS.models(reg);parents = diag.whole.models()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real training and resume pilot required')
    start,beat,guard = runtime(cfg);receipts = []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        items = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
        for context,refs in items:
            packets = diag.fit_packets(refs,ancestor,beat)
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                data,parent,oref,pref = prepared(context,seed,packets,origins,parents)
                path = PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}.pt.gz'
                kw = dict(data=data,settings=cfg['training'],seed=seed,identity=dict(context=context,seed=seed,
                    registration_sha256=sha(REG),origin_sha256=oref['sha256'],parent_sha256=pref['sha256']),
                    path=path,heartbeat=beat,guard=guard)
                if pilot and not path.exists():api.fit(**kw,stop_at=10)
                state = api.fit(**kw,stop_at=20 if pilot else None,resume=path.exists() and (resume or pilot))
                base.core.exact(state['initial_model'],parent['initial_model'])
                if not any(not api.torch.equal(v,state['initial_model'][k]) for k,v in state['model'].items()):
                    raise ValueError('No real parameter update')
                if pilot:
                    direct_path = PRIVATE/'pilot/direct.pt.gz'
                    direct = api.fit(**{**kw,'path':direct_path},stop_at=20,resume=resume and direct_path.exists())
                    for k in state:
                        if k!='seconds':base.core.exact(state[k],direct[k])
                    once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,actual_updates=40,
                        exact_resume=True,estimated_full_seconds=state['seconds']*100*18,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        engineering_pass=True,registration_sha256=sha(REG)))
                else:
                    diagnostics = {}
                    for name,s in (('exact_gradient',parent),('cost_sensitive',state)):
                        model = base.Policy(s['dimension'],s['settings']['width'],s['seed']);model.load_state_dict(s['model'])
                        diagnostics[name] = api.gradient_diagnostic(model,data,api.whole.layout(data),s['dual'])
                    receipt = PUBLIC/'fits'/f'{context}_seed{seed}.json'
                    once(receipt,dict(context=context,seed=seed,step=state['step'],seconds=state['seconds'],
                        parameters=state['parameters'],full_gradient_rows=state['full_gradient_rows'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                        first_trace=state['trace'][0],final_trace=state['trace'][-1],gradient_diagnostic=diagnostics,
                        registration_sha256=sha(REG),result_source='fresh_run_neural_policy_TRAIN_only',new_forecaster=False))
                    receipts.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)))
                    beat(state='fit_complete',fits=len(receipts),total=18)
        if not pilot:
            once(finish,dict(fits=receipts,count=18,optimizer_steps=36000,pid=os.getpid(),
                seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                registration_sha256=sha(REG),inner_held_scored=False,independent_roles_read=False))
        beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text());out = {}
    if freeze['count']!=18 or freeze['registration_sha256']!=sha(REG):raise ValueError('18 frozen finals required')
    for ref in freeze['fits']:
        path = ROOT/ref['path'];row = json.loads(path.read_text())
        if sha(path)!=ref['sha256'] or sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256']:
            raise ValueError('Frozen fit changed')
        if row['step']!=2000:raise ValueError('Incomplete fit')
        out[row['context'],row['seed']] = row
    if len(out)!=18:raise ValueError('Complete distinct fits required')
    return out


def evaluate(check=False,resume=False):
    reg,cfg,ancestor = admission();learned = models();parents = diag.whole.models()
    _,origins = diag.direct.ORIGINS.models(reg)
    if (PUBLIC/'complete.json').exists() and not check:raise ValueError('Completed phase immutable')
    previous = json.loads((diag.PUBLIC/'complete.json').read_text())
    oldrefs = {Path(r['path']).name:r for r in previous['groups']}
    start,beat,guard = runtime(cfg);rows,refs = [],[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context,items in sorted(reg['contexts'].items()):
            packets = diag.fit_packets(items,ancestor,beat)
            for seed in reg['seeds']:
                origin = base.core.read_checkpoint(ROOT/origins[context,seed,'self']['checkpoint']['path'])
                predictions = diag.direct.origin_predictions(origin,packets)
                states = dict(cost_sensitive=base.core.read_checkpoint(ROOT/learned[context,seed]['checkpoint']['path']),
                    exact_gradient=base.core.read_checkpoint(ROOT/parents[context,seed,'exact_gradient']['checkpoint']['path']))
                for source,args in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_fit.json'
                    if path.exists() and not (check or resume):raise ValueError('Explicit resume required')
                    eligible = base.shared.eligible(args)
                    masks = dict(origin=base.actions(predictions[source],eligible))
                    ranks = dict(origin=predictions[source][:,0]-predictions[source][:,1])
                    for name,s in states.items():
                        ranks[name],masks[name] = base.predict(s,source,args[0],args[1],predictions[source],eligible)
                    result = readout.evaluate(args,masks,ranks)
                    ref = oldrefs[path.name];oldpath = ROOT/ref['path']
                    if sha(oldpath)!=ref['sha256']:raise ValueError('Frozen diagnostic changed')
                    old = json.loads(oldpath.read_text())['result']
                    for arm in ('origin','exact_gradient'):
                        if result['policies'][arm]!=old['policies'][arm]['observed']:raise ValueError('Frozen control changed')
                        for rec,d in result['recordings'].items():
                            if d[arm]!=old['recordings'][rec]['policies'][arm]['observed']:raise ValueError('Frozen record control changed')
                    row = dict(context=context,source=source,seed=seed,result=result,
                        action_sha256={k:base.fingerprint(v) for k,v in masks.items()},
                        rank_sha256={k:base.fingerprint(v) for k,v in ranks.items()},registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),parent_fit_sha256=sha(oldpath),
                        result_source='fresh_run_fitting_only_controls_cached_verified',inner_held_scored=False)
                    if check:
                        if row!=json.loads(path.read_text()):raise ValueError('Readout replay differs')
                    else:once(path,row)
                    rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='fit_readout_progress',views=len(rows),total=72);guard()
        summary = readout.summarize(rows)
        if check:
            done = json.loads((PUBLIC/'complete.json').read_text())
            if summary!=json.loads((PUBLIC/'summary.json').read_text()) or refs!=done['groups']:raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(PUBLIC/'complete.json',dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),
                registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                pid=os.getpid(),seconds=time.monotonic()-start,inner_held_scored=False,independent_roles_read=False))
        beat(state='exact_replay_complete' if check else 'fit_readout_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','pilot','train','evaluate','check'))
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register':register()
    elif a.action in ('pilot','train'):train(a.action=='pilot',a.resume)
    else:evaluate(a.action=='check',a.resume)
