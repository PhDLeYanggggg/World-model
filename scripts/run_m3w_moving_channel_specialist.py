"""Fixed source-held channel-specific regression heads; no new held readout."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_boundary_audit as audit
from src.evaluation import m3w_channel_specialist as method
from src.evaluation.m3w_boundary_audit import summarize as boundary

source, rel, np = audit.source, audit.rel, audit.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_channel_specialist_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_channel_specialist_v1'
CONFIG = ROOT/'configs/m3w_moving_channel_specialist_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/evaluation/m3w_channel_specialist.py',
    ROOT/'tests/test_m3w_channel_specialist.py', ROOT/'src/evaluation/m3w_boundary_audit.py',
    ROOT/'scripts/run_m3w_moving_boundary_audit.py', ROOT/'tests/test_m3w_boundary_audit.py')


def register():
    cfg = source.read(CONFIG)
    if (cfg['arms'] != list(method.ARMS) or cfg['primary'] != 'separate64_minus_shared64'
            or cfg['features'] != 'original120' or cfg['minimum_joint_cost_reduction'] != .01
            or cfg['minimum_passing_contexts'] != 2 or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed model repair and original rules required')
    report = source.read(audit.PUBLIC/'analysis.json')
    source.once(REG, source.stable(dict(config=cfg, tree=method.parent.TREE,
        files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        data_audit=source.artifact(audit.PUBLIC/'analysis.json'), parent_registration=source.artifact(audit.old.REG),
        parent_result=source.artifact(audit.old.PUBLIC/'analysis.json'),
        data=[dict(job=c['job'], cache=c['cache'], array_hashes=c['array_hashes']) for c in report['contexts']],
        result_role='exposed_own_B_leave_source_out', new_V_or_H=False, original_rules_unchanged=True,
        identity_certification=False, independent_confirmation=False)))
    return cfg, report


def inputs(context):
    with np.load(source.checked(context['cache']), allow_pickle=False) as saved:
        data = {k:saved[k].copy() for k in saved.files}
    for k,v in data.items(): rel.base.exact(rel.parent.kin.digest(v), context['array_hashes'][k])
    rel.base.exact(len(data['x']), context['audit']['known_proposed_rows'])
    return data


def cell(data, context, site, arm, guard, *, resume=False, verify=False, replay=False, kind='fits'):
    train, held = data['sites'] != site, data['sites'] == site
    identity = dict(registration_sha256=source.sha(REG), job=context['job'], source=str(site), arm=arm,
        feature_hash=context['array_hashes']['x'], target_hash=context['array_hashes']['target'],
        fitting_ids_hash=rel.parent.kin.digest(data['ids'][train]), held_ids_hash=rel.parent.kin.digest(data['ids'][held]))
    if arm == 'shared64':
        d = dict(xs={'summary120':data['x']}, target=data['target'], sites=data['sites'],
            ids=data['ids'], weights=data['weights'])
        state, ref = audit.old.parent.old.cached_control(d, {'key':context['job']}, site, refit=verify)
        pred = method.parent.predict(state, data['x'][held]); seconds = None
        rel.base.exact(pred, data['old_prediction'][held])
        nodes = sum(t.tree_.node_count for t in state['model'].estimators_)
    else:
        path = PRIVATE/kind/(context['job']+'_'+str(site)+'_'+arm+'.pt.gz'); saved = None
        if path.exists():
            if not(resume or verify or replay): raise ValueError('Explicit resume or verification required')
            saved = rel.base.read_checkpoint(path); rel.base.exact(saved['identity'], identity)
        elif verify or replay: raise ValueError('Cannot replay missing model')
        if saved is None or verify:
            guard(); start = time.monotonic()
            state = method.fit(data['x'], data['target'], train, data['weights'], arm)
            fitted = dict(identity=identity, state=state, prediction=method.predict(state, data['x'][held]),
                seconds=time.monotonic()-start)
            if saved is None:
                path.parent.mkdir(parents=True, exist_ok=True); rel.base.save_checkpoint(path, fitted); saved=fitted
            else:
                rel.base.exact(method.fingerprint(state), method.fingerprint(saved['state']))
                rel.base.exact(fitted['prediction'], saved['prediction'])
        state, ref, seconds = saved['state'], source.artifact(path), saved['seconds']
        pred = method.predict(state, data['x'][held]); nodes = state['nodes']
        rel.base.exact(pred, saved['prediction'])
        if arm == 'shared32':
            fold = next(f for f in context['folds'] if f['source'] == site)
            control = rel.base.read_checkpoint(source.checked(fold['control']))['state']
            control['model'].estimators_ = control['model'].estimators_[:32]
            rel.base.exact(pred, method.parent.predict(control, data['x'][held]))
    rel.base.exact(state['target_scale'], data['target_scale'][held][0])
    return source.stable(dict(**identity, checkpoint=ref, seconds=seconds, nodes=nodes,
        result_source='cached_verified' if arm == 'shared64' else 'fresh_run',
        rows=int(held.sum()), queries=len(set(data['query'][held])), fitting_sources=sorted(set(data['sites'][train])),
        prediction_hash=rel.parent.kin.digest(pred), target_scale=state['target_scale'].tolist(),
        metrics=method.parent.previous.metrics(pred, data['target'][held], data['weights'][held], state['target_scale']),
        boundary=boundary(data['target'][held], data['old_prediction'][held], pred, data['moments'][held],
            data['weights'][held], state['target_scale'])))


def execute(report, guard, beat, *, resume=False, verify=False, replay=False):
    if not source.read(PUBLIC/'pilot.json')['exact_repeat']: raise ValueError('Full-model pilot required')
    contexts=[]; completed=0
    for context in report['contexts']:
        data=inputs(context); folds=[]
        for site in sorted(set(data['sites'])):
            fold=dict(source=str(site), arms={})
            for arm in ('shared64', *method.ARMS):
                record=cell(data,context,site,arm,guard,resume=resume,verify=verify,replay=replay)
                path=PUBLIC/'fits'/(context['job']+'_'+str(site)+'_'+arm+'.json')
                if replay: rel.base.exact(source.read(path),record)
                else: source.once(path,record)
                fold['arms'][arm]=record; completed+=1
                beat(state='cell_complete',cells=completed,total=24,job=context['job'],arm=arm)
            folds.append(fold)
        contexts.append(dict(job=context['job'],audit=context['audit'],folds=folds,summary=method.summarize(folds)))
    passing=sum(c['summary']['passes_information_screen'] for c in contexts)
    result=source.stable(dict(contexts=contexts,fresh_model_objects=16,fresh_forests=24,cached_controls=8,
        passing_contexts=passing,neural_followup_supported=passing>=2,
        result_source='fresh_run_source_held_channel_specialist_training',data_source='cached_verified',
        original_rules_unchanged=True,identity_certification=False,new_V_or_H=False,
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay: rel.base.exact(source.read(PUBLIC/'analysis.json'),result)
    else: source.once(PUBLIC/'analysis.json',result)
    if verify: source.once(PUBLIC/'verification.json',dict(all24_cells_refit_exact=True,
        target_scales_unchanged=True,shared32_prefix_exact=True,analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','pilot','run','verify','replay'))
    p.add_argument('--resume',action='store_true'); args=p.parse_args(); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime=audit.old.parent.old.previous.old.run.old.old.previous.teacher.lineage; runtime.PRIVATE=PRIVATE
        start,guard,beat=runtime.runtime({},source.read(CONFIG),args.phase)
        before={str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()} if args.phase=='replay' else None
        cfg,report=register()
        if args.phase=='register': return
        for file in (*FILES,REG):
            blob=source.subprocess.run(['git','show','HEAD:'+str(file.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if source.hashlib.sha256(blob).hexdigest()!=source.sha(file): raise ValueError('Commit protocol before fitting')
        receipt=(PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        if receipt.exists(): raise ValueError('Completed phase immutable')
        if args.phase=='pilot':
            context=report['contexts'][-1]; data=inputs(context); site=sorted(set(data['sites']))[0]
            a=cell(data,context,site,'separate64',guard,resume=args.resume,kind='pilot')
            b=cell(data,context,site,'separate64',guard,verify=True,kind='pilot'); rel.base.exact(a,b)
            source.once(PUBLIC/'pilot.json',dict(exact_repeat=True,model=a,local_cost_reasonable=True))
        else:
            if args.phase=='replay':
                def forbidden(*args,**kwargs): raise RuntimeError('Read-only replay cannot fit models')
                method.fit=forbidden; method.parent.fit_predict=forbidden
            execute(report,guard,beat,resume=args.resume,verify=args.phase=='verify',replay=args.phase=='replay')
        unchanged=None
        if before is not None:
            rel.base.exact(before,{str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()}); unchanged=len(before)
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            unchanged_public_files=unchanged))
        beat(state='complete')


if __name__=='__main__': main()
