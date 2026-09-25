"""Source-only paired risk-loss ablation with frozen forecasts and utility."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_protected_motion as prior
from scripts import run_m3w_european_symmetric_utility as utility
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.evaluation.m3w_symmetric_utility import complete_budget, verify_matched_checkpoint
from src.evaluation.m3w_symmetric_risk import validate_config, risk_diagnostic
from src.world_model.m3w_european_conditional_risk import nonnegative_moments, event_labels
from src.world_model.m3w_native_gain_harm import fit_neural, predict_neural, build_head
from src.world_model.m3w_european_protected_motion import candidate_decisions
from src.world_model.m3w_european_cv_reference import choose_errors, CONTROL_ARMS
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_symmetric_risk_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_risk_v1'
CONFIG = 'configs/m3w_european_symmetric_risk_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_symmetric_risk.py',
    'src/evaluation/m3w_symmetric_risk.py', 'tests/test_m3w_symmetric_risk.py',
    'tests/test_m3w_utility_objective_semantics.py',
    'outputs/publication_readiness_2026_09/european_symmetric_risk_v1/registration.md')


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Symmetric-risk registration changed: '+path)
    utility.assert_identity(identity['previous_identity'])
    if digest(utility.PUBLIC/'analysis.json') != identity['previous_analysis_sha256']:
        raise ValueError('Frozen control analysis changed')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    oldreg, data, oldidentity, designs, qmask = utility.load()
    validate_config(reg, oldreg)
    for name in ('verification.json', 'checkpoint_replay.json', 'accounting_audit.json'):
        r = json.loads((utility.PUBLIC/name).read_text())
        if r['analysis_sha256'] != digest(utility.PUBLIC/'analysis.json') or not r.get('all_passed', True):
            raise ValueError('Verified prior experiment required')
    identity = dict(bindings={p:digest(ROOT/p) for p in FILES}, previous_identity=oldidentity,
        previous_analysis_sha256=digest(utility.PUBLIC/'analysis.json'), folds=oldidentity['folds'],
        numpy=np.__version__, torch=torch.__version__, query_mask_sha256=array_hash(qmask))
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'matrix.json', dict(identity=identity, new_heads=36,
        frozen_utility_heads=18, frozen_ridge_heads=36, old_neural_risk_heads=36, new_neural_updates=72000,
        expected_policy_views=48, source_rows=len(data['sites']), joint_rows=int(qmask.sum()),
        reserved_roles_opened=False))
    return reg, data, identity, designs, qmask


def checked_head(candidate, key, event, identity):
    r = json.loads((PRIVATE/'heads'/f'{candidate}_{key}_{event}'/'complete.json').read_text())
    if r['identity']['identity'] != identity or not prior.previous.prior.previous.artifacts_ok(r):
        raise ValueError('Changed or incomplete symmetric risk head')
    complete_budget({'head':r}, expected=1)
    return r


def train(reg, data, identity, designs, *, resume, pilot):
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        for candidate in reg['candidates']:
            a = prior.assemble(candidate, data, identity['previous_identity']['previous_identity'], key, design, ti)
            for event in reg['events']:
                train_event(reg,data,identity,key,design,ti,candidate,event,a,resume=resume,pilot=pilot)
                if pilot:
                    return
            del a


def train_event(reg,data,identity,key,design,ti,candidate,event,a,*,resume,pilot):
    y, pr, lineage = prior.task_data(a, data, design, event)
    old = prior.checked_head(key, event, 'neural_underharm4', candidate, identity['previous_identity']['previous_identity'])
    fit, held = design['train_ids'], design['held_ids']
    name = f'{candidate}_{key}_{event}'
    directory = PRIVATE/'heads'/name
    hid = dict(identity=identity, lineage=lineage, arm='mse', candidate=candidate,event=event,
        old_checkpoint=old['artifacts']['checkpoint'])
    if (directory/'complete.json').exists():
        if checked_head(candidate,key,event,identity)['identity'] != hid:
            raise ValueError('Completed risk lineage changed')
        beat('verified_completed_head', trial=name)
        return
    start = time.monotonic()
    beat('risk_training', trial=name, train_rows=len(fit), held_rows=len(held))
    model, report = fit_neural(a['x'][fit], y, data['sites'][fit], np.zeros(len(fit),bool), pr,
        seed=ti['seed'], arm='mse', settings=reg['training'], identity=hid,
        directory=directory, resume=resume, stop_at=100 if pilot else None,
        heartbeat=lambda **kw:beat(trial=name, **kw))
    cp = directory/'checkpoint.pt'
    if not report['complete']:
        immutable_json(PRIVATE/'pilot.json', dict(identity=hid, fit=report,
            checkpoint=artifact(cp), result_source='fresh_run_inside_fixed_budget'))
        assert_identity(identity)
        beat('pilot_complete_requires_resume', trial=name, step=report['step'])
        return
    state = torch.load(cp, map_location='cpu', weights_only=False)
    oldstate = torch.load(ROOT/old['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
    verify_matched_checkpoint(state, oldstate)
    values = nonnegative_moments(predict_neural(model, a['x'][held], np.zeros(len(held),bool), pr), a['same'][held])
    path = directory/'scores.npz'
    prior.previous.prior.atomic_arrays(path, ids=held, moments=values)
    immutable_json(directory/'complete.json', dict(identity=hid, fit=report,
        matched_draws_and_preprocessing=True, wall_seconds=time.monotonic()-start,
        artifacts=dict(checkpoint=artifact(cp), scores=artifact(path))))
    assert_identity(identity)
    beat('risk_complete', trial=name, updates=report['step'])


def decisions(reg, data, a, held, qmask, utility_score, moments, name, identity, support, guard, verify):
    path = PRIVATE/'decisions'/(name+'.npz')
    binding = dict(identity=identity, name=name, source_support=support, guard=guard)
    if path.with_suffix('.json').exists():
        r = json.loads(path.with_suffix('.json').read_text())
        if r['identity'] != binding or r['sha256'] != digest(path):
            raise ValueError('Changed decisions')
        with np.load(path, allow_pickle=False) as z:
            out = {k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(out['pointwise_ids'], held)
        np.testing.assert_array_equal(out['ids'], held[qmask[held]])
        return out, r
    if verify:
        raise ValueError('Missing decisions cannot be verified')
    available = guard == 'no_guard' or support['gate_available']
    cached = '_ridge_' in name
    if cached:
        oldpath = utility.PRIVATE/'decisions'/(name+'.npz')
        oldr = json.loads(oldpath.with_suffix('.json').read_text())
        if digest(oldpath)!=oldr['sha256']:
            raise ValueError('Frozen ridge decisions changed')
        with np.load(oldpath,allow_pickle=False) as z:
            out = {k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(out['pointwise_ids'],held)
        np.testing.assert_array_equal(out['ids'],held[qmask[held]])
        queries = oldr['queries']
    else:
        out, queries = candidate_decisions(reg,data,a,held,qmask,utility_score,moments,available)
    prior.previous.prior.atomic_arrays(path, **out)
    r = dict(identity=binding,path=str(path.relative_to(ROOT)),sha256=digest(path),queries=queries,
        source_support_available=available,used_future_inputs=False,
        result_source='cached_verified_ridge_decisions' if cached else 'fresh_run_neural_risk_decisions')
    immutable_json(path.with_suffix('.json'),r)
    return out,r


def evaluate(reg, data, identity, designs, qmask, verify):
    heads = {f'{candidate}_{key}_{event}':checked_head(candidate,key,event,identity) for candidate in reg['candidates']
        for key,_,ti in designs if ti['kind']=='complement' for event in reg['events']}
    updates = complete_budget(heads,expected=36)
    utilityheads = {f'{candidate}_{key}':utility.checked_head(candidate,key,identity['previous_identity'])
        for candidate in reg['candidates'] for key,_,ti in designs if ti['kind']=='complement'}
    oldheads = {f'{candidate}_{key}_{task}_{arm}':prior.checked_head(key,task,arm,candidate,identity['previous_identity']['previous_identity'])
        for candidate in reg['candidates'] for key,_,ti in designs if ti['kind']=='complement'
        for task in reg['events'] for arm in ('ridge','neural_underharm4')}
    n = len(data['sites'])
    cv, cvf = np.asarray(data['baseline_ade'][:,1]), np.asarray(data['baseline_fde'][:,1])
    population, outputs, receipts, cost_metrics = {}, {}, [], {}
    for key,design,ti in designs:
        if ti['kind'] != 'complement':
            continue
        held, seed = design['held_ids'], ti['seed']
        p = population.setdefault(seed, dict(easy=np.zeros(n,bool),hard=np.zeros(n,bool),
            strong=np.full(n,np.nan),strongf=np.full(n,np.nan),candidates={}))
        p['easy'][held] = (cv[held]>0)&(cv[held]<=design['easy_cut'])
        p['hard'][held] = cv[held]>=design['hard_cut']
        p['strong'][held] = data['baseline_ade'][held,design['baseline_index']]
        p['strongf'][held] = data['baseline_fde'][held,design['baseline_index']]
        for candidate in reg['candidates']:
            a = prior.assemble(candidate,data,identity['previous_identity']['previous_identity'],key,design,ti)
            s = p['candidates'].setdefault(candidate,dict(ade=np.full(n,np.nan),fde=np.full(n,np.nan),seen=np.zeros(n,bool)))
            if s['seen'][held].any():
                raise ValueError('Repeated outer rows')
            s['seen'][held] = True
            s['ade'][held],s['fde'][held] = native_errors(a['p'][held].astype(float)+data['origin'][held,None],
                data['target_eval'][held],data['valid'][held],np.ones(len(held)))
            utility_r = utilityheads[f'{candidate}_{key}']
            cost = prior.scores(utility_r,'utility',held)
            u = (cost[:,0]-cost[:,1])/utility_r['identity']['lineage']['cost_scale']
            support = utility_r['identity']['lineage']['source_support']
            for event in reg['events']:
                truth = event_labels(cv[held],np.maximum(s['ade'][held]-cv[held],0),easy_cut=design['easy_cut'],event=event)
                for arm in reg['arms']:
                    risk_head = oldheads[f'{candidate}_{key}_{event}_ridge'] if arm=='ridge' else heads[f'{candidate}_{key}_{event}']
                    moments = prior.scores(risk_head,event,held)
                    oldarm = 'ridge' if arm=='ridge' else 'neural_underharm4'
                    oldmoments = prior.scores(oldheads[f'{candidate}_{key}_{event}_{oldarm}'],event,held)
                    for guard in reg['support_controls']:
                        name = f'{candidate}_{key}_{event}_{arm}_{guard}'
                        beat('fixed_policy_readout',trial=name,verify=verify)
                        choice,r = decisions(reg,data,a,held,qmask,u,moments,name,identity,support,guard,verify)
                        receipts.append(artifact(PRIVATE/'decisions'/(name+'.json')))
                        v = outputs.setdefault(f'{seed}_{candidate}_{event}_{arm}_{guard}',dict(seed=seed,candidate=candidate,
                            bits={k:np.zeros(n,bool) for k in ('pointwise',*CONTROL_ARMS)},
                            oldbits={k:np.zeros(n,bool) for k in ('pointwise',*CONTROL_ARMS)},
                            seen=np.zeros(n,bool),nonzero=np.zeros(n,bool),queries=[]))
                        if v['seen'][held].any():
                            raise ValueError('Repeated policy rows')
                        v['seen'][held] = True
                        v['bits']['pointwise'][held] = choice['pointwise']
                        for k in CONTROL_ARMS:
                            v['bits'][k][choice['ids']] = choice[k]
                        v['nonzero'][choice['ids']] = choice['matched_nonzero']
                        v['queries'].extend(r['queries'])
                        oldname = f'{candidate}_{key}_{event}_{oldarm}_{guard}'
                        oldpath = utility.PRIVATE/'decisions'/(oldname+'.npz')
                        oldr = json.loads(oldpath.with_suffix('.json').read_text())
                        if digest(oldpath) != oldr['sha256']:
                            raise ValueError('Old decision bank changed')
                        with np.load(oldpath,allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['pointwise_ids'],held)
                            np.testing.assert_array_equal(z['ids'],choice['ids'])
                            v['oldbits']['pointwise'][held] = z['pointwise']
                            for k in CONTROL_ARMS:
                                v['oldbits'][k][z['ids']] = z[k]
                            for site in sorted(set(data['sites'][held])):
                                use = data['sites'][held]==site
                                cost_metrics[f'{name}_{site}'] = {
                                    'new':risk_diagnostic(moments[use],truth[use],choice['pointwise'][use]),
                                    'old':risk_diagnostic(oldmoments[use],truth[use],z['pointwise'][use])}
            del a
    roster, full = sorted(identity['folds']), np.ones(n,bool)
    def metric(m,r,mask):
        return paired_scene_metrics(m[mask],r[mask],data['sites'][mask],expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
    def describe(p,ade,fde,bits,mask):
        easy = metric(ade,cv,mask&p['easy'])
        zero = mask&np.isfinite(cv)&(cv==0)
        return dict(ADE_vs_CV=metric(ade,cv,mask),FDE_vs_CV=metric(fde,cvf,mask),
            ADE_vs_training_selected_baseline=metric(ade,p['strong'],mask),
            hard_ADE_vs_CV=metric(ade,cv,mask&p['hard']),positive_easy_ADE_vs_CV=easy,
            complete_ADE_vs_CV=metric(ade,cv,mask&data['valid'].all(1)),
            zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((ade[zero]>0).sum()),total_added_error=float(ade[zero].sum())),
            switch_rate=float(bits[mask].mean()),indexed_rows=int(mask.sum()),
            safety_observed_pass=bool(easy['worst_scene_gain_percent'] is not None
                and easy['worst_scene_gain_percent']>=-2 and not np.any(ade[zero]>0)),
            zero_safety_supported=bool(zero.any()),calibrated_safety=False)
    policies, errorbanks, old_contrasts, changes = {}, {}, {}, {}
    for name,v in outputs.items():
        p = population[v['seed']]
        s = p['candidates'][v['candidate']]
        if not v['seen'].all() or not s['seen'].all():
            raise ValueError('Missing complete policy population')
        errors = {k:choose_errors(bits,s['ade'],cv) for k,bits in v['bits'].items()}
        old = {k:choose_errors(bits,s['ade'],cv) for k,bits in v['oldbits'].items()}
        errorbanks[name] = errors
        policies[name] = dict(full=describe(p,errors['pointwise'],choose_errors(v['bits']['pointwise'],s['fde'],cvf),v['bits']['pointwise'],full),
            joint_population={k:describe(p,errors[k],choose_errors(v['bits'][k],s['fde'],cvf),v['bits'][k],qmask) for k in CONTROL_ARMS},
            comparisons=dict(joint_vs_independent=metric(errors['joint'],errors['independent'],qmask),
                joint_exact_vs_independent=metric(errors['joint_exact'],errors['independent'],qmask&v['nonzero']),
                joint_exact_vs_unary=metric(errors['joint_exact'],errors['unary_exact'],qmask&v['nonzero'])),
            queries=dict(total=len(v['queries']),matched=sum(q['matched'] for q in v['queries']),
                matched_nonzero=sum(q['matched_nonzero'] for q in v['queries']),
                solver_failures={k:sum(not q['arms'][k]['solver_optimal'] for q in v['queries']) for k in CONTROL_ARMS}))
        old_contrasts[name] = {k:metric(errors[k],old[k],mask) for k,mask in
            [('pointwise',full),*[(k,qmask) for k in CONTROL_ARMS]]}
        old_contrasts[name]['pointwise_hard'] = metric(errors['pointwise'],old['pointwise'],p['hard'])
        old_contrasts[name]['pointwise_easy'] = metric(errors['pointwise'],old['pointwise'],p['easy'])
        changes[name] = {k:dict(added=int((v['bits'][k]&~v['oldbits'][k]).sum()),
            removed=int((~v['bits'][k]&v['oldbits'][k]).sum())) for k in v['bits']}
    contrasts = {}
    for seed in reg['seeds']:
        for event in reg['events']:
            for arm in reg['arms']:
                for guard in reg['support_controls']:
                    key = f'{seed}_{event}_{arm}_{guard}'
                    left,right = (errorbanks[f'{seed}_{c}_{event}_{arm}_{guard}'] for c in reg['candidates'])
                    contrasts[key] = {k:metric(left[k],right[k],mask) for k,mask in
                        [('pointwise',full),*[(k,qmask) for k in CONTROL_ARMS]]}
                    contrasts[key]['pointwise_hard'] = metric(left['pointwise'],right['pointwise'],population[seed]['hard'])
                    contrasts[key]['pointwise_easy'] = metric(left['pointwise'],right['pointwise'],population[seed]['easy'])
    result = dict(result_source='fresh_run_symmetric_risk_heads_and_decisions_cached_verified_ridge_controls',identity=identity,
        training={k:dict(fit=r['fit'],lineage=r['identity']['lineage'],artifacts=r['artifacts'],result_source='fresh_run') for k,r in heads.items()},
        frozen_controls={k:r['artifacts'] for k,r in oldheads.items()},
        frozen_utility={k:dict(artifacts=r['artifacts'],lineage=r['identity']['lineage']) for k,r in utilityheads.items()},controls=receipts,
        source_rows=n,joint_rows=int(qmask.sum()),policies=policies,neural_vs_damping=contrasts,
        symmetric_vs_asymmetric=old_contrasts,decision_changes=changes,risk_reliability=cost_metrics,
        new_heads=36,cached_heads=90,new_neural_updates=updates,new_forecaster_training=False,
        independent_reserved_readout=False,calibrated_safety=False,deployment_changed=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:
        immutable_json(PUBLIC/'verification.json',dict(result_source='cached_verified',
            analysis_sha256=digest(PUBLIC/'analysis.json'),metrics_recomputed=True,new_training=False))
    beat('symmetric_readout_complete',policies=len(policies),verified=verify)


def replay(reg,data,identity,designs):
    if not (PUBLIC/'analysis.json').exists():
        raise ValueError('Complete analysis required')
    checks = []
    for key,design,ti in designs:
        if ti['kind']!='complement':
            continue
        for candidate in reg['candidates']:
            a = prior.assemble(candidate,data,identity['previous_identity']['previous_identity'],key,design,ti)
            for event in reg['events']:
                checks.append(replay_event(reg,data,identity,key,design,ti,candidate,event,a))
            del a
    if len(checks)!=36:
        raise ValueError('All thirty-six new heads required')
    assert_identity(identity)
    immutable_json(PUBLIC/'checkpoint_replay.json',dict(result_source='fresh_run_checkpoint_inference',
        analysis_sha256=digest(PUBLIC/'analysis.json'),checks=checks,all_passed=True,new_training=False))
    beat('symmetric_replay_complete',heads=len(checks))


def replay_event(reg,data,identity,key,design,ti,candidate,event,a):
    r = checked_head(candidate,key,event,identity)
    old = prior.checked_head(key,event,'neural_underharm4',candidate,identity['previous_identity']['previous_identity'])
    cp = torch.load(ROOT/r['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
    oldcp = torch.load(ROOT/old['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
    verify_matched_checkpoint(cp,oldcp)
    if cp['identity']!=r['identity']:
        raise ValueError('Checkpoint identity mismatch')
    _,pr,lineage = prior.task_data(a,data,design,event)
    if lineage!=r['identity']['lineage']:
        raise ValueError('Recomputed lineage mismatch')
    for k in ('mean','std','constant','weights','known'):
        np.testing.assert_array_equal(cp['preprocess'][k],pr[k])
    held = design['held_ids'][:4096]
    model = build_head(reg['training']['width'],pr,ti['seed'])
    model.load_state_dict(cp['model'])
    fresh = nonnegative_moments(predict_neural(model,a['x'][held],np.zeros(len(held),bool),pr),a['same'][held])
    saved = prior.scores(r,event,design['held_ids'])[:len(held)]
    np.testing.assert_array_equal(fresh,saved)
    return dict(trial=f'{candidate}_{key}_{event}',rows=len(held),exact=True,max_difference=0.,
        paired_sampler_exact=True,total_draws=int(cp['draws'].sum()),unknown_draws=0,
        checkpoint_sha256=r['artifacts']['checkpoint']['sha256'],ids_sha256=array_hash(held))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare','pilot','train','resume','evaluate','verify','replay'):
        parser.add_argument('--'+name,action='store_true')
    args = parser.parse_args()
    if not any((args.prepare,args.pilot,args.train,args.evaluate,args.verify,args.replay)):
        parser.error('Explicit phase required')
    if args.pilot and any((args.train,args.evaluate,args.verify,args.replay)):
        parser.error('Pilot is separate')
    for path in (PRIVATE,PUBLIC):
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('Symlinked destination')
        path.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        reg,data,identity,designs,qmask = load()
        beat('inputs_verified',rows=len(data['sites']),architecture=platform.machine(),threads=torch.get_num_threads(),workers=0)
        if args.train or args.pilot:
            train(reg,data,identity,designs,resume=args.resume,pilot=args.pilot)
        if args.evaluate or args.verify:
            evaluate(reg,data,identity,designs,qmask,args.verify)
        if args.replay:
            replay(reg,data,identity,designs)
        beat('phase_complete')


if __name__=='__main__':
    main()
