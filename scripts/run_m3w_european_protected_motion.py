"""Registered same-risk motion versus neural control; no reserved-role readout."""
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
from scripts import run_m3w_european_conditional_risk as previous
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.world_model.m3w_european_protected_motion import motion_candidate_inputs, candidate_decisions
from src.world_model.m3w_european_conditional_risk import event_labels, fitting_support, nonnegative_moments
from src.world_model.m3w_european_source_intervention import paired_cost_labels
from src.world_model.m3w_european_cv_reference import choose_errors, CONTROL_ARMS
from src.world_model.m3w_native_gain_harm import preprocess, fit_ridge, predict_ridge, fit_neural, predict_neural, build_head
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_protected_motion_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_protected_motion_v1'
CONFIG = 'configs/m3w_european_protected_motion_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_protected_motion.py',
    'src/world_model/m3w_european_protected_motion.py', 'tests/test_m3w_european_protected_motion.py',
    'src/world_model/m3w_event_risk_feasibility.py', 'tests/test_m3w_event_risk_feasibility.py',
    'outputs/publication_readiness_2026_09/european_protected_motion_v1/registration.md')


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def assert_identity(identity):
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen protected-motion experiment changed: '+path)
    previous.assert_identity(identity['previous_identity'])


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    oldreg, data, oldidentity, designs, qmask = previous.load()
    for key in ('seeds', 'events', 'arms', 'support_controls', 'ridge_alpha', 'training',
                'predicted_risk_budget', 'pair_weight', 'edge_radius_bbox_widths',
                'proximity_threshold_bbox_widths', 'solver_seconds', 'bootstrap_resamples', 'bootstrap_seed'):
        if reg[key] != oldreg[key]:
            raise ValueError('Unregistered simultaneous change: '+key)
    if (reg['candidates'] != ['neural', 'damping097'] or reg['utility_head'] !=
            'candidate_specific_neural_underharm4_same_budget' or any(reg[k] for k in
            ('independent_reserved_readout', 'deployment_promotion', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only candidate comparison required')
    for name in ('verification.json', 'checkpoint_replay.json', 'accounting_audit.json'):
        r = json.loads((previous.PUBLIC/name).read_text())
        if r['analysis_sha256'] != digest(previous.PUBLIC/'analysis.json') or not r.get('all_passed', True):
            raise ValueError('Verified previous analysis required')
    identity = dict(bindings={p:digest(ROOT/p) for p in FILES}, previous_identity=oldidentity,
        previous_analysis_sha256=digest(previous.PUBLIC/'analysis.json'), folds=oldidentity['folds'],
        numpy=np.__version__, torch=torch.__version__, query_mask_sha256=array_hash(qmask))
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'matrix.json', dict(identity=identity, new_heads=45,
        new_ridge_heads=18, new_neural_heads=27, new_neural_updates=54000,
        cached_verified_neural_candidate_heads=45, expected_policy_views=48,
        source_rows=len(data['sites']), joint_rows=int(qmask.sum()), reserved_roles_opened=False))
    return reg, data, identity, designs, qmask


def specs(reg):
    return [('utility', 'neural_underharm4')]+[(e,a) for e in reg['events'] for a in reg['arms']]


def assemble(candidate, data, identity, key, design, ti):
    if candidate == 'neural':
        return previous.prior.assemble(data, identity['previous_identity']['previous_identity'], key, design, ti)
    if candidate != 'damping097':
        raise ValueError('Undeclared candidate')
    fit, held = design['train_ids'], design['held_ids']
    train_sites, held_sites = sorted(set(data['sites'][fit])), sorted(set(data['sites'][held]))
    if set(train_sites) & set(held_sites):
        raise ValueError('Source-fold leakage')
    a = motion_candidate_inputs(data['geometry'], data['history'], data['origin'])
    err, fde = native_errors(a['p']+data['origin'][:, None], data['target_eval'], data['valid'], np.ones(len(a['p'])))
    np.testing.assert_array_equal(err, data['baseline_ade'][:, 3])
    np.testing.assert_array_equal(fde, data['baseline_fde'][:, 3])
    cv = np.asarray(data['baseline_ade'][:, 1])
    a['y'] = paired_cost_labels(cv[fit], err[fit])
    a['pr'] = preprocess(a['x'][fit], a['y'], cv[fit], data['sites'][fit], held_sites[0])
    a['lineage'] = dict(outer_fold=ti['fold'], seed=ti['seed'], training_sites=train_sites, held_sites=held_sites,
        final_producer='fixed_causal_damping097_no_fitted_teacher', nested_producers={}, baseline_index=1,
        candidate_index=3, train_ids_sha256=array_hash(fit), held_ids_sha256=array_hash(held),
        feature_train_sha256=array_hash(a['x'][fit]), labels_train_sha256=array_hash(a['y']),
        cost_scale=a['pr']['cost_scale'], easy_cut=design['easy_cut'], hard_cut=design['hard_cut'],
        supported_training_rows=int(a['pr']['known'].sum()), unknown_training_rows=int((~a['pr']['known']).sum()))
    return a


def task_data(a, data, design, task):
    fit = design['train_ids']
    cv = np.asarray(data['baseline_ade'][fit, 1])
    y = a['y'] if task == 'utility' else event_labels(cv, a['y'][:, 1], easy_cut=design['easy_cut'], event=task)
    pr = preprocess(a['x'][fit], y, cv, data['sites'][fit], sorted(set(data['sites'][design['held_ids']]))[0])
    for k in ('mean', 'std', 'weights', 'known'):
        np.testing.assert_array_equal(pr[k], a['pr'][k])
    if pr['cost_scale'] != a['pr']['cost_scale']:
        raise ValueError('Different CV cost scale')
    lineage = dict(a['lineage'], task=task, task_labels_sha256=array_hash(y),
        source_support=fitting_support(cv, data['sites'][fit]))
    return y, pr, lineage


def checked_head(key, task, arm, candidate, identity):
    if candidate == 'neural':
        if task == 'utility':
            return previous.prior.checked_head(key+'_'+arm, identity['previous_identity']['previous_identity'])
        return previous.checked_head(key+'_'+task+'_'+arm, identity['previous_identity'])
    name = key+'_'+task+'_'+arm
    r = json.loads((PRIVATE/'heads'/name/'complete.json').read_text())
    if r['identity']['identity'] != identity or not previous.prior.previous.artifacts_ok(r):
        raise ValueError('Incomplete or changed protected-motion head')
    if 'neural' in arm and (r['fit']['step'] != 2000 or not r['fit']['complete']):
        raise ValueError('Unfinished neural budget')
    return r


def scores(r, task, held):
    field = 'predicted_costs' if 'predicted_costs' in r['artifacts'] else 'scores'
    with np.load(ROOT/r['artifacts'][field]['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], held)
        return z['costs' if task == 'utility' else 'moments'].copy()


def train(reg, data, identity, designs, resume, pilot):
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        a = assemble('damping097', data, identity, key, design, ti)
        fit, held = design['train_ids'], design['held_ids']
        for task, arm in specs(reg):
            name = key+'_'+task+'_'+arm
            directory = PRIVATE/'heads'/name
            y, pr, lineage = task_data(a, data, design, task)
            hid = dict(identity=identity, lineage=lineage, arm=arm, candidate='damping097')
            if (directory/'complete.json').exists():
                if checked_head(key, task, arm, 'damping097', identity)['identity'] != hid:
                    raise ValueError('Completed lineage changed')
                beat('verified_completed_head', trial=name)
                continue
            start = time.monotonic()
            beat('motion_head_training', trial=name, train_rows=len(fit), held_rows=len(held))
            zero = a['same'] if task == 'utility' else np.zeros(len(a['p']), bool)
            if arm == 'ridge':
                head = fit_ridge(a['x'][fit], y, pr, alpha=reg['ridge_alpha'])
                raw = predict_ridge(head, a['x'][held], zero[held], pr)
                cp = directory/'ridge.pt'
                previous.prior.previous.save_state(cp, dict(identity=hid, head=head, preprocess=pr))
                report = dict(method='weighted_ridge_closed_form', gradient_updates=0,
                              supported_rows=int(pr['known'].sum()))
            else:
                model, report = fit_neural(a['x'][fit], y, data['sites'][fit], zero[fit], pr,
                    seed=ti['seed'], arm='underharm4', settings=reg['training'], identity=hid,
                    directory=directory, resume=resume, stop_at=100 if pilot else None,
                    heartbeat=lambda **kw:beat(trial=name, **kw))
                cp = directory/'checkpoint.pt'
                if not report['complete']:
                    immutable_json(PRIVATE/'pilot.json', dict(identity=hid, fit=report,
                        checkpoint=artifact(cp), result_source='fresh_run_inside_fixed_budget'))
                    assert_identity(identity)
                    beat('pilot_complete_requires_resume', trial=name, step=report['step'])
                    return
                raw = predict_neural(model, a['x'][held], zero[held], pr)
            values = np.maximum(raw, 0) if task == 'utility' else nonnegative_moments(raw, a['same'][held])
            path = directory/'scores.npz'
            previous.prior.atomic_arrays(path, ids=held, **{'costs' if task=='utility' else 'moments':values})
            immutable_json(directory/'complete.json', dict(identity=hid, fit=report,
                wall_seconds=time.monotonic()-start, artifacts=dict(checkpoint=artifact(cp), scores=artifact(path))))
            assert_identity(identity)
            beat('motion_head_complete', trial=name)
        del a


def get_decisions(reg, data, a, held, qmask, u, m, name, identity, support, guard, verify):
    path = PRIVATE/'decisions'/(name+'.npz')
    binding = dict(identity=identity, name=name, source_support=support, guard=guard)
    if path.with_suffix('.json').exists():
        r = json.loads(path.with_suffix('.json').read_text())
        if r['identity'] != binding or r['sha256'] != digest(path):
            raise ValueError('Changed control decisions')
        with np.load(path, allow_pickle=False) as z:
            out = {k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(out['pointwise_ids'], held)
        np.testing.assert_array_equal(out['ids'], held[qmask[held]])
        return out, r
    if verify:
        raise ValueError('Cannot verify missing decisions')
    available = guard == 'no_guard' or support['gate_available']
    out, queries = candidate_decisions(reg, data, a, held, qmask, u, m, available)
    previous.prior.atomic_arrays(path, **out)
    r = dict(identity=binding, path=str(path.relative_to(ROOT)), sha256=digest(path), queries=queries,
        source_support_available=available, used_future_inputs=False)
    immutable_json(path.with_suffix('.json'), r)
    return out, r


def evaluate(reg, data, identity, designs, qmask, verify):
    heads = {f'{c}_{key}_{task}_{arm}':checked_head(key, task, arm, c, identity)
        for c in reg['candidates'] for key, _, ti in designs if ti['kind']=='complement'
        for task, arm in specs(reg)}
    if len(heads) != 90:
        raise ValueError('45 new and45 verified cached heads required before readout')
    n = len(data['sites'])
    cv, cvf = np.asarray(data['baseline_ade'][:, 1]), np.asarray(data['baseline_fde'][:, 1])
    population, outputs, receipts, old_changes = {}, {}, [], {}
    for key, design, ti in designs:
        if ti['kind'] != 'complement':
            continue
        held, seed = design['held_ids'], ti['seed']
        p = population.setdefault(seed, dict(easy=np.zeros(n, bool), hard=np.zeros(n, bool),
            strong=np.full(n, np.nan), strongf=np.full(n, np.nan), candidates={}))
        p['easy'][held] = (cv[held]>0)&(cv[held]<=design['easy_cut'])
        p['hard'][held] = cv[held]>=design['hard_cut']
        p['strong'][held], p['strongf'][held] = data['baseline_ade'][held, design['baseline_index']], data['baseline_fde'][held, design['baseline_index']]
        for candidate in reg['candidates']:
            a = assemble(candidate, data, identity, key, design, ti)
            s = p['candidates'].setdefault(candidate, dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan), seen=np.zeros(n, bool)))
            if s['seen'][held].any():
                raise ValueError('Repeated outer readout')
            s['seen'][held] = True
            s['ade'][held], s['fde'][held] = native_errors(a['p'][held].astype(float)+data['origin'][held,None],
                data['target_eval'][held], data['valid'][held], np.ones(len(held)))
            utility_r = heads[f'{candidate}_{key}_utility_neural_underharm4']
            cost = scores(utility_r, 'utility', held)
            u = (cost[:,0]-cost[:,1])/utility_r['identity']['lineage']['cost_scale']
            support = fitting_support(data['baseline_ade'][design['train_ids'],1], data['sites'][design['train_ids']])
            for event in reg['events']:
                truth = event_labels(cv[held], np.maximum(s['ade'][held]-cv[held],0), easy_cut=design['easy_cut'], event=event)
                for arm in reg['arms']:
                    r = heads[f'{candidate}_{key}_{event}_{arm}']
                    m = scores(r, event, held)
                    for guard in reg['support_controls']:
                        name = f'{candidate}_{key}_{event}_{arm}_{guard}'
                        beat('candidate_control_readout', trial=name, verify=verify)
                        choice, receipt = get_decisions(reg,data,a,held,qmask,u,m,name,identity,support,guard,verify)
                        receipts.append(artifact(PRIVATE/'decisions'/(name+'.json')))
                        v = outputs.setdefault(f'{seed}_{candidate}_{event}_{arm}_{guard}', dict(seed=seed,candidate=candidate,
                            bits={k:np.zeros(n,bool) for k in ('pointwise',*CONTROL_ARMS)},
                            seen=np.zeros(n,bool), nonzero=np.zeros(n,bool), queries=[], moment_errors={}))
                        if v['seen'][held].any():
                            raise ValueError('Repeated policy population')
                        v['seen'][held] = True
                        v['bits']['pointwise'][held] = choice['pointwise']
                        for k in CONTROL_ARMS:v['bits'][k][choice['ids']] = choice[k]
                        v['nonzero'][choice['ids']] = choice['matched_nonzero']
                        v['queries'].extend(receipt['queries'])
                        for site in sorted(set(data['sites'][held])):
                            use = (data['sites'][held]==site)&np.isfinite(truth).all(1)
                            v['moment_errors'][site] = dict(rows=int(use.sum()), predicted_mean=m[use].mean(0).tolist(),
                                true_mean=truth[use].mean(0).tolist(), mean_absolute_error=np.abs(m[use]-truth[use]).mean(0).tolist())
                        if candidate == 'neural':
                            oldpath = previous.PRIVATE/'decisions'/(key+'_'+event+'_'+arm+'_'+guard+'.npz')
                            oldreceipt = json.loads(oldpath.with_suffix('.json').read_text())
                            if digest(oldpath)!=oldreceipt['sha256']:
                                raise ValueError('Old decisions changed')
                            with np.load(oldpath,allow_pickle=False) as z:
                                np.testing.assert_array_equal(z['ids'],choice['ids'])
                                np.testing.assert_array_equal(z['pointwise_ids'],choice['pointwise_ids'])
                                old_changes[name] = {k:int(np.sum(z[k]!=choice[k])) for k in ('pointwise',*CONTROL_ARMS)}
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
            ADE_vs_fixed_damping097=metric(ade,data['baseline_ade'][:,3],mask),
            hard_ADE_vs_CV=metric(ade,cv,mask&p['hard']),positive_easy_ADE_vs_CV=easy,
            complete_ADE_vs_CV=metric(ade,cv,mask&data['valid'].all(1)),
            zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((ade[zero]>0).sum()),total_added_error=float(ade[zero].sum())),
            switch_rate=float(bits[mask].mean()),indexed_rows=int(mask.sum()),
            safety_observed_pass=bool(easy['worst_scene_gain_percent'] is not None
                and easy['worst_scene_gain_percent']>=-2 and not np.any(ade[zero]>0)),
            zero_safety_supported=bool(zero.any()),calibrated_safety=False)
    references, summaries = {}, {}
    for seed,p in population.items():
        if not all(s['seen'].all() for s in p['candidates'].values()):raise ValueError('Missing candidate rows')
        raw = dict(CV=(cv,cvf),training_selected_baseline=(p['strong'],p['strongf']),
            **{k:(s['ade'],s['fde']) for k,s in p['candidates'].items()})
        references[str(seed)] = {k:{'full':describe(p,ade,fde,np.zeros(n,bool),full),
            'joint_population':describe(p,ade,fde,np.zeros(n,bool),qmask)} for k,(ade,fde) in raw.items()}
    errorbanks = {}
    for name,v in outputs.items():
        if not v['seen'].all():raise ValueError('Missing policy rows')
        p = population[v['seed']]; s = p['candidates'][v['candidate']]
        e = {k:choose_errors(b,s['ade'],cv) for k,b in v['bits'].items()}
        errorbanks[name] = e
        summaries[name] = dict(full=describe(p,e['pointwise'],choose_errors(v['bits']['pointwise'],s['fde'],cvf),v['bits']['pointwise'],full),
            joint_population={k:describe(p,e[k],choose_errors(v['bits'][k],s['fde'],cvf),v['bits'][k],qmask) for k in CONTROL_ARMS},
            comparisons=dict(joint_vs_independent=metric(e['joint'],e['independent'],qmask),
                joint_exact_vs_independent=metric(e['joint_exact'],e['independent'],qmask&v['nonzero']),
                joint_exact_vs_unary=metric(e['joint_exact'],e['unary_exact'],qmask&v['nonzero'])),
            moment_errors=v['moment_errors'],queries=dict(total=len(v['queries']),
                matched=sum(q['matched'] for q in v['queries']),matched_nonzero=sum(q['matched_nonzero'] for q in v['queries']),
                with_edges=sum(q['edges']>0 for q in v['queries']),pruned_infeasible=sum(q['pruned_infeasible'] for q in v['queries']),
                solver_failures={k:sum(not q['arms'][k]['solver_optimal'] for q in v['queries']) for k in CONTROL_ARMS}))
    contrasts = {}
    for seed in reg['seeds']:
        for event in reg['events']:
            for arm in reg['arms']:
                for guard in reg['support_controls']:
                    key = f'{seed}_{event}_{arm}_{guard}'
                    left,right = (errorbanks[f'{seed}_{c}_{event}_{arm}_{guard}'] for c in reg['candidates'])
                    p = population[seed]
                    contrasts[key] = {k:metric(left[k],right[k],mask) for k,mask in
                        [('pointwise',full),*[(k,qmask) for k in CONTROL_ARMS]]}
                    contrasts[key]['pointwise_hard'] = metric(left['pointwise'],right['pointwise'],p['hard'])
                    contrasts[key]['pointwise_easy'] = metric(left['pointwise'],right['pointwise'],p['easy'])
    result = dict(result_source='fresh_run_damping_heads_and_both_candidate_decisions',identity=identity,
        training={k:dict(fit=r['fit'],lineage=r['identity']['lineage'],artifacts=r['artifacts'],
            result_source='cached_verified' if k.startswith('neural_') else 'fresh_run') for k,r in heads.items()},
        controls=receipts,source_rows=n,joint_rows=int(qmask.sum()),references=references,policies=summaries,
        neural_vs_damping=contrasts,old_neural_decision_changes=old_changes,
        new_heads=45,cached_heads=45,new_neural_updates=54000,new_forecaster_training=False,
        independent_reserved_readout=False,calibrated_safety=False,deployment_changed=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:immutable_json(PUBLIC/'verification.json',dict(result_source='cached_verified',
        analysis_sha256=digest(PUBLIC/'analysis.json'),metrics_recomputed=True,new_training=False))
    beat('protected_motion_readout_complete',policies=len(summaries),verified=verify)


def replay(reg,data,identity,designs):
    if not (PUBLIC/'analysis.json').exists():raise ValueError('Complete analysis required')
    checks, draws = [], []
    for key,design,ti in designs:
        if ti['kind']!='complement':continue
        shared_draws,shared_rng,shared_known,shared_weights,shared_scale = None,None,None,None,None
        for candidate in reg['candidates']:
            a = assemble(candidate,data,identity,key,design,ti)
            held = design['held_ids'][:4096]
            for task,arm in specs(reg):
                r = checked_head(key,task,arm,candidate,identity)
                _,pr,_ = task_data(a,data,design,task)
                cp = torch.load(ROOT/r['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
                if cp['identity']!=r['identity']:raise ValueError('Replay identity mismatch')
                for k in ('mean','std','constant','weights','known'):np.testing.assert_array_equal(cp['preprocess'][k],pr[k])
                zero = a['same'][held] if task=='utility' else np.zeros(len(held),bool)
                if arm=='ridge':raw = predict_ridge(cp['head'],a['x'][held],zero,pr)
                else:
                    if shared_draws is None:
                        shared_draws,shared_rng = cp['draws'],cp['sampler_rng']
                        shared_known,shared_weights,shared_scale = pr['known'],pr['weights'],pr['cost_scale']
                    np.testing.assert_array_equal(cp['draws'],shared_draws)
                    np.testing.assert_array_equal(cp['sampler_rng'],shared_rng)
                    np.testing.assert_array_equal(pr['known'],shared_known)
                    np.testing.assert_array_equal(pr['weights'],shared_weights)
                    if pr['cost_scale']!=shared_scale:raise ValueError('Candidate CV scale mismatch')
                    model = build_head(reg['training']['width'],pr,ti['seed'])
                    model.load_state_dict(cp['model'])
                    raw = predict_neural(model,a['x'][held],zero,pr)
                fresh = np.maximum(raw,0) if task=='utility' else nonnegative_moments(raw,a['same'][held])
                saved = scores(r,task,design['held_ids'])[:len(held)]
                np.testing.assert_array_equal(fresh,saved)
                checks.append(dict(trial=f'{candidate}_{key}_{task}_{arm}',rows=len(held),exact=True,max_difference=0.,
                    ids_sha256=array_hash(held),checkpoint_sha256=r['artifacts']['checkpoint']['sha256']))
            del a
        draws.append(dict(fold=ti['fold'],seed=ti['seed'],exact=True,
            compared_neural_heads=6,total_draws=int(shared_draws.sum()),unique_rows=int((shared_draws>0).sum())))
    assert_identity(identity)
    immutable_json(PUBLIC/'checkpoint_replay.json',dict(result_source='fresh_run_checkpoint_inference',
        analysis_sha256=digest(PUBLIC/'analysis.json'),checks=checks,sampler_checks=draws,all_passed=True,new_training=False))
    beat('protected_motion_replay_complete',heads=len(checks),sampler_groups=len(draws))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare','pilot','train','resume','evaluate','verify','replay'):parser.add_argument('--'+name,action='store_true')
    args = parser.parse_args()
    if not any((args.prepare,args.pilot,args.train,args.evaluate,args.verify,args.replay)):parser.error('Explicit phase required')
    if args.pilot and any((args.train,args.evaluate,args.verify,args.replay)):parser.error('Pilot is separate')
    for path in (PRIVATE,PUBLIC):
        if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Symlinked output destination')
        path.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        reg,data,identity,designs,qmask = load()
        beat('inputs_verified',rows=len(data['sites']),architecture=platform.machine(),threads=torch.get_num_threads(),workers=0)
        if args.train or args.pilot:train(reg,data,identity,designs,args.resume,args.pilot)
        if args.evaluate or args.verify:evaluate(reg,data,identity,designs,qmask,args.verify)
        if args.replay:replay(reg,data,identity,designs)
        beat('phase_complete')


if __name__=='__main__':main()
