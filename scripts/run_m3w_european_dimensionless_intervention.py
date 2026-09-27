"""Matched source-only risk heads on the frozen dimensionless forecast bank."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
import numpy as np
import torch
from scripts import run_m3w_european_dimensionless_refit as parent
from scripts.run_m3w_native_forecast import json_write, array_hash
from src.world_model import m3w_geometric_cost_head as head
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_fixed_producer_roles import role_indices
from src.world_model.m3w_floor_relative import relative_targets
from src.world_model.m3w_european_source_intervention import query_subset
from src.world_model.m3w_dimensionless_intervention import inference_features, allowed, joint_controls
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_partial_neighbor_refit import slice_metrics, masks
from src.evaluation.m3w_agent_track_refit import paired_localities

PUBLIC = parent.previous.BASE/'european_dimensionless_intervention_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_dimensionless_intervention_v1'
CONFIG = 'configs/m3w_european_dimensionless_intervention_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_dimensionless_intervention.py',
    'src/world_model/m3w_dimensionless_intervention.py', 'tests/test_m3w_dimensionless_intervention.py',
    'src/world_model/m3w_european_source_intervention.py', 'src/world_model/m3w_geometric_cost_head.py',
    'src/world_model/m3w_native_gain_harm.py', 'src/world_model/m3w_floor_relative.py',
    'src/world_model/m3w_fixed_producer_roles.py', 'src/world_model/m3w_scaled_risk_controls.py',
    'src/world_model/m3w_interaction_controls.py', 'src/world_model/m3w_joint_intervention.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, committed = parent.artifact, parent.digest, parent.immutable_json, parent.committed
FULL_POLICIES = ['raw', 'point', 'constant']
JOINT_POLICIES = ['point', 'scene_uniform', 'half_independent', 'half_unary', 'half_joint']


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def load(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    seal_path = parent.PUBLIC/'verification.json'
    assert digest(seal_path) == cfg['parent_seal_sha256']
    seal = json.loads(seal_path.read_text())
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p) == h, p
    _, _, data, jobs, _ = parent.load()
    roster = [sorted({s for j in jobs if j['old_identity']['fold'] == f for s in j['old_identity']['fit_sites']})
              for f in range(3)]
    assert cfg['seeds'] == [17, 29, 43] and cfg['head_count'] == 108
    assert cfg['candidates'] == ['dimensionless', 'damped'] and cfg['tasks'] == ['utility', 'all', 'easy']
    assert not any(cfg[k] for k in ('readout_tuning', 'independent_roles_read', 'deployment_changed',
                                    'stage5c_executed', 'smc_enabled'))
    frozen = json.loads((parent.PUBLIC/'prediction_freeze.json').read_text())
    assert len(frozen['predictions']) == len(jobs) == 9
    for j, ref in zip(jobs, frozen['predictions']):
        assert j['key'] == ref['key'] and artifact(ROOT/ref['dimensionless']['path']) == ref['dimensionless']
        j['prediction_ref'] = ref['dimensionless']
    qmask = query_subset(data['sites'], data['recordings'], data['frames'],
                         cfg['query_count_per_locality'], cfg['query_salt'])
    identity = dict(parent_seal=artifact(seal_path), predictions=artifact(parent.PUBLIC/'prediction_freeze.json'),
        bindings={p: digest(ROOT/p) for p in FILES}, rosters=roster,
        query_mask_sha256=array_hash(qmask), source_rows=len(data['sites']), query_rows=int(qmask.sum()))
    if create: immutable_json(PUBLIC/'registration.json', identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == identity
        committed(PUBLIC/'registration.json')
    return cfg, data, jobs, identity, qmask


def candidates(j, data):
    ids = j['design']['held_ids']
    cv = baseline_numpy(data['history'][ids], 1)-data['origin'][ids, None]
    with np.load(ROOT/j['prediction_ref']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        neural = z['prediction'].copy()
    damped = baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
    return ids, cv, dict(dimensionless=neural, damped=damped)


def group_name(j, controller, candidate):
    return j['key']+f'_controller{controller}_'+candidate


def read_done(path, identity):
    r = json.loads(path.read_text())
    assert r['identity']['experiment'] == identity and r['fit']['complete'] and r['fit']['step'] == 2000
    for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    return r


def constant_scores(pr, envelope, task):
    z = np.tile(pr['constant'], (len(envelope), 1))
    if task == 'utility':
        mass = z.sum(1)
        z *= np.minimum(1., np.divide(envelope, mass, out=np.ones(len(mass)), where=mass > 0))[:, None]
    else: z[:, 1] = np.minimum(z[:, 1], envelope)
    return z


def train(cfg, data, jobs, identity, *, resume, pilot=False, replay=False):
    refs = []; checks = []
    for j in jobs:
        bank_ids, cv_rollout, bank = candidates(j, data)
        for candidate, prediction in bank.items():
            x, envelope, _ = inference_features(data['geometry'][bank_ids], cv_rollout, prediction)
            for controller in range(3):
                if controller == j['old_identity']['fold']: continue
                roles = role_indices(data['sites'], identity['rosters'], j['old_identity']['fold'], controller)
                fit_ids, held_ids = roles['controller'], roles['readout']
                fit_pos, held_pos = np.searchsorted(bank_ids, fit_ids), np.searchsorted(bank_ids, held_ids)
                np.testing.assert_array_equal(bank_ids[fit_pos], fit_ids)
                np.testing.assert_array_equal(bank_ids[held_pos], held_ids)
                name = group_name(j, controller, candidate)
                cv = data['baseline_ade'][fit_ids, 1]
                errors = native_errors(prediction[fit_pos].astype(float)+data['origin'][fit_ids, None],
                    data['target_eval'][fit_ids], data['valid'][fit_ids], np.ones(len(fit_ids)))[0]
                utility, all_risk = relative_targets(cv, cv, errors, reference='cv', event='all', easy_cut=j['design']['easy_cut'])
                _, easy = relative_targets(cv, cv, errors, reference='cv', event='easy', easy_cut=j['design']['easy_cut'])
                for task, y in zip(cfg['tasks'], (utility, all_risk, easy)):
                    home = PRIVATE/'heads'/(name+'_'+task); done = home/'complete.json'
                    pr = preprocess(x[fit_pos], y, cv, data['sites'][fit_ids], '__excluded_readout__')
                    hid = dict(experiment=identity, group=name, task=task, seed=j['old_identity']['seed'],
                        train_ids_sha256=array_hash(fit_ids), train_features_sha256=array_hash(x[fit_pos]),
                        train_labels_sha256=array_hash(y), held_features_sha256=array_hash(held_ids, x[held_pos]),
                        easy_cut=j['design']['easy_cut'], prediction=j['prediction_ref'] if candidate == 'dimensionless' else 'causal_damping_index3')
                    if done.exists():
                        rec = read_done(done, identity); assert rec['identity'] == hid
                    else:
                        if replay: raise ValueError('Replay requires completed heads')
                        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; resume saved heads')
                        model, fit = head.fit(x[fit_pos], y, data['sites'][fit_ids], envelope[fit_pos], pr,
                            seed=j['old_identity']['seed'], task=task, settings=cfg['head_training'], identity=hid,
                            directory=home, resume=resume, stop_at=cfg['pilot_updates'] if pilot else None,
                            heartbeat=lambda **kw: beat(head=name+'_'+task, **kw))
                        if pilot:
                            immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'),
                                resumes_inside_budget=True)); return
                        scores = head.predict(model, x[held_pos], envelope[held_pos], pr)
                        const = constant_scores(pr, envelope[held_pos], task)
                        tmp = home/'scores.tmp.npz'; np.savez(tmp, ids=held_ids, scores=scores, constant=const)
                        os.replace(tmp, home/'scores.npz')
                        assert fit['unknown_rows_sampled'] == 0
                        rec = dict(identity=hid, fit=fit, cost_scale=pr['cost_scale'],
                            artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'), scores=artifact(home/'scores.npz')))
                        immutable_json(done, rec)
                        beat('head_complete', head=name+'_'+task, seconds=fit['seconds'])
                    refs.append(artifact(done))
                    if replay:
                        state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
                        assert state['identity'] == hid and state['step'] == 2000
                        for k in ('mean', 'std', 'weights', 'known', 'constant'):
                            np.testing.assert_array_equal(pr[k], state['preprocess'][k])
                        model = head.initialize_head(cfg['head_training']['width'], pr, j['old_identity']['seed'], task, state['mean_envelope'])
                        model.load_state_dict(state['model'])
                        with np.load(home/'scores.npz', allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['ids'], held_ids)
                            np.testing.assert_array_equal(z['scores'], head.predict(model, x[held_pos], envelope[held_pos], pr))
                            np.testing.assert_array_equal(z['constant'], constant_scores(pr, envelope[held_pos], task))
                        checks.append(dict(head=name+'_'+task, rows=len(held_ids), exact=True))
            del x
    assert len(refs) == cfg['head_count']
    immutable_json(PUBLIC/'training_freeze.json', dict(identity=identity, heads=refs, heads_count=len(refs),
        updates=len(refs)*2000, independent_roles_read=False, readout_targets_used=False))
    if replay: immutable_json(PUBLIC/'head_replay.json', dict(training=artifact(PUBLIC/'training_freeze.json'),
                                                           checks=checks, all_exact=True))


def scores(name, identity):
    learned, constant, scale = {}, {}, None
    for task in ('utility', 'all', 'easy'):
        home = PRIVATE/'heads'/(name+'_'+task)
        rec = read_done(home/'complete.json', identity)
        with np.load(home/'scores.npz', allow_pickle=False) as z:
            ids = z['ids'].copy(); learned[task] = z['scores'].copy(); constant[task] = z['constant'].copy()
        if scale is None: scale = rec['cost_scale']
        assert scale == rec['cost_scale']
    return ids, learned, constant, scale


def decision_group(cfg, data, j, controller, candidate, bank_ids, cv, prediction, identity, qmask):
    name = group_name(j, controller, candidate)
    ids, learned, constant, scale = scores(name, identity)
    roles = role_indices(data['sites'], identity['rosters'], j['old_identity']['fold'], controller)
    np.testing.assert_array_equal(ids, roles['readout'])
    pos = np.searchsorted(bank_ids, ids)
    moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
    point = allowed(learned['utility'], learned['all'], learned['easy'], moving, cfg['risk_budget'])
    const = allowed(constant['utility'], constant['all'], constant['easy'], moving, cfg['risk_budget'])
    take = np.flatnonzero(qmask[ids]); qids = ids[take]
    decisions = {k: np.zeros(len(take), bool) for k in JOINT_POLICIES}
    matched = np.zeros(len(take), bool); active = matched.copy()
    keys = np.column_stack((data['recordings'][qids], data['frames'][qids]))
    unique, inv = np.unique(keys, axis=0, return_inverse=True); reports = []
    for qi, key in enumerate(unique):
        loc = np.flatnonzero(inv == qi); ix = take[loc]; glob = ids[ix]; ip = pos[ix]
        dec, rpt = joint_controls(learned['utility'][ix], learned['all'][ix], learned['easy'][ix], moving[ix],
            data['origin'][glob], data['width'][glob], cv[ip].astype(float)+data['origin'][glob, None],
            prediction[ip].astype(float)+data['origin'][glob, None], scale, cfg)
        for k in decisions: decisions[k][loc] = dec[k]
        matched[loc] = rpt['matched']; active[loc] = rpt['nonadditive_supported_edges'] > 0
        reports.append(dict(site=str(data['sites'][glob[0]]), recording=int(key[0]), frame=int(key[1]), **rpt))
        if qi % 192 == 0: beat('joint_decisions', group=name, query=qi, total=len(unique))
    np.testing.assert_array_equal(decisions['point'], point[take])
    arrays = dict(ids=ids, point=point, constant=const, query_ids=qids, matched=matched, active=active,
                  **{'query_'+k: v for k, v in decisions.items()})
    return arrays, reports


def decide(cfg, data, jobs, identity, qmask, *, replay=False, resume=False):
    refs = []
    for j in jobs:
        bank_ids, cv, bank = candidates(j, data)
        for controller in range(3):
            if controller == j['old_identity']['fold']: continue
            for candidate, pred in bank.items():
                name = group_name(j, controller, candidate); home = PRIVATE/'decisions'
                home.mkdir(parents=True, exist_ok=True); path = home/(name+'.json')
                if path.exists() and not replay:
                    if not resume: raise ValueError('Existing decisions require --resume')
                    rec = json.loads(path.read_text()); assert rec['identity'] == identity
                    for ref in rec['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
                else:
                    arrays, reports = decision_group(cfg, data, j, controller, candidate, bank_ids, cv, pred, identity, qmask)
                    npz = path.with_suffix('.npz'); queries = home/(name+'_queries.json')
                    if replay:
                        with np.load(npz, allow_pickle=False) as z:
                            assert set(z.files) == set(arrays)
                            for k, v in arrays.items(): np.testing.assert_array_equal(z[k], v)
                        assert json.loads(queries.read_text()) == reports
                    else:
                        temp = npz.with_suffix('.tmp.npz'); np.savez(temp, **arrays); os.replace(temp, npz)
                        immutable_json(queries, reports)
                    immutable_json(path, dict(identity=identity, group=name, rows=len(arrays['ids']),
                        query_rows=len(arrays['query_ids']), queries=len(reports),
                        artifacts=dict(arrays=artifact(npz), queries=artifact(queries)), future_labels_used=False))
                    beat('decisions_replayed' if replay else 'decisions_frozen', group=name, queries=len(reports))
                refs.append(artifact(path))
    assert len(refs) == 36
    immutable_json(PUBLIC/'decision_freeze.json', dict(identity=identity, training=artifact(PUBLIC/'training_freeze.json'),
        groups=refs, future_labels_used=False, independent_roles_read=False))
    if replay: immutable_json(PUBLIC/'decision_replay.json', dict(decisions=artifact(PUBLIC/'decision_freeze.json'), all_exact=True))


def eval_group(cfg, data, j, controller, identity):
    bank_ids, cv_rollout, bank = candidates(j, data)
    roles = role_indices(data['sites'], identity['rosters'], j['old_identity']['fold'], controller)
    ids = roles['readout']; pos = np.searchsorted(bank_ids, ids)
    errors, decisions, query_reports = {}, {}, {}
    for candidate, p in bank.items():
        errors[candidate] = native_errors(p[pos].astype(float)+data['origin'][ids, None],
            data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
        name = group_name(j, controller, candidate)
        rec = json.loads((PRIVATE/'decisions'/(name+'.json')).read_text()); assert rec['identity'] == identity
        for ref in rec['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
        with np.load(ROOT/rec['artifacts']['arrays']['path'], allow_pickle=False) as z:
            decisions[candidate] = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(decisions[candidate]['ids'], ids)
        query_reports[candidate] = json.loads((ROOT/rec['artifacts']['queries']['path']).read_text())
    rows, contrasts, joint_rows = [], [], []
    subset = masks(data['baseline_ade'][ids, 1], j['design']['easy_cut'], j['design']['hard_cut'])
    roster = identity['rosters'][roles['readout_fold']]
    for endpoint, ei in [('ADE', 0), ('FDE', 1)]:
        cv = data['baseline_'+endpoint.lower()][ids, 1]
        ref = data['baseline_'+endpoint.lower()][ids, j['design']['baseline_index']]
        for scope, policies in [('full', FULL_POLICIES), ('query', JOINT_POLICIES)]:
            use = np.arange(len(ids)) if scope == 'full' else np.searchsorted(ids, decisions['dimensionless']['query_ids'])
            err, bits = {}, {}
            for candidate in bank:
                if scope == 'query': np.testing.assert_array_equal(decisions[candidate]['query_ids'], ids[use])
                for policy in policies:
                    choice = (np.ones(len(ids), bool) if policy == 'raw' else decisions[candidate][policy]) if scope == 'full' else decisions[candidate]['query_'+policy]
                    bits[candidate, policy] = choice
                    err[candidate, policy] = np.where(choice, errors[candidate][ei][use], cv[use])
            for site in roster:
                site_mask = data['sites'][ids[use]] == site
                for sub, full_mask in subset.items():
                    mask = site_mask & full_mask[use]
                    for candidate in bank:
                        for policy in policies:
                            a = err[candidate, policy]
                            m = slice_metrics(a, cv[use], ref[use], cv[use], mask)
                            m.update(switch_rate=float(bits[candidate, policy][mask].mean()) if mask.any() else None,
                                absolute_zero_harmed=int((mask & (cv[use] == 0) & (a > 0)).sum()))
                            rows.append(dict(trial=j['key'], controller=controller, candidate=candidate, policy=policy,
                                scope=scope, endpoint=endpoint, subset=sub, site=site, metric=m))
                    for policy in policies:
                        m = slice_metrics(err['dimensionless', policy], err['damped', policy], ref[use], cv[use], mask)
                        contrasts.append(dict(trial=j['key'], controller=controller, policy=policy, scope=scope,
                            endpoint=endpoint, subset=sub, site=site, metric=m))
                    if scope == 'query':
                        for candidate in bank:
                            for control in ('half_independent', 'half_unary'):
                                match = decisions[candidate]['matched']
                                for population, extra in [('all_queries', np.ones(len(use), bool)), ('matched', match),
                                        ('matched_nonadditive', match & decisions[candidate]['active'])]:
                                    m = slice_metrics(err[candidate, 'half_joint'], err[candidate, control],
                                        ref[use], cv[use], mask & extra)
                                    joint_rows.append(dict(trial=j['key'], controller=controller, candidate=candidate,
                                        control=control, endpoint=endpoint, subset=sub, population=population, site=site, metric=m))
    return rows, contrasts, joint_rows, query_reports


def evaluate(cfg, data, jobs, identity, *, verify=False):
    committed(PUBLIC/'decision_freeze.json')
    frozen = json.loads((PUBLIC/'decision_freeze.json').read_text()); assert frozen['identity'] == identity
    for r in frozen['groups']: assert artifact(ROOT/r['path']) == r
    rows, contrasts, joint_rows, queries = [], [], [], []
    for j in jobs:
        for controller in range(3):
            if controller == j['old_identity']['fold']: continue
            rr, cc, jj, qq = eval_group(cfg, data, j, controller, identity)
            rows.extend(rr); contrasts.extend(cc); joint_rows.extend(jj)
            for candidate, query in qq.items():
                queries.extend(dict(trial=j['key'], controller=controller, candidate=candidate, **q) for q in query)
            beat('readout_group', trial=j['key'], controller=controller)
    sites = sorted(set(data['sites']))
    def aggregate(table, keys, measures):
        groups = sorted(set(tuple(r[k] for k in keys) for r in table))
        return [{**dict(zip(keys, values)), 'metrics': {m: paired_localities(
            [r for r in table if tuple(r[k] for k in keys) == values], sites, m,
            cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for m in measures}} for values in groups]
    summary = aggregate(rows, ['scope','candidate','policy','endpoint','subset'],
        ['gain_vs_CV_percent','gain_vs_reference_percent','new_mean','mean_positive_harm_vs_legacy','new_p95','switch_rate'])
    paired = aggregate(contrasts, ['scope','policy','endpoint','subset'], ['gain_vs_legacy_percent'])
    joint_summary = aggregate(joint_rows, ['candidate','control','endpoint','subset','population'], ['gain_vs_legacy_percent'])
    def find(table, **kw): return next(r for r in table if all(r[k] == v for k, v in kw.items()))['metrics']
    p = find(paired, scope='full', policy='point', endpoint='ADE', subset='all')['gain_vs_legacy_percent']
    neural = find(summary, scope='full', candidate='dimensionless', policy='point', endpoint='ADE', subset='all')['gain_vs_CV_percent']
    easy = find(summary, scope='full', candidate='dimensionless', policy='point', endpoint='ADE', subset='positive_easy')['gain_vs_CV_percent']
    hard = find(summary, scope='full', candidate='dimensionless', policy='point', endpoint='ADE', subset='hard')['gain_vs_CV_percent']
    zero = [r['metric']['absolute_zero_harmed'] for r in rows if r['scope']=='full' and r['candidate']=='dimensionless'
            and r['policy']=='point' and r['endpoint']=='ADE' and r['subset']=='zero_CV']
    gates = dict(primary_neural_over_protected_damping=p['ci95'] is not None and p['ci95'][0] > 0,
        positive_neural_gain_over_CV=neural['point'] is not None and neural['point'] > 0,
        every_locality_easy_preserved=all(v is not None and v >= -2 for v in easy['by_site'].values()),
        no_zero_reference_harm=sum(zero)==0, hard_nonnegative=hard['point'] is not None and hard['point'] >= 0)
    gates['exploratory_useful_safe_screen'] = all(gates.values())
    gates.update(independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    result = dict(identity=identity, result_source='fresh_heads_and_decisions_cached_verified_forecasts',
        summaries=summary, neural_vs_protected_damping=paired, joint_contrasts=joint_summary, gates=gates,
        rows=rows, paired_rows=contrasts, joint_rows=joint_rows, query_diagnostics=queries,
        total_unique_source_rows=len(data['sites']), heads=108, head_updates=216000,
        uncertainty='exploratory_12_source_localities_not_independent_confirmation', future_inputs=False)
    immutable_json(PUBLIC/'evaluation.json', result)
    immutable_json(PUBLIC/'gates.json', gates)
    if verify: immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, evaluation_sha256=digest(PUBLIC/'evaluation.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['register','pilot','train','decide','replay_heads','replay_decisions','evaluate','verify_eval'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('phase_started', phase=args.phase)
        cfg, data, jobs, identity, qmask = load(create=args.phase=='register')
        if args.phase in ('pilot','train','replay_heads'):
            train(cfg, data, jobs, identity, resume=args.resume, pilot=args.phase=='pilot', replay=args.phase=='replay_heads')
        elif args.phase in ('decide','replay_decisions'):
            decide(cfg, data, jobs, identity, qmask, replay=args.phase=='replay_decisions', resume=args.resume)
        elif args.phase in ('evaluate','verify_eval'):
            evaluate(cfg, data, jobs, identity, verify=args.phase=='verify_eval')
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
