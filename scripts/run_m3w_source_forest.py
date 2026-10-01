"""Frozen European nonlinear cost-regression control on source-only partitions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_source_policy_selection as policy
from scripts import report_m3w_inner_separability as accounting
from scripts.report_m3w_source_checkpoint import locality_interval, query_average_squared_error
from scripts.export_m3w_easy_hurdle_create import closure
from src.world_model import m3w_source_forest as api
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds

parent = policy.parent
inner = parent.parent.parent
core = api.core
base = policy.base
digest, immutable = policy.digest, policy.immutable
NAME = 'european_source_forest_v1'
PUBLIC = parent.PUBLIC.parent / NAME
PRIVATE = parent.PRIVATE.parent / NAME
CONFIG = ROOT / 'configs' / ('m3w_' + NAME + '.json')
POLICIES = ('forest', 'neural', 'prior', 'forest_matched', 'neural_matched', 'screened', 'floor')


def registration():
    cfg = json.loads(CONFIG.read_text()); seal = parent.PUBLIC / 'verification.json'
    assert digest(seal) == cfg['parent_seal_sha256']
    v = json.loads(seal.read_text())
    for k, h in v['source_bindings'].items():
        assert digest(ROOT / k) == h
    for k, h in v['artifacts'].items():
        assert digest(parent.PUBLIC / k) == h
    paths = closure(ROOT, ['scripts.run_m3w_source_forest'])
    paths += [CONFIG, PUBLIC / 'protocol.md', ROOT / 'tests/test_m3w_source_forest.py']
    return cfg, dict(parent_seal_sha256=digest(seal),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in paths}, independent_roles_read=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE / 'heartbeat.json', row)
    with (PRIVATE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def guard(cfg):
    if shutil.disk_usage(PRIVATE).free < cfg['disk_reserve_bytes'] + 32 * 2**20:
        raise OSError('Preserve original10GiB disk reserve; resume or move to CREATE, do not downgrade')


def quality(p, y, pr, data, ids):
    score = api.signed_score(p, y, pr, data['sites'][ids], data['recordings'][ids], data['frames'][ids])
    independent = query_average_squared_error(p, y, pr['scale'], pr['rms'][5:],
                                               data['recordings'][ids], data['frames'][ids])
    np.testing.assert_allclose(score, independent, rtol=1e-10, atol=1e-10)
    return score


def validation(c, at, ids, x, env, y, val, neural, forest, data):
    p, support = parent.api.paired_predictions(neural, x[val], env[val])
    prior, check = core.predict(neural, x[val], env[val], initial=True)
    np.testing.assert_array_equal(support, check)
    fp, check = api.predict(forest, x[val], env[val])
    np.testing.assert_array_equal(support, check)
    p = dict(forest=fp, neural=p['validation'], final=p['final'], prior=prior)
    vid = ids[val]
    actions = {k: policy.causal_action(v, c['moving'][at][val], support,
        data['recordings'][vid], data['frames'][vid], vid) for k, v in p.items()}
    return dict(ids_hash=base.inter.array_hash(vid),
        scores={k: quality(v, y[val], neural['preprocess'], data, vid) for k, v in p.items()},
        prediction_hashes={k: base.inter.array_hash(v) for k, v in p.items()},
        action_hashes={k: base.inter.array_hash(v) for k, v in actions.items()},
        completion_screen=completion_bounds(y[val], actions['forest'], env[val]))


def train(cfg, data, jobs, oid, pilot=False, resume=False, replay=False):
    fits = policy.fit_docs(); refs = []; began = time.monotonic()
    causal = {k: data[k] for k in inner.old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal, jobs, oid):
        for site in inner.sources(c):
            guard(cfg); name = c['name'] + '_fit_' + site; doc = fits[name]
            neural = core.read_checkpoint(ROOT / doc['checkpoint']['path'])
            at, ids, x, env, y, _, upstream = inner.training_arrays(c, data, site)
            assert upstream == neural['identity']['upstream']
            tr, val, partition = parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
            assert partition == neural['partition']
            identity = dict(registration_sha256=digest(PUBLIC/'registration.json'),
                group=name, upstream=upstream, parent_checkpoint=doc['checkpoint'], partition=partition)
            home = PRIVATE / ('replay' if replay else 'heads') / name
            cp, done = home/'checkpoint.joblib', home/'complete.json'
            if done.exists() and resume and not replay:
                complete = json.loads(done.read_text()); assert complete['identity'] == identity
                assert base.artifact(cp) == complete['checkpoint']
            else:
                forest = api.fit(x[tr], env[tr], y[tr], data['sites'][ids][tr],
                    data['recordings'][ids][tr], data['frames'][ids][tr], neural['preprocess'],
                    settings=cfg['estimator'], seed=neural['seed'], identity=identity, path=cp,
                    heartbeat=lambda **kw: beat(state='forest_training', group=name, **kw),
                    resume=resume and not replay, stop_at=cfg['pilot_trees'] if pilot else None)
                if pilot:
                    # Bound each remaining forest by its source row count, not by the largest source alone.
                    ns = [d['partition']['train_rows'] for d in fits.values()]
                    leaf = cfg['estimator']['min_samples_leaf']; depth = cfg['estimator']['max_depth']
                    nodes = [min(2**(depth+1)-1, max(1, 2*(n//leaf)-1)) for n in ns]
                    upper = cfg['estimator']['trees'] * (sum(nodes)+2*max(nodes)) * 176 + 64*2**20
                    # Include row-count-scaled time only as an estimate; a large pilot may overestimate.
                    elapsed = time.monotonic()-began
                    estimate = forest['seconds'] * cfg['estimator']['trees']/cfg['pilot_trees'] * sum(ns)/int(tr.sum())
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                    immutable(PUBLIC/'pilot.json', dict(group=name, trees=cfg['pilot_trees'],
                        seconds=elapsed, fit_seconds=forest['seconds'], checkpoint_bytes=cp.stat().st_size,
                        peak_RSS_bytes=rss, estimated_full_fit_seconds=estimate, estimate_not_measurement=True,
                        conservative_remaining_storage_bytes=upper, free_bytes=shutil.disk_usage(PRIVATE).free,
                        storage_sufficient=shutil.disk_usage(PRIVATE).free-upper>cfg['disk_reserve_bytes'],
                        local_memory_feasible=rss<40*2**30, optimization_rows=int(tr.sum()),
                        known_optimization_rows=forest['known_training_rows']))
                    return
                if replay:
                    original = joblib.load(PRIVATE/'heads'/name/'checkpoint.joblib')
                    api.exact_forest(forest, original)
                    np.testing.assert_array_equal(api.predict(forest, x[val], env[val])[0],
                                                  api.predict(original, x[val], env[val])[0])
                    immutable(PUBLIC/'fit_replay.json', dict(group=name, exact_except_elapsed=True,
                        all72_retrained=False, trees=cfg['estimator']['trees'], seconds=time.monotonic()-began))
                    return
                result = validation(c, at, ids, x, env, y, val, neural, forest, data)
                complete = dict(identity=identity, checkpoint=base.artifact(cp), checkpoint_bytes=cp.stat().st_size, partition=partition,
                    source=site, seed=neural['seed'], known_training_rows=forest['known_training_rows'],
                    trees=len(forest['model'].estimators_), trace=forest['trace'], seconds=forest['seconds'],
                    validation=result)
                immutable(done, complete)
            refs.append(base.artifact(done))
            beat(state='forest_fit_complete', fits=len(refs), group=name)
    assert len(refs) == cfg['source_fits']
    immutable(PUBLIC/'training_freeze.json', dict(groups=refs, unique_fits=len(refs),
        trees=cfg['source_fits']*cfg['estimator']['trees'], seconds=time.monotonic()-began,
        result_source='fresh_run', independent_roles_read=False, deployment_changed=False))


def forest_docs():
    base.inter.committed(PUBLIC/'training_freeze.json')
    out = {}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['groups']:
        assert base.artifact(ROOT/ref['path']) == ref
        d = json.loads((ROOT/ref['path']).read_text())
        assert base.artifact(ROOT/d['checkpoint']['path']) == d['checkpoint']
        out[d['identity']['group']] = d
    assert len(out) == 72
    return out


def views(data, jobs, oid):
    docs = forest_docs()
    for c, at, ids, neural_p, _, pr, meta in parent.views(data, jobs, oid):
        doc = docs[c['name']+'_fit_'+meta['source']]
        state = joblib.load(ROOT/doc['checkpoint']['path'])
        fp, support = api.predict(state, c['x'][at], c['env'][at])
        old = doc['identity']['parent_checkpoint']
        assert base.artifact(ROOT/old['path']) == old
        neural = core.read_checkpoint(ROOT/old['path'])
        prior, check = core.predict(neural, c['x'][at], c['env'][at], initial=True)
        np.testing.assert_array_equal(support, check)
        pp = dict(forest=fp, neural=neural_p['validation'], prior=prior)
        action = core.decisions(dict(affine=pp['neural'], nonlinear=fp), c['moving'][at], support,
                               data['recordings'][ids], data['frames'][ids], ids)
        acts = {k: action[v] for k, v in [('forest','nonlinear'),('neural','affine'),
                    ('forest_matched','nonlinear_matched'),('neural_matched','affine_matched')]}
        acts['prior'] = policy.causal_action(prior, c['moving'][at], support, data['recordings'][ids], data['frames'][ids], ids)
        allowed = doc['validation']['completion_screen']['finite_completion_supported']
        acts['screened'] = acts['forest'].copy() if allowed else np.zeros(len(ids), bool)
        out = dict(view=meta['view'], site=meta['site'], source=meta['source'], seed=meta['seed'],
            ids_hash=meta['ids_hash'], source_screen_pass=allowed,
            prediction_hashes={k:base.inter.array_hash(v) for k,v in pp.items()},
            action_hashes={k:base.inter.array_hash(v) for k,v in acts.items()})
        yield c, at, ids, pp, acts, pr, out


def decide(data, jobs, oid):
    rows = []
    for *_, meta in views(data, jobs, oid):
        rows.append(meta)
        if len(rows)%36 == 0:
            beat(state='causal_action_freeze', views=len(rows))
    assert len(rows) == 216
    immutable(PUBLIC/'decision_freeze.json', dict(rows=rows, causal_only=True, threshold_search=False))


def evaluate(data, jobs, oid, replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json'); began = time.monotonic()
    frozen = {r['view']:r for r in json.loads((PUBLIC/'decision_freeze.json').read_text())['rows']}
    rows, scores = [], []; checks = queries = 0
    for c, at, ids, pp, actions, pr, meta in views(data, jobs, oid):
        assert meta == frozen[meta['view']]
        cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, at)
        y = core.targets(cv, floor, neural, c['job']['design']['easy_cut'])
        for arm, p in pp.items():
            scores.append(dict(view=meta['view'], site=meta['site'], seed=meta['seed'], arm=arm,
                               score=quality(p, y, pr, data, ids)))
        _, group = np.unique(np.rec.fromarrays([data['recordings'][ids],data['frames'][ids]]), return_inverse=True)
        count = {k:np.bincount(group,weights=v.astype(int)) for k,v in actions.items()}
        expected = np.minimum(count['neural'],count['forest'])
        np.testing.assert_array_equal(count['neural_matched'],expected)
        np.testing.assert_array_equal(count['forest_matched'],expected); queries += len(expected)
        known = np.isfinite(cv); easy = known & (cv>0) & (cv<=c['job']['design']['easy_cut'])
        for arm, take in {**actions,'floor':np.zeros(len(ids),bool)}.items():
            m = base.floor_api.metric(cv,floor,neural,cf,ff,nf,np.where(take,neural,floor),
                np.where(take,nf,ff),take,data['valid'][ids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            chosen = easy & take; ref = float(floor[chosen].sum())
            m['selected_easy_positive_harm_ratio'] = float(np.maximum(neural-floor,0)[chosen].sum())/ref if ref>0 else None
            checks += accounting.audit_metric(m,cv,floor,neural,cf,ff,nf,take,data['valid'][ids],
                                               c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            rows.append(dict(**meta,policy=arm,metric=m))
        if len(scores)%108 == 0:
            beat(state='development_transfer_readout',views=len(scores)//3)
    assert len(rows)==1512 and len(scores)==648
    immutable(PUBLIC/'readout.json',dict(rows=rows,quality=scores,independent_metric_checks=checks,
        query_count_checks=queries,independent_confirmation=False,deployment_changed=False))
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(seconds=time.monotonic()-began,exact_readout_replay=replay,all216_actions_reverified=True))


def report(cfg, reg):
    assert json.loads((PUBLIC/'evaluation_replay.json').read_text())['exact_readout_replay']
    assert json.loads((PUBLIC/'fit_replay.json').read_text())['exact_except_elapsed']
    docs = list(forest_docs().values()); readout = json.loads((PUBLIC/'readout.json').read_text())
    ci = lambda pairs: locality_interval(pairs,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    primary = ci([(d['source'],d['validation']['scores']['forest']-d['validation']['scores']['neural']) for d in docs])
    q = {(r['view'],r['arm']):r for r in readout['quality']}
    metrics = {(r['view'],r['policy']):r for r in readout['rows']}; contrast=[]; transfer=[]
    for (view, arm), r in q.items():
        if arm!='forest':continue
        transfer.append((r['site'],r['score']-q[view,'neural']['score']))
        a,b=(metrics[view,k]['metric']['error_sum'] for k in ('neural_matched','forest_matched'))
        contrast.append(dict(site=r['site'],seed=r['seed'],gain=100*(a-b)/a if a>0 else None))
    policies={}
    for p in POLICIES:
        rows=[r for r in readout['rows'] if r['policy']==p]
        policies[p]={k:ci([(r['site'],r['metric'][k]) for r in rows]) for k in
            ('all_gain_floor','easy_gain_floor','hard_gain_floor','FDE_gain_floor','intervention_rate')}
        for label,key in [('all','selected_positive_harm_ratio'),('easy','selected_easy_positive_harm_ratio')]:
            policies[p][label+'_risk']=accounting.risk_coverage(rows,key)
        policies[p]['worst_whole_easy_degradation_percent']=max(-r['metric']['easy_gain_floor'] for r in rows
            if r['metric']['easy_gain_floor'] is not None)
    summary=dict(diagnostic_primary=primary,source_validation_scores={k:ci([(d['source'],d['validation']['scores'][k])
        for d in docs]) for k in ('forest','neural','final','prior')},
        source_forest_better_than_neural=sum(d['validation']['scores']['forest']<d['validation']['scores']['neural'] for d in docs),
        source_forest_screen_pass=sum(d['validation']['completion_screen']['finite_completion_supported'] for d in docs),
        transferred_signed_MSE_difference=ci(transfer),
        matched_ADE_improvement=ci([(r['site'],r['gain']) for r in contrast]),
        by_seed={str(s):ci([(r['site'],r['gain']) for r in contrast if r['seed']==s]) for s in (17,29,43)},
        policies=policies,unique_fits=72,trees=9216,
        cumulative_fit_seconds=sum(d['seconds'] for d in docs),
        checkpoint_bytes=sum(d['checkpoint_bytes'] for d in docs),
        independent_metric_checks=readout['independent_metric_checks'],query_count_checks=readout['query_count_checks'],
        independent_quality_checks=648+288,result_source='fresh_run_with_exact_fit_and_readout_replay',
        independent_confirmation=False,deployment_changed=False)
    immutable(PUBLIC/'summary.json',summary)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_m3w_source_forest.py',
        'tests/test_m3w_source_checkpoint.py','tests/test_m3w_source_checkpoint_report.py',
        'tests/test_m3w_unknown_outcome_bounds.py','-q'],cwd=ROOT,capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(tests.stdout+tests.stderr)
    immutable(PUBLIC/'verification.json',dict(status='verified_development_estimator_control',source_bindings=reg['bindings'],
        artifacts={p.name:digest(p) for p in PUBLIC.iterdir() if p.is_file() and p.name!='verification.json'},
        first_fit_exact_replay=True,all216_readouts_exact_replay=True,full_legacy_suite='not_run',
        independent_confirmation=False,deployment_changed=False))
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','pilot','train','replay_fit','decide','evaluate','replay_evaluate','report'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    cfg,reg=registration()
    if a.phase=='register':immutable(PUBLIC/'registration.json',reg);print('Registered fixed European five-moment forest');return
    assert json.loads((PUBLIC/'registration.json').read_text())==reg
    base.inter.committed(PUBLIC/'registration.json')
    core.torch.set_num_threads(cfg['cpu_threads']);core.torch.set_num_interop_threads(cfg['interop_threads'])
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard(cfg);beat(state='starting',phase=a.phase)
        if a.phase=='report':report(cfg,reg);return
        _,_,data,jobs,oid,_,_,_=inner.old.load()
        if a.phase in ('pilot','train','replay_fit'):
            if a.phase=='train':
                pilot=json.loads((PUBLIC/'pilot.json').read_text())
                assert pilot['storage_sufficient'] and pilot['local_memory_feasible'], 'Use approved CREATE path after local resource blocker'
            train(cfg,data,jobs,oid,a.phase=='pilot',a.resume,a.phase=='replay_fit')
        elif a.phase=='decide':decide(data,jobs,oid)
        else:evaluate(data,jobs,oid,a.phase=='replay_evaluate')


if __name__=='__main__':main()
