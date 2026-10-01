"""Registered source-only component calibration; no target-selected rule."""
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
from scripts import run_m3w_crossed_head_seed as parent
from src.world_model import m3w_component_calibration as api

NAME = 'european_component_calibration_v1'
PUBLIC = parent.PUBLIC.parent / NAME; PRIVATE = parent.PRIVATE.parent / NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
base, core, inner = parent.base, parent.core, parent.inner
forest = parent.parent
digest, immutable = parent.digest, parent.immutable


def registration():
    cfg = json.loads(CONFIG.read_text()); parent.registration()
    seal = parent.PUBLIC/'verification.json'
    assert digest(seal) == cfg['parent_seal_sha256']
    v = json.loads(seal.read_text())
    for name, h in v['source_bindings'].items(): assert digest(ROOT/name) == h
    for name, h in v['artifacts'].items(): assert digest(parent.PUBLIC/name) == h
    paths = forest.closure(ROOT, ['scripts.run_m3w_component_calibration'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_component_calibration.py']
    return cfg, dict(parent_seal_sha256=digest(seal), bindings={str(p.relative_to(ROOT)):digest(p) for p in paths},
        independent_roles_read=False, neural_parameter_updates=0)


def beat(**kw):
    d = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', d)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(d)+'\n')
    print(json.dumps(d), flush=True)


def guard(cfg):
    if shutil.disk_usage(PRIVATE).free < cfg['disk_reserve_bytes'] + 32*2**20:
        raise OSError('Preserve10GiB reserve; use isolated CREATE if required')


def fit(cfg, data, jobs, oid, pilot=False, replay=False, resume=False):
    fits = parent.docs(); refs = []; began = time.monotonic()
    for c in parent.contexts(data, jobs, oid):
        for site in inner.sources(c):
            group = c['name']+'_fit_'+site
            at, ids, x, env, y, _, upstream = inner.training_arrays(c, data, site)
            _, val, partition = forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
            vid = ids[val]
            for seed in parent.api.SEEDS:
                guard(cfg); original = fits[group, seed]
                state = joblib.load(ROOT/original['checkpoint']['path'])
                assert state['identity']['upstream'] == upstream and original['partition'] == partition
                p, support = forest.api.predict(state, x[val], env[val])
                assert base.inter.array_hash(p) == original['validation']['prediction_hashes']['forest']
                moving = c['moving'][at][val]
                raw = api.eligible(p, moving, support)
                assert base.inter.array_hash(raw) == original['validation']['action_hashes']['forest']
                identity = dict(group=group, head_seed=seed, checkpoint=original['checkpoint'],
                    registration_sha256=digest(PUBLIC/'registration.json'), partition=partition,
                    validation_ids_hash=base.inter.array_hash(vid), source=site,
                    target_hash=base.inter.array_hash(y[val]), prediction_hash=base.inter.array_hash(p),
                    envelope_hash=base.inter.array_hash(env[val]))
                path = PRIVATE/'fits'/(group+'_head'+str(seed)+'.json')
                if path.exists() and not (resume or replay):
                    raise ValueError('Existing calibration requires explicit resume or replay')
                if path.exists() and resume and not replay:
                    record = json.loads(path.read_text()); assert record['identity'] == identity
                else:
                    record, oof = api.calibrate(p, y[val], env[val], moving, support,
                                               data['recordings'][vid], cfg['empirical_quantile'])
                    record.update(identity=identity, oof_action_hashes={m:base.inter.array_hash(a) for m,a in oof.items()},
                        parent_source_screen=original['validation']['completion_screen']['finite_completion_supported'])
                    immutable(path, record)
                refs.append(base.artifact(path))
                beat(state='source_calibration', heads=len(refs), group=group, head_seed=seed)
                if pilot:
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                    needed = 72*path.stat().st_size*2+20*2**20
                    immutable(PUBLIC/'pilot.json', dict(seconds=time.monotonic()-began, peak_RSS_bytes=rss,
                        calibration_bytes=path.stat().st_size, estimated_storage_bytes=needed,
                        storage_sufficient=shutil.disk_usage(PRIVATE).free-needed>cfg['disk_reserve_bytes'],
                        memory_feasible=rss<40*2**30, one_real_source_calibration=True))
                    return
    assert len(refs) == 72
    immutable(PUBLIC/'calibration_freeze.json', dict(groups=refs, heads=72,
        correction_arms=list(api.MODES), source_only=True, independent_calibration=False))
    immutable(PUBLIC/('calibration_replay.json' if replay else 'calibration_runtime.json'),
        dict(seconds=time.monotonic()-began, exact_all72_replay=replay))


def docs():
    base.inter.committed(PUBLIC/'calibration_freeze.json')
    out = {}
    for ref in json.loads((PUBLIC/'calibration_freeze.json').read_text())['groups']:
        assert base.artifact(ROOT/ref['path']) == ref
        d = json.loads((ROOT/ref['path']).read_text()); k=d['identity']
        out[k['group'], k['head_seed']] = d
    assert len(out) == 72
    return out


def views(data, jobs, oid):
    fitted = docs()
    for c, at, ids, predictions, previous, pr, oldmeta in parent.views(data, jobs, oid):
        support = forest.api.causal_inputs(c['x'][at], c['env'][at], pr)[1]
        for seed in parent.api.SEEDS:
            record = fitted[c['name']+'_fit_'+oldmeta['source'], seed]
            p = predictions[seed]; e=c['env'][at]
            q = {m:api.adjust(p, e, record['final'], m) for m in api.MODES}
            raw = api.eligible(p, c['moving'][at], support)
            np.testing.assert_array_equal(raw, previous['seed'+str(seed)])
            act = dict(raw=raw, parent=previous['screen'+str(seed)])
            screens = {}
            for mode in api.MODES:
                take = api.eligible(q[mode], c['moving'][at], support)
                assert not (take & ~raw).any()
                screens[mode] = record['source'][mode]['oof']['finite_completion_supported']
                act[mode] = take
                act[mode+'_screen'] = take.copy() if screens[mode] else np.zeros(len(ids), bool)
            act['parent_matched'],act['joint_matched'] = api.matched(act['parent'], act['joint_screen'],
                core.signed(p)[:,0], core.signed(q['joint'])[:,0], data['recordings'][ids],data['frames'][ids],ids)
            meta = dict(view=oldmeta['view']+'_head'+str(seed), parent_view=oldmeta['view'],
                site=oldmeta['site'], source=oldmeta['source'], head_seed=seed, upstream_seed=43,
                ids_hash=oldmeta['ids_hash'], raw_prediction_hash=oldmeta['prediction_hashes'][str(seed)],
                adjusted_hashes={m:base.inter.array_hash(z) for m,z in q.items()},
                action_hashes={k:base.inter.array_hash(v) for k,v in act.items()},
                source_screens=screens, raw_count=int(raw.sum()),
                zero_joint_easy_reference=int((q['joint'][:,3]<=0).sum()))
            yield c, at, ids, act, meta


def decide(data,jobs,oid):
    rows=[]
    for *_, meta in views(data,jobs,oid):
        rows.append(meta)
        if len(rows)%36==0: beat(state='causal_action_freeze', views=len(rows))
    assert len(rows)==216
    immutable(PUBLIC/'decision_freeze.json',dict(rows=rows,causal_only=True,transfer_threshold_selection=False))


def evaluate(data,jobs,oid,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json'); began=time.monotonic()
    frozen={v['view']:v for v in json.loads((PUBLIC/'decision_freeze.json').read_text())['rows']}
    old={(v['view'],v['policy']):v['metric'] for v in json.loads((parent.PUBLIC/'readout.json').read_text())['rows']}
    rows=[];checks=queries=parent_checks=0;last=None;costs=None
    for c,at,ids,actions,meta in views(data,jobs,oid):
        assert meta==frozen[meta['view']]
        if last!=meta['parent_view']:
            costs=base.floor_api.costs(c,data,at);last=meta['parent_view']
        cv,cf,(floor,ff),(neural,nf)=costs
        easy=np.isfinite(cv)&(cv>0)&(cv<=c['job']['design']['easy_cut'])
        _,groups=np.unique(np.rec.fromarrays([data['recordings'][ids],data['frames'][ids]]),return_inverse=True)
        counts={k:np.bincount(groups,weights=a.astype(int)) for k,a in actions.items()}
        want=np.minimum(counts['parent'],counts['joint_screen'])
        np.testing.assert_array_equal(counts['parent_matched'],want)
        np.testing.assert_array_equal(counts['joint_matched'],want);queries+=len(want)
        for name,take in {**actions,'floor':np.zeros(len(ids),bool)}.items():
            m=base.floor_api.metric(cv,floor,neural,cf,ff,nf,np.where(take,neural,floor),np.where(take,nf,ff),take,
                data['valid'][ids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            chosen=easy&take;ref=float(floor[chosen].sum())
            m['selected_easy_positive_harm_ratio']=float(np.maximum(neural-floor,0)[chosen].sum())/ref if ref>0 else None
            checks+=forest.accounting.audit_metric(m,cv,floor,neural,cf,ff,nf,take,data['valid'][ids],
                c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            if name in ('raw','parent','floor'):
                key={'raw':'seed'+str(meta['head_seed']),'parent':'screen'+str(meta['head_seed']),'floor':'floor'}[name]
                core.exact(m,old[meta['parent_view'],key]);parent_checks+=1
            rows.append(dict(view=meta['view'],site=meta['site'],source=meta['source'],head_seed=meta['head_seed'],policy=name,metric=m))
        if len(rows)%396==0: beat(state='outcome_readout',views=len(rows)//11)
    assert len(rows)==2376 and parent_checks==648
    immutable(PUBLIC/'readout.json',dict(rows=rows,independent_metric_checks=checks,query_count_checks=queries,
        parent_metric_views_exact=parent_checks,independent_confirmation=False,deployment_changed=False))
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(seconds=time.monotonic()-began,exact_readout_replay=replay))


def report(cfg,reg):
    assert json.loads((PUBLIC/'calibration_replay.json').read_text())['exact_all72_replay']
    assert json.loads((PUBLIC/'evaluation_replay.json').read_text())['exact_readout_replay']
    d=json.loads((PUBLIC/'readout.json').read_text());rows=d['rows'];fitted=docs()
    ci=lambda pairs:forest.locality_interval(pairs,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    m={(r['view'],r['policy']):r for r in rows}
    policies={}
    for name in ('raw','parent','harm','reference','joint','harm_screen','reference_screen','joint_screen','parent_matched','joint_matched','floor'):
        rr=[r for r in rows if r['policy']==name]
        z={k:ci([(r['site'],r['metric'][k]) for r in rr]) for k in
            ('all_gain_floor','hard_gain_floor','easy_gain_floor','FDE_gain_floor','intervention_rate')}
        z.update(easy_risk=forest.accounting.risk_coverage(rr,'selected_easy_positive_harm_ratio'),
                 all_risk=forest.accounting.risk_coverage(rr,'selected_positive_harm_ratio'),
                 worst_whole_easy_degradation_percent=max(-r['metric']['easy_gain_floor'] for r in rr if r['metric']['easy_gain_floor'] is not None))
        policies[name]=z
    contrasts=[(r['site'],100*(r['metric']['error_sum']-m[v,'joint_matched']['metric']['error_sum'])/r['metric']['error_sum'])
        for (v,k),r in m.items() if k=='parent_matched']
    source={mode:dict(oof_screen_pass=sum(v['source'][mode]['oof']['finite_completion_supported'] for v in fitted.values()),
        resubstitution_screen_pass=sum(v['source'][mode]['full_fit_resubstitution']['finite_completion_supported'] for v in fitted.values())) for mode in api.MODES}
    summary=dict(policies=policies,matched_ADE_improvement=ci(contrasts),source_screens=source,
        full_calibrators_supported=sum(v['final']['supported'] for v in fitted.values()),
        total_oof_folds=sum(len(v['folds']) for v in fitted.values()),
        unsupported_oof_folds=sum(not f['calibration']['supported'] for v in fitted.values() for f in v['folds']),
        by_head_seed={str(s):{k:forest.accounting.risk_coverage([r for r in rows if r['head_seed']==s and r['policy']==k],
            'selected_easy_positive_harm_ratio') for k in ('parent','harm_screen','reference_screen','joint_screen')} for s in parent.api.SEEDS},
        calibration_models=72,neural_parameter_updates=0,independent_confirmation=False,deployment_changed=False,
        independent_metric_checks=d['independent_metric_checks'],query_count_checks=d['query_count_checks'],parent_metric_views_exact=d['parent_metric_views_exact'])
    immutable(PUBLIC/'summary.json',summary)
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_m3w_component_calibration.py','tests/test_m3w_crossed_seed.py',
        'tests/test_m3w_source_forest.py','tests/test_m3w_unknown_outcome_bounds.py','-q'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(test.stdout+test.stderr)
    immutable(PUBLIC/'verification.json',dict(source_bindings=reg['bindings'],
        artifacts={p.name:digest(p) for p in PUBLIC.iterdir() if p.is_file() and p.name!='verification.json'},
        all72_calibrators_exact_replay=True,full_readout_exact_replay=True,independent_confirmation=False,
        deployment_changed=False,full_legacy_suite='not_run'))
    print(json.dumps(dict(calibrators=72,views=216,verified=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','pilot','fit','replay_fit','decide','evaluate','replay_evaluate','report'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    cfg,reg=registration()
    if a.phase=='register':immutable(PUBLIC/'registration.json',reg);print('Registered component calibration control');return
    assert json.loads((PUBLIC/'registration.json').read_text())==reg
    base.inter.committed(PUBLIC/'registration.json')
    core.torch.set_num_threads(4);core.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard(cfg);beat(state='starting',phase=a.phase)
        if a.phase=='report':report(cfg,reg);return
        _,_,data,jobs,oid,_,_,_=inner.old.load()
        if a.phase in ('pilot','fit','replay_fit'):
            if a.phase=='fit':
                pilot=json.loads((PUBLIC/'pilot.json').read_text());assert pilot['storage_sufficient'] and pilot['memory_feasible']
            fit(cfg,data,jobs,oid,a.phase=='pilot',a.phase=='replay_fit',a.resume)
        elif a.phase=='decide':decide(data,jobs,oid)
        else:evaluate(data,jobs,oid,a.phase=='replay_evaluate')


if __name__=='__main__':main()
