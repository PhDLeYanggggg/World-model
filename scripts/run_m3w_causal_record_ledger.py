"""Evaluate a past-only recording budget ledger on the frozen A bank."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_nested_action_calibration as nested
from src.world_model import m3w_causal_record_ledger as model

run, np = nested.run, nested.np
PUBLIC = run.PUBLIC/'causal_ledger'; PRIVATE = run.PRIVATE/'causal_ledger'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_causal_record_ledger.py')
ARMS = ('raw', 'query_matched', 'ledger_matched', 'ledger_free', 'calibrated_filter', 'calibrated_ledger_free')
MATCHED = ARMS[:3]


def register():
    oracle = run.PUBLIC/'selected_positive_harm'/'oracle'
    result, replay = oracle/'analysis.json', oracle/'replay_execution.json'
    a = run.read(result)
    if run.read(replay)['artifact']['sha256'] != run.sha(result) or not a['no_label_aware_choices_exported']:
        raise ValueError('Require verified oracle diagnosis, never row decisions')
    parents = (result, replay, run.REG, Path(run.__file__), Path(run.model.__file__), nested.REG,
               nested.PUBLIC/'freeze.json', nested.PUBLIC/'analysis.json', Path(nested.cal.__file__),
               nested.base.REG, nested.base.PUBLIC/'freeze.json', nested.base.PUBLIC/'analysis.json')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES], parents=[run.artifact(p) for p in parents],
        arms=ARMS, original_budget=.02, query_cap=2, same_counts_arms=MATCHED,
        initialize_recording_slack=[0., 0.], chronological=True, independent_confirmation=False)))


def blocks(data, frames):
    out = {}
    for r in np.unique(data['recordings']):
        ix = np.flatnonzero(data['recordings']==r)
        q = data['queries'][ix]; ids = data['global_ids'][ix]; f = frames[ix]
        groups = [g[np.argsort(ids[g], kind='stable')] for g in run.old.prior.groups(q)]
        groups.sort(key=lambda g: f[g[0]])
        times = [f[g[0]] for g in groups]
        if (len(set(times)) != len(groups) or not all((f[g] == f[g[0]]).all() for g in groups)
                or len(np.unique(data['sites'][ix])) != 1):
            raise ValueError('Each query is one current frame within a single source recording')
        out[str(r)] = (ix, groups)
    return out


def select(d, refs, by_record, beat, replay):
    priors = {r['seed']:r for r in run.read(nested.base.PUBLIC/'freeze.json')['choices']}
    cal = {r['seed']:r for r in run.read(nested.PUBLIC/'freeze.json')['cells']}
    ids = d['global_ids']; files = []
    for cell in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = cell['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); coef = dict(zip(z['names'], z['coefficients']))
        with np.load(run.checked(priors[seed]['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); raw = z['choice']
        with np.load(run.checked(cal[seed]['decisions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); old_cal = z['choices'][3]
        margins = {r['source']: r['margin'] for r in cal[seed]['calibrators']}
        choice = np.full((len(ARMS),len(ids)), -1, dtype=np.int16); info = []
        for record, (ix,g) in by_record.items():
            source = str(d['sites'][ix[0]]); key = PRIVATE/f'{seed}_{record}'
            cache = key.with_suffix('.npz'); mark = key.with_suffix('.json')
            identity = dict(registration=run.sha(REG), seed=seed, recording=record, source=source)
            if replay or not mark.exists():
                h, cv, p = d['history'][ix], floor[ix], paths[ix]
                x, scale = run.model.descriptors(h,p,cv,d['crowd'][ix]); u,a,e,ok = run.model.decision_costs(x,coef[source],p,cv,scale)
                parent = nested.base.select(g,u,a,e,scale,ok); np.testing.assert_array_equal(parent, raw[ix])
                n = [(parent[j] >= 0).sum() for j in g]; value = u/scale[:,None]
                q, qi = model.stream(g,value,a,e,ok,counts=n,carry=False,witness=parent)
                l, li = model.stream(g,value,a,e,ok,counts=n,witness=parent)
                f, fi = model.stream(g,value,a,e,ok)
                radius = np.linalg.norm(p-cv[:,None],axis=-1).max(2)
                if margins[source] is None:
                    ac, ec = a, e; allowed = np.zeros_like(ok)
                else:
                    ac, ec = nested.cal.adjusted_excess('joint',a,e,radius,margins[source]); allowed = ok
                cp = nested.base.select(g,u,ac,ec,scale,allowed); np.testing.assert_array_equal(cp, old_cal[ix])
                cf, ci = model.stream(g,value,ac,ec,allowed)
                all_choices = np.stack((parent,q,l,f,cp,cf))
                if replay:
                    with np.load(run.checked(run.read(mark)['cache']),allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],ids[ix]); np.testing.assert_array_equal(z['choices'],all_choices)
                else:
                    tmp = cache.with_suffix('.pending.npz'); np.savez_compressed(tmp,ids=ids[ix],choices=all_choices); tmp.replace(cache)
                run.once(mark,run.stable(dict(identity=identity,cache=run.artifact(cache),
                    optimization=dict(query_matched=qi,ledger_matched=li,ledger_free=fi,calibrated_ledger_free=ci))))
            doc = run.read(mark)
            if doc['identity'] != identity:
                raise ValueError('Cannot resume another recording or calibration')
            with np.load(run.checked(doc['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix]); choice[:,ix] = z['choices']
            info.append(doc)
        cache = PRIVATE/f'{seed}_all.npz'
        if not cache.exists():
            tmp = cache.with_suffix('.pending.npz'); np.savez_compressed(tmp,ids=ids,choices=choice); tmp.replace(cache)
        else:
            with np.load(cache,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids); np.testing.assert_array_equal(z['choices'],choice)
        files.append(dict(seed=seed,choices=run.artifact(cache),recordings=info))
        beat(state='decisions_frozen',seed=seed,recordings=len(info),replay=replay)
    run.once(PUBLIC/'freeze.json',dict(seeds=files,before_outcome_readout=True,chronological=True,
                                       no_future_labels_or_later_queries_in_decisions=True))


def score(d, refs, by_record, beat):
    freeze = run.read(PUBLIC/'freeze.json')
    if len(freeze['seeds']) != 3 or not freeze['before_outcome_readout']:
        raise ValueError('Require all 3 frozen banks')
    label = run.old.prior.arrays(('target_eval','valid')); y = label['target_eval']-d['origin'][:,None]; valid = label['valid']
    sites, q = d['sites'], d['queries']; p = run.parent; cuts = run.cuts_by_source(); records = []
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(query_vs_raw=('raw','query_matched',p.CV), ledger_vs_raw=('raw','ledger_matched',p.CV),
        ledger_vs_query=('query_matched','ledger_matched',p.CV), calibrated_ledger_vs_filter=('calibrated_filter','calibrated_ledger_free',p.CV))
    old = {(r['seed'],r['recording']): r['metrics']['filter'] for r in run.read(nested.base.PUBLIC/'analysis.json')['records']}
    old_cal = {(r['seed'],r['recording']): r['metrics']['joint'] for r in run.read(nested.PUBLIC/'analysis.json')['records']}
    for cell in freeze['seeds']:
        seed = cell['seed']; cv, bank = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids']); choices = z['choices']
        for r,(ix,groups) in by_record.items():
            ch = choices[:,ix]; s = str(sites[ix[0]]); c = cuts[s]
            for g in groups:
                n = (ch[:len(MATCHED),g]>=0).sum(1)
                if not (n == n[0]).all():
                    raise ValueError('One query count changed in the matched comparison')
            paths = {a:run.model.selected_paths(z,bank[ix],cv[ix]) for a,z in zip(ARMS,ch)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],q[ix],r,
                easy_cut=c['easy'],hard_cut=c['hard'])
            if out['metrics']['raw'] != old[seed,r] or out['metrics']['calibrated_filter'] != old_cal[seed,r]:
                raise ValueError('Both pointwise controls must replay exactly')
            interaction = {}
            for a in MATCHED:
                vals = [run.model.pair_error(d['history'][ix[g]],paths[a][g],y[ix[g]],valid[ix[g]]) for g in groups]
                interaction[a] = dict(error_sum=float(sum(v[0] for v in vals if v[0] is not None)),
                    supported_queries=sum(v[0] is not None for v in vals), observed_pair_steps=sum(v[1] for v in vals))
            records.append(dict(seed=seed,source=s,interaction=interaction,**out))
        beat(state='scored_seed',seed=seed,views=len(records))
    if len(records) != 102:
        raise ValueError('All required recording views')
    names = np.unique(sites); draw = np.random.default_rng(73107).integers(len(names),size=(3000,len(names)))
    grouped = [[r for r in records if r['source']==s and r['seed']==seed] for s in names for seed in run.kin.prior.SEEDS]
    def ci(v):
        if any(x is None for x in v):
            return None
        z = np.asarray(v).reshape(len(names),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    summary = {}
    for k in pairs:
        m = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in grouped]
        summary[k] = {f:ci([v[f] for v in m]) for f in ('known_pp','completion_lower_pp')}
    def pair_delta(a,b):
        def value(g):
            n = sum(r['interaction'][a]['supported_queries'] for r in g)
            return sum(r['interaction'][a]['error_sum']-r['interaction'][b]['error_sum'] for r in g)/n if n else None
        return ci([value(g) for g in grouped])
    risks = {}
    for a in ARMS:
        views = [r['metrics'][a]['risk'] for r in records]
        risks[a] = dict(passes=sum(v['supported'] for v in views),switches=sum(v['selected'] for v in views),
            known_easy_over_2=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in views),
            easy_over_2=sum(v['completion_easy_risk_upper'] is not None and v['completion_easy_risk_upper']>.02 for v in views),
            undefined_easy=sum(v['completion_easy_risk_upper'] is None for v in views),
            unknown_switches=sum(v['unknown_selected'] for v in views))
    opt = {a:[r['optimization'][a] for f in freeze['seeds'] for r in f['recordings']] for a in
           ('query_matched','ledger_matched','ledger_free','calibrated_ledger_free')}
    opt_summary = {a:dict(queries_using_credit=sum(v['queries_using_credit'] for v in rows),
        witness_fallback_queries=sum(v['witness_fallback_queries'] for v in rows),
        min_record_final_slack=np.min([v['predicted_final_slack'] for v in rows],axis=0).tolist()) for a,rows in opt.items()}
    run.once(PUBLIC/'analysis.json',run.stable(dict(comparisons=summary,risk=risks,records=records,
        observed_pair_error_reduction_ledger_vs_query=pair_delta('query_matched','ledger_matched'),
        observed_pair_error_reduction_ledger_vs_raw=pair_delta('raw','ledger_matched'),optimization=opt_summary,
        both_controls_exact=True,chronological=True,original_rules_unchanged=True,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','freeze','decision_replay','score','replay'))
    phase = p.parse_args().phase; register()
    if phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the causal allocation design first')
    PRIVATE.mkdir(parents=True,exist_ok=True); start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from saved recording decisions')
            pkt = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',pkt); print(pkt,flush=True)
        beat(state='start'); d = run.population(); refs = run.register()
        by_record = blocks(d,run.old.prior.arrays(('frames',))['frames']); decisions = phase in ('freeze','decision_replay')
        if decisions:
            select(d,refs,by_record,beat,phase=='decision_replay')
        else:
            score(d,refs,by_record,beat)
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifact=run.artifact(PUBLIC/('freeze.json' if decisions else 'analysis.json'))))


if __name__ == '__main__':
    main()
