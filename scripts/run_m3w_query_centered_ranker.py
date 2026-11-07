"""Matched current-query ranking on the fixed full forecast bank."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_resolved_motion_feedback as base
from src.world_model import m3w_query_centered_ranker as ranker

run,np,prior = base.run,base.np,base.prior
PUBLIC,PRIVATE = prior.PUBLIC/'query_ranker',prior.PRIVATE/'query_ranker'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(ranker.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_query_centered_ranker.py')
ARMS = ('static','pointwise','centered')


def register():
    paths = (prior.REG,prior.PUBLIC/'freeze.json',prior.PUBLIC/'analysis.json',
        prior.PUBLIC/'replay_execution.json',run.REG,run.PUBLIC/'freeze.json',
        Path(prior.model.__file__),Path(run.model.__file__),Path(base.__file__),
        Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths],arms=ARMS,seeds=run.kin.prior.SEEDS,
        ridge=.002,same_query_counts=True,original_risk_limit=.02,query_bootstrap=False,
        paired_source_bootstraps=3000,independent_outcomes_opened=False)))


def fit(d,refs,by_record,beat):
    ids,sites = d['global_ids'],d['sites']
    split = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    control = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    control_fits = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cells = []
    for seed in run.kin.prior.SEEDS:
        mark = PRIVATE/f'{seed}.json'
        if not mark.exists():
            cv,bank = run.candidates(d,refs,seed)
            with np.load(run.checked(control[seed]['choices']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                raw,static = z['choices'][[0,2]]
            with np.load(run.checked(control_fits[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                prior_weights = dict(zip(z['names'],z['coefficients']))
            moments = {}
            for s,ix in split.items():
                c,b = cv[ix],bank[ix]
                x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
                y = ranker.reward(b,c,labels['target_eval'][ix]-d['origin'][ix,None],
                                  labels['valid'][ix],scale,cuts[s]['easy'])
                legal = np.any(b!=c[:,None],axis=(2,3))
                moments[s] = ranker.moments(x,y,legal,d['queries'][ix])
            weights = [{s:ranker.fit([v[j] for t,v in moments.items() if s!=t],bank.shape[1])
                        for s in split} for j in range(2)]
            choice = np.full((len(ARMS),len(ids)),-1,np.int16)
            choice[0] = static
            fallbacks = [0,0]
            for ix,groups in by_record.values():
                s = sites[ix[0]]
                c,b = cv[ix],bank[ix]
                x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
                _,a,e,legal = run.model.decision_costs(x,prior_weights[s],b,c,scale)
                count = [(raw[ix[g]]>=0).sum() for g in groups]
                for j,w in enumerate(weights):
                    v = ranker.score(x,w[s])
                    choice[j+1,ix],info = prior.model.stream(groups,v,a,e,legal,counts=count,witness=raw[ix])
                    fallbacks[j] += info['witness_fallback_queries']
            path = PRIVATE/f'{seed}.npz'
            base.write_array(path,ids=ids,choices=choice,sources=np.asarray(list(split)),
                coefficients=np.stack([[w[s] for s in split] for w in weights]))
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(path),
                               fits=2*len(split),witness_fallbacks=fallbacks))
        c = run.read(mark)
        if c['registration']!=run.sha(REG) or c['seed']!=seed:
            raise ValueError('Saved source cross-fit identity mismatch')
        run.checked(c['cache'])
        cells.append(c)
        beat(state='seed_frozen',seed=seed)
    run.once(PUBLIC/'freeze.json',dict(cells=cells,decisions_frozen_before_scoring=True))


def score(d,refs,by_record,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['decisions_frozen_before_scoring']:
        raise ValueError('Require all three frozen banks')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(centered_vs_pointwise=('pointwise','centered',p.CV),
                 centered_vs_static=('static','centered',p.CV),
                 pointwise_vs_static=('static','pointwise',p.CV))
    old = {(r['seed'],r['recording']):r['metrics']['ledger_matched']
           for r in run.read(prior.PUBLIC/'analysis.json')['records']}
    records = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choice = z['choices']
        for r,(ix,groups) in by_record.items():
            ch = choice[:,ix]
            for g in groups:
                count = (ch[:,g]>=0).sum(1)
                if not (count==count[0]).all():
                    raise ValueError('Exact query counts required')
            s = str(d['sites'][ix[0]])
            paths = {a:run.model.selected_paths(z,bank[ix],cv[ix]) for a,z in zip(ARMS,ch)}
            metrics = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            if metrics['metrics']['static']!=old[seed,r]:
                raise ValueError('Frozen static control failed exact replay')
            interaction = {}
            for a in ARMS:
                v = [run.model.pair_error(d['history'][ix[g]],paths[a][g],y[ix[g]],valid[ix[g]]) for g in groups]
                interaction[a] = dict(total=float(sum(e for e,_ in v if e is not None)),
                                      queries=sum(e is not None for e,_ in v))
            change = dict(replaced_rows=int(((ch[1]>=0)!=(ch[2]>=0)).sum()),
                          same_row_changed_action=int(((ch[1]>=0)&(ch[2]>=0)&(ch[1]!=ch[2])).sum()))
            records.append(dict(seed=seed,source=s,**metrics,interaction=interaction,change=change))
        beat(state='seed_scored',seed=seed)
    if len(records)!=102:
        raise ValueError('Every recording view must be represented')
    sources = np.unique(d['sites'])
    groups = [[r for r in records if r['source']==s and r['seed']==seed]
              for s in sources for seed in run.kin.prior.SEEDS]
    draws = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draws].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        x = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in groups]
        comparisons[k] = {key:ci([v[key] for v in x]) for key in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risk = {}
    for a in ARMS:
        rows = [r['metrics'][a]['risk'] for r in records]
        risk[a] = dict(passes=sum(x['supported'] for x in rows),switches=sum(x['selected'] for x in rows),
            unknown_switches=sum(x['unknown_selected'] for x in rows),
            all_over_2=sum(x['completion_all_risk_upper'] is not None and x['completion_all_risk_upper']>.02 for x in rows),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in rows),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in rows),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in rows))
    def hard(arm,g):
        z = [r['metrics'][arm]['subsets']['hard'] for r in g if r['metrics'][arm]['subsets']['hard']['known_rows']]
        ref = sum(v['floor_ADE_pixel']*v['known_rows'] for v in z)
        return 100*sum((v['floor_ADE_pixel']-v['model_ADE_pixel'])*v['known_rows'] for v in z)/ref if ref else None
    def pair_delta(g):
        n = sum(r['interaction']['pointwise']['queries'] for r in g)
        return sum(r['interaction']['pointwise']['total']-r['interaction']['centered']['total'] for r in g)/n if n else None
    change = {k:sum(r['change'][k] for r in records) for k in ('replaced_rows','same_row_changed_action')}
    run.once(PUBLIC/'analysis.json',run.stable(dict(comparisons=comparisons,risk=risk,records=records,
        pair_error_reduction_centered_vs_pointwise=ci([pair_delta(g) for g in groups]),
        hard_known_gain_vs_cv={a:ci([hard(a,g) for g in groups]) for a in ARMS},change=change,
        original_static_control_exact=True,original_risk_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','fit','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for file in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(file.relative_to(ROOT))],cwd=ROOT)!=file.read_bytes():
            raise ValueError('Commit the design before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from the last frozen seed')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        beat(state='start')
        d,refs = run.population(),run.register()
        by_record = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (fit if phase=='fit' else score)(d,refs,by_record,beat)
        target = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                      artifact=run.artifact(target)))


if __name__=='__main__':
    main()
