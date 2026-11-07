"""Learn to correct a donor proposal whose score was fitted outside both sources."""
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
from scripts import run_m3w_feasible_teacher as base

run,np,prior = base.run,base.np,base.prior
PUBLIC,PRIVATE = base.PUBLIC/'nested',base.PRIVATE/'nested'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),PUBLIC/'protocol.md')
ARMS = base.ARMS


def register():
    paths = (base.REG,base.PUBLIC/'analysis.json',base.PUBLIC/'replay_execution.json',
             Path(base.__file__),Path(base.teacher.__file__),Path(base.parent.ranker.__file__),
             Path(prior.model.__file__),Path(run.model.__file__),prior.REG,run.REG)
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths],arms=ARMS,seeds=run.kin.prior.SEEDS,
        donor_proposal_excludes_both_sources=True,original_budget=.02,
        paired_source_draws=3000,independent_outcomes_opened=False)))


def fit(d,refs,by_record,beat):
    ids,sites = d['global_ids'],d['sites']
    splits = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    cut = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    old = {c['seed']:c for c in run.read(prior.PUBLIC/'freeze.json')['seeds']}
    old_heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cells = []
    for seed in run.kin.prior.SEEDS:
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(old[seed]['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            raw,original = z['choices'][[0,2]]
        with np.load(run.checked(old_heads[seed]['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            old_coef = dict(zip(z['names'],z['coefficients']))
        data,m = {},{}
        for s,ix in splits.items():
            b,c = bank[ix],cv[ix]
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            y,valid = labels['target_eval'][ix]-d['origin'][ix,None],labels['valid'][ix]
            _,a,e = base.teacher.actual(b,c,y,valid,cut[s]['easy'])
            q = d['queries'][ix]
            data[s] = (b,c,x,scale,a,e,q,run.old.prior.groups(q))
            m[s] = run.model.fitting_moments(x,b,c,y,valid,scale,1./d['crowd'][ix],cut[s]['easy'])
        decisions = np.full((len(ARMS),len(ids)),-1,np.int16)
        decisions[0] = original
        summaries = []
        for held,ix in splits.items():
            mark = PRIVATE/f'{seed}_{held}.json'
            identity = dict(seed=seed,held=str(held),registration=run.sha(REG))
            if not mark.exists():
                active = []
                detail = []
                for donor in splits:
                    if donor==held:
                        continue
                    b,c,x,scale,a,e,q,g = data[donor]
                    coef = run.model.coefficients([m[s] for s in splits if s not in (held,donor)])
                    u,_,_,legal = run.model.decision_costs(x,coef,b,c,scale)
                    y,mask,info = base.teacher.examples(g,u/scale[:,None],a,e,legal)
                    if info['active_queries']:
                        active.append([base.parent.ranker.moments(x,z,mask,q)[1] for z in y])
                    detail.append(dict(donor=str(donor),**info))
                if not active:
                    raise ValueError('No training demonstrations for this held source')
                w = np.stack([base.parent.ranker.fit([r[j] for r in active],bank.shape[1]) for j in range(2)])
                ch = np.full((2,len(ix)),-1,np.int16)
                for ri,gs in by_record.values():
                    if sites[ri[0]]!=held:
                        continue
                    b,c = bank[ri],cv[ri]
                    x,scale = run.model.descriptors(d['history'][ri],b,c,d['crowd'][ri])
                    _,a,e,legal = run.model.decision_costs(x,old_coef[held],b,c,scale)
                    count = [(raw[ri[g]]>=0).sum() for g in gs]
                    for j in range(2):
                        ch[j,np.searchsorted(ix,ri)],_ = prior.model.stream(gs,
                            base.parent.ranker.score(x,w[j]),a,e,legal,counts=count,witness=raw[ri])
                cache = mark.with_suffix('.npz')
                base.parent.base.write_array(cache,ids=ids[ix],choices=ch,coefficients=w)
                run.once(mark,dict(identity=identity,cache=run.artifact(cache),donors=detail))
            v = run.read(mark)
            if v['identity']!=identity:
                raise ValueError('Saved double exclusion identity mismatch')
            with np.load(run.checked(v['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix])
                decisions[1:,ix] = z['choices']
            summaries.append(dict(held=str(held),active=sum(z['active_queries'] for z in v['donors']),
                                  changed=sum(z['changed_queries'] for z in v['donors'])))
            beat(state='held_source_frozen',seed=seed,fold=len(summaries),changed=summaries[-1]['changed'])
        path = PRIVATE/f'{seed}_all.npz'
        if not path.exists():
            base.parent.base.write_array(path,ids=ids,choices=decisions)
        else:
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['choices'],decisions)
        cells.append(dict(seed=seed,cache=run.artifact(path),summaries=summaries))
    run.once(PUBLIC/'freeze.json',dict(cells=cells,before_readout=True,no_donor_actions_exported=True))


def score(d,refs,by_record,beat):
    f = run.read(PUBLIC/'freeze.json')
    if not f['before_readout'] or len(f['cells'])!=3:
        raise ValueError('Require complete frozen matrix')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update(feasible_vs_gain=('gain_teacher','feasible_teacher',p.CV),
                 feasible_vs_original=('original','feasible_teacher',p.CV),
                 gain_vs_original=('original','gain_teacher',p.CV))
    old = {(r['seed'],r['recording']):r['metrics']['original']
           for r in run.read(base.PUBLIC/'analysis.json')['records']}
    results = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch = z['choices']
        for r,(ix,gs) in by_record.items():
            s = str(d['sites'][ix[0]])
            v = ch[:,ix]
            for g in gs:
                n = (v[:,g]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Matched count changed')
            paths = {a:run.model.selected_paths(z,bank[ix],cv[ix]) for a,z in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],
                d['queries'][ix],r,easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            if out['metrics']['original']!=old[seed,r]:
                raise ValueError('Original control mismatch')
            pair = {}
            for a in ARMS:
                z = [run.model.pair_error(d['history'][ix[g]],paths[a][g],y[ix[g]],valid[ix[g]]) for g in gs]
                pair[a] = [float(sum(v0 for v0,_ in z if v0 is not None)),sum(v0 is not None for v0,_ in z)]
            results.append(dict(seed=seed,source=s,**out,pair=pair,
                replaced_selected=int(((v[1]>=0)&(v[2]<0)).sum()),
                changed_actions=int((v[1]!=v[2]).sum())))
        beat(state='seed_scored',seed=seed)
    if len(results)!=102:
        raise ValueError('Every recording view is required')
    sources = np.unique(d['sites'])
    groups = [[r for r in results if r['source']==s and r['seed']==seed]
              for s in sources for seed in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        b = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in groups]
        comparisons[k] = {key:ci([v[key] for v in b]) for key in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risks = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risks[a] = dict(passes=sum(z['supported'] for z in v),switches=sum(z['selected'] for z in v),
            easy_over_2=sum(z['completion_easy_risk_upper'] is not None and z['completion_easy_risk_upper']>.02 for z in v),
            all_over_2=sum(z['completion_all_risk_upper'] is not None and z['completion_all_risk_upper']>.02 for z in v),
            undefined_easy=sum(z['completion_easy_risk_upper'] is None for z in v),
            easy_net_over_2=sum(z['known_easy_net_degradation'] is not None and z['known_easy_net_degradation']>.02 for z in v))
    def pair_delta(g):
        n = sum(r['pair']['gain_teacher'][1] for r in g)
        return sum(r['pair']['gain_teacher'][0]-r['pair']['feasible_teacher'][0] for r in g)/n if n else None
    def hard(a,g):
        z = [r['metrics'][a]['subsets']['hard'] for r in g if r['metrics'][a]['subsets']['hard']['known_rows']]
        ref = sum(v['known_rows']*v['floor_ADE_pixel'] for v in z)
        return 100*sum(v['known_rows']*(v['floor_ADE_pixel']-v['model_ADE_pixel']) for v in z)/ref if ref else None
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=results,risk=risks,comparisons=comparisons,
        hard_known_gain={a:ci([hard(a,g) for g in groups]) for a in ARMS},
        observed_pair_reduction=ci([pair_delta(g) for g in groups]),
        donor_queries=sum(v['active'] for c in f['cells'] for v in c['summaries']),
        teacher_different_queries=sum(v['changed'] for c in f['cells'] for v in c['summaries']),
        replaced_selected=sum(r['replaced_selected'] for r in results),
        changed_actions=sum(r['changed_actions'] for r in results),
        original_control_exact=True,original_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','fit','score','replay'))
    phase = parser.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the registered design first')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from the last frozen held source')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        beat(state='start')
        d,refs = run.population(),run.register()
        records = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (fit if phase=='fit' else score)(d,refs,records,beat)
        target = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                      artifact=run.artifact(target)))


if __name__=='__main__':
    main()
