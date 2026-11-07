"""Select a release guard on disjoint inner sources and score held sources once."""
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
from scripts import run_m3w_veto_risk_refit as parent
from src.world_model import m3w_release_guard as guard

run,np,prior = parent.run,parent.np,parent.prior
PUBLIC,PRIVATE = parent.PUBLIC/'guard',parent.PRIVATE/'guard'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(guard.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_release_guard.py')
ARMS = ('base','unguarded','learned','radius')


def register():
    parents = (parent.REG,parent.PUBLIC/'analysis.json',parent.PUBLIC/'replay_execution.json',
        parent.PUBLIC/'tradeoff_attribution.json',Path(parent.__file__),Path(parent.model.__file__),
        run.REG,Path(run.model.__file__),prior.REG,Path(prior.model.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,seeds=run.kin.prior.SEEDS,
        grid=guard.GRIDS,held_sources=1,calibration_sources=3,fitting_sources=10,
        original_budget=.02,matched_query_counts=True,independent_outcomes_opened=False)))


def original_context(groups,x,scale,b,c,coef):
    u,a,e,legal = run.model.decision_costs(x,coef,b,c,scale)
    w = prior.nested.base.select(groups,u,a,e,scale,legal)
    v = u/scale[:,None]
    n = [int((w[g]>=0).sum()) for g in groups]
    ch,_ = prior.model.stream(groups,v,a,e,legal,counts=n,witness=w)
    before = parent.model.prefix(groups,a,e,ch)
    z,veto = parent.model.descriptors(x,scale,a,e,before,legal)
    return v,a,e,legal,w,ch,before,z,veto


def context(d,ix,groups,cv,bank,coef,correction,harm):
    c,b = cv[ix],bank[ix]
    x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
    v,a,e,legal,w,ch,before,z,veto = original_context(groups,x,scale,b,c,coef)
    radius = np.linalg.norm(b-c[:,None],axis=-1).max(2)
    aa,ee,re = parent.model.reopen(z,correction,scale,radius,a,e,before,veto,w)
    scores = dict(learned=np.einsum('nad,ad->na',z,harm).clip(0.,1.),radius=radius/scale[:,None])
    return (groups,v,a,e,legal,w,aa,ee),ch,re,scores


def calibrate(d,records,cal,cv,bank,coef,correction,harm,y,valid,cuts):
    p = run.parent
    by_arm = {(arm,j):[] for arm,grid in guard.GRIDS.items() for j in range(len(grid))}
    for r,(ix,g) in records.items():
        s = str(d['sites'][ix[0]])
        if s not in cal:
            continue
        ctx,ch,re,scores = context(d,ix,g,cv,bank,coef,correction,harm)
        c,b = cv[ix],bank[ix]
        original = run.model.selected_paths(ch,b,c)
        prediction = {p.CV:c,'base':original}
        pairs = {}
        keys = {}
        decisions = {}
        for arm,j in by_arm:
            accepted = guard.keep(arm,j,scores[arm],re)
            key = accepted.tobytes()
            if key not in decisions:
                decisions[key] = 'p'+str(len(decisions))
                name = decisions[key]
                ch0 = guard.allocation(*ctx,accepted) if accepted.any() else ch
                prediction[name] = run.model.selected_paths(ch0,b,c)
                pairs[name] = ('base',name,p.CV)
            keys[arm,j] = decisions[key]
        result = p.readout.score_recording(prediction,pairs,y[ix],valid[ix],d['queries'][ix],r,
            easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
        metrics = result['metrics']
        for key,name in keys.items():
            by_arm[key].append((s,guard.preserves(metrics['base']['risk'],metrics[name]['risk']),
                                result['comparisons'][name]))
    detail = {}
    for arm,grid in guard.GRIDS.items():
        rows = []
        for j in range(len(grid)):
            v = by_arm[arm,j]
            if {x[0] for x in v}!=set(cal):
                raise ValueError('Require all calibration sources')
            lower = [p.readout.merge_comparisons([x[2] for x in v if x[0]==s])['completion_lower_pp'] for s in cal]
            rows.append(dict(preserved=all(x[1] for x in v),views=len(v),
                lower_pp=None if any(x is None for x in lower) else float(np.mean(lower))))
        detail[arm] = dict(selected=guard.choose_threshold(rows),candidates=rows)
    return detail


def fit(d,refs,records,beat):
    ids,sites = d['global_ids'],d['sites']
    splits = {str(s):np.flatnonzero(sites==s) for s in np.unique(sites)}
    by_source = {s:[] for s in splits}
    for ri,g in records.values():
        s = str(sites[ri[0]])
        by_source[s].append((g,np.searchsorted(splits[s],ri)))
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    cells = []
    for seed in run.kin.prior.SEEDS:
        cv,bank = run.candidates(d,refs,seed)
        data,m = {},{}
        for s,ix in splits.items():
            b,c = bank[ix],cv[ix]
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            _,a,e = parent.base.nested.base.teacher.actual(b,c,y[ix],valid[ix],cuts[s]['easy'])
            ref,_ = run.model.errors(c,y[ix],valid[ix])
            easy = np.where((ref>0.)&(ref<=cuts[s]['easy']),ref,0.)
            radius = np.linalg.norm(b-c[:,None],axis=-1).max(2)
            h = (e+.02*easy[:,None])/radius.clip(1e-8)
            data[s] = (b,c,x,scale,a,e,h.clip(0.,1.))
            m[s] = run.model.fitting_moments(x,b,c,y[ix],valid[ix],scale,1./d['crowd'][ix],cuts[s]['easy'])
        choices = np.full((len(ARMS),len(ids)),-1,np.int16)
        selected_release = np.zeros((len(ARMS),len(ids)),bool)
        selection = []
        for held,ix in splits.items():
            mark = PRIVATE/f'{seed}_{held}.json'
            learn,cal = guard.roles(splits,held)
            ident = dict(seed=seed,held=held,fitting=list(learn),calibration=list(cal),registration=run.sha(REG))
            if not mark.exists():
                residual,guard_harm = [],[]
                for donor in learn:
                    b,c,x,s,true_a,true_e,h = data[donor]
                    coef = run.model.coefficients([m[k] for k in learn if k!=donor])
                    z = np.empty((*x.shape[:2],x.shape[2]+4))
                    veto = np.zeros(x.shape[:2],bool)
                    u = np.empty((*x.shape[:2],2))
                    for g,j in by_source[donor]:
                        _,a,e,_,_,_,_,z0,veto0 = original_context(g,x[j],s[j],b[j],c[j],coef)
                        z[j],veto[j] = z0,veto0
                        u[j] = np.stack((true_a[j]-a,true_e[j]-e),axis=-1)/s[j,None,None]
                    w = 1./d['crowd'][splits[donor]]
                    residual.append(parent.model.moments(z,u,veto,w))
                    guard_harm.append(parent.model.moments(z,np.stack((h,np.zeros_like(h)),axis=-1),veto,w))
                coef = run.model.coefficients([m[s] for s in learn])
                correction = parent.model.coefficients(residual)
                harm = parent.model.coefficients(guard_harm)[...,0]
                decisions = calibrate(d,records,cal,cv,bank,coef,correction,harm,y,valid,cuts)
                ch = np.full((len(ARMS),len(ix)),-1,np.int16)
                used = np.zeros_like(ch,bool)
                for ri,g in records.values():
                    if str(sites[ri[0]])!=held:
                        continue
                    ctx,old,re,scores = context(d,ri,g,cv,bank,coef,correction,harm)
                    j = np.searchsorted(ix,ri)
                    ch[0,j] = old
                    for k,a in enumerate(ARMS[1:],1):
                        keep = re if a=='unguarded' else guard.keep(a,decisions[a]['selected'],scores[a],re)
                        v = guard.allocation(*ctx,keep) if keep.any() else old
                        ch[k,j] = v
                        rows = np.flatnonzero(v>=0)
                        used[k,j[rows]] = keep[rows,v[rows]]
                cache = mark.with_suffix('.npz')
                parent.base.nested.base.parent.base.write_array(cache,ids=ids[ix],choices=ch,selected_release=used,
                    baseline_coefficients=coef,residual_coefficients=correction,guard_coefficients=harm)
                run.once(mark,dict(identity=ident,cache=run.artifact(cache),selection=decisions))
            cell = run.read(mark)
            if cell['identity']!=ident:
                raise ValueError('Saved inner roles differ')
            with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids[ix])
                choices[:,ix],selected_release[:,ix] = z['choices'],z['selected_release']
            selection.append(cell)
            beat(state='held_source_frozen',seed=seed,fold=len(selection),
                thresholds=[cell['selection'][a]['selected'] for a in ARMS[2:]])
        cache = PRIVATE/f'{seed}_all.npz'
        if not cache.exists():
            parent.base.nested.base.parent.base.write_array(cache,ids=ids,choices=choices,selected_release=selected_release)
        else:
            with np.load(cache,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['choices'],choices)
                np.testing.assert_array_equal(z['selected_release'],selected_release)
        cells.append(dict(seed=seed,cache=run.artifact(cache),selection=selection))
    run.once(PUBLIC/'freeze.json',dict(cells=cells,outer_decisions_before_scoring=True))


def score(d,refs,records,beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['outer_decisions_before_scoring']:
        raise ValueError('Complete frozen matrix required')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    pairs = {a+'_vs_base':('base',a,p.CV) for a in ARMS[1:]}
    pairs.update(learned_vs_unguarded=('unguarded','learned',p.CV),
        radius_vs_unguarded=('unguarded','radius',p.CV),learned_vs_radius=('radius','learned',p.CV))
    results = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch,re = z['choices'],z['selected_release']
        for r,(ix,g) in records.items():
            v = ch[:,ix]
            for j in g:
                n = (v[:,j]>=0).sum(1)
                if not (n==n[0]).all():
                    raise ValueError('Exact per-query counts required')
            s = str(d['sites'][ix[0]])
            b,c = bank[ix],cv[ix]
            paths = {a:run.model.selected_paths(z,b,c) for a,z in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:c,**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            results.append(dict(seed=seed,source=s,**out,
                release={a:int(z[ix].sum()) for a,z in zip(ARMS,re)}))
        beat(state='seed_scored',seed=seed)
    if len(results)!=102:
        raise ValueError('Full recording roster required')
    sites = sorted({r['source'] for r in results})
    grouped = [[r for r in results if r['source']==s and r['seed']==k]
               for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    def ci(x):
        if any(v is None for v in x):
            return None
        z = np.asarray(x).reshape(len(sites),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for key in pairs:
        v = [p.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in grouped]
        comparisons[key] = {k:ci([x[k] for x in v]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risk[a] = dict(passes=sum(x['supported'] for x in v),switches=sum(x['selected'] for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v),
            selected_release=sum(r['release'][a] for r in results))
    choices = {a:np.bincount([v['selection'][a]['selected'] for c in f['cells'] for v in c['selection']],
                            minlength=len(guard.GRIDS[a])).tolist() for a in ARMS[2:]}
    run.once(PUBLIC/'analysis.json',run.stable(dict(records=results,risk=risk,comparisons=comparisons,
        threshold_histogram=choices,outer_outcomes_not_used_for_selection=True,original_rules_unchanged=True,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','fit','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the design before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Resume from the last saved held source')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        beat(state='start')
        d,refs = run.population(),run.register()
        records = prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (fit if phase=='fit' else score)(d,refs,records,beat)
        name = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                     artifact=run.artifact(name)))


if __name__=='__main__':
    main()
