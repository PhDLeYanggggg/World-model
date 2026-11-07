"""Compare direct amplitude increments with differenced absolute costs."""
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
from scripts import run_m3w_bounded_path_bank as parent
from src.world_model import m3w_incremental_amplitude as model

run, np = parent.run,parent.np
bank_model = parent.model
cal = parent.prior.nested.cal
PUBLIC, PRIVATE = parent.PUBLIC/'incremental', parent.PRIVATE/'incremental'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_incremental_amplitude.py')
ARMS = ('original','quarter','absolute','incremental','matched_absolute')


def register():
    bindings = (parent.REG,parent.PUBLIC/'freeze.json',parent.PUBLIC/'analysis.json',
        parent.PUBLIC/'replay_execution.json',Path(parent.__file__),Path(bank_model.__file__),
        Path(run.model.__file__),Path(cal.__file__),run.PUBLIC/'freeze.json')
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in bindings],arms=ARMS,levels=model.LEVELS,
        seeds=run.kin.prior.SEEDS,ridge=.002,bootstrap_draws=3000,
        development_only=True,independent_outcomes_opened=False,original_budget=.02)))


def solve(d, refs, records, beat):
    ids, sites = d['global_ids'],d['sites']
    split = {s:np.flatnonzero(sites==s) for s in np.unique(sites)}
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval','valid'))
    old = {r['seed']:r for r in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cells = []
    for c in run.read(parent.PUBLIC/'freeze.json')['cells']:
        seed = c['seed']
        mark = PRIVATE/f'{seed}.json'
        if not mark.exists():
            cv, bank = run.candidates(d,refs,seed)
            width = bank.shape[1]
            with np.load(run.checked(c['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                original = z['choices'][3]
                smaller = {k:dict(zip(z['sources'],v)) for k,v in zip((.5,.25),z['coefficients'])}
            with np.load(run.checked(old[seed]['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                smaller[1.] = dict(zip(z['names'],z['coefficients']))
            moments = {a:{} for a in model.LEVELS[1:]}
            for s,ix in split.items():
                h, b, p = d['history'][ix],cv[ix],bank[ix]
                y, valid = labels['target_eval'][ix]-d['origin'][ix,None],labels['valid'][ix]
                prev = None
                for a in model.LEVELS:
                    path = bank_model.blend(p,b,a)
                    z = model.outcomes(path,b,y,valid,cuts[s]['easy'])
                    if prev is not None:
                        x, scale = run.model.descriptors(h,path,b,d['crowd'][ix])
                        moments[a][s] = model.fit_moments(x,model.increment(prev,z),scale,1./d['crowd'][ix])
                    prev = z
            coef = {a:{s:run.model.coefficients([m for t,m in moments[a].items() if t!=s])
                       for s in split} for a in model.LEVELS[1:]}
            chosen = np.broadcast_to(original,(len(ARMS),len(ids))).copy()
            taken = np.flatnonzero(original>=0)
            pred = np.zeros((len(taken),2,2,3),dtype=np.float64)
            exchangeable = 0
            for ix,groups in records.values():
                selected = ix[original[ix]>=0]
                if not len(selected):
                    continue
                s = sites[ix[0]]
                h,b,p = d['history'][selected],cv[selected],bank[selected]
                family = original[selected]%width
                upper = 2-original[selected]//width
                rr = np.arange(len(selected))
                prev, prev_radius = None,None
                direct, absolute = [],[]
                for a in model.LEVELS:
                    path = bank_model.blend(p,b,a)
                    x,scale = run.model.descriptors(h,path,b,d['crowd'][selected])
                    radius = np.linalg.norm(path-b[:,None],axis=-1).max(2)
                    utility,*_ = run.model.decision_costs(x,smaller[a][s],path,b,scale)
                    harm = cal.predicted_components(x,smaller[a][s],scale,radius)
                    z = np.stack((utility,harm[...,0],harm[...,2]),axis=-1)[rr,family]
                    if prev is not None:
                        dr = np.maximum(radius[rr,family]-prev_radius,0.)
                        pr = np.einsum('nd,ndc->nc',x[rr,family],coef[a][s][family])*scale[:,None]
                        direct.append(model.project(pr,dr))
                        absolute.append(model.project(model.increment(prev,z),dr))
                    prev,prev_radius = z,radius[rr,family]
                a, m = np.stack(absolute,axis=1),np.stack(direct,axis=1)
                sa, sm = model.scores(a),model.scores(m)
                inc = model.choose(upper,sm)
                j = {i:k for k,i in enumerate(selected)}
                query = [np.asarray([j[i] for i in ix[g] if i in j],dtype=int) for g in groups]
                matched,n = model.match([g for g in query if len(g)],upper,inc,sa)
                exchangeable += n
                for k, z in enumerate((np.zeros_like(upper),model.choose(upper,sa),inc,matched),1):
                    chosen[k,selected] = family+width*(2-z)
                pred[np.searchsorted(taken,selected)] = np.stack((a,m),axis=1)
            path = PRIVATE/f'{seed}.npz'
            parent.feedback.write_array(path,ids=ids,choices=chosen,predictions=pred,
                fitted_sources=np.asarray(list(split)),
                coefficients=np.stack([[coef[a][s] for s in split] for a in model.LEVELS[1:]]))
            run.once(mark,dict(seed=seed,registration=run.sha(REG),cache=run.artifact(path),
                               fits=2*len(split),exchangeable_queries=exchangeable))
        x = run.read(mark)
        if x['seed']!=seed or x['registration']!=run.sha(REG):
            raise ValueError('Saved fit identity mismatch')
        run.checked(x['cache'])
        cells.append(x)
        beat(state='seed_frozen',seed=seed,exchangeable=x['exchangeable_queries'])
    run.once(PUBLIC/'freeze.json',dict(cells=cells,decisions_frozen_before_readout=True))


def score(d, refs, by_record, beat):
    f = run.read(PUBLIC/'freeze.json')
    if len(f['cells'])!=3 or not f['decisions_frozen_before_readout']:
        raise ValueError('Require all three frozen arms')
    labels = run.old.prior.arrays(('target_eval','valid'))
    y, valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    p = run.parent
    cuts = run.cuts_by_source()
    pairs = {a+'_vs_cv':(p.CV,a,p.CV) for a in ARMS}
    pairs.update({f'incremental_vs_{a}':(a,'incremental',p.CV) for a in ARMS if a!='incremental'})
    original = {(r['seed'],r['recording']):r['metrics']['bounded'] for r in run.read(parent.PUBLIC/'analysis.json')['records']}
    records = []
    for c in f['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        width = bank.shape[1]
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choices,pred = z['choices'],z['predictions']
        taken = np.flatnonzero(choices[0]>=0)
        for r,(ix,groups) in by_record.items():
            ch = choices[:,ix]
            s = str(d['sites'][ix[0]])
            selected = ch[0]>=0
            if (not ((ch>=0)==selected).all() or not ((ch%width)==ch[0]%width).all()
                    or (ch<ch[0]).any()):
                raise ValueError('Row, forecast family, or amplitude cap changed')
            for g in groups:
                np.testing.assert_array_equal(np.sort(ch[3,g]//width),np.sort(ch[4,g]//width))
            paths = {a:bank_model.selected(bank[ix],cv[ix],z) for a,z in zip(ARMS,ch)}
            out = p.readout.score_recording({p.CV:cv[ix],**paths},pairs,y[ix],valid[ix],d['queries'][ix],r,
                easy_cut=cuts[s]['easy'],hard_cut=cuts[s]['hard'])
            if out['metrics']['original']!=original[seed,r]:
                raise ValueError('Original outcomes do not replay')
            j = ix[selected]
            p0 = bank[j,ch[0,selected]%width][:,None]
            terms = [model.outcomes(bank_model.blend(p0,cv[j],a),cv[j],y[j],valid[j],cuts[s]['easy'])[:,0]
                     for a in model.LEVELS]
            t = np.stack([model.increment(a,b) for a,b in zip(terms,terms[1:])],axis=1)
            z = pred[np.searchsorted(taken,j)]
            use = np.arange(1,3)[None,:] <= (2-ch[0,selected]//width)[:,None]
            dist = np.linalg.norm(p0[:,0]-cv[j],axis=-1).max(1)
            residual = (z-t[:,None])/dist[:,None,None,None]
            w = use/d['crowd'][j,None]
            e = dict(weight=float(w.sum()),squared=(residual**2*w[:,None,:,None]).sum((0,2)).tolist(),
                     bias=(residual*w[:,None,:,None]).sum((0,2)).tolist())
            records.append(dict(seed=seed,source=s,**out,errors=e,
                matched_different_rows=int((ch[3]!=ch[4]).sum()),
                amplitude_counts={a:np.bincount(z[z>=0]//width,minlength=3).tolist() for a,z in zip(ARMS,ch)}))
        beat(state='seed_scored',seed=seed)
    if len(records)!=102:
        raise ValueError('Full roster required')
    sources = np.unique(d['sites'])
    grouped = [[r for r in records if r['source']==s and r['seed']==seed]
               for s in sources for seed in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def ci(x):
        if any(z is None for z in x):
            return None
        z = np.asarray(x).reshape(len(sources),3).mean(1)
        return dict(point=float(z.mean()),interval95=np.quantile(z[draw].mean(1),[.025,.975]).tolist())
    comparisons = {}
    for k in pairs:
        block = [p.readout.merge_comparisons([r['comparisons'][k] for r in group]) for group in grouped]
        comparisons[k] = {f:ci([b[f] for b in block]) for f in ('known_pp','completion_lower_pp','completion_upper_pp')}
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in records]
        risk[a] = dict(passes=sum(x['supported'] for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper']>.02 for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation']>.02 for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            amplitude_counts=np.sum([r['amplitude_counts'][a] for r in records],axis=0).tolist())
    w = sum(r['errors']['weight'] for r in records)
    error = {k:(np.sum([r['errors'][k] for r in records],axis=0)/w).tolist() for k in ('squared','bias')}
    changed = sum(r['matched_different_rows'] for r in records)
    if not changed:
        comparisons['incremental_vs_matched_absolute'] = None
    run.once(PUBLIC/'analysis.json',run.stable(dict(comparisons=comparisons,risk=risk,records=records,
        error=error,exchangeable_queries=sum(c['exchangeable_queries'] for c in f['cells']),
        changed_at_matched_amplitudes=changed,original_risk_rules_unchanged=True,
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
        by_record = parent.prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
        (solve if phase=='fit' else score)(d,refs,by_record,beat)
        target = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
                                                      artifact=run.artifact(target)))


if __name__=='__main__':
    main()
