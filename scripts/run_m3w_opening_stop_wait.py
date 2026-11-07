"""Learn and test next-query continuation on the original source-separated roster."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_easy_joint_anchor as anchor
from scripts.audit_m3w_easy_opening_opportunity import inspect
from src.world_model import m3w_opening_stop_wait as model

run, np, prior, timing = anchor.run,anchor.np,anchor.prior,anchor.timing
PUBLIC, PRIVATE = prior.PUBLIC/'opening_stop_wait', prior.PRIVATE/'opening_stop_wait'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_opening_stop_wait.py')
ARMS = ('direct','stop','wait','diagnostic_stop','diagnostic_wait')
WIDTH = 10


def register():
    a = anchor.PUBLIC/'analysis.json'
    proof = anchor.PUBLIC/'replay_execution.json'
    if run.read(proof)['artifact']['sha256']!=run.sha(a):
        raise ValueError('Require the verified opening study')
    parents = (anchor.REG,a,proof,anchor.PUBLIC/'freeze.json',anchor.PUBLIC/'fit_execution.json',
               anchor.PUBLIC/'opportunity_diagnostic.json',anchor.SIZE.REG,
               anchor.SIZE.PUBLIC/'freeze.json',timing.PAST.PUBLIC/'freeze.json',
               Path(anchor.model.__file__),Path(run.model.__file__),Path(prior.model.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,width=WIDTH,
        original_budget=.02,independent_outcomes_opened=False)))


def options(x,scale,b,c,g,co,rank):
    u,a,e,ok = run.model.decision_costs(x,co,b,c,scale)
    mint = prior.nested.base.select(g,u,a,e,scale,ok)
    direct,_ = prior.model.stream(g,u/scale[:,None],a,e,ok,
                                counts=[int((mint[j]>=0).sum()) for j in g],witness=mint)
    joint,marginal = anchor.model.predict(x,rank)
    raw = anchor.model.prefix_options(g,direct,joint,ok)
    rows = []
    for step,(s,i,j,q) in enumerate(raw):
        z = np.array([1.,s,marginal[i,j],np.arcsinh(u[i,j]/scale[i]),
            np.arcsinh(a[i,j]/scale[i]),np.arcsinh(e[i,j]/scale[i]),np.log1p(len(q)),
            np.log1p(scale[i]),np.log1p(np.linalg.norm(b[i,j]-c[i],axis=-1).max()/scale[i]),
            np.log1p(step)])
        rows.append((s,i,j,z))
    return direct,rows


def fit(d,refs,records,y,valid,beat):
    groups = {str(s):np.flatnonzero(d['sites']==s) for s in np.unique(d['sites'])}
    by_source = {s:[] for s in groups}
    for rec,(ix,g) in records.items():
        s = str(d['sites'][ix[0]])
        by_source[s].append((rec,ix,g,np.searchsorted(groups[s],ix)))
    cuts = run.cuts_by_source()
    controls = {c['seed']:c['selection'] for c in run.read(anchor.SIZE.PUBLIC/'freeze.json')['cells']}
    previous = {(c['seed'],f['identity']['held']):f for c in run.read(anchor.PUBLIC/'freeze.json')['cells']
                for f in c['folds']}
    parent = run.read(timing.PAST.PUBLIC/'freeze.json')['cells']
    writer = anchor.SIZE.base.parent.base.nested.base.parent.base.write_array
    frozen = []
    for cell in parent:
        seed = cell['seed']
        c,b = run.candidates(d,refs,seed)
        direct,coefs = timing.parents(d,cell,controls)
        atoms,desc,scales,rank_m,orig_m = {},{},{},{},{}
        for s,ix in groups.items():
            e = cuts[s]['easy']
            x,scale = run.model.descriptors(d['history'][ix],b[ix],c[ix],d['crowd'][ix])
            label,atoms[s],_ = anchor.model.labels(c[ix],b[ix],y[ix],valid[ix],e)
            desc[s],scales[s] = x,scale
            ok = np.linalg.norm(b[ix]-c[ix,None],axis=-1).max(2)>0.
            rank_m[s] = anchor.model.fitting(x,label,ok,d['crowd'][ix])
            orig_m[s] = run.model.fitting_moments(x,b[ix],c[ix],y[ix],valid[ix],
                                                 scale,1./d['crowd'][ix],e)
        folds = []
        for control in controls[seed]:
            i = control['identity']
            held,train,cal = i['held'],i['fit'],i['cal_a']+i['cal_b']
            if len(set(train+cal+[held]))!=14 or len(train)!=7 or len(cal)!=6:
                raise ValueError('No source role overlap allowed')
            identity = dict(held=held,seed=seed,fit=train,cal=cal,registration=run.sha(REG))
            mark = PRIVATE/f'{seed}_{held}.json'
            if not mark.exists():
                samples = {}
                for s in train:
                    other = [k for k in train if k!=s]
                    co = run.model.coefficients([orig_m[k] for k in other])
                    rank = anchor.model.fit([rank_m[k] for k in other])
                    samples[s] = []
                    for _,ix,g,j in by_source[s]:
                        _,opt = options(desc[s][j],scales[s][j],b[ix],c[ix],g,co,rank)
                        if opt:
                            samples[s].append(model.sample(opt,atoms[s][j]))
                stop,wait = model.fit(samples,WIDTH)
                with np.load(run.checked(previous[seed,held]['cache']),allow_pickle=False) as cache:
                    rank = cache['coefficients'].copy()
                co = coefs[held]
                donors = [[],[]]
                for s in cal:
                    for _,ix,g,j in by_source[s]:
                        dc,opt = options(desc[s][j],scales[s][j],b[ix],c[ix],g,co,rank)
                        acts = model.plan(opt,stop,wait)
                        z = atoms[s][j]
                        k = np.flatnonzero(dc>=0)
                        mass = z[k,dc[k]].sum(0)
                        for l,t in enumerate(acts):
                            donors[l].append(dict(mass=mass,options=t,atoms=z,
                                easy=float(z[:,0,1].sum()),crowd=d['crowd'][ix]))
                cuts2 = [anchor.model.calibrate(v) for v in donors]
                ix = groups[held]
                ch = np.broadcast_to(direct[ix],(len(ARMS),len(ix))).copy()
                for _,ri,g,j in by_source[held]:
                    dc,opt = options(desc[held][j],scales[held][j],b[ri],c[ri],g,co,rank)
                    np.testing.assert_array_equal(dc,direct[ri])
                    acts = model.plan(opt,stop,wait)
                    for k,(lst,cut) in enumerate(zip(acts*2,[v[0] for v in cuts2]+[-np.inf]*2),1):
                        selected = anchor.model.first(lst,cut)
                        if selected is not None:
                            _,row,a,_ = selected
                            ch[k,j[row]] = a
                cache = mark.with_suffix('.npz')
                writer(cache,ids=d['global_ids'][ix],choices=ch,stop=stop,wait=wait)
                run.once(mark,run.stable(dict(identity=identity,cache=run.artifact(cache),
                    thresholds=[v[0] for v in cuts2],calibration_tables=[v[1] for v in cuts2],
                    fit_query_rows=sum(len(x) for rows in samples.values() for x,_ in rows))))
            fold = run.read(mark)
            if fold['identity']!=identity:
                raise ValueError('Wrong saved fit cell')
            folds.append(fold)
            beat(state='fold_frozen',seed=seed,complete=len(folds))
        frozen.append(dict(seed=seed,folds=folds))
    run.once(PUBLIC/'freeze.json',run.stable(dict(cells=frozen,held_excluded_from_fitting_and_calibration=True)))


def score(d,refs,records,y,valid,beat):
    freeze = PUBLIC/'freeze.json'
    f = run.read(freeze)
    if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256']!=run.sha(freeze):
        raise ValueError('Require frozen choices')
    p = run.parent
    pairs = dict(stop_vs_direct=('direct','stop',p.CV),wait_vs_direct=('direct','wait',p.CV),
                 wait_vs_stop=('stop','wait',p.CV),
                 diagnostic_wait_vs_stop=('diagnostic_stop','diagnostic_wait',p.CV))
    base = {(r['seed'],r['recording']):r['metrics']['direct'] for r in
            run.read(timing.PAST.PUBLIC/'analysis.json')['records']}
    cuts = run.cuts_by_source()
    rows = []
    for cell in f['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        ch = np.full((len(ARMS),len(cv)),-1,np.int16)
        for fold in cell['folds']:
            at = d['sites']==fold['identity']['held']
            with np.load(run.checked(fold['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
                ch[:,at] = z['choices']
        for rec,(ix,g) in records.items():
            s = str(d['sites'][ix[0]])
            e,h = cuts[s]['easy'],cuts[s]['hard']
            b,c = bank[ix],cv[ix]
            v = ch[:,ix]
            dc = v[0]>=0
            if (v[:,dc]!=v[0,dc]).any() or ((v>=0)&~dc).sum(1).max()>1:
                raise ValueError('Do not change direct decisions or add more than one action')
            paths = {k:run.model.selected_paths(t,b,c) for k,t in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:c,**paths},pairs,y[ix],valid[ix],d['queries'][ix],
                    rec,easy_cut=e,hard_cut=h)
            if out['metrics']['direct']!=base[seed,rec]:
                raise ValueError('Direct control must replay exactly')
            cat = None if dc.any() else inspect(c,b,y[ix],valid[ix],g,e,h,rec)['category']
            rows.append(dict(seed=seed,source=s,category=cat,contrasts=out['comparisons'],
                delayed=int(not np.array_equal(v[3],v[4]) and int((v[4]>=0).sum())>int(dc.sum())),
                risk={k:out['metrics'][k]['risk'] for k in ARMS}))
        beat(state='seed_scored',seed=seed)
    sites = sorted({r['source'] for r in rows})
    classes = Counter(r['category'] for r in rows if r['category'])
    expected = run.read(anchor.PUBLIC/'opportunity_diagnostic.json')['categories']
    if len(rows)!=102 or len(sites)!=14 or dict(classes)!=expected:
        raise ValueError('Require the registered complete evaluation population')
    groups = [[r for r in rows if r['source']==s and r['seed']==k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14,size=(3000,14))
    ci = {}
    for name in pairs:
        vals = [p.readout.merge_comparisons([r['contrasts'][name] for r in g]) for g in groups]
        x = np.asarray([r['completion_lower_pp'] for r in vals],float).reshape(14,3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('Require all source estimands')
        ci[name] = dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    risk = {}
    for arm in ARMS:
        vals = [r['risk'][arm] for r in rows]
        risk[arm] = dict(switches=sum(x['selected'] for x in vals),passes=sum(x['supported'] for x in vals),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in vals),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation']>.02 for x in vals),
            by_zero_category={k:dict(opened=sum(r['risk'][arm]['selected']>0 for r in rows if r['category']==k),
                                     passed=sum(r['risk'][arm]['supported'] for r in rows if r['category']==k))
                              for k in classes})
    thresholds = [dict(Counter(str(v['thresholds'][j]) for c in f['cells'] for v in c['folds'])) for j in (0,1)]
    run.once(PUBLIC/'analysis.json',run.stable(dict(views=len(rows),risk=risk,contrasts=ci,
        thresholds=dict(zip(('stop','wait'),thresholds)),diagnostic_delay_actions=sum(r['delayed'] for r in rows),
        zero_categories=dict(classes),original_rules_unchanged=True,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','fit','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the experiment first')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if time.monotonic()-start>43200 or shutil.disk_usage(ROOT).free<2*1024**3:
                raise OSError('Resume from saved fitting cells')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        d,refs = run.population(),run.register()
        label = run.old.prior.arrays(('target_eval','valid','frames'))
        records = prior.blocks(d,label['frames'])
        y,valid = label['target_eval']-d['origin'][:,None],label['valid']
        (fit if phase=='fit' else score)(d,refs,records,y,valid,beat)
        out = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),
            seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__=='__main__':
    main()
