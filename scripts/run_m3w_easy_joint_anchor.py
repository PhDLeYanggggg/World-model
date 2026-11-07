"""Source-disjoint fit/calibration for a single learned easy-support opening."""
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
from scripts import run_m3w_timing_capacity as timing
from src.world_model import m3w_easy_joint_anchor as model

run, np, prior = timing.run,timing.np,timing.prior
PUBLIC, PRIVATE = prior.PUBLIC/'easy_joint_anchor', prior.PRIVATE/'easy_joint_anchor'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_easy_joint_anchor.py')
ARMS = ('direct','joint','marginal','diagnostic_joint','diagnostic_original')
SIZE = timing.PAST.size
FIT_KEYS = ('joint','marginal')


def register():
    path = timing.PUBLIC/'analysis.json'
    if run.read(timing.PUBLIC/'replay_execution.json')['artifact']['sha256']!=run.sha(path):
        raise ValueError('Require the verified timing study')
    inputs = (timing.REG,path,SIZE.REG,SIZE.PUBLIC/'freeze.json',timing.PAST.PUBLIC/'freeze.json',
              timing.PAST.PUBLIC/'analysis.json',Path(timing.__file__),Path(run.model.__file__),
              Path(prior.model.__file__),Path(run.parent.readout.__file__),Path(prior.nested.base.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        inputs=[run.artifact(p) for p in inputs],thresholds=model.THRESHOLDS,arms=ARMS,
        original_budget=.02,independent_outcomes_opened=False)))


def record_options(d,ix,g,b,c,weights,co):
    x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
    u,a,e,ok = run.model.decision_costs(x,co,b,c,scale)
    mint = prior.nested.base.select(g,u,a,e,scale,ok)
    n = [int((mint[j]>=0).sum()) for j in g]
    direct,_ = prior.model.stream(g,u/scale[:,None],a,e,ok,counts=n,witness=mint)
    options = [model.prefix_options(g,direct,z,ok) for z in model.predict(x,weights)]
    return direct,options,u/scale[:,None],ok


def fit(d,refs,records,cells,controls,y,valid,beat):
    sources = {str(s):np.flatnonzero(d['sites']==s) for s in np.unique(d['sites'])}
    by_source = {s:[] for s in sources}
    for rec,(ix,g) in records.items():
        s = str(d['sites'][ix[0]])
        by_source[s].append((rec,ix,g,np.searchsorted(sources[s],ix)))
    cuts = run.cuts_by_source()
    writer = SIZE.base.parent.base.nested.base.parent.base.write_array
    frozen = []
    for parent in cells:
        seed = parent['seed']
        c,b = run.candidates(d,refs,seed)
        direct,coef = timing.parents(d,parent,controls)
        atoms,stat = {},{}
        for s,ix in sources.items():
            x,_ = run.model.descriptors(d['history'][ix],b[ix],c[ix],d['crowd'][ix])
            label,atoms[s],_ = model.labels(c[ix],b[ix],y[ix],valid[ix],cuts[s]['easy'])
            legal = np.linalg.norm(b[ix]-c[ix,None],axis=-1).max(2)>0.
            stat[s] = model.fitting(x,label,legal,d['crowd'][ix])
        marks = []
        for control in controls[seed]:
            i = control['identity']
            held = i['held']
            train,cal = i['fit'], i['cal_a']+i['cal_b']
            if len(set(train+cal+[held]))!=14 or len(train)!=7 or len(cal)!=6:
                raise ValueError('Training and calibration must exclude the held source')
            identity = dict(held=held,seed=seed,fit=train,cal=cal,registration=run.sha(REG))
            mark = PRIVATE/f'{seed}_{held}.json'
            if not mark.exists():
                w = model.fit([stat[s] for s in train])
                donors = {k:[] for k in FIT_KEYS}
                co = coef[held]
                for s in cal:
                    for _,ix,g,j in by_source[s]:
                        dc,options,_,_ = record_options(d,ix,g,b[ix],c[ix],w,co)
                        z = atoms[s][j]
                        k = np.flatnonzero(dc>=0)
                        mass = z[k,dc[k]].sum(0)
                        for name,opt in zip(FIT_KEYS,options):
                            donors[name].append(dict(mass=mass,options=opt,atoms=z,
                                easy=float(z[:,0,1].sum()),crowd=d['crowd'][ix]))
                picked = {k:model.calibrate(v) for k,v in donors.items()}
                ix = sources[held]
                choice = np.broadcast_to(direct[ix],(len(ARMS),len(ix))).copy()
                for _,ri,g,j in by_source[held]:
                    dc,options,u,ok = record_options(d,ri,g,b[ri],c[ri],w,co)
                    np.testing.assert_array_equal(dc,direct[ri])
                    for k,(name,opt) in enumerate(zip(FIT_KEYS,options),1):
                        cand = model.first(opt,picked[name][0])
                        if cand is not None:
                            _,row,a,_ = cand
                            choice[k,j[row]] = a
                    cand = model.first(options[0],-np.inf)
                    if cand is not None:
                        _,row,a,query = cand
                        choice[3,j[row]] = a
                        r,k = np.unravel_index(np.where(ok[query],u[query],-np.inf).argmax(),ok[query].shape)
                        choice[4,j[query[r]]] = k
                cache = mark.with_suffix('.npz')
                writer(cache,ids=d['global_ids'][ix],choices=choice,coefficients=w)
                run.once(mark,run.stable(dict(identity=identity,cache=run.artifact(cache),
                    threshold={k:v[0] for k,v in picked.items()},
                    calibration_table={k:v[1] for k,v in picked.items()})))
            cell = run.read(mark)
            if cell['identity']!=identity:
                raise ValueError('Cannot resume a different fitting cell')
            marks.append(cell)
            beat(state='fold_frozen',seed=seed,complete=len(marks))
        frozen.append(dict(seed=seed,folds=marks))
    run.once(PUBLIC/'freeze.json',run.stable(dict(cells=frozen,held_not_used_for_fit_or_threshold=True)))


def score(d,refs,records,y,valid,beat):
    freeze = PUBLIC/'freeze.json'
    f = run.read(freeze)
    if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256']!=run.sha(freeze):
        raise ValueError('All choices must freeze before held-source scoring')
    base = {(r['seed'],r['recording']):r['metrics']['direct'] for r in
            run.read(timing.PAST.PUBLIC/'analysis.json')['records']}
    cuts = run.cuts_by_source()
    p = run.parent
    pairs = dict(joint_vs_direct=('direct','joint',p.CV),
                 marginal_vs_direct=('direct','marginal',p.CV),
                 joint_vs_marginal=('marginal','joint',p.CV),
                 diagnostic_joint_vs_original=('diagnostic_original','diagnostic_joint',p.CV))
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
            c,b = cv[ix],bank[ix]
            z = ch[:,ix]
            counts = z>=0
            if (z[:,counts[0]]!=z[0,counts[0]]).any() or ((counts & ~counts[0]).sum(1)>1).any():
                raise ValueError('An opening cannot replace or add multiple direct actions')
            if any((counts[3,j].sum()!=counts[4,j].sum()) for j in g):
                raise ValueError('Diagnostic control must match openings query by query')
            paths = {k:run.model.selected_paths(a,b,c) for k,a in zip(ARMS,z)}
            e,h = cuts[s]['easy'],cuts[s]['hard']
            res = p.readout.score_recording({p.CV:c,**paths},pairs,y[ix],valid[ix],d['queries'][ix],
                                             rec,easy_cut=e,hard_cut=h)
            if res['metrics']['direct']!=base[seed,rec]:
                raise ValueError('Exact original direct control required')
            rows.append(dict(seed=seed,source=s,recording=rec,zero_direct=not counts[0].any(),
                risk={k:res['metrics'][k]['risk'] for k in ARMS},contrasts=res['comparisons']))
        beat(state='seed_scored',seed=seed)
    sources = sorted({r['source'] for r in rows})
    if len(rows)!=102 or len(sources)!=14 or sum(r['zero_direct'] for r in rows)!=18:
        raise ValueError('Incomplete registered 102-view population')
    groups = [[r for r in rows if r['source']==s and r['seed']==k]
              for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14,size=(3000,14))
    ci = {}
    for name in pairs:
        vals = [p.readout.merge_comparisons([r['contrasts'][name] for r in g]) for g in groups]
        x = np.asarray([r['completion_lower_pp'] for r in vals],dtype=float).reshape(14,3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('Paired lower utility must be estimable')
        ci[name] = dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    risk = {}
    for k in ARMS:
        a = [r['risk'][k] for r in rows]
        risk[k] = dict(switches=sum(x['selected'] for x in a),passes=sum(x['supported'] for x in a),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in a),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation']>.02 for x in a),
            zero_views_opened=sum(r['zero_direct'] and r['risk'][k]['selected']>0 for r in rows),
            zero_views_passed=sum(r['zero_direct'] and r['risk'][k]['supported'] for r in rows))
    chosen = {k:dict(Counter(str(r['threshold'][k]) for c in f['cells'] for r in c['folds'])) for k in FIT_KEYS}
    run.once(PUBLIC/'analysis.json',run.stable(dict(views=len(rows),risk=risk,contrasts=ci,thresholds=chosen,
        original_rules_unchanged=True,independent_confirmation=False,deployment_changed=False,
        stage5c_executed=False,smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','fit','score','replay'))
    phase = parser.parse_args().phase
    register()
    if phase=='register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)!=f.read_bytes():
            raise ValueError('Commit experiment definition first')
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
        if phase=='fit':
            cells = run.read(timing.PAST.PUBLIC/'freeze.json')['cells']
            controls = {c['seed']:c['selection'] for c in run.read(SIZE.PUBLIC/'freeze.json')['cells']}
            fit(d,refs,records,cells,controls,y,valid,beat)
        else:
            score(d,refs,records,y,valid,beat)
        out = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__=='__main__':
    main()
