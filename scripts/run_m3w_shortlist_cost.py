"""Freeze a common shortlist and ablate its conditional risk terms."""
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
from scripts import run_m3w_opening_stop_wait as source
from src.world_model import m3w_shortlist_cost as model

run,np,prior,anchor,timing = source.run,source.np,source.prior,source.anchor,source.timing
PUBLIC, PRIVATE = prior.PUBLIC/'shortlist_cost',prior.PRIVATE/'shortlist_cost'
PAIR = prior.PUBLIC/'opening_pair_rank'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_shortlist_cost.py')
ARMS = ('direct','gain','risk','diagnostic_old','diagnostic_gain','diagnostic_risk')


def register():
    result = PAIR/'analysis.json'
    for path in (PAIR/'replay_execution.json',PAIR/'diverse_shortlist.json'):
        d = run.read(path)
        if (d.get('analysis_sha256') or d['artifact']['sha256'])!=run.sha(result):
            raise ValueError('Require verified prior shortlist measurements')
    parents = (PAIR/'registration.json',result,PAIR/'diverse_shortlist.json',
               anchor.REG,anchor.PUBLIC/'freeze.json',anchor.SIZE.PUBLIC/'freeze.json',
               timing.PAST.PUBLIC/'freeze.json',Path(anchor.model.__file__),Path(run.model.__file__),
               Path(prior.nested.base.__file__),Path(prior.model.__file__),Path(run.parent.readout.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],arms=ARMS,shortlist=8,prefix_weight=4.,
        original_budget=.02,independent_outcomes_opened=False)))


def banks():
    parent = run.read(timing.PAST.PUBLIC/'freeze.json')['cells']
    controls = {c['seed']:c['selection'] for c in run.read(anchor.SIZE.PUBLIC/'freeze.json')['cells']}
    rank = {(c['seed'],f['identity']['held']):f['cache']
            for c in run.read(anchor.PUBLIC/'freeze.json')['cells'] for f in c['folds']}
    return parent,controls,rank


def proposals(x,scale,b,c,g,co,rank):
    u,a,e,ok = run.model.decision_costs(x,co,b,c,scale)
    w = prior.nested.base.select(g,u,a,e,scale,ok)
    direct,_ = prior.model.stream(g,u/scale[:,None],a,e,ok,
                                    counts=[int((w[j]>=0).sum()) for j in g],witness=w)
    return direct,model.features(x,scale,u,a,e,ok,rank,b,c,g)


def options(direct,examples,coef):
    gain,risk = [],[]
    old = []
    for q,i,j,x in examples:
        if (direct[q]>=0).any():
            break
        old.append((0.,int(i[0]),int(j[0]),q))
        for v,pick in zip((gain,risk),model.winners((q,i,j,x),coef)):
            v.append((*pick,q))
    return gain,risk,old


def fit(d,refs,records,y,valid,beat):
    source_ix = {str(s):np.flatnonzero(d['sites']==s) for s in np.unique(d['sites'])}
    group = {s:[] for s in source_ix}
    for ix,g in records.values():
        s = str(d['sites'][ix[0]])
        group[s].append((ix,g,np.searchsorted(source_ix[s],ix)))
    cuts = run.cuts_by_source()
    parent,controls,old = banks()
    writer = anchor.SIZE.base.parent.base.nested.base.parent.base.write_array
    frozen = []
    for cell in parent:
        seed = cell['seed']
        c,b = run.candidates(d,refs,seed)
        direct,coefs = timing.parents(d,cell,controls)
        raw,scale,atoms,original,ranking = {},{},{},{},{}
        for s,ix in source_ix.items():
            e = cuts[s]['easy']
            x,z = run.model.descriptors(d['history'][ix],b[ix],c[ix],d['crowd'][ix])
            raw[s],scale[s] = x,z
            label,atoms[s],_ = anchor.model.labels(c[ix],b[ix],y[ix],valid[ix],e)
            legal = np.linalg.norm(b[ix]-c[ix,None],axis=-1).max(2)>0.
            ranking[s] = anchor.model.fitting(x,label,legal,d['crowd'][ix])
            original[s] = run.model.fitting_moments(x,b[ix],c[ix],y[ix],valid[ix],z,1./d['crowd'][ix],e)
        folds = []
        for control in controls[seed]:
            i = control['identity']
            held,learn,cal = i['held'],i['fit'],i['cal_a']+i['cal_b']
            if len(learn)!=7 or len(cal)!=6 or len(set(learn+cal+[held]))!=14:
                raise ValueError('Require the original separated source roles')
            ident = dict(seed=seed,held=held,fit=learn,cal=cal,registration=run.sha(REG))
            mark = PRIVATE/f'{seed}_{held}.json'
            if not mark.exists():
                stats = []
                for s in learn:
                    rest = [t for t in learn if t!=s]
                    co = run.model.coefficients([original[t] for t in rest])
                    rank = anchor.model.fit([ranking[t] for t in rest])
                    own = []
                    for ix,g,j in group[s]:
                        dc,examples = proposals(raw[s][j],scale[s][j],b[ix],c[ix],g,co,rank)
                        targets = model.targets(atoms[s][j],scale[s][j],examples)
                        own.append(model.moments(examples,targets,dc))
                    stats.append(own)
                weights = model.fit(stats)
                with np.load(run.checked(old[seed,held]),allow_pickle=False) as z:
                    rank = z['coefficients'].copy()
                co = coefs[held]
                donors = [[],[]]
                for s in cal:
                    for ix,g,j in group[s]:
                        dc,examples = proposals(raw[s][j],scale[s][j],b[ix],c[ix],g,co,rank)
                        z = atoms[s][j]
                        k = np.flatnonzero(dc>=0)
                        mass = z[k,dc[k]].sum(0)
                        for q,opts in enumerate(options(dc,examples,weights)[:2]):
                            donors[q].append(dict(mass=mass,options=opts,atoms=z,
                                easy=float(z[:,0,1].sum()),crowd=d['crowd'][ix]))
                thresholds = [anchor.model.calibrate(v) for v in donors]
                ix = source_ix[held]
                ch = np.broadcast_to(direct[ix],(len(ARMS),len(ix))).copy()
                for ri,g,j in group[held]:
                    dc,examples = proposals(raw[held][j],scale[held][j],b[ri],c[ri],g,co,rank)
                    np.testing.assert_array_equal(dc,direct[ri])
                    gn,rk,old_opts = options(dc,examples,weights)
                    for k,(opts,cut) in enumerate(zip((gn,rk,old_opts,gn,rk),
                        [q[0] for q in thresholds]+[-np.inf]*3),1):
                        a = anchor.model.first(opts,cut)
                        if a is not None:
                            _,row,fam,_ = a
                            ch[k,j[row]] = fam
                cache = mark.with_suffix('.npz')
                writer(cache,ids=d['global_ids'][ix],choices=ch,coefficients=weights)
                run.once(mark,run.stable(dict(identity=ident,cache=run.artifact(cache),
                    thresholds=[v[0] for v in thresholds],calibration=[v[1] for v in thresholds])))
            cell_out = run.read(mark)
            if cell_out['identity']!=ident:
                raise ValueError('Incompatible fold checkpoint')
            folds.append(cell_out)
            beat(state='fold_frozen',seed=seed,complete=len(folds))
        frozen.append(dict(seed=seed,folds=folds))
    run.once(PUBLIC/'freeze.json',run.stable(dict(cells=frozen,held_excluded=True)))


def score(d,refs,records,y,valid,beat):
    f = PUBLIC/'freeze.json'
    frozen = run.read(f)
    if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256']!=run.sha(f):
        raise ValueError('Require complete frozen choices')
    p = run.parent
    cuts = run.cuts_by_source()
    parent,controls,old = banks()
    parent = {c['seed']:c for c in parent}
    base = {(r['seed'],r['recording']):r['metrics']['direct'] for r in
            run.read(timing.PAST.PUBLIC/'analysis.json')['records']}
    pairs = dict(risk_vs_gain=('gain','risk',p.CV),
                 risk_vs_direct=('direct','risk',p.CV),gain_vs_direct=('direct','gain',p.CV),
                 diagnostic_risk_vs_gain=('diagnostic_gain','diagnostic_risk',p.CV),
                 diagnostic_risk_vs_old=('diagnostic_old','diagnostic_risk',p.CV))
    rows,recall = [],[]
    for cell in frozen['cells']:
        seed = cell['seed']
        c,b = run.candidates(d,refs,seed)
        direct,co = timing.parents(d,parent[seed],controls)
        ch = np.full((len(ARMS),len(c)),-1,np.int16)
        weights = {}
        for fold in cell['folds']:
            s = fold['identity']['held']
            at = d['sites']==s
            with np.load(run.checked(fold['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
                ch[:,at] = z['choices']
                weights[s] = z['coefficients'].copy()
        np.testing.assert_array_equal(ch[0],direct)
        for rec,(ix,g) in records.items():
            s = str(d['sites'][ix[0]])
            e,h = cuts[s]['easy'],cuts[s]['hard']
            z = ch[:,ix]
            dc = z[0]>=0
            on = z>=0
            if (z[:,dc]!=z[0,dc]).any() or (on&~dc).sum(1).max()>1:
                raise ValueError('An opening cannot modify direct decisions')
            for q in g:
                counts = on[3:,q].sum(1)
                if not (counts==counts[0]).all():
                    raise ValueError('Require matched diagnostic queries and counts')
            paths = {k:run.model.selected_paths(v,b[ix],c[ix]) for k,v in zip(ARMS,z)}
            out = p.readout.score_recording({p.CV:c[ix],**paths},pairs,y[ix],valid[ix],d['queries'][ix],
                    rec,easy_cut=e,hard_cut=h)
            if out['metrics']['direct']!=base[seed,rec]:
                raise ValueError('Original observed control must replay exactly')
            rows.append(dict(seed=seed,source=s,zero=not dc.any(),contrasts=out['comparisons'],
                risk={k:out['metrics'][k]['risk'] for k in ARMS}))
            if not dc.any():
                raw,scale = run.model.descriptors(d['history'][ix],b[ix],c[ix],d['crowd'][ix])
                with np.load(run.checked(old[seed,s]),allow_pickle=False) as cache:
                    rank = cache['coefficients']
                _,examples = proposals(raw,scale,b[ix],c[ix],g,co[s],rank)
                _,atoms,_ = anchor.model.labels(c[ix],b[ix],y[ix],valid[ix],e)
                targets = model.targets(atoms,scale,examples)
                cases = {}
                for label,k in (('first',1),('first_two',2),('any',len(examples))):
                    hits = dict(shortlist=False,old=False,gain=False,risk=False)
                    for sample,t in zip(examples[:k],targets[:k]):
                        _,i,j,_ = sample
                        u = t[:,4]>0.
                        hits['shortlist'] |= bool(u.any())
                        hits['old'] |= bool(u[0])
                        for n,(_,row,a) in zip(('gain','risk'),model.winners(sample,weights[s])):
                            hits[n] |= bool(u[(i==row)&(j==a)].any())
                    cases[label] = hits
                recall.append(cases)
        beat(state='seed_scored',seed=seed)
    sources = sorted({r['source'] for r in rows})
    if len(rows)!=102 or len(sources)!=14 or len(recall)!=18:
        raise ValueError('Incomplete registered rosters')
    hits = {k:{n:sum(r[k][n] for r in recall) for n in recall[0][k]} for k in recall[0]}
    prev = run.read(PAIR/'diverse_shortlist.json')['recall']
    if any(v['shortlist']!=prev[k]['old']['family'] for k,v in hits.items()):
        raise ValueError('Shortlist recall does not match the frozen family-diverse control')
    groups = [[r for r in rows if r['source']==s and r['seed']==k]
              for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14,size=(3000,14))
    ci = {}
    for name in pairs:
        vals = [p.readout.merge_comparisons([r['contrasts'][name] for r in g]) for g in groups]
        x = np.asarray([r['completion_lower_pp'] for r in vals],float).reshape(14,3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('All original source contrasts required')
        ci[name] = dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    risk = {}
    for arm in ARMS:
        v = [r['risk'][arm] for r in rows]
        risk[arm] = dict(switches=sum(x['selected'] for x in v),passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation']>.02 for x in v),
            zero_open=sum(r['zero'] and r['risk'][arm]['selected']>0 for r in rows),
            zero_pass=sum(r['zero'] and r['risk'][arm]['supported'] for r in rows))
    thresholds = {a:dict(Counter(str(f['thresholds'][j]) for c in frozen['cells'] for f in c['folds']))
                  for j,a in enumerate(('gain','risk'))}
    run.once(PUBLIC/'analysis.json',run.stable(dict(views=102,zero_views=18,thresholds=thresholds,risk=risk,
        contrasts=ci,recall=hits,original_rules_unchanged=True,independent_confirmation=False,
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
            raise ValueError('Register the experiment in Git before running')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if time.monotonic()-start>43200 or shutil.disk_usage(ROOT).free<2*1024**3:
                raise OSError('Resume from saved source checkpoints')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            print(msg,flush=True)
        d,refs = run.population(),run.register()
        lab = run.old.prior.arrays(('target_eval','valid','frames'))
        records = prior.blocks(d,lab['frames'])
        y,valid = lab['target_eval']-d['origin'][:,None],lab['valid']
        (fit if phase=='fit' else score)(d,refs,records,y,valid,beat)
        path = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),
            seconds=time.monotonic()-start,artifact=run.artifact(path)))


if __name__=='__main__':
    main()
