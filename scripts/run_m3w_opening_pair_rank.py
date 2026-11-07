"""Fit matched Torch rankers and test shortlist recall and unchanged observed risk."""
import argparse
from collections import Counter
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opening_stop_wait as stop
from src.world_model import m3w_opening_pair_rank as model

run,np,prior,anchor,timing = stop.run,stop.np,stop.prior,stop.anchor,stop.timing
PUBLIC, PRIVATE = prior.PUBLIC/'opening_pair_rank', prior.PRIVATE/'opening_pair_rank'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(model.__file__),PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_opening_pair_rank.py')
HEADS = ('pointwise','pair')
ARMS = ('direct',*HEADS,'diagnostic_old','diagnostic_pointwise','diagnostic_pair')
K = (1,3,8)


def register():
    doc = stop.PUBLIC/'analysis.json'
    proof = stop.PUBLIC/'replay_execution.json'
    if run.read(proof)['artifact']['sha256']!=run.sha(doc):
        raise ValueError('Require previous full verified scoring')
    parents = (stop.REG,doc,proof,stop.PUBLIC/'rank_recall.json',anchor.REG,
               anchor.PUBLIC/'freeze.json',anchor.SIZE.PUBLIC/'freeze.json',
               timing.PAST.PUBLIC/'freeze.json',Path(run.model.__file__),Path(anchor.model.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents],steps=model.STEPS,
        batch_queries=model.QUERIES_PER_UPDATE,pair_weight=model.PAIR_WEIGHT,arms=ARMS,
        original_budget=.02,torch_threads=4,loader_workers=0,independent_outcomes_opened=False)))


def observed(g,b,c,raw,scale,co,rank,nets):
    u,a,e,ok = run.model.decision_costs(raw,co,b,c,scale)
    mint = prior.nested.base.select(g,u,a,e,scale,ok)
    dc,_ = prior.model.stream(g,u/scale[:,None],a,e,ok,counts=[int((mint[j]>=0).sum()) for j in g],witness=mint)
    x = model.features(raw)
    scores = [model.predict(net,x) for net in nets]
    opts = [anchor.model.prefix_options(g,dc,v,ok) for v in scores]
    old = anchor.model.prefix_options(g,dc,anchor.model.predict(raw,rank)[0],ok)
    return dc,opts,old


def fit(d,refs,records,y,valid,beat):
    writer = anchor.SIZE.base.parent.base.nested.base.parent.base.write_array
    groups = {str(s):np.flatnonzero(d['sites']==s) for s in np.unique(d['sites'])}
    recs = {s:[] for s in groups}
    for rec,(ix,g) in records.items():
        s = str(d['sites'][ix[0]])
        recs[s].append((rec,ix,g,np.searchsorted(groups[s],ix)))
    cuts = run.cuts_by_source()
    controls = {c['seed']:c['selection'] for c in run.read(anchor.SIZE.PUBLIC/'freeze.json')['cells']}
    old = {(c['seed'],f['identity']['held']):f for c in run.read(anchor.PUBLIC/'freeze.json')['cells']
           for f in c['folds']}
    frozen = []
    for parent in run.read(timing.PAST.PUBLIC/'freeze.json')['cells']:
        seed = parent['seed']
        c,b = run.candidates(d,refs,seed)
        direct,coefs = timing.parents(d,parent,controls)
        raw,scale,atoms,useful,legal = {},{},{},{},{}
        for s,ix in groups.items():
            raw[s],scale[s] = run.model.descriptors(d['history'][ix],b[ix],c[ix],d['crowd'][ix])
            label,z,_ = anchor.model.labels(c[ix],b[ix],y[ix],valid[ix],cuts[s]['easy'])
            atoms[s],useful[s] = z,label[...,0] & (z[...,5]>0.)
            legal[s] = np.linalg.norm(b[ix]-c[ix,None],axis=-1).max(2)>0.
        folds = []
        for cell in controls[seed]:
            ident = cell['identity']
            held,learn,cal = ident['held'],ident['fit'],ident['cal_a']+ident['cal_b']
            identity = dict(held=held,seed=seed,fit=learn,cal=cal,registration=run.sha(REG))
            if len(learn)!=7 or len(cal)!=6 or len(set(learn+cal+[held]))!=14:
                raise ValueError('Require disjoint fitting, calibration and held sources')
            mark = PRIVATE/f'{seed}_{held}.json'
            if not mark.exists():
                with np.load(run.checked(old[seed,held]['cache']),allow_pickle=False) as z:
                    rank = z['coefficients'].copy()
                pools,fs = [],[]
                offset = 0
                for s in learn:
                    x = model.features(raw[s])
                    fs.append(x.reshape(-1,x.shape[-1]))
                    q = [j[g0] for _,_,g,j in recs[s] for g0 in g]
                    priority = anchor.model.predict(raw[s],rank)[0]
                    p = model.query_pool(q,legal[s],useful[s],priority)
                    pools.append([tuple(v+offset for v in t) for t in p])
                    offset += len(fs[-1])
                x = np.concatenate(fs)
                del fs
                plan = model.batch_plan(pools,seed)
                checksum = hashlib.sha256()
                for batch in plan:
                    for pair in batch:
                        for v in pair:
                            checksum.update(np.r_[len(v),v].astype('<i8').tobytes())
                nets,checkpoints = [],[]
                for name,w in zip(HEADS,(0.,model.PAIR_WEIGHT)):
                    cp = mark.with_name(mark.stem+'_'+name+'.pt')
                    nets.append(model.train(x,plan,seed,w,cp,
                        lambda **z:beat(state='training',seed=seed,fold=len(folds)+1,arm=name,**z)))
                    checkpoints.append(run.artifact(cp))
                del x
                co = coefs[held]
                donors = [[],[]]
                for s in cal:
                    for _,ix,g,j in recs[s]:
                        dc,options,_ = observed(g,b[ix],c[ix],raw[s][j],scale[s][j],co,rank,nets)
                        z = atoms[s][j]
                        k = np.flatnonzero(dc>=0)
                        mass = z[k,dc[k]].sum(0)
                        for index,opts in enumerate(options):
                            donors[index].append(dict(mass=mass,options=opts,atoms=z,
                                easy=float(z[:,0,1].sum()),crowd=d['crowd'][ix]))
                thresholds = [anchor.model.calibrate(v) for v in donors]
                ix = groups[held]
                ch = np.broadcast_to(direct[ix],(len(ARMS),len(ix))).copy()
                for _,ri,g,j in recs[held]:
                    dc,opts,base = observed(g,b[ri],c[ri],raw[held][j],scale[held][j],co,rank,nets)
                    np.testing.assert_array_equal(dc,direct[ri])
                    for k,(items,cut) in enumerate(zip(opts+[base]+opts,[a[0] for a in thresholds]+[-np.inf]*3),1):
                        pick = anchor.model.first(items,cut)
                        if pick is not None:
                            _,i,a,_ = pick
                            ch[k,j[i]] = a
                cache = mark.with_suffix('.npz')
                writer(cache,ids=d['global_ids'][ix],choices=ch)
                run.once(mark,run.stable(dict(identity=identity,cache=run.artifact(cache),
                    checkpoints=checkpoints,plan_sha256=checksum.hexdigest(),
                    thresholds=[v[0] for v in thresholds],calibration=[v[1] for v in thresholds],
                    fitting_queries=sum(map(len,pools)))))
            r = run.read(mark)
            if r['identity']!=identity:
                raise ValueError('Saved fitting cell mismatch')
            folds.append(r)
            beat(state='fold_frozen',seed=seed,complete=len(folds))
        frozen.append(dict(seed=seed,folds=folds))
    run.once(PUBLIC/'freeze.json',run.stable(dict(cells=frozen,held_excluded=True)))


def recall(raw,c,b,g,y,valid,cut,nets):
    labels,z,_ = anchor.model.labels(c,b,y,valid,cut)
    good = labels[...,0] & (z[...,5]>0.)
    legal = np.linalg.norm(b-c[:,None],axis=-1).max(2)>0.
    queries = [q for q in g if len(q)>1 and legal[q].any()]
    x = model.features(raw)
    out = {}
    scores = [model.predict(net,x) for net in nets]
    for n,subset in (('first',queries[:1]),('first_two',queries[:2]),('any',queries)):
        hit = {'bank':any(good[j].any() for j in subset)}
        for arm,s in zip(HEADS,scores):
            p = {k:False for k in K}
            for q in subset:
                ix = np.flatnonzero(legal[q].ravel())
                pick = ix[np.argsort(-s[q].ravel()[ix],kind='stable')[:max(K)]]
                for k in K:
                    p[k] |= bool(good[q].ravel()[pick[:k]].any())
            hit.update({f'{arm}_top{k}':v for k,v in p.items()})
        out[n] = hit
    return out


def score(d,refs,records,y,valid,beat):
    freeze = PUBLIC/'freeze.json'
    f = run.read(freeze)
    if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256']!=run.sha(freeze):
        raise ValueError('Require frozen decisions')
    p = run.parent
    cuts = run.cuts_by_source()
    base = {(r['seed'],r['recording']):r['metrics']['direct'] for r in
            run.read(timing.PAST.PUBLIC/'analysis.json')['records']}
    pairs = dict(pointwise_vs_direct=('direct','pointwise',p.CV),pair_vs_direct=('direct','pair',p.CV),
        pair_vs_pointwise=('pointwise','pair',p.CV),
        diagnostic_pair_vs_old=('diagnostic_old','diagnostic_pair',p.CV),
        diagnostic_pair_vs_pointwise=('diagnostic_pointwise','diagnostic_pair',p.CV))
    rows,recalled = [],[]
    for cell in f['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        ch = np.full((len(ARMS),len(cv)),-1,np.int16)
        cache = {}
        for fold in cell['folds']:
            s = fold['identity']['held']
            at = d['sites']==s
            with np.load(run.checked(fold['cache']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
                ch[:,at] = z['choices']
            cache[s] = fold['checkpoints']
        for rec,(ix,g) in records.items():
            s = str(d['sites'][ix[0]])
            e,h = cuts[s]['easy'],cuts[s]['hard']
            c,b = cv[ix],bank[ix]
            v = ch[:,ix]
            direct = v[0]>=0
            chosen = v>=0
            if (v[:,direct]!=v[0,direct]).any() or (chosen&~direct).sum(1).max()>1:
                raise ValueError('Only one additional opening is allowed')
            for j in g:
                a = chosen[3:,j].sum(1)
                if not (a==a[0]).all():
                    raise ValueError('Diagnostic queries and counts must match')
            paths = {k:run.model.selected_paths(a,b,c) for k,a in zip(ARMS,v)}
            out = p.readout.score_recording({p.CV:c,**paths},pairs,y[ix],valid[ix],d['queries'][ix],
                                             rec,easy_cut=e,hard_cut=h)
            if out['metrics']['direct']!=base[seed,rec]:
                raise ValueError('Require exact direct control')
            rows.append(dict(seed=seed,source=s,zero=not direct.any(),
                             risk={k:out['metrics'][k]['risk'] for k in ARMS},contrasts=out['comparisons']))
            if not direct.any():
                raw,_ = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
                width = raw.shape[2]-1+raw.shape[1]
                nets = []
                for cp in cache[s]:
                    net = model.network(width,seed)
                    t = model.torch.load(run.checked(cp),map_location='cpu',weights_only=True)
                    if t['step']!=model.STEPS:
                        raise ValueError('Incomplete fit')
                    net.load_state_dict(t['network'])
                    nets.append(net.eval())
                recalled.append(recall(raw,c,b,g,y[ix],valid[ix],e,nets))
        beat(state='seed_scored',seed=seed)
    sites = sorted({r['source'] for r in rows})
    if len(rows)!=102 or len(sites)!=14 or len(recalled)!=18:
        raise ValueError('Require the full registered populations')
    hit = {k:{n:sum(r[k][n] for r in recalled) for n in recalled[0][k]} for k in recalled[0]}
    old = run.read(stop.PUBLIC/'rank_recall.json')['counts']
    if any(hit[k]['bank']!=v['bank'] for k,v in old.items()):
        raise ValueError('The frozen bank and positive label definition must agree')
    groups = [[r for r in rows if r['source']==s and r['seed']==k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14,size=(3000,14))
    ci = {}
    for name in pairs:
        vals = [p.readout.merge_comparisons([r['contrasts'][name] for r in g]) for g in groups]
        x = np.asarray([r['completion_lower_pp'] for r in vals],float).reshape(14,3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('All source contrasts required')
        ci[name] = dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    risk = {}
    for a in ARMS:
        vals = [r['risk'][a] for r in rows]
        risk[a] = dict(switches=sum(x['selected'] for x in vals),passes=sum(x['supported'] for x in vals),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in vals),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation']>.02 for x in vals),
            zero_pass=sum(r['zero'] and r['risk'][a]['supported'] for r in rows),
            zero_open=sum(r['zero'] and r['risk'][a]['selected']>0 for r in rows))
    threshold = {a:dict(Counter(str(z['thresholds'][j]) for c in f['cells'] for z in c['folds']))
                 for j,a in enumerate(HEADS)}
    run.once(PUBLIC/'analysis.json',run.stable(dict(views=102,zero_views=18,recall=hit,old_recall=old,risk=risk,
        contrasts=ci,thresholds=threshold,training_jobs=84,updates_per_job=model.STEPS,
        original_rules_unchanged=True,independent_confirmation=False,deployment_changed=False,
        stage5c_executed=False,smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','fit','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit definition before running')
    model.torch.set_num_threads(4)
    model.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            if time.monotonic()-start>43200 or shutil.disk_usage(ROOT).free<2*1024**3:
                raise OSError('Resume from saved fold and update checkpoints')
            msg = dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            if v['state']!='training' or v['step']==model.STEPS:
                print(msg,flush=True)
        d,refs = run.population(),run.register()
        label = run.old.prior.arrays(('target_eval','valid','frames'))
        records = prior.blocks(d,label['frames'])
        y,valid = label['target_eval']-d['origin'][:,None],label['valid']
        (fit if phase=='fit' else score)(d,refs,records,y,valid,beat)
        path = PUBLIC/('freeze.json' if phase=='fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            artifact=run.artifact(path),torch_threads=model.torch.get_num_threads()))


if __name__=='__main__':
    main()
