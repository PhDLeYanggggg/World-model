"""Broader query-local intervention with predicted, shared risk slack."""
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_record_allocation as parent

rank,screen,base,prior,model = parent.rank,parent.screen,parent.base,parent.prior,parent.model
np,torch,read,sha,once,stable,artifact,checked = (
    base.np,base.torch,base.read,base.sha,base.once,base.stable,base.artifact,base.checked)
OUT = parent.OUT.parent/'moving_cv_pooled_coverage_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
GRID = tuple((cap,m,p) for cap in (8,32,None) for m in (0.,.005,.02) for p in (.10,.25))
ARMS = ('sparse','marginal','unpooled','pool8','pool32','pool_full','gain_match','gain_full','calibrated')


def independent(gain, legal, groups, limit=None, match=None):
    best = np.where(legal,gain,-np.inf).argmax(1)
    row = np.arange(len(best))
    value = np.where(legal[row,best],gain[row,best],-np.inf)
    out = np.full(len(row),-1,dtype=np.int8)
    for g in groups:
        k = len(g) if limit is None else limit
        if match is not None:
            k = int((match[g]>=0).sum())
        picked = g[np.argsort(-value[g],kind='stable')[:k]]
        picked = picked[np.isfinite(value[picked])]
        if match is not None and len(picked)!=k:
            raise ValueError('Gain control lost a matched edit')
        out[picked] = best[picked]
    return out


def pool_query(score,cost,easy,legal,limit,support):
    """Seed predicted support and spend only the current query's slack."""
    out = np.full(len(score),-1,dtype=np.int8)
    if not (legal & (score>0.)).any():
        return out
    donor = legal & (easy>=support) & (cost<=0.).all(2)
    # A donor can start the pool even when it is not the best action on its row.
    source_score = np.where(donor,score,-np.inf)
    i,j = np.unravel_index(source_score.argmax(),source_score.shape)
    if not donor[i,j]:
        return out
    balance = -cost[i,j].copy()
    out[i] = j
    total = score[i,j]
    count = 1
    order = np.argsort(-np.where(legal,score,-np.inf).max(1),kind='stable')
    for k in order:
        if count>=limit:
            break
        if k==i:
            continue
        feasible = legal[k] & (score[k]>0.) & (cost[k]<=balance).all(1)
        if not feasible.any():
            continue
        a = int(np.where(feasible,score[k],-np.inf).argmax())
        balance -= cost[k,a]
        total += score[k,a]
        out[k] = a
        count += 1
    if total<=0.:
        out.fill(-1)
    return out


def pooled(pred,groups,scale,limit,margin,support):
    g,r,e,legal = pred
    out = np.full(len(g),-1,dtype=np.int8)
    support_only = 0
    for ix in groups:
        v = g[ix]*scale[ix,None]
        cost = (r[ix]+margin)*scale[ix,None,None]
        pick = pool_query(v,cost,e[ix],legal[ix],len(ix) if limit is None else limit,support)
        at = np.flatnonzero(pick>=0)
        a = pick[at]
        if len(at):
            if (cost[at,a].sum(0)>1e-7).any() or not (e[ix][at,a]>=support).any():
                raise ValueError('Predicted slack or support invariant failed')
            support_only += int(((cost[at,a]>0.).any(1) | (e[ix][at,a]<support)).sum())
        out[ix] = pick
    return out,support_only


def calibrate(pred,scale,groups,cv,ols,bank,d,cut):
    out = []
    expected = len(np.unique(d['recording']))
    for limit,margin,support in GRID:
        a,_ = pooled(pred,groups,scale,limit,margin,support)
        rows = base.metrics(cv,ols,bank,d,{'strict':a},cut)
        lower = [[prior.prior.parent.readout.merge_comparisons(
            [r['comparisons'][f'strict_vs_{k}'] for r in rows if r['source']==s])['completion_lower_pp']
            for k in ('cv','ols4')] for s in sorted(set(d['sites']))]
        out.append(dict(limit=limit,margin=margin,min_support=support,edits=int((a>=0).sum()),
            passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
            lower_pp=np.mean(lower,axis=0).tolist()))
    good = [r for r in out if r['edits'] and r['passes']==expected and min(r['lower_pp'])>0.]
    return out,max(good,key=lambda r:(min(r['lower_pp']),-r['edits'])) if good else None


def cell(old, alloc, d, donor):
    ident = old['identity']
    if ident!=alloc['identity'] or old['fits'][0]['identity']!=dict(ident,head='risk'):
        raise ValueError('Risk fit lineage changed')
    roles = [set(ident[k]) for k in ('train','calibration','held')]
    if any(len(s)!=3 for s in roles) or len(set.union(*roles))!=9:
        raise ValueError('Expected disjoint fitting roles')
    fold,seed = ident['fold'],ident['seed']
    _,c = base.subset(d,ident['calibration'])
    hi,h = base.subset(d,ident['held'])
    w = prior.study.motion.features(c['history'][:1],'prefix')[0].shape[1]+25
    net = screen.load_head(old['fits'][0],w,8)
    cal = prior.study.motion.bank(c['history'])
    cb = base.expanded(cal[:,0],cal[:,1:])
    cp = screen.predictions(net,cal[:,0],cb,c['history'])
    ladder,selected = calibrate(cp,prior.model.scale(c['history']),prior.query_groups(c['query']),
                               cal[:,0],cal[:,1],cb,c,old['cut'])
    cv,_,paths = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    pred = screen.predictions(net,cv,base.expanded(cv,paths),h['history'])
    groups = prior.query_groups(h['query'])
    scale = prior.model.scale(h['history'])
    g,r,e,legal = pred
    with np.load(checked(alloc['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        sparse = z['local']
    a = dict(sparse=sparse,marginal=independent(g,model.admissible(g,r,e,legal,*screen.RELAXED),groups),
        unpooled=independent(g,model.admissible(g,r,e,legal,0.,.10),groups))
    pooled_only = {}
    for label,cap in (('pool8',8),('pool32',32),('pool_full',None)):
        a[label],pooled_only[label] = pooled(pred,groups,scale,cap,0.,.10)
    a.update(gain_match=independent(g,legal,groups,match=a['pool_full']),
             gain_full=independent(g,legal & (g>0.),groups),
             calibrated=(np.full_like(sparse,-1) if selected is None else pooled(pred,groups,scale,
                 selected['limit'],selected['margin'],selected['min_support'])[0]))
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=h['ids'],**a))
    return dict(identity=ident,cut=old['cut'],actions=path,selection=selected,ladder=ladder,pooled_only=pooled_only)


def evaluate(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    contrast = {f'{a}_vs_{k}':(b,a) for a in ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    contrast.update(pool_vs_gain=('gain_match','pool_full'),pool_vs_marginal=('marginal','pool_full'),
                    pool_vs_unpooled=('unpooled','pool_full'))
    geo_keys = ('pool_vs_gain','pool_vs_marginal')
    rows, geo = [],{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            act = {k:z[k] for k in ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            c0,o = cv[ix],ols[ix]
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            paths = {base.CV:c0,base.OLS:o,**{k:prior.apply(c0,bank[ix],a[ix]) for k,a in act.items()}}
            r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            effects = {k:prior.study.motion.full_masses(paths[a],paths[b],c0,y,v,1./n[inv]) for k,(a,b) in contrast.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=effects))
            z = geo.setdefault((str(source),seed),np.zeros((2,4)))
            past = h['history'][ix]
            for block in prior.query_groups(q):
                if (act['pool_full'][ix][block]>=0).sum()!=(act['gain_match'][ix][block]>=0).sum():
                    raise ValueError('Per-query count changed')
                arr = np.stack([paths[k][block] for k in ('gain_match','marginal','pool_full')])
                _,v0,_,(steps,known,_,_) = prior.geometry.query(past[block],arr,y[block],v[block],((0,2),(1,2)))
                if steps: z[:,1] += v0[:,1]; z[:,3] += 1
                if known: z[:,0] += v0[:,0]; z[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['sparse']['switches'])!=(72,9,2317):
        raise ValueError('Unexpected held fitting roster')
    sources = sorted(set(r['source'] for r in rows))
    g = np.asarray([[geo[s,k] for k in base.SEEDS] for s in sources])
    f = np.divide(g[...,:2],g[...,2:],out=np.full_like(g[...,:2],np.nan),where=g[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    geometry = {k:dict(observed_pixel=prior.geometry.bootstrap(f[:,:,i,0],draw),
                      completion_lower_pixel=prior.geometry.bootstrap(f[:,:,i,1],draw)) for i,k in enumerate(geo_keys)}
    return summary,geometry


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if 'phase' in v:
                print(info,flush=True)

        reg,donor,_ = base.register()
        a,b = base.OUT/'freeze.json',parent.OUT/'freeze.json'
        train,alloc = read(a),read(b)
        if (train['registration']!=reg or alloc['registration']!=sha(parent.OUT/'registration.json')
                or not all(x['before_held_scoring'] and len(x['cells'])==9 for x in (train,alloc))):
            raise ValueError('Require frozen fitting-source assets')
        register = OUT/'registration.json'
        once(register,stable(dict(parents=[artifact(a),artifact(b)],files=[artifact(k) for k in
            (Path(__file__),Path(screen.__file__),OUT/'protocol.md')],
            grid=GRID,seeds=base.SEEDS,original_2pct=True,independent_confirmation=False)))
        d = base.fit_population(beat)
        freeze = OUT/'freeze.json'
        if not freeze.exists():
            cells = []
            for c,s in zip(train['cells'],alloc['cells'],strict=True):
                z = cell(c,s,d,donor)
                cells.append(z)
                beat(phase='choice_frozen',fold=z['identity']['fold'],seed=z['identity']['seed'])
            once(freeze,stable(dict(registration=sha(register),cells=cells,before_held_scoring=True)))
        b = read(freeze)
        if b['registration']!=sha(register) or not b['before_held_scoring'] or len(b['cells'])!=9:
            raise ValueError('All predictions must freeze first')
        result,geom = evaluate(b['cells'],d,donor,beat)
        once(OUT/'readout.json',stable(dict(freeze=sha(freeze),summary=result,geometry=geom,optimizer_updates=0,
            qualifying=sum(c['selection'] is not None for c in b['cells']),
            calibration_max_passes=[max(r['passes'] for r in c['ladder']) for c in b['cells']],
            pooled_only={k:sum(c['pooled_only'][k] for c in b['cells']) for k in ('pool8','pool32','pool_full')},
            original_2pct=True,independent_confirmation=False,a_labels_opened=False)))
        beat(phase='complete')


if __name__=='__main__':
    main()
