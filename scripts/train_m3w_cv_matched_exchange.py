"""Learn and evaluate recipient exchanges without spending additional query budget."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_fixed_recipient_support as prior_run
from src.world_model import m3w_cv_matched_exchange as model

old,base,prior,np,torch = prior_run.old,prior_run.base,prior_run.prior,prior_run.np,prior_run.torch
parent = prior_run.parent
OUT = prior_run.OUT.parent/'moving_cv_matched_recipient_exchange_v1'
PRIVATE = prior_run.PRIVATE.parent/OUT.name
STEPS = 1400
MODES = ('geometry','learned','guard')
ARMS = ('half_anchor',*MODES,'calibrated')
GRID = tuple((m,t) for m in MODES for t in (0.,.02,.10))


def make_sampler(tr,cut):
    cv,ols,b = old.teacher.analytic(tr['history'])
    past,s = prior_run.context.context(tr)
    diff = model.peak(ols-cv)
    legal = (b!=cv[:,None]).any((2,3))
    groups = [g for g in prior.query_groups(tr['query']) if len(g)>=2 and (diff[g]>0.).any()]
    sites = np.asarray([tr['sites'][g[0]] for g in groups])
    roles = [np.flatnonzero(sites==v) for v in sorted(set(tr['sites']))]
    if any(not len(g) for g in roles):
        raise ValueError('Each training source must contain analytic exchange pairs')

    def draw(rng):
        q = prior.draw(roles,rng)
        ij = np.asarray([rng.choice(groups[i],2,replace=False) for i in q])
        i,j = ij.T
        k = rng.integers(b.shape[1],size=len(i))
        peak = .5*model.peak(b[i,k]-cv[i])
        ok = legal[i,k] & (peak<=diff[j]) & (peak>0.)
        i,j,k,peak = (x[ok] for x in (i,j,k,peak))
        if not len(i):
            raise ValueError('No legal training swaps in sampled queries')
        u = cv[i]+.5*(b[i,k]-cv[i])
        p = model.incoming_path(cv[j],ols[j],peak)
        x = model.features(past,tr['history'],cv,ols,s,u,p,i,j)
        y,v = tr['target'],tr['valid']
        gain = model.targets(cv[j],ols[j],p,y[j],v[j],cut)-model.targets(cv[i],ols[i],u,y[i],v[i],cut)
        return torch.from_numpy(x),torch.from_numpy((gain/((s[i]+s[j])*.5)[:,None]).clip(-8.,8.).astype(np.float32))

    return draw,2*(past.shape[1]+37)+10


@torch.inference_mode()
def candidates(net,d,c,o,anchor):
    selected = (anchor!=c).any((1,2))
    g = prior.query_groups(d['query'])
    i,j,q = model.shortlist(c,o,anchor,selected,g)
    r = model.peak(anchor[i]-c[i])
    p = model.incoming_path(c[j],o[j],r)
    past,s = prior_run.context.context(d)
    scale = (s[i]+s[j])*.5
    old_credit = (np.linalg.norm(o[i]-c[i],axis=-1)-np.linalg.norm(o[i]-anchor[i],axis=-1)).mean(1)
    new_credit = (np.linalg.norm(o[j]-c[j],axis=-1)-np.linalg.norm(o[j]-p,axis=-1)).mean(1)
    geom = (new_credit-old_credit)/scale
    est = np.empty((len(i),5),np.float32)
    for start in range(0,len(i),4096):
        z = slice(start,start+4096)
        x = model.features(past,d['history'],c,o,s,anchor[i[z]],p[z],i[z],j[z])
        est[z] = net(torch.from_numpy(x)).numpy()
    if not np.isfinite(est).all():
        raise ValueError('Invalid exchange estimates')
    guard = (est[:,:2]>0.).all(1)&(est[:,2:]<=0.).all(1)
    utility = est[:,:2].min(1)-est[:,2:].clip(0.).sum(1)
    return i,j,q,p,{'geometry':geom,'learned':utility,'guard':np.where(guard,utility,-np.inf)}


def calibrate(c,o,u,d,choices,cut):
    scorer = prior.prior.parent.readout
    i,j,q,p,s = choices
    table = []
    recs = np.unique(d['recording'])
    for arm,t in GRID:
        k = model.decode(s[arm],q,t)
        z = model.apply(c,u,i,j,p,k)
        values = []
        for rec in recs:
            ix = d['recording']==rec
            y,v,w = (d[k][ix] for k in ('target','valid','query'))
            r = scorer.score_recording({base.CV:c[ix],'p':z[ix]}, {}, y,v,w,str(rec),
                easy_cut=cut[0],hard_cut=cut[1])
            s0, = np.unique(d['sites'][ix])
            g = [prior.study.motion.full_masses(b[ix],z[ix],c[ix],y,v,old.weight(w)) for b in (c,o)]
            values.append((s0,r['metrics']['p']['risk']['supported'],g))
        lo = [[scorer.merge_comparisons([x[2][k] for x in values if x[0]==source])['completion_lower_pp']
               for k in range(2)] for source in sorted(set(d['sites']))]
        table.append(dict(arm=arm,margin=t,exchanges=len(k),passes=sum(x[1] for x in values),
            lower_pp=np.mean(lo,axis=0).tolist()))
    good = [x for x in table if x['passes']==len(recs) and min(x['lower_pp'])>0.]
    return table,max(good,key=lambda x:(min(x['lower_pp']),-x['exchanges'])) if good else None


def cell(d,donor,frozen,lead,signed,reg,beat,replay):
    ident = frozen['identity']
    fold,seed = ident['fold'],ident['seed']
    cut = frozen['cut']
    if any(k['identity']!=ident or k['cut']!=cut for k in (lead,signed)):
        raise ValueError('Calibration parent roles changed')
    train = old.teacher.fit_role(d,ident)
    draw,w = make_sampler(train,cut[0])
    net,fit,updates = prior.prefix.trainer.fit(dict(ident,registration=reg,arm='exchange'),f'exchange_{fold}_{seed}',
        seed,lambda:prior.model.head(w,5),STEPS,draw,model.objective,replay,beat)
    _,cal = base.subset(d,ident['calibration'])
    cv,ols,b = old.teacher.analytic(cal['history'])
    width = prior.study.motion.features(cal['history'][:1],'prefix')[0].shape[1]+25
    a = old.cal_anchor(cal,cv,ols,b,signed,lead,width)
    u = cv+.5*(old.slate.path_for(cv,b,a)-cv)
    table,sel = calibrate(cv,ols,u,cal,candidates(net,cal,cv,ols,u),cut)
    hi,h = base.subset(d,ident['held'])
    cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    b = base.expanded(cv,raw)
    with np.load(old.checked(frozen['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        a = z['frozen']
    u = cv+.5*(old.slate.path_for(cv,b,a)-cv)
    i,j,q,p,s = candidates(net,h,cv,ols,u)
    arms = {m:model.decode(s[m],q,0. if m=='geometry' else .02) for m in MODES}
    if sel is not None:
        arms['calibrated'] = model.decode(s[sel['arm']],q,sel['margin'])
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',
        dict(ids=h['ids'],anchor=a,old=i,new=j,query=q,**arms))
    return dict(identity=ident,cut=cut,fit=fit,ladder=table,selected=sel,choices=path),updates


def upper(cv,ols,u,y,v,q,i,j,p):
    w = old.weight(q)
    _,_,base_gain,_ = model.candidate_terms(cv,ols,u[:,None],y,v)
    ref,offset,_,_ = model.candidate_terms(cv,ols,cv[:,None],y,v)
    norm = w@ref
    if norm<=0.:
        return None
    _,_,new,_ = model.candidate_terms(cv[j],ols[j],p[:,None],y[j],v[j])
    a = new[:,0]*w[j,None]-base_gain[i,0]*w[i,None]
    best = np.zeros((len(np.unique(q)),2))
    k = np.searchsorted(np.unique(q),q[i])
    np.maximum.at(best,k,a)
    z = np.array([0.,w@offset])+w@base_gain[:,0]
    return (100.*(z+best.sum(0))/norm).tolist()


def evaluate(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{m}_vs_{tag}':(base0,m) for m in ARMS for tag,base0 in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update({f'{m}_vs_anchor':('half_anchor',m) for m in MODES})
    pairs['guard_vs_geometry'] = ('geometry','guard')
    rows,limits,counts,geo = [],[],dict.fromkeys(MODES,0),{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(old.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            u = cv+.5*(old.slate.path_for(cv,b,z['anchor'])-cv)
            i,j,q = (z[k] for k in ('old','new','query'))
            p = model.incoming_path(cv[j],ols[j],model.peak(u[i]-cv[i]))
            paths = {base.CV:cv,base.OLS:ols,'half_anchor':u}
            for k in MODES:
                counts[k] += len(z[k])
                paths[k] = model.apply(cv,u,i,j,p,z[k])
            paths['calibrated'] = cv if c['selected'] is None else model.apply(cv,u,i,j,p,z['calibrated'])
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            z = {k:a[ix] for k,a in paths.items()}
            c0,o0 = cv[ix],ols[ix]
            r = scorer.score_recording(z,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k:prior.study.motion.full_masses(z[l],z[r],c0,y,v,w) for k,(l,r) in pairs.items()}
            s, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(s),seed=seed,comparisons=terms))
            ix2 = ix[i]
            at = np.flatnonzero(ix)
            limits.append(upper(c0,o0,u[ix],y,v,q,np.searchsorted(at,i[ix2]),
                np.searchsorted(at,j[ix2]),p[ix2]))
            agg = geo.setdefault((str(s),seed),np.zeros((len(MODES),4)))
            history = h['history'][ix]
            for g in prior.query_groups(q):
                radii = [model.peak(z[k][g]-c0[g]) for k in ('half_anchor',*MODES)]
                if len({int((x>0.).sum()) for x in radii})!=1:
                    raise ValueError('A query lost its count match')
                np.testing.assert_allclose([x.sum() for x in radii],radii[0].sum(),rtol=1e-6,atol=1e-8)
                a = np.stack([z['half_anchor'][g],*(z[k][g] for k in MODES)])
                _,l,_,(steps,known,_,_) = prior.geometry.query(history[g],a,y[g],v[g],tuple((0,k) for k in range(1,4)))
                if steps: agg[:,1] += l[:,1]; agg[:,3] += 1
                if known: agg[:,0] += l[:,0]; agg[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if summary['views']!=72 or any(summary['risk'][k]['switches']!=54748 for k in ('half_anchor',*MODES)):
        raise ValueError('Frozen intervention roster changed')
    sites = sorted(set(roster.values()))
    x = np.asarray([[geo[s,k] for k in base.SEEDS] for s in sites])
    p = np.divide(x[...,:2],x[...,2:],out=np.full_like(x[...,:2],np.nan),where=x[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    g = {k:dict(observed_pixel=prior.geometry.bootstrap(p[:,:,i,0],draw),
        completion_lower_pixel=prior.geometry.bootstrap(p[:,:,i,1],draw)) for i,k in enumerate(MODES)}
    a = np.asarray([v for v in limits if v is not None]).reshape(-1,2)
    cap = dict(scored=len(a),cv_excluded=int((a[:,0]<=0.).sum()),ols4_excluded=int((a[:,1]<=0.).sum()),
        either_excluded=int((a.min(1)<=0.).sum()),median_upper_pp=np.median(a,axis=0).tolist())
    return summary,g,counts,cap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',action='store_true')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            z = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',z)
            if v.get('phase') and (v['phase']!='fit' or v.get('step') in (100,STEPS)):
                print(z,flush=True)
        refs = (old.OUT/'freeze.json',old.OUT/'readout.json',parent.OUT/'freeze.json',
                parent.parent.OUT/'freeze.json',prior_run.OUT/'readout.json')
        f,r,l,s,previous = map(old.read,refs)
        if (r['freeze']!=old.sha(refs[0]) or any(z['a_labels_opened'] or z['independent_confirmation']
                or not z['original_2pct'] for z in (r,previous))):
            raise ValueError('Expected source-isolated development references')
        reg = OUT/'registration.json'
        old.once(reg,old.stable(dict(parents=[old.artifact(x) for x in refs],
            files=[old.artifact(x) for x in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
            steps=STEPS,arms=ARMS,grid=GRID,seeds=base.SEEDS,original_2pct=True)))
        _,donor,_ = base.register()
        d = base.fit_population(beat)
        cells,updates = [],0
        for a,b,c in zip(f['cells'],l['cells'],s['cells'],strict=True):
            x,n = cell(d,donor,a,b,c,old.sha(reg),beat,args.replay)
            cells.append(x)
            updates += n
            beat(phase='cell_frozen',fold=x['identity']['fold'],seed=x['identity']['seed'],updates=updates)
        fp = OUT/'freeze.json'
        old.once(fp,old.stable(dict(registration=old.sha(reg),cells=cells,optimizer_steps=9*STEPS,before_held_scoring=True)))
        summary,g,counts,cap = evaluate(old.read(fp)['cells'],d,donor,beat)
        if summary['risk']['half_anchor']!=previous['summary']['risk']['uniform']:
            raise ValueError('Previous shrinkage control did not reproduce')
        old.once(OUT/'readout.json',old.stable(dict(freeze=old.sha(fp),summary=summary,geometry=g,
            exchanges=counts,one_exchange_upper=cap,qualified=sum(x['selected'] is not None for x in cells),
            calibration_max_passes=[max(x['passes'] for x in c['ladder']) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
