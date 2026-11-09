"""Fit and calibrate causal equal-peak replacements on frozen recipients."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_slate_supervision as old
from scripts import train_m3w_cv_minimum_budget as context
from src.world_model import m3w_cv_fixed_recipient_support as model

base,prior,np,torch = old.base,old.prior,old.np,old.torch
parent = old.parent
OUT = old.OUT.parent/'moving_cv_fixed_recipient_support_v1'
PRIVATE = old.PRIVATE.parent/OUT.name
MODES = ('uniform','pooled','consensus')
ARMS = ('frozen',*MODES,'calibrated')
ALPHAS = (.25,.50,.75)
STEPS = 1000


def sampling(tr,cut):
    c,o,b = old.teacher.analytic(tr['history'])
    h,s = context.context(tr)
    legal = (b!=c[:,None]).any((2,3))
    groups = [np.flatnonzero((tr['sites']==site)&(legal.sum(1)>=2)) for site in sorted(set(tr['sites']))]
    if any(not len(g) for g in groups):
        raise ValueError('Source needs analytic path pairs')

    def sampler(rng,g):
        ix = prior.draw(g,rng)
        key = np.where(legal[ix],rng.random(legal[ix].shape),-1.)
        a,j = np.argsort(key,axis=1)[:,-2:].T
        anchor,donor = b[ix,a],b[ix,j]
        alpha = np.asarray(ALPHAS)[rng.integers(len(ALPHAS),size=len(ix))]
        delta,d = anchor-c[ix],donor-c[ix]
        u = c[ix]+alpha[:,None,None]*delta
        rr,r = model.peak(u-c[ix]),model.peak(d)
        p = c[ix]+(rr/r)[:,None,None]*d
        p[r<rr] = u[r<rr]
        x = model.features(h[ix],c[ix],o[ix],u,p,s[ix])
        y = model.targets(c[ix],o[ix],u,p,tr['target'][ix],tr['valid'][ix],s[ix],cut)
        return torch.from_numpy(x),torch.from_numpy(y)

    return sampler,groups,h.shape[1]+74


@torch.inference_mode()
def predict(nets,d,cv,ols,b,anchor,alpha):
    past,scale = context.context(d)
    at = np.flatnonzero(anchor>=0)
    out = {k:np.zeros(len(at),np.int8) for k in MODES}
    for start in range(0,len(at),2048):
        ix = at[start:start+2048]
        paths,legal = model.project(cv[ix],b[ix,anchor[ix]],b[ix],alpha)
        preds = np.empty((len(nets),*legal.shape,8),np.float32)
        for j in range(paths.shape[1]):
            x = torch.from_numpy(model.features(past[ix],cv[ix],ols[ix],paths[:,0],paths[:,j],scale[ix]))
            for n,net in enumerate(nets):
                preds[n,:,j] = net(x).numpy()
        if not np.isfinite(preds).all():
            raise ValueError('Nonfinite action predictions')
        for k in MODES[1:]:
            out[k][start:start+len(ix)] = model.choose(preds[0],preds[1:],legal,k)
    return out


def paths(cv,b,anchor,choice,alpha):
    at = np.flatnonzero(anchor>=0)
    if len(choice)!=len(at):
        raise ValueError('Direction choices do not match frozen recipients')
    p = cv.copy()
    for start in range(0,len(at),4096):
        ix = at[start:start+4096]
        a = choice[start:start+len(ix)]
        candidate,legal = model.project(cv[ix],b[ix,anchor[ix]],b[ix],alpha)
        row = np.arange(len(ix))
        if not legal[row,a].all():
            raise ValueError('Direction is not eligible at requested radius')
        p[ix] = candidate[row,a]
        np.testing.assert_allclose(model.peak(p[ix]-cv[ix]),model.peak(candidate[:,0]-cv[ix]),rtol=1e-6,atol=1e-8)
    np.testing.assert_array_equal((p!=cv).any((1,2)),anchor>=0)
    return p


def calibrate(nets,d,c,o,b,a,cut):
    scorer = prior.prior.parent.readout
    tab = []
    rec = np.unique(d['recording'])
    for alpha in ALPHAS:
        p = predict(nets,d,c,o,b,a,alpha)
        for arm,z in p.items():
            yhat = paths(c,b,a,z,alpha)
            rows = []
            for k in rec:
                ix = d['recording']==k
                y,v,q = (d[key][ix] for key in ('target','valid','query'))
                score = scorer.score_recording({base.CV:c[ix],'p':yhat[ix]}, {}, y,v,q,str(k),
                    easy_cut=cut[0],hard_cut=cut[1])
                w = old.weight(q)
                g = [prior.study.motion.full_masses(base0[ix],yhat[ix],c[ix],y,v,w) for base0 in (c,o)]
                source, = np.unique(d['sites'][ix])
                rows.append((source,score['metrics']['p']['risk']['supported'],g))
            gains = [[scorer.merge_comparisons([z[2][j] for z in rows if z[0]==s])['completion_lower_pp']
                      for j in range(2)] for s in sorted(set(d['sites']))]
            tab.append(dict(arm=arm,alpha=alpha,edits=int((a>=0).sum()),changed_directions=int((z>0).sum()),
                passes=sum(z[1] for z in rows),lower_pp=np.mean(gains,axis=0).tolist()))
    good = [x for x in tab if x['edits'] and x['passes']==len(rec) and min(x['lower_pp'])>0.]
    return tab,max(good,key=lambda r:(min(r['lower_pp']),-r['changed_directions'])) if good else None


def fit_cell(data,donor,frozen,lead,signed,reg,beat,replay):
    ident = frozen['identity']
    fold,seed = ident['fold'],ident['seed']
    if any(c['identity']!=ident or c['cut']!=frozen['cut'] for c in (lead,signed)):
        raise ValueError('Frozen source/seed roles changed')
    cut = frozen['cut']
    tr = old.teacher.fit_role(data,ident)
    sample,groups,width = sampling(tr,cut[0])
    fits,nets,updates = [],[],0
    for key,g in [('pooled',groups),*((f'source_{i}',[x]) for i,x in enumerate(groups))]:
        fit_id = dict(ident,arm=key,registration=reg)
        net,receipt,u = prior.prefix.trainer.fit(fit_id,f'{key}_{fold}_{seed}',seed,
            lambda:prior.model.head(width,8),STEPS,lambda rng:sample(rng,g),model.objective,replay,beat)
        fits.append(receipt)
        nets.append(net)
        updates += u
    _,d = base.subset(data,ident['calibration'])
    c,o,b = old.teacher.analytic(d['history'])
    w = prior.study.motion.features(d['history'][:1],'prefix')[0].shape[1]+25
    a = old.cal_anchor(d,c,o,b,signed,lead,w)
    table,sel = calibrate(nets,d,c,o,b,a,cut)
    hi,h = base.subset(data,ident['held'])
    c,o,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
    b = base.expanded(c,raw)
    with np.load(old.checked(frozen['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        anchor = z['frozen']
    p = predict(nets,h,c,o,b,anchor,.5)
    if sel is not None:
        t = p if sel['alpha']==.5 else predict(nets,h,c,o,b,anchor,sel['alpha'])
        p['calibrated'] = t[sel['arm']]
    artifact = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',
        dict(ids=h['ids'],anchor=anchor,**p))
    return dict(identity=ident,cut=cut,fits=fits,ladder=table,selected=sel,choices=artifact),updates


def upper(c,o,b,a,y,v,q):
    """Upper envelope for each comparator separately, not a joint witness."""
    w = old.weight(q)
    ref,offset,_,_ = model.candidate_terms(c,o,c[:,None],y,v)
    norm = w@ref
    if norm<=0.:
        return None
    at = np.flatnonzero(a>=0)
    total = np.array([0.,w@offset])
    for start in range(0,len(at),2048):
        ix = at[start:start+2048]
        bank,legal = model.project(c[ix],b[ix,a[ix]],b[ix],.5)
        _,_,gain,_ = model.candidate_terms(c[ix],o[ix],bank,y[ix],v[ix])
        total += w[ix]@np.where(legal[...,None],gain,-np.inf).max(1)
    return (100.*total/norm).tolist()


def score(cells,data,donor,beat):
    scorer = prior.prior.parent.readout
    primary = dict(pooled_vs_uniform=('uniform','pooled'),consensus_vs_uniform=('uniform','consensus'))
    pairs = {f'{a}_vs_{k}':(b,a) for a in ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(primary)
    rows,geo,limits,changes = [],{},[],dict.fromkeys(MODES,0)
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(data,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
        b = base.expanded(cv,raw)
        with np.load(old.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = z['anchor']
            p = {k:paths(cv,b,a,z[k],.5) for k in MODES}
            for k in MODES:
                changes[k] += int((z[k]>0).sum())
            p['calibrated'] = cv if c['selected'] is None else paths(cv,b,a,z['calibrated'],c['selected']['alpha'])
        p.update({base.CV:cv,base.OLS:ols,'frozen':old.slate.path_for(cv,b,a)})
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            z = {k:p[k][ix] for k in p}
            r = scorer.score_recording(z,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k:prior.study.motion.full_masses(z[l],z[r],cv[ix],y,v,w) for k,(l,r) in pairs.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=terms))
            limits.append(upper(cv[ix],ols[ix],b[ix],a[ix],y,v,q))
            agg = geo.setdefault((str(source),seed),np.zeros((2,4)))
            history = h['history'][ix]
            for g in prior.query_groups(q):
                arr = np.stack([z[k][g] for pair in primary.values() for k in pair])
                _,u,_,(steps,known,_,_) = prior.geometry.query(history[g],arr,y[g],v[g],((0,1),(2,3)))
                if steps: agg[:,1] += u[:,1]; agg[:,3] += 1
                if known: agg[:,0] += u[:,0]; agg[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if summary['views']!=72 or any(summary['risk'][k]['switches']!=54748 for k in (*MODES,'frozen')):
        raise ValueError('Held frozen recipients changed')
    sites = sorted(set(roster.values()))
    x = np.asarray([[geo[s,k] for k in base.SEEDS] for s in sites])
    p = np.divide(x[...,:2],x[...,2:],out=np.full_like(x[...,:2],np.nan),where=x[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    g = {k:dict(observed_pixel=prior.geometry.bootstrap(p[:,:,i,0],draw),
        completion_lower_pixel=prior.geometry.bootstrap(p[:,:,i,1],draw)) for i,k in enumerate(primary)}
    lim = np.asarray([z for z in limits if z is not None]).reshape(-1,2)
    cap = dict(scored_views=len(lim),undefined=len(limits)-len(lim),
        cv_nonpositive=int((lim[:,0]<=0.).sum()),ols4_nonpositive=int((lim[:,1]<=0.).sum()),
        either_nonpositive=int((lim.min(1)<=0.).sum()),median_upper_pp=np.median(lim,axis=0).tolist())
    return summary,g,cap,changes


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
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if v.get('phase') and (v['phase']!='fit' or v.get('step') in (100,STEPS)):
                print(info,flush=True)
        refs = (old.OUT/'freeze.json',old.OUT/'readout.json',parent.OUT/'freeze.json',parent.parent.OUT/'freeze.json')
        f,r,l,s = map(old.read,refs)
        if (r['freeze']!=old.sha(refs[0]) or r['a_labels_opened'] or r['independent_confirmation']
                or not r['original_2pct'] or not all(z['before_held_scoring'] for z in (f,l,s))):
            raise ValueError('Expected frozen fitting-only controls')
        reg = OUT/'registration.json'
        old.once(reg,old.stable(dict(parents=[old.artifact(x) for x in refs],files=[old.artifact(x) for x in
            (Path(__file__),Path(model.__file__),OUT/'protocol.md')],steps=STEPS,seeds=base.SEEDS,
            alphas=ALPHAS,modes=MODES,original_2pct=True)))
        _,donor,_ = base.register()
        data = base.fit_population(beat)
        cells,updates = [],0
        for x,y,z in zip(f['cells'],l['cells'],s['cells'],strict=True):
            c,u = fit_cell(data,donor,x,y,z,old.sha(reg),beat,args.replay)
            cells.append(c)
            updates += u
            beat(phase='cell_frozen',fold=c['identity']['fold'],seed=c['identity']['seed'],updates=updates)
        fp = OUT/'freeze.json'
        old.once(fp,old.stable(dict(registration=old.sha(reg),cells=cells,optimizer_steps=36*STEPS,before_held_scoring=True)))
        summary,g,cap,changes = score(old.read(fp)['cells'],data,donor,beat)
        old.once(OUT/'readout.json',old.stable(dict(freeze=old.sha(fp),summary=summary,geometry=g,
            fixed_recipient_capacity=cap,changed_directions=changes,qualified=sum(c['selected'] is not None for c in cells),
            calibration_max_passes=[max(x['passes'] for x in c['ladder']) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
