"""Train direct and shape-constrained magnitude curves and calibrate decisions."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_fixed_recipient_support as parent
from src.world_model import m3w_cv_amplitude_response as model

old,base,prior,np,torch = parent.old,parent.base,parent.prior,parent.np,parent.torch
OUT = parent.OUT.parent/'moving_cv_amplitude_response_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
STEPS = 1400
MODES = ('direct','shape')
ARMS = ('frozen','frozen_boost','direct_ind','shape_count','direct_joint','shape_joint',
        'independent_match','uniform_mass','calibrated')
GRID = tuple((mode,cap,slack) for mode in MODES for cap in (.25,.50,1.) for slack in (0.,.05))
PRIMARY = dict(shape_at_count=('direct_ind','shape_count'),portfolio_at_count=('independent_match','shape_joint'),
    amplitude_at_mass=('uniform_mass','shape_joint'),shape_vs_direct_joint=('direct_joint','shape_joint'))


def analytic(h):
    a = prior.study.motion.bank(h)
    return a[:,0],a[:,1],a[:,1:]


def sampler(d,cut):
    c,o,b = analytic(d['history'])
    past,s = parent.context.context(d)
    l = (b!=c[:,None]).any((2,3))
    groups = [np.flatnonzero((d['sites']==g)&l.any(1)) for g in sorted(set(d['sites']))]
    if any(not len(x) for x in groups):
        raise ValueError('Need analytic rays in all training sources')

    def draw(rng):
        i = prior.draw(groups,rng)
        j = np.where(l[i],rng.random(l[i].shape),-1.).argmax(1)
        path = b[i,j]
        x = parent.context.model.action_features(past[i],c[i],o[i],path,s[i])
        return tuple(torch.from_numpy(x) for x in (x,*model.targets(c[i],o[i],path,d['target'][i],
                                                                  d['valid'][i],s[i],cut)))

    return draw,past.shape[1]+37


@torch.inference_mode()
def predictions(nets,d,c,o,b):
    past,s = parent.context.context(d)
    l = (b!=c[:,None]).any((2,3))
    pred = {}
    for k,net in nets.items():
        p = np.empty((*l.shape,model.M,5),np.float32)
        m = np.empty((*l.shape,2),np.float32)
        for j in range(b.shape[1]):
            for start in range(0,len(c),4096):
                ix = slice(start,start+4096)
                x = parent.context.model.action_features(past[ix],c[ix],o[ix],b[ix,j],s[ix])
                v,ref,_ = model.response(net(torch.from_numpy(x)),k=='shape')
                p[ix,j],m[ix,j] = v.numpy(),ref.numpy()
        pred[k] = model.decode(p,m,l,s)
        if any(not np.isfinite(a).all() for a in pred[k]):
            raise ValueError('Invalid causal magnitude predictions')
    return pred


def calibration(pred,d,cv,ols,b,anchor,cut):
    scorer = prior.prior.parent.readout
    groups = prior.query_groups(d['query'])
    recs = np.unique(d['recording'])
    table = []

    def measure(path,kind,**kw):
        rows = []
        for rec in recs:
            ix = d['recording']==rec
            y,v,q = (d[k][ix] for k in ('target','valid','query'))
            r = scorer.score_recording({base.CV:cv[ix],'p':path[ix]}, {},y,v,q,str(rec),
                easy_cut=cut[0],hard_cut=cut[1])
            w = old.weight(q)
            g = [prior.study.motion.full_masses(a[ix],path[ix],cv[ix],y,v,w) for a in (cv,ols)]
            site, = np.unique(d['sites'][ix])
            rows.append((site,r['metrics']['p']['risk']['supported'],g))
        gains = [[scorer.merge_comparisons([x[2][j] for x in rows if x[0]==site])['completion_lower_pp']
                  for j in range(2)] for site in sorted(set(d['sites']))]
        table.append(dict(kind=kind,passes=sum(x[1] for x in rows),lower_pp=np.mean(gains,axis=0).tolist(),
            edits=int((path!=cv).any((1,2)).sum()),**kw))

    for mode,cap,slack in GRID:
        a = model.portfolio(*pred[mode],groups,cap,slack)
        measure(model.apply(cv,b,a),'joint',mode=mode,cap=cap,slack=slack)
    for k in MODES:
        a = model.independent(*pred[k][:3],groups,.25)
        measure(model.apply(cv,b,a),'independent',mode=k,cap=.25)
    for scale in (.5,1.,2.):
        measure(cv+scale*(anchor-cv),'fixed_scale',scale=scale)
    good = [r for r in table if r['edits'] and r['passes']==len(recs) and min(r['lower_pp'])>0.]
    return table,max(good,key=lambda r:(min(r['lower_pp']),-r['edits'])) if good else None


def cell(d,donor,frozen,lead,signed,reg,beat,replay):
    ident,cut = frozen['identity'],frozen['cut']
    fold,seed = ident['fold'],ident['seed']
    if any(x['identity']!=ident or x['cut']!=cut for x in (lead,signed)):
        raise ValueError('Calibration parent roles changed')
    tr = old.teacher.fit_role(d,ident)
    draw,width = sampler(tr,cut[0])
    fits,nets,updates,rng0 = [],{},0,None
    for key in MODES:
        def loss(net,*x):
            return model.loss(net,*x,key=='shape')
        net,fit,n = prior.prefix.trainer.fit(dict(ident,arm=key,registration=reg),f'{key}_{fold}_{seed}',seed,
            lambda:prior.model.head(width,model.WIDTH),STEPS,draw,loss,replay,beat)
        rng = torch.load(old.checked(fit['artifact']),map_location='cpu',weights_only=True)['rng']
        if rng0 is not None and rng!=rng0:
            raise ValueError('Amplitude ablation did not share training draws')
        rng0 = rng
        fits.append(fit)
        nets[key] = net
        updates += n
    _,cal = base.subset(d,ident['calibration'])
    c,o,raw = analytic(cal['history'])
    b = base.expanded(c,raw)
    w = prior.study.motion.features(cal['history'][:1],'prefix')[0].shape[1]+25
    a = old.cal_anchor(cal,c,o,b,signed,lead,w)
    anchor = old.slate.path_for(c,b,a)
    table,sel = calibration(predictions(nets,cal,c,o,raw),cal,c,o,raw,anchor,cut)
    hi,h = base.subset(d,ident['held'])
    c,o,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    pred = predictions(nets,h,c,o,raw)
    q = prior.query_groups(h['query'])
    with np.load(old.checked(frozen['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        a = z['frozen']
    p = {'direct_ind':model.independent(*pred['direct'][:3],q,.25),'frozen':a}
    p['shape_count'] = model.independent(*pred['shape'][:3],q,.25,match=p['direct_ind'])
    for k in MODES:
        p[f'{k}_joint'] = model.portfolio(*pred[k],q,.50,.0)
    p['independent_match'] = model.independent(*pred['shape'][:3],q,.50,match=p['shape_joint'])
    if sel is not None and sel['kind']!='fixed_scale':
        mode,cap = sel['mode'],sel['cap']
        p['calibrated'] = (model.portfolio(*pred[mode],q,cap,sel['slack']) if sel['kind']=='joint' else
                           model.independent(*pred[mode][:3],q,cap))
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=h['ids'],**p))
    return dict(identity=ident,cut=cut,fits=fits,ladder=table,selected=sel,choices=path),updates


def read_paths(c,h,cv,ols,raw):
    with np.load(old.checked(c['choices']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        ref = old.slate.path_for(cv,base.expanded(cv,raw),z['frozen'])
        paths = {base.CV:cv,base.OLS:ols,'frozen':ref,'frozen_boost':cv+2.*(ref-cv)}
        for k in ('direct_ind','shape_count','direct_joint','shape_joint','independent_match'):
            paths[k] = model.apply(cv,raw,z[k])
        paths['uniform_mass'] = model.uniform(cv,raw,z['shape_joint'],prior.query_groups(h['query']))
        a = c['selected']
        paths['calibrated'] = (cv if a is None else cv+a['scale']*(ref-cv) if a['kind']=='fixed_scale' else
                               model.apply(cv,raw,z['calibrated']))
    return paths


def evaluate(cells,d,donor,beat,*,export_rows=False):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{k}':(b,a) for a in ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(PRIMARY)
    rows,geo,peak = [],{},{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        p = read_paths(c,h,cv,ols,raw)
        for k in ARMS:
            r = np.linalg.norm(p[k]-cv,axis=-1).max(1)
            peak.setdefault(k,np.zeros(2))
            peak[k] += [(r>0.).sum(),r.sum()]
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            c0 = cv[ix]
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            z = {k:x[ix] for k,x in p.items()}
            r = scorer.score_recording(z,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k:prior.study.motion.full_masses(z[l],z[r],c0,y,v,w) for k,(l,r) in pairs.items()}
            s, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(s),seed=seed,comparisons=terms))
            if export_rows:
                continue
            a = geo.setdefault((str(s),seed),np.zeros((len(PRIMARY),4)))
            history = h['history'][ix]
            r = {k:np.linalg.norm(z[k]-c0,axis=-1).max(1) for k in ARMS}
            np.testing.assert_array_equal(r['uniform_mass']>0.,r['shape_joint']>0.)
            for g in prior.query_groups(q):
                for l,k in PRIMARY.values():
                    if (l,k)!=('direct_joint','shape_joint') and (r[l][g]>0.).sum()!=(r[k][g]>0.).sum():
                        raise ValueError('Query counts changed')
                np.testing.assert_allclose(r['uniform_mass'][g].sum(),r['shape_joint'][g].sum(),rtol=1e-6,atol=1e-8)
                arr = np.stack([z[k][g] for pair in PRIMARY.values() for k in pair])
                _,u,_,(steps,known,_,_) = prior.geometry.query(history[g],arr,y[g],v[g],
                    tuple((2*i,2*i+1) for i in range(len(PRIMARY))))
                if steps: a[:,1] += u[:,1]; a[:,3] += 1
                if known: a[:,0] += u[:,0]; a[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    if export_rows:
        return rows
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if summary['views']!=72 or summary['risk']['frozen']['switches']!=54748:
        raise ValueError('Original held control changed')
    sites = sorted(set(roster.values()))
    x = np.asarray([[geo[s,k] for k in base.SEEDS] for s in sites])
    p = np.divide(x[...,:2],x[...,2:],out=np.full_like(x[...,:2],np.nan),where=x[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    geometry = {k:dict(observed_pixel=prior.geometry.bootstrap(p[:,:,i,0],draw),
        completion_lower_pixel=prior.geometry.bootstrap(p[:,:,i,1],draw)) for i,k in enumerate(PRIMARY)}
    radius = {k:dict(total_peak_pixel=float(p[1]),mean_peak_pixel=float(p[1]/p[0]) if p[0] else None)
              for k,p in peak.items()}
    return summary,geometry,radius


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
        refs = (old.OUT/'freeze.json',old.OUT/'readout.json',parent.parent.OUT/'freeze.json',
                parent.parent.parent.OUT/'freeze.json')
        f,r,l,s = map(old.read,refs)
        if (r['freeze']!=old.sha(refs[0]) or r['a_labels_opened'] or r['independent_confirmation']
                or not r['original_2pct'] or not all(z['before_held_scoring'] for z in (f,l,s))):
            raise ValueError('Require frozen, development-only controls')
        reg = OUT/'registration.json'
        old.once(reg,old.stable(dict(parents=[old.artifact(x) for x in refs],
            files=[old.artifact(x) for x in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
            amplitudes=model.AMPLITUDES,prices=model.PRICES,grid=GRID,steps=STEPS,arms=ARMS,
            seeds=base.SEEDS,original_2pct=True)))
        _,donor,_ = base.register()
        d = base.fit_population(beat)
        cells,updates = [],0
        for a,b,c in zip(f['cells'],l['cells'],s['cells'],strict=True):
            x,n = cell(d,donor,a,b,c,old.sha(reg),beat,args.replay)
            cells.append(x)
            updates += n
            beat(phase='cell_frozen',fold=x['identity']['fold'],seed=x['identity']['seed'],updates=updates)
        fp = OUT/'freeze.json'
        old.once(fp,old.stable(dict(registration=old.sha(reg),cells=cells,optimizer_steps=18*STEPS,before_held_scoring=True)))
        summary,geo,peak = evaluate(old.read(fp)['cells'],d,donor,beat)
        old.once(OUT/'readout.json',old.stable(dict(freeze=old.sha(fp),summary=summary,geometry=geo,peak=peak,
            qualified=sum(x['selected'] is not None for x in cells),
            calibration_max_passes=[max(x['passes'] for x in c['ladder']) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
