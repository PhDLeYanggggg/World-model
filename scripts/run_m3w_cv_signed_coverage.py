"""Factorial replacement of predicted risk and gain on the frozen path bank."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_pooled_coverage as prior_pool
from src.world_model import m3w_cv_signed_coverage as signed

base,prior,screen,model = prior_pool.base,prior_pool.prior,prior_pool.screen,prior_pool.model
np,torch,read,sha,once,stable,artifact,checked = (
    base.np,base.torch,base.read,base.sha,base.once,base.stable,base.artifact,base.checked)
OUT = prior_pool.OUT.parent/'moving_cv_signed_coverage_v1'
PRIVATE = prior_pool.PRIVATE.parent/OUT.name
STEPS = 1000
RISKS = ('direct','conditional')
FACTORIAL = ('old_old','old_swap','direct_old','direct_swap','conditional_old','conditional_swap')
MATCH = ('direct_gain_match','conditional_gain_match')
ARMS = (*FACTORIAL,*MATCH,'calibrated')
GRID = tuple((m,p) for m in (0.,.005,.02) for p in (.10,.25))


def setup(d,cv,ols,bank,easy):
    past,_ = prior.study.motion.features(d['history'],'prefix')
    scale = prior.model.scale(d['history'])
    groups = [np.flatnonzero(d['sites']==s) for s in sorted(set(d['sites']))]
    queries = [q for q in prior.query_groups(d['query']) if len(q)>=2]
    qsite = np.array([d['sites'][q[0]] for q in queries])
    qgroups = [np.flatnonzero(qsite==s) for s in sorted(set(d['sites']))]
    if any(not len(g) for g in qgroups):
        raise ValueError('At least one multi-agent query per fitting source is required')

    def risk_sample(rng):
        ix = prior.draw(groups,rng)
        j = rng.integers(bank.shape[1],size=len(ix))
        p,c,o,y,v,s = bank[ix,j],cv[ix],ols[ix],d['target'][ix],d['valid'][ix],scale[ix]
        _,r,cat = signed.labels(c,o,p[:,None],y,v,s,easy)
        x = model.features(past[ix],p-c,s)
        return tuple(map(torch.from_numpy,(x,r[:,0].clip(-8.,8.).astype(np.float32),cat.astype(np.int64))))

    def gain_sample(rng):
        qi = prior.draw(qgroups,rng)
        ix = np.array([rng.choice(queries[k],2,replace=False) for k in qi])
        a = rng.integers(bank.shape[1],size=ix.shape)
        shape = ix.shape
        ix,a = ix.ravel(),a.ravel()
        p,c,o,y,v,s = bank[ix,a],cv[ix],ols[ix],d['target'][ix],d['valid'][ix],scale[ix]
        gain,_,_ = signed.labels(c,o,p[:,None],y,v,s,easy)
        x = model.features(past[ix],p-c,s)
        pref = signed.signed_swap(*(z.reshape(*shape,*z.shape[1:]) for z in (c,p,y,v,s)))
        return tuple(map(torch.from_numpy,(x.reshape(*shape,-1),gain.reshape(*shape,2).clip(-8.,8.).astype(np.float32),
            pref.astype(np.float32),s.reshape(shape).astype(np.float32))))
    return risk_sample,gain_sample,past.shape[1]+25


@torch.inference_mode()
def infer(nets,cv,bank,history,old):
    past,_ = prior.study.motion.features(history,'prefix')
    scale = prior.model.scale(history)
    n,k = bank.shape[:2]
    gain0,r0,p0,legal = screen.predictions(old,cv,bank,history)
    costs = {a:np.empty((n,k,2)) for a in RISKS}
    support = {a:np.empty((n,k)) for a in RISKS}
    gain = np.empty((n,k))
    for j in range(k):
        for a in range(0,n,4096):
            ix = slice(a,a+4096)
            delta = bank[ix,j]-cv[ix]
            x = torch.from_numpy(model.features(past[ix],delta,scale[ix]))
            radius = torch.from_numpy((np.linalg.norm(delta,axis=-1).max(1)/scale[ix]).astype(np.float32))
            for key in RISKS:
                r,p = signed.decode_risk(nets[key](x),radius,key=='conditional')
                costs[key][ix,j],support[key][ix,j] = r.numpy(),p.numpy()
            gain[ix,j] = nets['swap'](x).mean(1).numpy()
    risks = dict(old=(r0,p0),**{a:(costs[a],support[a]) for a in RISKS})
    scores = dict(old=gain0,swap=gain)
    return {name:(scores[g],*risks[r],legal) for name in FACTORIAL for r,g in [name.split('_')]}


def choose(pred,groups,scale,m=0.,p=.10):
    return prior_pool.pooled(pred,groups,scale,None,m,p)[0]


def calibrate(pred,groups,scale,cv,ols,bank,d,cut):
    trial = []
    expected = len(np.unique(d['recording']))
    scorer = prior.prior.parent.readout
    for name in FACTORIAL:
        for m,p in GRID:
            z = choose(pred[name],groups,scale,m,p)
            rows = base.metrics(cv,ols,bank,d,{'strict':z},cut)
            gains = [[scorer.merge_comparisons([r['comparisons'][f'strict_vs_{k}'] for r in rows if r['source']==s]
                       )['completion_lower_pp'] for k in ('cv','ols4')] for s in sorted(set(d['sites']))]
            trial.append(dict(arm=name,margin=m,min_support=p,edits=int((z>=0).sum()),
                passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
                lower_pp=np.mean(gains,axis=0).tolist()))
    feasible = [r for r in trial if r['edits'] and r['passes']==expected and min(r['lower_pp'])>0.]
    return trial,max(feasible,key=lambda v:(min(v['lower_pp']),-v['edits'])) if feasible else None


def cell(old,pooled,d,donor,reg,beat,replay):
    ident = old['identity']
    if ident!=pooled['identity'] or old['fits'][0]['identity']!=dict(ident,head='risk'):
        raise ValueError('Parent model lineage changed')
    roles = [set(ident[k]) for k in ('train','calibration','held')]
    if any(len(s)!=3 for s in roles) or len(set.union(*roles))!=9:
        raise ValueError('Source roles are not disjoint')
    fold,seed = ident['fold'],ident['seed']
    _,tr = base.subset(d,ident['train'])
    _,cal = base.subset(d,ident['calibration'])
    hi,h = base.subset(d,ident['held'])
    motion = prior.study.motion
    t = motion.bank(tr['history'])
    draw,swap,w = setup(tr,t[:,0],t[:,1],base.expanded(t[:,0],t[:,1:]),old['cut'][0])
    del t
    nets,fits,updates,state = {},[],0,None
    for arm in (*RISKS,'swap'):
        def loss(net,*x):
            return signed.gain_loss(net,*x) if arm=='swap' else signed.risk_loss(net,*x,arm=='conditional')
        name = f'{arm}_{fold}_{seed}'
        net,fit,u = prior.prefix.trainer.fit(dict(ident,arm=arm,registration=reg),name,seed,
            lambda:prior.model.head(w,2 if arm=='swap' else 5),STEPS,swap if arm=='swap' else draw,loss,replay,beat)
        rng = torch.load(checked(fit['artifact']),weights_only=True,map_location='cpu')['rng']
        if arm=='conditional' and rng!=state:
            raise ValueError('Risk ablation training samples diverged')
        state = rng
        nets[arm] = net
        fits.append(fit)
        updates += u
    net = screen.load_head(old['fits'][0],w,8)
    c = motion.bank(cal['history'])
    bank = base.expanded(c[:,0],c[:,1:])
    p = infer(nets,c[:,0],bank,cal['history'],net)
    ladder,selected = calibrate(p,prior.query_groups(cal['query']),prior.model.scale(cal['history']),
                                c[:,0],c[:,1],bank,cal,old['cut'])
    cv,_,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    p = infer(nets,cv,base.expanded(cv,raw),h['history'],net)
    q,s = prior.query_groups(h['query']),prior.model.scale(h['history'])
    a = {k:choose(v,q,s) for k,v in p.items()}
    with np.load(checked(pooled['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        np.testing.assert_array_equal(a['old_old'],z['pool_full'])
    for k in RISKS:
        a[k+'_gain_match'] = prior_pool.independent(p[k+'_old'][0],p[k+'_old'][3],q,match=a[k+'_old'])
    a['calibrated'] = (np.full_like(a['old_old'],-1) if selected is None else
        choose(p[selected['arm']],q,s,selected['margin'],selected['min_support']))
    donor_arrays = {k:((p[k+'_old'][1]<=0.).all(2)&(p[k+'_old'][2]>=.10)&p[k+'_old'][3])
                    for k in ('old',*RISKS)}
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choice_{fold}_{seed}.npz',dict(ids=h['ids'],**a,**donor_arrays))
    return dict(identity=ident,cut=old['cut'],fits=fits,ladder=ladder,selected=selected,actions=path),updates


def evaluate(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{tag}':(b,a) for a in ARMS for tag,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(direct_vs_old=('old_old','direct_old'),conditional_vs_direct=('direct_old','conditional_old'),
        direct_vs_gain_match=('direct_gain_match','direct_old'),
        conditional_vs_gain_match=('conditional_gain_match','conditional_old'),
        swap_on_old=('old_old','old_swap'),swap_on_direct=('direct_old','direct_swap'),
        swap_on_conditional=('conditional_old','conditional_swap'))
    geokeys = ('direct_vs_gain_match','conditional_vs_gain_match','swap_on_conditional')
    rows, geo, stats = [],{},np.zeros((3,5),np.int64)
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        y,v,q = (h[k] for k in ('target','valid','query'))
        cut = c['cut']
        ref,_ = scorer.errors(cv,y,v)
        truth = (v.any(1)&(ref>0.)&(ref<=cut[0]))[:,None] & (np.linalg.norm(bank-cv[:,None],axis=-1).max(2)>1e-8)
        truth &= (prior.utility.risk_excess(cv,bank,y,v,cut[0])<=0.).all(2)
        groups = prior.query_groups(q)
        with np.load(checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            acts = {k:z[k] for k in ARMS}
            for i,key in enumerate(('old',*RISKS)):
                p = z[key]
                for ix in groups:
                    guess,real = p[ix].any(),truth[ix].any()
                    stats[i] += (1,guess,real,guess and real,(p[ix]&truth[ix]).any())
        for ix in groups:
            for k in RISKS:
                if (acts[k+'_old'][ix]>=0).sum()!=(acts[k+'_gain_match'][ix]>=0).sum():
                    raise ValueError('Gain control changed query count')
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q0,y0,v0,cv0 = q[ix],y[ix],v[ix],cv[ix]
            paths = {base.CV:cv0,base.OLS:ols[ix],**{k:prior.apply(cv0,bank[ix],a[ix]) for k,a in acts.items()}}
            r = scorer.score_recording(paths,{},y0,v0,q0,str(rec),easy_cut=cut[0],hard_cut=cut[1])
            _,inv,n = np.unique(q0,return_inverse=True,return_counts=True)
            cmp = {k:prior.study.motion.full_masses(paths[a],paths[b],cv0,y0,v0,1./n[inv]) for k,(a,b) in pairs.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=cmp))
            g = geo.setdefault((str(source),seed),np.zeros((len(geokeys),4)))
            past = h['history'][ix]
            for b in prior.query_groups(q0):
                z = np.stack([paths[a][b] for k in geokeys for a in pairs[k]])
                _,val,_,(steps,known,_,_) = prior.geometry.query(past[b],z,y0[b],v0[b],((0,1),(2,3),(4,5)))
                if steps: g[:,1] += val[:,1]; g[:,3] += 1
                if known: g[:,0] += val[:,0]; g[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['old_old']['switches'])!=(72,9,244):
        raise ValueError('Expected unchanged fitting-source roster and frozen control')
    sources = sorted(set(r['source'] for r in rows))
    g = np.asarray([[geo[s,k] for k in base.SEEDS] for s in sources])
    ratio = np.divide(g[...,:2],g[...,2:],out=np.full_like(g[...,:2],np.nan),where=g[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    geometry = {k:dict(observed_pixel=prior.geometry.bootstrap(ratio[:,:,i,0],draw),
                      completion_lower_pixel=prior.geometry.bootstrap(ratio[:,:,i,1],draw)) for i,k in enumerate(geokeys)}
    ds = {k:dict(zip(('views','predicted','actual','query_overlap','same_action_overlap'),map(int,s),strict=True))
          for k,s in zip(('old',*RISKS),stats,strict=True)}
    return summary,geometry,ds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if v.get('phase') and v.get('step',STEPS) in (100,STEPS):
                print(info,flush=True)

        old_reg,donor,_ = base.register()
        a,b = base.OUT/'freeze.json',prior_pool.OUT/'freeze.json'
        old,pooled = read(a),read(b)
        if (old['registration']!=old_reg or pooled['registration']!=sha(prior_pool.OUT/'registration.json')
                or not all(x['before_held_scoring'] and len(x['cells'])==9 for x in (old,pooled))):
            raise ValueError('Require frozen parent heads and choices')
        reg = OUT/'registration.json'
        once(reg,stable(dict(parents=[artifact(a),artifact(b)],files=[artifact(x) for x in
            (Path(__file__),Path(signed.__file__),OUT/'protocol.md',Path(prior_pool.__file__))],
            grid=GRID,steps=STEPS,seeds=base.SEEDS,arms=FACTORIAL,original_2pct=True,independent_confirmation=False)))
        d = base.fit_population(beat)
        cells,updates = [],0
        for c,z in zip(old['cells'],pooled['cells'],strict=True):
            x,u = cell(c,z,d,donor,sha(reg),beat,args.replay)
            cells.append(x)
            updates += u
            beat(phase='cell_frozen',fold=x['identity']['fold'],seed=x['identity']['seed'],updates=updates)
        freeze = OUT/'freeze.json'
        once(freeze,stable(dict(registration=sha(reg),cells=cells,optimizer_steps=27*STEPS,before_held_scoring=True)))
        summary,geometry,donors = evaluate(read(freeze)['cells'],d,donor,beat)
        once(OUT/'readout.json',stable(dict(freeze=sha(freeze),summary=summary,geometry=geometry,donors=donors,
            qualified=sum(v['selected'] is not None for v in cells),
            calibration_max_passes=[max(v['passes'] for v in c['ladder']) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
