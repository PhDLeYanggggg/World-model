"""Train a smooth lead-dependent proposal and its parameter-matched control."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_record_allocation as parent
from src.world_model import m3w_lead_mixture as model

rank, screen, base, prior = parent.rank, parent.screen, parent.base, parent.prior
np, torch, read, sha, once, stable, artifact, checked = (
    base.np,base.torch,base.read,base.sha,base.once,base.stable,base.artifact,base.checked)
OUT = parent.OUT.parent/'moving_cv_lead_mixture_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
FAMILIES = ('constant','lead')
ARMS = ('frozen','constant','lead','constant_radius','lead_radius','constant_calibrated','lead_calibrated')
STEPS = 1000


def sampler(d, bank, x):
    scale = prior.model.scale(d['history']).astype(np.float32)
    y,v = d['target'].astype(np.float32),d['valid']
    groups = [np.flatnonzero(d['sites']==s) for s in sorted(set(d['sites']))]

    def sample(rng):
        i = prior.draw(groups,rng)
        return tuple(map(torch.from_numpy,(x[i],bank[i],y[i],v[i],scale[i])))
    return sample


@torch.inference_mode()
def predict(net,x,bank,arm):
    pred = np.concatenate([net(torch.from_numpy(x[i:i+4096]),torch.from_numpy(bank[i:i+4096]),
                               arm=='lead').numpy() for i in range(0,len(x),4096)])
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite proposal')
    return pred


def peaks(p,cv):
    return np.linalg.norm(p-cv,axis=-1).max(-1)


def apply(cv,path,mask,alpha):
    out = cv.copy()
    if alpha:
        out[mask] = cv[mask]+alpha*(path[mask]-cv[mask])
    return out


def match_radius(cv,p,q,selected):
    a,b = peaks(p,cv),peaks(q,cv)
    cap = np.minimum(a,b)
    if (cap[selected]<=1e-8).any():
        raise ValueError('Radius-matched comparison lost an edit')
    out = [cv+(z-cv)*np.divide(cap,r,out=np.zeros_like(cap),where=r>0.)[:,None,None] for z,r in ((p,a),(q,b))]
    np.testing.assert_allclose(peaks(out[0],cv),peaks(out[1],cv),atol=1e-7)
    return out


def cal_mask(old,pair,cal,cv,ols,bank,width):
    risk = screen.load_head(old['fits'][0],width+25,8)
    net = screen.load_head(pair['fits'][1],width+49,2)
    paths = base.expanded(cv,bank)
    pred = screen.predictions(risk,cv,paths,cal['history'])
    score = rank.forecast(net,cv,ols,paths,cal['history'])
    g = prior.query_groups(cal['query'])
    before = base.choices(screen.nominate(pred),g,screen.RELAXED)
    return screen.choose(rank.nomination(score,pred,screen.RELAXED),g,before)>=0


def calibrate(cv,ols,proposal,mask,d,cut):
    scorer = prior.prior.parent.readout
    rows = []
    for a in base.FRACTIONS:
        p = apply(cv,proposal,mask,a)
        by,passes = {},0
        for rec in np.unique(d['recording']):
            ix = d['recording']==rec
            y,v,q = (d[k][ix] for k in ('target','valid','query'))
            c,o,z = cv[ix],ols[ix],p[ix]
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            r = scorer.score_recording({base.CV:c,'candidate':z},{},y,v,q,str(rec),easy_cut=cut[0],hard_cut=cut[1])
            passes += r['metrics']['candidate']['risk']['supported']
            site, = np.unique(d['sites'][ix])
            by.setdefault(site,[]).append([prior.study.motion.full_masses(f,z,c,y,v,1./n[inv]) for f in (c,o)])
        gain = [[scorer.merge_comparisons([q[i] for q in v])['completion_lower_pp']
                 for i in range(2)] for v in by.values()]
        rows.append(dict(alpha=a,passes=passes,gain_pp=np.mean(gain,axis=0).tolist()))
    feasible = [r for r in rows if r['passes']==len(np.unique(d['recording'])) and min(r['gain_pp'])>0.]
    return dict(alpha=max(feasible,key=lambda r:(min(r['gain_pp']),-r['alpha']))['alpha'] if feasible else 0.,ladder=rows)


def fit_cell(old,pair,alloc,d,reg,beat,replay):
    ident = old['identity']
    if not (ident==pair['identity']==alloc['identity']):
        raise ValueError('Parent identities do not agree')
    roles = [set(ident[k]) for k in ('train','calibration','held')]
    if any(len(s)!=3 for s in roles) or len(set.union(*roles))!=9:
        raise ValueError('Expected disjoint fitting-source roles')
    fold,seed = ident['fold'],ident['seed']
    _,tr = base.subset(d,ident['train'])
    _,cal = base.subset(d,ident['calibration'])
    _,held = base.subset(d,ident['held'])
    ft,fc,fh = [prior.study.motion.features(z['history'],'prefix')[0] for z in (tr,cal,held)]
    bt,bc,bh = [prior.study.motion.bank(z['history']) for z in (tr,cal,held)]
    width,expert = ft.shape[1],bt.shape[1]
    expected = dict(ident,arm='paired',registration=sha(rank.OUT/'registration.json'),weight=rank.RANK_WEIGHT)
    if pair['fits'][1]['identity']!=expected or old['fits'][0]['identity']!=dict(ident,head='risk'):
        raise ValueError('Risk or paired model lineage changed')
    cm = cal_mask(old,pair,cal,bc[:,0],bc[:,1],bc[:,1:],width)
    draw = sampler(tr,bt.astype(np.float32),ft)
    cut = old['cut']
    fits,selections,updates,state = [],{},0,None
    arrays = dict(ids=held['ids'])
    for arm in FAMILIES:
        def loss(net,x,bank,y,v,s):
            return model.objective(net(x,bank,arm=='lead'),bank[:,0],bank[:,1],y,v,s,cut[0])
        net,fit,u = prior.prefix.trainer.fit(dict(ident,arm=arm,registration=reg),
            f'{arm}_{fold}_{seed}',seed,lambda:model.LeadMixture(width,expert),STEPS,draw,loss,replay,beat)
        rng = torch.load(checked(fit['artifact']),weights_only=True,map_location='cpu')['rng']
        if state is not None and rng!=state:
            raise ValueError('Training samples diverged across arms')
        state = rng
        fits.append(fit)
        updates += u
        cp = predict(net,fc,bc.astype(np.float32),arm)
        selections[arm] = calibrate(bc[:,0],bc[:,1],cp,cm,cal,cut)
        arrays[arm] = predict(net,fh,bh.astype(np.float32),arm)
    path = prior.study.parent.parent.save_actions(PRIVATE/f'paths_{fold}_{seed}.npz',arrays)
    return dict(identity=ident,cut=cut,fits=fits,selections=selections,paths=path,allocation=alloc['actions']),updates


def readout(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    contrasts = {f'{a}_vs_{k}':(b,a) for a in ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    contrasts.update(lead_vs_constant=('constant','lead'),lead_radius_vs_constant=('constant_radius','lead_radius'),
                     lead_vs_frozen=('frozen','lead'),constant_vs_frozen=('frozen','constant'))
    pair_keys = ('lead_vs_constant','lead_radius_vs_constant')
    rows, geom, count = [], {}, 0
    for cell in cells:
        ident = cell['identity']
        seed = ident['seed']
        hi,h = base.subset(d,ident['held'])
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        with np.load(checked(cell['allocation']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            action = z['local']
        at = action>=0
        with np.load(checked(cell['paths']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            p,q = [z[k] for k in FAMILIES]
        radius = match_radius(cv,p,q,at)
        paths = {base.CV:cv,base.OLS:ols,'frozen':prior.apply(cv,base.expanded(cv,raw),action),
            **{k:apply(cv,v,at,1.) for k,v in zip(FAMILIES,(p,q),strict=True)},
            **{k+'_radius':apply(cv,v,at,1.) for k,v in zip(FAMILIES,radius,strict=True)},
            **{k+'_calibrated':apply(cv,v,at,cell['selections'][k]['alpha'])
               for k,v in zip(FAMILIES,(p,q),strict=True)}}
        if any(not np.array_equal(peaks(paths[k],cv)>0.,at) for k in ARMS[:5]):
            raise ValueError('Matched recipients changed')
        count += int(at.sum())
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            y,v,q = (h[k][ix] for k in ('target','valid','query'))
            local = {k:v[ix] for k,v in paths.items()}
            c = cv[ix]
            s, = np.unique(h['sites'][ix])
            r = scorer.score_recording(local,{},y,v,q,str(rec),easy_cut=cell['cut'][0],hard_cut=cell['cut'][1])
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            comparison = {k:prior.study.motion.full_masses(local[a],local[b],c,y,v,1./n[inv])
                          for k,(a,b) in contrasts.items()}
            rows.append(dict(r,source=str(s),seed=seed,comparisons=comparison))
            g = geom.setdefault((str(s),seed),np.zeros((2,4)))
            hx = h['history'][ix]
            for block in prior.query_groups(q):
                z = np.stack([local[k][block] for k in ('constant','lead','constant_radius','lead_radius')])
                _,val,_,(steps,known,_,_) = prior.geometry.query(hx[block],z,y[block],v[block],((0,1),(2,3)))
                if steps: g[:,1] += val[:,1]; g[:,3] += 1
                if known: g[:,0] += val[:,0]; g[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    if (summary['views'],summary['source_groups'],count)!=(72,9,2317):
        raise ValueError('Matched roster changed')
    sources = sorted({r['source'] for r in rows})
    g = np.asarray([[geom[s,k] for k in base.SEEDS] for s in sources])
    ratio = np.divide(g[...,:2],g[...,2:],out=np.full_like(g[...,:2],np.nan),where=g[...,2:]>0.)
    draws = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    geometry = {k:dict(observed_pixel=prior.geometry.bootstrap(ratio[:,:,i,0],draws),
                      completion_lower_pixel=prior.geometry.bootstrap(ratio[:,:,i,1],draws)) for i,k in enumerate(pair_keys)}
    return summary,geometry


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

        parent_reg,donor,_ = base.register()
        paths = [x.OUT/'freeze.json' for x in (base,rank,parent)]
        a,b,c = [read(k) for k in paths]
        if (a['registration']!=parent_reg or b['registration']!=sha(rank.OUT/'registration.json')
                or c['registration']!=sha(parent.OUT/'registration.json')
                or not all(z['before_held_scoring'] and len(z['cells'])==9 for z in (a,b,c))):
            raise ValueError('Expected verified development freezes')
        registration = OUT/'registration.json'
        once(registration,stable(dict(parents=[artifact(k) for k in paths],
            files=[artifact(k) for k in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
            steps=STEPS,seed=base.SEEDS,families=FAMILIES,original_2pct=True,independent_confirmation=False)))
        reg = sha(registration)
        d = base.fit_population(beat)
        cells,updates = [],0
        for old,pair,alloc in zip(a['cells'],b['cells'],c['cells'],strict=True):
            cell,u = fit_cell(old,pair,alloc,d,reg,beat,args.replay)
            cells.append(cell)
            updates += u
            beat(phase='cell_frozen',fold=cell['identity']['fold'],seed=cell['identity']['seed'],updates=updates)
        freeze = OUT/'freeze.json'
        once(freeze,stable(dict(registration=reg,cells=cells,before_held_scoring=True,optimizer_steps=18*STEPS)))
        summary,geometry = readout(read(freeze)['cells'],d,donor,beat)
        once(OUT/'readout.json',stable(dict(freeze=sha(freeze),summary=summary,geometry=geometry,
            qualified={k:sum(c['selections'][k]['alpha']>0. for c in cells) for k in FAMILIES},
            calibration_max_passes={k:[max(r['passes'] for r in c['selections'][k]['ladder']) for c in cells] for k in FAMILIES},
            original_2pct=True,a_labels_opened=False,independent_confirmation=False)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
