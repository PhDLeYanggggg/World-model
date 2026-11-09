"""A count-matched factorial of easy-risk asymmetry and lead-aware missingness."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_signed_coverage as parent
from src.world_model import m3w_cv_lead_risk as lead

base,prior,screen,model = parent.base,parent.prior,parent.screen,parent.model
np,torch,read,sha,once,stable,artifact,checked = (
    base.np,base.torch,base.read,base.sha,base.once,base.stable,base.artifact,base.checked)
OUT = parent.OUT.parent/'moving_cv_lead_risk_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
STEPS = 1000
FACTORIAL = ('cond_flat','cond_lead','tail_flat','tail_lead')
MATCHED = tuple(k+'_matched' for k in FACTORIAL)
ARMS = ('legacy',*FACTORIAL,*MATCHED,'calibrated')


def samplers(d,cv,bank):
    past,_ = prior.study.motion.features(d['history'],'prefix')
    scale = prior.model.scale(d['history'])
    groups = [np.flatnonzero(d['sites']==k) for k in sorted(set(d['sites']))]
    valid = d['valid']

    def availability(rng):
        i = prior.draw(groups,rng)
        return torch.from_numpy(past[i]),torch.from_numpy(valid[i].astype(np.float32))

    def benefit(rng):
        i = prior.draw(groups,rng)
        j = rng.integers(bank.shape[1],size=len(i))
        p,c,y,v,s = bank[i,j],cv[i],d['target'][i],valid[i],scale[i]
        target = lead.observed_targets(c,p,y,v,s).clip(-8.,8.).astype(np.float32)
        return tuple(map(torch.from_numpy,(model.features(past[i],p-c,s),target,v)))
    return availability,benefit,past.shape[1]


@torch.inference_mode()
def inference(nets,cv,ols,bank,history):
    past,_ = prior.study.motion.features(history,'prefix')
    scale = prior.model.scale(history)
    n,k,t = bank.shape[:3]
    visible = np.concatenate([nets['visibility'](torch.from_numpy(past[a:a+4096])).sigmoid().numpy()
                              for a in range(0,n,4096)])
    gain = {key:np.empty((n,k)) for key in ('legacy','flat','lead')}
    risk = {key:np.empty((n,k,2)) for key in ('cond','tail')}
    easy = {key:np.empty((n,k)) for key in risk}
    legal = np.empty((n,k),bool)
    for j in range(k):
        for a in range(0,n,4096):
            ix = slice(a,a+4096)
            c,o,p = cv[ix],ols[ix],bank[ix,j]
            r = np.linalg.norm(p-c,axis=-1).max(1)
            x = torch.from_numpy(model.features(past[ix],p-c,scale[ix]))
            radius = torch.from_numpy((r/scale[ix]).astype(np.float32))
            for key in risk:
                u,v = parent.signed.decode_risk(nets[key](x),radius,True)
                risk[key][ix,j],easy[key][ix,j] = u.numpy(),v.numpy()
            gain['legacy'][ix,j] = nets['swap'](x).mean(1).numpy()
            gain['flat'][ix,j],gain['lead'][ix,j] = lead.score(c,o,p,nets['benefit'](x).numpy(),visible[ix],scale[ix])
            legal[ix,j] = r>1e-8
    result = {name:(gain[g],risk[r],easy[r],legal) for name in FACTORIAL for r,g in [name.split('_')]}
    result['legacy'] = gain['legacy'],risk['cond'],easy['cond'],legal
    return result,visible


def matched_query(p,count):
    """Largest common achievable prefix encountered by deterministic cap shrinkage."""
    while count>0:
        a = [parent.prior_pool.pool_query(*z,count,.10) for z in p]
        used = [(z>=0).sum() for z in a]
        if min(used)==max(used):
            return a
        count = min(used)
    return [np.full(len(z[0]),-1,dtype=np.int8) for z in p]


def calibrate(pred,groups,scale,cv,ols,bank,d,cut):
    scorer = prior.prior.parent.readout
    sources = sorted(set(d['sites']))
    n = len(np.unique(d['recording']))
    table = []
    for arm in FACTORIAL:
        for m,p in parent.GRID:
            action = parent.choose(pred[arm],groups,scale,m,p)
            rows = base.metrics(cv,ols,bank,d,{'strict':action},cut)
            gains = [[scorer.merge_comparisons([r['comparisons'][f'strict_vs_{k}'] for r in rows if r['source']==s]
                        )['completion_lower_pp'] for k in ('cv','ols4')] for s in sources]
            table.append(dict(arm=arm,margin=m,support=p,edits=int((action>=0).sum()),
                passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
                lower_pp=np.mean(gains,axis=0).tolist()))
    good = [z for z in table if z['edits'] and z['passes']==n and min(z['lower_pp'])>0.]
    return table,max(good,key=lambda z:(min(z['lower_pp']),-z['edits'])) if good else None


def cell(c,d,donor,reg,beat,replay):
    ident = c['identity']
    groups = [set(ident[k]) for k in ('train','calibration','held')]
    if any(len(g)!=3 for g in groups) or len(set.union(*groups))!=9:
        raise ValueError('Source roles must be disjoint')
    fold,seed = ident['fold'],ident['seed']
    _,tr = base.subset(d,ident['train'])
    _,cal = base.subset(d,ident['calibration'])
    hi,h = base.subset(d,ident['held'])
    motion = prior.study.motion
    b = motion.bank(tr['history'])
    bank = base.expanded(b[:,0],b[:,1:])
    risk,_,width = parent.setup(tr,b[:,0],b[:,1],bank,c['cut'][0])
    mask,gain,hw = samplers(tr,b[:,0],bank)
    old_cond,old_swap = c['fits'][1:3]
    for old,key in ((old_cond,'conditional'),(old_swap,'swap')):
        expected = dict(ident,arm=key,registration=sha(parent.OUT/'registration.json'))
        if old['identity']!=expected:
            raise ValueError('Old head identity changed')
    nets = dict(cond=screen.load_head(old_cond,width,5),swap=screen.load_head(old_swap,width,2))
    fits,updates = [],0
    bce = lambda net,x,y:torch.nn.functional.binary_cross_entropy_with_logits(net(x),y)
    for arm,draw,w,o,loss in (('tail',risk,width,5,lead.tail_loss),('visibility',mask,hw,12,bce),
                              ('benefit',gain,width,12,lead.benefit_loss)):
        net,f,u = prior.prefix.trainer.fit(dict(ident,arm=arm,registration=reg),f'{arm}_{fold}_{seed}',seed,
            lambda:prior.model.head(w,o),STEPS,draw,loss,replay,beat)
        if arm=='tail':
            states = [torch.load(checked(z['artifact']),weights_only=True,map_location='cpu')['rng']
                      for z in (f,old_cond)]
            if states[0]!=states[1]:
                raise ValueError('Tail-risk draws differ from the control')
        nets[arm] = net
        fits.append(f)
        updates += u
    x = motion.bank(cal['history'])
    cb = base.expanded(x[:,0],x[:,1:])
    cp,_ = inference(nets,x[:,0],x[:,1],cb,cal['history'])
    ladder,selected = calibrate(cp,prior.query_groups(cal['query']),prior.model.scale(cal['history']),
                                x[:,0],x[:,1],cb,cal,c['cut'])
    cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    p,visible = inference(nets,cv,ols,base.expanded(cv,raw),h['history'])
    q,s = prior.query_groups(h['query']),prior.model.scale(h['history'])
    actions = {k:parent.choose(v,q,s) for k,v in p.items()}
    with np.load(checked(c['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        np.testing.assert_array_equal(actions['legacy'],z['conditional_swap'])
    for k in MATCHED:
        actions[k] = np.full_like(actions['legacy'],-1)
    for ix in q:
        n = min((actions[k][ix]>=0).sum() for k in FACTORIAL)
        data = [(p[k][0][ix]*s[ix,None],p[k][1][ix]*s[ix,None,None],p[k][2][ix],p[k][3][ix])
                for k in FACTORIAL]
        for k,a in zip(MATCHED,matched_query(data,n),strict=True):
            actions[k][ix] = a
    actions['calibrated'] = (np.full_like(actions['legacy'],-1) if selected is None else
        parent.choose(p[selected['arm']],q,s,selected['margin'],selected['support']))
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',
        dict(ids=h['ids'],visible=visible,**actions))
    return dict(identity=ident,cut=c['cut'],fits=fits,ladder=ladder,selected=selected,actions=path),updates


def evaluate(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    contrasts = {f'{a}_vs_{tag}':(b,a) for a in ARMS for tag,b in (('cv',base.CV),('ols4',base.OLS))}
    primary = dict(lead_matched=('cond_flat_matched','cond_lead_matched'),
        tail_matched=('cond_lead_matched','tail_lead_matched'),
        lead_under_tail=('tail_flat_matched','tail_lead_matched'))
    contrasts.update(primary)
    rows,geometry,mask = [],{},{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            acts = {k:z[k] for k in ARMS}
            visibility = z['visible']
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            cv0 = cv[ix]
            paths = {base.CV:cv0,base.OLS:ols[ix],**{k:prior.apply(cv0,bank[ix],a[ix]) for k,a in acts.items()}}
            r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            w = 1./n[inv]
            terms = {k:prior.study.motion.full_masses(paths[a],paths[b],cv0,y,v,w) for k,(a,b) in contrasts.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=terms))
            key = str(source),seed
            m = mask.setdefault(key,np.zeros((3,12)))
            vis = visibility[ix]
            m[0] += w@np.square(vis.mean(1,keepdims=True)-v)
            m[1] += w@np.square(vis-v)
            m[2] += w.sum()
            g = geometry.setdefault(key,np.zeros((len(primary),4)))
            past = h['history'][ix]
            for block in prior.query_groups(q):
                if len({int((acts[k][ix][block]>=0).sum()) for k in MATCHED})!=1:
                    raise ValueError('Query-matched counts diverged')
                z = np.stack([paths[a][block] for pair in primary.values() for a in pair])
                _,u,_,(steps,known,_,_) = prior.geometry.query(past[block],z,y[block],v[block],((0,1),(2,3),(4,5)))
                if steps: g[:,1] += u[:,1]; g[:,3] += 1
                if known: g[:,0] += u[:,0]; g[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['legacy']['switches'])!=(72,9,73477):
        raise ValueError('Unexpected held roster or legacy edits')
    sources = sorted(set(roster.values()))
    z = np.array([[geometry[s,k] for k in base.SEEDS] for s in sources])
    ratio = np.divide(z[...,:2],z[...,2:],out=np.full_like(z[...,:2],np.nan),where=z[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    geom = {k:dict(observed_pixel=prior.geometry.bootstrap(ratio[:,:,i,0],draw),
                   completion_lower_pixel=prior.geometry.bootstrap(ratio[:,:,i,1],draw)) for i,k in enumerate(primary)}
    m = np.array([[mask[s,k] for k in base.SEEDS] for s in sources])
    brier = m[:,:,:2,:]/m[:,:,2:,:]
    mask_scores = dict(flat=brier[:,:,0,:].mean((0,1)).tolist(),lead=brier[:,:,1,:].mean((0,1)).tolist(),
        lead_brier_reduction=prior.geometry.bootstrap((brier[:,:,0]-brier[:,:,1]).mean(2),draw))
    return summary,geom,mask_scores


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

        _,donor,_ = base.register()
        parent_fp,parent_result = parent.OUT/'freeze.json',parent.OUT/'readout.json'
        old,stats = read(parent_fp),read(parent_result)
        if (stats['freeze']!=sha(parent_fp) or old['registration']!=sha(parent.OUT/'registration.json')
                or not old['before_held_scoring'] or len(old['cells'])!=9 or stats['a_labels_opened']
                or stats['independent_confirmation']):
            raise ValueError('Require verified signed-risk controls')
        reg = OUT/'registration.json'
        once(reg,stable(dict(parents=[artifact(x) for x in (parent_fp,parent_result)],files=[artifact(x) for x in
            (Path(__file__),Path(lead.__file__),OUT/'protocol.md',Path(parent.__file__))],grid=parent.GRID,
            steps=STEPS,seeds=base.SEEDS,factorial=FACTORIAL,original_2pct=True,independent_confirmation=False)))
        d = base.fit_population(beat)
        cells,updates = [],0
        for old_cell in old['cells']:
            c,u = cell(old_cell,d,donor,sha(reg),beat,args.replay)
            cells.append(c)
            updates += u
            beat(phase='cell_frozen',fold=c['identity']['fold'],seed=c['identity']['seed'],updates=updates)
        fp = OUT/'freeze.json'
        once(fp,stable(dict(registration=sha(reg),cells=cells,before_held_scoring=True,optimizer_steps=27*STEPS)))
        summary,geo,mask = evaluate(read(fp)['cells'],d,donor,beat)
        once(OUT/'readout.json',stable(dict(freeze=sha(fp),summary=summary,geometry=geo,mask=mask,
            qualified=sum(c['selected'] is not None for c in cells),
            calibration_max_passes=[max(v['passes'] for v in c['ladder']) for c in cells],
            original_2pct=True,a_labels_opened=False,independent_confirmation=False)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
