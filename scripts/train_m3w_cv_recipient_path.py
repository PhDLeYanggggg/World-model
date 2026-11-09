"""Three-seed recipient-supervision and bounded path factorial on development data."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_cv_recipient_teachers as teachers
from scripts import train_m3w_cv_minimum_budget as previous
from src.world_model import m3w_cv_recipient_path as model

exp,np,base,prior,torch = previous.exp,previous.np,previous.base,previous.prior,previous.torch
OUT,PRIVATE = teachers.OUT,teachers.PRIVATE
STEPS = 1200
RECIPIENTS = ('cost','one','soft')
PATHS = ('old','pair')
MATCHED = tuple(f'{r}_{p}' for r in RECIPIENTS for p in PATHS)
ARMS = ('frozen',*MATCHED,'calibrated')
GRID = tuple((p,f,m) for p in PATHS for f in (.10,.25,.50) for m in (0.,.05))
PRIMARY = dict(single_recipient=('cost_old','one_old'),soft_recipient=('one_old','soft_old'),
    paired_path=('cost_old','cost_pair'),soft_path=('soft_old','soft_pair'),full=('cost_old','soft_pair'))


def samples(tr,teacher,cut):
    cv,ols,b = exp.teacher.analytic(tr['history'])
    past,scale = previous.context(tr)
    features = model.recipient_features(past,cv,ols,scale)
    with np.load(exp.checked(teacher['labels']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],tr['ids'])
        a,soft,valid = z['one'],z['soft'],z['valid']
    sites = [np.flatnonzero(tr['sites']==s) for s in sorted(set(tr['sites']))]
    mask = [g[valid[g]] for g in sites if valid[g].any()]
    positive = [g[soft[g]>0.] for g in sites if (soft[g]>0.).any()]
    legal = (b!=cv[:,None]).any((2,3))
    pairs = [g[legal[g].sum(1)>=2] for g in sites]
    if not mask or not positive or any(not len(x) for x in pairs):
        raise ValueError('Insufficient recipient or within-row action targets')

    def recipient(rng):
        i = prior.draw(mask,rng)
        j = prior.draw(positive,rng)[:len(i)//3]
        i[:len(j)] = j
        return tuple(torch.from_numpy(x) for x in (features[i],a[i].astype(np.float32),soft[i]))

    def path(rng):
        i = prior.draw(pairs,rng)
        key = np.where(legal[i],rng.random((len(i),b.shape[1])),-1.)
        j = np.argsort(key,axis=1)[:,-2:]
        at = np.repeat(i,2)
        p = b[at,j.ravel()]
        x = previous.model.action_features(past[at],cv[at],ols[at],p,scale[at])
        target = previous.model.targets(cv[at],ols[at],p,tr['target'][at],tr['valid'][at],scale[at],cut)
        return tuple(torch.from_numpy(z.reshape(len(i),2,-1)) for z in (x,target))

    return recipient,path,features.shape[1],past.shape[1]+37


def old_cost(fit,width):
    z = torch.load(exp.checked(fit['artifact']),map_location='cpu',weights_only=True)
    if z['identity']!=fit['identity'] or z['step']!=previous.STEPS or fit['identity']['arm']!='cost':
        raise ValueError('Frozen cost control changed')
    net = prior.model.head(width,6)
    net.load_state_dict(z['net'])
    return net.eval()


@torch.inference_mode()
def predict(nets,old,d,cv,ols,b):
    pred,legal,scale = previous.infer({'old':old,'pair':nets['pair']},d,cv,ols,b)
    radius = np.linalg.norm(b-cv[:,None],axis=-1).max(2)
    rank,cost,path = model.nominations(pred['old'],pred['pair'],legal,radius)
    feats = torch.from_numpy(model.recipient_features(previous.context(d)[0],cv,ols,scale))
    ranks = {'cost':rank}
    for a in ('one','soft'):
        p = nets[a](feats).flatten().sigmoid().numpy()
        ranks[a] = rank+.24*(p-.5)
    return ranks,{'old':cost,'pair':path},scale


def calibrate(p,d,c,o,b,cut):
    rank,path,scale = p
    q = prior.query_groups(d['query'])
    scorer = prior.prior.parent.readout
    tab = []
    n = len(np.unique(d['recording']))
    for a,f,m in GRID:
        action = model.select(rank['soft'],path[a],q,scale,fraction=f,margin=m)
        rows = base.metrics(c,o,b,d,{'strict':action},cut)
        utility = [[scorer.merge_comparisons([r['comparisons'][f'strict_vs_{key}'] for r in rows if r['source']==s])
                    ['completion_lower_pp'] for key in ('cv','ols4')] for s in sorted(set(d['sites']))]
        tab.append(dict(path=a,fraction=f,margin=m,passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
            edits=int((action>=0).sum()),lower=np.mean(utility,axis=0).tolist()))
    good = [r for r in tab if r['edits'] and r['passes']==n and min(r['lower'])>0.]
    return tab,max(good,key=lambda r:(min(r['lower']),-r['edits'])) if good else None


def cell(d,donor,parent,teacher,registration,beat,replay):
    ident = parent['identity']
    fold,seed = ident['fold'],ident['seed']
    if teacher['fold']!=fold or teacher['train']!=ident['train']:
        raise ValueError('Teacher escaped its training role')
    tr = exp.teacher.fit_role(d,ident)
    rec,path,w,a = samples(tr,teacher,parent['cut'][0])
    fits,nets,updates,rng0 = [],{},0,None
    for key in ('one','soft','pair'):
        def loss(net,*x):
            return model.paired_loss(net,*x) if key=='pair' else torch.nn.functional.binary_cross_entropy_with_logits(
                net(x[0]).flatten(),x[1 if key=='one' else 2])
        net,fit,u = prior.prefix.trainer.fit(dict(ident,arm=key,registration=registration),f'{key}_{fold}_{seed}',seed,
            lambda:prior.model.head(a if key=='pair' else w,6 if key=='pair' else 1),STEPS,
            path if key=='pair' else rec,loss,replay,beat)
        rng = torch.load(exp.checked(fit['artifact']),map_location='cpu',weights_only=True)['rng']
        if key=='soft' and rng!=rng0:
            raise ValueError('Single and ensemble sampling streams differ')
        rng0 = rng
        fits.append(fit)
        nets[key] = net
        updates += u
    old = old_cost(parent['fits'][0],a)
    _,cal = base.subset(d,ident['calibration'])
    c,o,b = exp.teacher.analytic(cal['history'])
    table,selected = calibrate(predict(nets,old,cal,c,o,b),cal,c,o,b,parent['cut'])
    hi,h = base.subset(d,ident['held'])
    c,o,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    b = base.expanded(c,raw)
    ranks,actions,scale = predict(nets,old,h,c,o,b)
    q = prior.query_groups(h['query'])
    with np.load(exp.checked(parent['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        match,anchor = z['cost'],z['frozen']
    arms = {f'{r}_{p}':model.select(ranks[r],actions[p],q,scale,match=match)
            for r in RECIPIENTS for p in PATHS}
    np.testing.assert_array_equal(arms['cost_old'],match)
    arms['frozen'] = anchor
    arms['calibrated'] = (np.full_like(match,-1) if selected is None else model.select(ranks['soft'],
        actions[selected['path']],q,scale,fraction=selected['fraction'],margin=selected['margin']))
    artifact = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=h['ids'],**arms))
    return dict(identity=ident,cut=parent['cut'],fits=fits,ladder=table,selected=selected,actions=artifact),updates


def score(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{k}':(b,a) for a in ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(PRIMARY)
    rows,geo = [],{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = {k:z[k] for k in ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            c0 = cv[ix]
            paths = {base.CV:c0,base.OLS:ols[ix],**{k:exp.slate.path_for(c0,b[ix],a[k][ix]) for k in ARMS}}
            row = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = exp.weight(q)
            terms = {k:prior.study.motion.full_masses(paths[b0],paths[a0],c0,y,v,w)
                     for k,(b0,a0) in pairs.items()}
            site, = np.unique(h['sites'][ix])
            rows.append(dict(row,source=str(site),seed=seed,comparisons=terms))
            for r in RECIPIENTS:
                np.testing.assert_array_equal(a[f'{r}_old'][ix]>=0,a[f'{r}_pair'][ix]>=0)
                radius = lambda k:np.linalg.norm(paths[k]-c0,axis=-1).max(1)
                if (radius(f'{r}_pair')>radius(f'{r}_old')+1e-9).any():
                    raise ValueError('Within-recipient path exceeded frozen radius')
            z = geo.setdefault((str(site),seed),np.zeros((len(PRIMARY),4)))
            pair_idx = [(2*i,2*i+1) for i in range(len(PRIMARY))]
            history = h['history'][ix]
            for g in prior.query_groups(q):
                if len({int((a[k][ix][g]>=0).sum()) for k in MATCHED})!=1:
                    raise ValueError('Matched query counts changed')
                arr = np.stack([paths[k][g] for pair in PRIMARY.values() for k in pair])
                _,u,_,(steps,known,_,_) = prior.geometry.query(history[g],arr,y[g],v[g],pair_idx)
                if steps: z[:,1] += u[:,1]; z[:,3] += 1
                if known: z[:,0] += u[:,0]; z[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['cost_old']['switches'])!=(72,9,45698):
        raise ValueError('Unexpected held development roster')
    sites = sorted(set(roster.values()))
    x = np.asarray([[geo[s,k] for k in base.SEEDS] for s in sites])
    p = np.divide(x[...,:2],x[...,2:],out=np.full_like(x[...,:2],np.nan),where=x[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    geometry = {k:dict(observed_pixel=prior.geometry.bootstrap(p[:,:,i,0],draw),
        completion_lower_pixel=prior.geometry.bootstrap(p[:,:,i,1],draw)) for i,k in enumerate(PRIMARY)}
    return summary,geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',action='store_true')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            z = dict(pid=os.getpid(),seconds=round(time.monotonic()-t,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',z)
            if v.get('phase')!='fit' or v.get('step') in (100,STEPS):
                print(z,flush=True)
        files = [OUT/'teacher_freeze.json',OUT/'teacher_registration.json',
                 previous.OUT/'training_freeze.json',previous.OUT/'training_readout.json']
        teach,treg,freeze,readout = map(exp.read,files)
        if (teach['registration']!=exp.sha(files[1]) or not teach['train_only'] or not treg['train_only']
                or readout['freeze']!=exp.sha(files[2]) or not freeze['before_held_scoring']
                or readout['a_labels_opened'] or readout['independent_confirmation'] or not readout['original_2pct']):
            raise ValueError('Need verified training roles and original CV rule')
        reg = OUT/'registration.json'
        exp.once(reg,exp.stable(dict(parents=[exp.artifact(x) for x in files],
            files=[exp.artifact(x) for x in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
            arms=ARMS,grid=GRID,seeds=base.SEEDS,steps=STEPS,original_2pct=True)))
        _,donor,_ = base.register()
        d = base.fit_population(beat)
        cells,updates = [],0
        for x in freeze['cells']:
            ident = x['identity']
            c,u = cell(d,donor,x,teach['cells'][ident['fold']],exp.sha(reg),beat,args.replay)
            cells.append(c)
            updates += u
            beat(phase='cell_frozen',fold=ident['fold'],seed=ident['seed'],updates=updates)
        f = OUT/'freeze.json'
        exp.once(f,exp.stable(dict(registration=exp.sha(reg),cells=cells,optimizer_steps=27*STEPS,before_held_scoring=True)))
        summary,geometry = score(exp.read(f)['cells'],d,donor,beat)
        exp.once(OUT/'readout.json',exp.stable(dict(freeze=exp.sha(f),summary=summary,geometry=geometry,
            qualified=sum(c['selected'] is not None for c in cells),
            calibration_max_passes=[max(x['passes'] for x in c['ladder']) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
