"""Train an analytic minimum-edit teacher and its cost-only matched control."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_minimum_budget as study
from src.world_model import m3w_cv_minimum_budget as model

exp,np,base,prior = study.exp,study.np,study.base,study.prior
torch = exp.torch
OUT,PRIVATE = study.OUT,study.PRIVATE
STEPS = 1200
MODELS = ('cost','teacher')
ARMS = ('frozen','cost','teacher','teacher_matched','teacher_guarded','calibrated')
GRID = tuple((a,f,p,g) for a in MODELS for f in (.10,.25,.50) for p in (1.,3.) for g in (False,True))


def context(d):
    p,_ = prior.study.motion.features(d['history'],'prefix')
    return model.query_context(p,d['query']),prior.model.scale(d['history'])


def sampler(tr,z,cut):
    c,o,b = exp.teacher.analytic(tr['history'])
    p,s = context(tr)
    with np.load(exp.checked(z['teacher']),allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],tr['ids'])
        action,valid = a['action'],a['verified']
    sites = [np.flatnonzero(tr['sites']==site) for site in sorted(set(tr['sites']))]
    chosen = [g[action[g]>=0] for g in sites if (action[g]>=0).any()]
    if not chosen:
        raise ValueError('Expected a verified minimum-edit teaching sample')
    y,v = tr['target'],tr['valid']

    def draw(rng):
        at = prior.draw(sites,rng)
        positive = prior.draw(chosen,rng)[:len(at)//4]
        at[:len(positive)] = positive
        j = rng.integers(b.shape[1],size=len(at))
        j[:len(positive)] = action[positive]
        x = model.action_features(p[at],c[at],o[at],b[at,j],s[at])
        target = model.targets(c[at],o[at],b[at,j],y[at],v[at],s[at],cut)
        return tuple(torch.from_numpy(a) for a in (x,target,action[at]==j,valid[at]))

    return draw,p.shape[1]+37


@torch.inference_mode()
def infer(nets,d,c,o,b):
    past,scale = context(d)
    n,k = b.shape[:2]
    predicted = {a:np.empty((n,k,6),np.float32) for a in nets}
    legal = (b!=c[:,None]).any((2,3))
    for j in range(k):
        for first in range(0,n,4096):
            ix = slice(first,first+4096)
            x = torch.from_numpy(model.action_features(past[ix],c[ix],o[ix],b[ix,j],scale[ix]))
            for a,net in nets.items():
                predicted[a][ix,j] = net(x).numpy()
    if any(not np.isfinite(x).all() for x in predicted.values()):
        raise ValueError('Nonfinite predictions')
    return predicted,legal,scale


def decode(pred,legal,groups,scale,arm,f,p,g,match=None):
    return model.choose(pred[arm],legal,groups,scale,f,p,arm=='teacher',match=match,guard=g)


def calibrate(pred,legal,groups,scale,c,o,b,d,cut):
    table = []
    scorer = prior.prior.parent.readout
    n = len(np.unique(d['recording']))
    for a,f,p,g in GRID:
        z = decode(pred,legal,groups,scale,a,f,p,g)
        rows = base.metrics(c,o,b,d,{'strict':z},cut)
        gain = [[scorer.merge_comparisons([r['comparisons'][f'strict_vs_{k}'] for r in rows if r['source']==s])
                    ['completion_lower_pp'] for k in ('cv','ols4')] for s in sorted(set(d['sites']))]
        table.append(dict(arm=a,fraction=f,penalty=p,guard=g,edits=int((z>=0).sum()),
            passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),lower=np.mean(gain,axis=0).tolist()))
    good = [x for x in table if x['edits'] and x['passes']==n and min(x['lower'])>0.]
    return table,max(good,key=lambda r:(min(r['lower']),-r['edits'])) if good else None


def cell(d,donor,old,t,reg,beat,replay):
    ident = old['identity']
    fold,seed = ident['fold'],ident['seed']
    if t['fold']!=fold or t['train']!=ident['train']:
        raise ValueError('Teacher escaped its training role')
    tr = exp.teacher.fit_role(d,ident)
    draw,w = sampler(tr,t,old['cut'][0])
    nets,fits,updates,state = {},[],0,None
    for a in MODELS:
        def loss(net,*x):
            return model.objective(net,*x,a=='teacher')
        net,fit,u = prior.prefix.trainer.fit(dict(ident,arm=a,registration=reg),f'{a}_{fold}_{seed}',seed,
            lambda:prior.model.head(w,6),STEPS,draw,loss,replay,beat)
        rng = torch.load(exp.checked(fit['artifact']),map_location='cpu',weights_only=True)['rng']
        if state is not None and rng!=state:
            raise ValueError('Teacher ablation sampling streams diverged')
        nets[a] = net
        fits.append(fit)
        updates += u
        state = rng
    _,cal = base.subset(d,ident['calibration'])
    c,o,b = exp.teacher.analytic(cal['history'])
    p,l,s = infer(nets,cal,c,o,b)
    table,sel = calibrate(p,l,prior.query_groups(cal['query']),s,c,o,b,cal,old['cut'])
    hi,h = base.subset(d,ident['held'])
    c,o,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    b = base.expanded(c,raw)
    p,l,s = infer(nets,h,c,o,b)
    q = prior.query_groups(h['query'])
    with np.load(exp.checked(old['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        a = {'frozen':z['frozen']}
    a['cost'] = decode(p,l,q,s,'cost',.25,1.,False)
    a['teacher'] = decode(p,l,q,s,'teacher',.25,1.,False)
    a['teacher_matched'] = decode(p,l,q,s,'teacher',.25,1.,False,match=a['cost'])
    a['teacher_guarded'] = decode(p,l,q,s,'teacher',.25,1.,True)
    a['calibrated'] = (np.full_like(a['cost'],-1) if sel is None else
        decode(p,l,q,s,sel['arm'],sel['fraction'],sel['penalty'],sel['guard']))
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=h['ids'],**a))
    return dict(identity=ident,cut=old['cut'],fits=fits,ladder=table,selected=sel,actions=path),updates


def score(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{k}':(b,a) for a in ARMS for k,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(teacher_at_match=('cost','teacher_matched'),guarded_vs_teacher=('teacher','teacher_guarded'))
    rows,geo = [],{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = {k:z[k] for k in ARMS}
        for r in np.unique(h['recording']):
            ix = h['recording']==r
            c0,o = cv[ix],ols[ix]
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            paths = {base.CV:c0,base.OLS:o,**{k:exp.slate.path_for(c0,bank[ix],a[k][ix]) for k in ARMS}}
            row = scorer.score_recording(paths,{},y,v,q,str(r),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = exp.weight(q)
            terms = {k:prior.study.motion.full_masses(paths[b],paths[a],c0,y,v,w) for k,(b,a) in pairs.items()}
            site, = np.unique(h['sites'][ix])
            rows.append(dict(row,source=str(site),seed=seed,comparisons=terms))
            acc = geo.setdefault((str(site),seed),np.zeros(4))
            history = h['history'][ix]
            for g in prior.query_groups(q):
                if int((a['cost'][ix][g]>=0).sum())!=int((a['teacher_matched'][ix][g]>=0).sum()):
                    raise ValueError('Held matched counts changed')
                arr = np.stack([paths[k][g] for k in ('cost','teacher_matched')])
                _,u,_,(steps,known,_,_) = prior.geometry.query(history[g],arr,y[g],v[g],((0,1),))
                if steps: acc[1] += u[0,1]; acc[3] += 1
                if known: acc[0] += u[0,0]; acc[2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['frozen']['switches'])!=(72,9,54748):
        raise ValueError('Expected the same held fitting-source roster')
    sites = sorted(set(roster.values()))
    z = np.array([[geo[s,k] for k in base.SEEDS] for s in sites])
    p = np.divide(z[...,:2],z[...,2:],out=np.full_like(z[...,:2],np.nan),where=z[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    geometry = dict(observed_pixel=prior.geometry.bootstrap(p[:,:,0],draw),
                    completion_lower_pixel=prior.geometry.bootstrap(p[:,:,1],draw))
    return summary,geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    a = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            z = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'train_heartbeat.json',z)
            if v.get('phase')!='train' or v.get('step') in (100,STEPS):
                print(z,flush=True)
        paths = (OUT/'teacher_freeze.json',OUT/'registration.json',exp.OUT/'freeze.json',exp.OUT/'readout.json')
        t,r,f,s = map(exp.read,paths)
        if (not t['train_only'] or t['registration']!=exp.sha(paths[1])
                or not f['before_held_scoring'] or s['freeze']!=exp.sha(paths[2])
                or s['independent_confirmation'] or s['a_labels_opened'] or not s['original_2pct']):
            raise ValueError('Require frozen training-only teacher and unchanged development roles')
        reg = OUT/'training_registration.json'
        exp.once(reg,exp.stable(dict(parents=[exp.artifact(x) for x in paths],
            files=[exp.artifact(x) for x in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
            models=MODELS,arms=ARMS,steps=STEPS,grid=GRID,seeds=base.SEEDS)))
        _,donor,_ = base.register()
        d = base.fit_population(lambda **v:None)
        cells,updates = [],0
        for old in f['cells']:
            z,u = cell(d,donor,old,t['cells'][old['identity']['fold']],exp.sha(reg),beat,a.replay)
            cells.append(z)
            updates += u
            beat(phase='cell_frozen',fold=z['identity']['fold'],seed=z['identity']['seed'],updates=updates)
        fp = OUT/'training_freeze.json'
        exp.once(fp,exp.stable(dict(registration=exp.sha(reg),cells=cells,optimizer_steps=18*STEPS,before_held_scoring=True)))
        summary,geometry = score(exp.read(fp)['cells'],d,donor,beat)
        exp.once(OUT/'training_readout.json',exp.stable(dict(freeze=exp.sha(fp),summary=summary,geometry=geometry,
            qualified=sum(c['selected'] is not None for c in cells),
            calibration_max_passes=[max(x['passes'] for x in c['ladder']) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=a.replay)


if __name__=='__main__':
    main()
