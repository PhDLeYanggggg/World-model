"""Control the realized query counts when replacing the gain model."""
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_signed_coverage as study

np,torch,base,prior,screen = study.np,study.torch,study.base,study.prior,study.screen
read,sha,artifact,once,stable,checked = study.read,study.sha,study.artifact,study.once,study.stable,study.checked


def matched_query(g0,g1,r,p,legal,limit):
    pool = study.prior_pool.pool_query
    cap = int(limit)
    while cap>0:
        a,b = [pool(g,r,p,legal,cap,.10) for g in (g0,g1)]
        ca,cb = (a>=0).sum(),(b>=0).sum()
        if ca==cb:
            return a,b
        cap = min(ca,cb)
    return (np.full(len(r),-1,dtype=np.int8),np.full(len(r),-1,dtype=np.int8))


def freeze_cell(c,old,d,donor):
    ident = c['identity']
    if ident!=old['identity']:
        raise ValueError('Parent identity changed')
    hi,h = base.subset(d,ident['held'])
    seed,fold = ident['seed'],ident['fold']
    cv,_,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    bank = base.expanded(cv,raw)
    width = prior.study.motion.features(h['history'][:1],'prefix')[0].shape[1]+25
    nets = {k:screen.load_head(z,width,2 if k=='swap' else 5)
            for k,z in zip((*study.RISKS,'swap'),c['fits'],strict=True)}
    legacy = screen.load_head(old['fits'][0],width,8)
    p = study.infer(nets,cv,bank,h['history'],legacy)
    scale = prior.model.scale(h['history'])
    groups = prior.query_groups(h['query'])
    arrays = dict(ids=h['ids'])
    with np.load(checked(c['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        for k in study.RISKS:
            g,r,e,legal = p[k+'_old']
            g1 = p[k+'_swap'][0]
            a0,b0 = z[k+'_old'],z[k+'_swap']
            a,b = np.full_like(a0,-1),np.full_like(b0,-1)
            for ix in groups:
                n = min((a0[ix]>=0).sum(),(b0[ix]>=0).sum())
                s = scale[ix,None]
                a[ix],b[ix] = matched_query(g[ix]*s,g1[ix]*s,r[ix]*s[:,:,None],e[ix],legal[ix],n)
            arrays[k+'_old'],arrays[k+'_swap'] = a,b
    path = prior.study.parent.parent.save_actions(study.PRIVATE/f'matched_gain_{fold}_{seed}.npz',arrays)
    return dict(identity=ident,cut=c['cut'],paths=path)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out,private = study.OUT,study.PRIVATE
    with (private/'matched_gain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        public,oldf = out/'readout.json',out/'freeze.json'
        x,y = read(public),read(oldf)
        if x['freeze']!=sha(oldf) or x['a_labels_opened'] or x['independent_confirmation']:
            raise ValueError('Expected fitting-only parents')
        reg = dict(parent=artifact(public),code=artifact(Path(__file__)),same_count_each_query=True)
        _,donor,_ = base.register()
        data = base.fit_population(lambda **v:None)
        fp = out/'matched_gain_freeze.json'
        if not fp.exists():
            old = read(base.OUT/'freeze.json')['cells']
            cells = []
            for c,v in zip(y['cells'],old,strict=True):
                cells.append(freeze_cell(c,v,data,donor))
                print(dict(phase='frozen',fold=c['identity']['fold'],seed=c['identity']['seed']),flush=True)
            once(fp,stable(dict(**reg,cells=cells,before_held_scoring=True)))
        f = read(fp)
        if any(f[k]!=v for k,v in reg.items()) or not f['before_held_scoring'] or len(f['cells'])!=9:
            raise ValueError('Matched gain freeze changed')
        scorer = prior.prior.parent.readout
        rows = []
        for c in f['cells']:
            ident = c['identity']
            hi,d = base.subset(data,ident['held'])
            seed = ident['seed']
            cv,_,raw = prior.bank(d['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
            bank = base.expanded(cv,raw)
            with np.load(checked(c['paths']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['ids'])
                actions = {f'{k}_{g}':z[f'{k}_{g}'] for k in study.RISKS for g in ('old','swap')}
            for rec in np.unique(d['recording']):
                ix = d['recording']==rec
                c0 = cv[ix]
                q,y,v = (d[k][ix] for k in ('query','target','valid'))
                for g in prior.query_groups(q):
                    for k in study.RISKS:
                        if (actions[k+'_old'][ix][g]>=0).sum()!=(actions[k+'_swap'][ix][g]>=0).sum():
                            raise ValueError('Counts are not matched')
                paths = {base.CV:c0,**{k:prior.apply(c0,bank[ix],a[ix]) for k,a in actions.items()}}
                r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
                _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
                pairs = {k:prior.study.motion.full_masses(paths[k+'_old'],paths[k+'_swap'],c0,y,v,1./n[inv])
                         for k in study.RISKS}
                source, = np.unique(d['sites'][ix])
                rows.append(dict(r,source=str(source),seed=seed,comparisons=pairs))
        summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
        if summary['views']!=72:
            raise ValueError('Incomplete matched gain readout')
        once(out/'matched_gain.json',stable(dict(freeze=artifact(fp),summary=summary,
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))


if __name__=='__main__':
    main()
