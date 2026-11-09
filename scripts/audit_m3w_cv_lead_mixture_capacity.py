"""Check whether new paths or more recipients are needed at frozen budgets."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_record_allocation_capacity as cap
from scripts import run_m3w_cv_lead_mixture as study

np,torch,rank,screen,base,prior = study.np,study.torch,study.rank,study.screen,study.base,study.prior
read,sha,artifact,once,stable,checked = study.read,study.sha,study.artifact,study.once,study.stable,study.checked


def perfect_upper(cv,ols,y,valid,q,selected):
    """Per-reference maxima with arbitrary paths, before any risk constraint."""
    clean = np.where(valid[...,None],y,0.)
    ec,eo = [np.linalg.norm(z-clean,axis=-1) for z in (cv,ols)]
    missing = np.where(valid,0.,np.linalg.norm(cv-ols,axis=-1)).sum(1)
    _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
    w = 1./n[inv]
    ref = np.where(valid,ec,0.).sum(1)
    denom = float(w@ref)
    if denom<=0.:
        return None
    # For each reference separately, a perfect edit equals labels when known
    # and that reference at every missing step. This relaxes the common path.
    offset = float(w@(np.where(valid,eo-ec,0.).sum(1)-missing))
    utility = np.column_stack((ref,ref+missing))*w[:,None]
    fixed = utility[selected].sum(0)
    eligible = np.concatenate([g[np.argsort(-utility[g,1],kind='stable')[:base.CAP]]
                               for g in prior.query_groups(q)])
    k = int(selected.sum())
    # Per-reference top-k can differ even within each query.
    c = np.concatenate([g[np.argsort(-utility[g,0],kind='stable')[:base.CAP]] for g in prior.query_groups(q)])
    free = [np.sort(utility[c,0])[::-1][:k].sum(),np.sort(utility[eligible,1])[::-1][:k].sum()]
    return dict(fixed_pp=((fixed+[0.,offset])*100./denom).tolist(),
                free_pp=((np.asarray(free)+[0.,offset])*100./denom).tolist())


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (study.PRIVATE/'capacity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fp,rp = study.OUT/'freeze.json',study.OUT/'readout.json'
        old_capacity = study.parent.OUT/'hindsight_capacity.json'
        f,r,old = [read(z) for z in (fp,rp,old_capacity)]
        if (r['freeze']!=sha(fp) or not f['before_held_scoring'] or r['independent_confirmation']
                or r['a_labels_opened'] or old['parent']['sha256']!=sha(study.parent.OUT/'readout.json')):
            raise ValueError('Expected fitting-only proposal and allocation evidence')
        _,donor,_ = base.register()
        data = base.fit_population(lambda **v:None)
        a,b = [read(s.OUT/'freeze.json')['cells'] for s in (base,rank)]
        code = artifact(Path(__file__))
        modes = ('shared_addback','lead_addback','both_addback')
        aggregate = {k:Counter() for k in modes}
        upper = []
        for i,(c,old_risk,pair) in enumerate(zip(f['cells'],a,b,strict=True)):
            ident = c['identity']
            if not (ident==old_risk['identity']==pair['identity']):
                raise ValueError('Parent identity changed')
            hi,d = base.subset(data,ident['held'])
            seed = ident['seed']
            cv,ols,raw = prior.bank(d['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
            bank = base.expanded(cv,raw)
            h = d['history']
            width = prior.study.motion.features(h[:1],'prefix')[0].shape[1]
            risk = screen.load_head(old_risk['fits'][0],width+25,8)
            net = screen.load_head(pair['fits'][1],width+49,2)
            nominee = rank.nomination(rank.forecast(net,cv,ols,bank,h),screen.predictions(risk,cv,bank,h),screen.RELAXED)[0]
            path = prior.apply(cv,bank,nominee)
            with np.load(checked(c['allocation']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['ids'])
                chosen = z['local']>=0
                np.testing.assert_array_equal(nominee[chosen],z['local'][chosen])
            with np.load(checked(c['paths']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],d['ids'])
                proposals = [np.where((nominee>=0)[:,None,None],z[k],cv) for k in study.FAMILIES]
            for rec in np.unique(d['recording']):
                ix = d['recording']==rec
                cr,orr,y,v,q = cv[ix],ols[ix],d['target'][ix],d['valid'][ix],d['query'][ix]
                bank = np.stack((path[ix],*[p[ix] for p in proposals]),axis=1)
                active = chosen[ix]
                bound = perfect_upper(cr,orr,y,v,q,active)
                upper.append(bound)
                _, inv = np.unique(q,return_inverse=True)
                counts = np.bincount(inv[active],minlength=int(inv.max())+1)
                cache = study.PRIVATE/f'capacity_{i}_{rec}.json'
                identity = dict(cell=i,recording=int(rec),parent=artifact(rp),code=code)
                menus = dict(zip(modes,(bank[:,[0,1]],bank[:,[0,2]],bank),strict=True))
                if not cache.exists():
                    decisions = {k:cap.witness(cap.m.coefficients(cr,orr,p,y,v,q,c['cut'][0]),
                        (cr,orr,p,y,v,q,*c['cut']),active,counts,'dual','record_exact') for k,p in menus.items()}
                    once(cache,stable(dict(identity=identity,decisions=decisions)))
                record = read(cache)
                if record['identity']!=identity:
                    raise ValueError('Capacity checkpoint changed')
                for k,p in menus.items():
                    z = record['decisions'][k]
                    if z['status']=='verified':
                        verified = cap.audit.verify(z['actions'],cr,orr,p,y,v,q,*c['cut'],base.CAP)
                        if not verified['verified'] or verified['edits']!=active.sum():
                            raise ValueError('Invalid replayed witness')
                    aggregate[k][z['status']] += 1
                prior.donor.prior.prior.source.atomic(study.PRIVATE/'capacity_heartbeat.json',
                    dict(pid=os.getpid(),cell=i,recording=int(rec)))
            print(dict(pid=os.getpid(),cell=i,seed=seed),flush=True)
        if len(upper)!=72:
            raise ValueError('Wrong recording roster')
        result = dict(parent=artifact(rp),prior_capacity=artifact(old_capacity),code=code,
            original=old['counts']['dual_record_exact'],same_screened_recipients=True,same_exact_record_budget=True,
            proposed={k:dict(v) for k,v in aggregate.items()},
            perfect_bound={k:dict(positive_dual=sum(min(u[k])>cap.m.POSITIVE_PP for u in upper if u is not None),
                                  upper_ols_pp_median=float(np.median([u[k][1] for u in upper if u is not None])))
                           for k in ('fixed_pp','free_pp')},
            missing_reference=sum(v is None for v in upper),uses_outcomes_only_as_hindsight=True,
            original_2pct=True,independent_confirmation=False,a_labels_opened=False)
        once(study.OUT/'hindsight_capacity.json',stable(result))
        print(result['proposed'],result['perfect_bound'],flush=True)


if __name__=='__main__':
    main()
