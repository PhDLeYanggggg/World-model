"""Change edit timing at equal per-agent displacement, without tuning outcomes."""
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_lead_risk as study

np,base,prior = study.np,study.base,study.prior
read,sha,artifact,once,stable,checked = study.read,study.sha,study.artifact,study.once,study.stable,study.checked
PARENT = ('cond_lead_matched','tail_lead_matched')
WEIGHT = (1.,1.,1.,1.,.8,.6,.4,.2,0.,0.,0.,0.)


def redistribute(cv,path):
    d = path-cv
    w = np.linalg.norm(d,axis=-1)
    mass = w.sum(1)
    budget = w@np.asarray(WEIGHT)
    alpha = np.divide(budget,mass,out=np.zeros_like(mass),where=mass>0.)
    tapered = cv+d*np.asarray(WEIGHT)[None,:,None]
    uniform = cv+d*alpha[:,None,None]
    np.testing.assert_allclose(np.linalg.norm(tapered-cv,axis=-1).sum(1),
                              np.linalg.norm(uniform-cv,axis=-1).sum(1),atol=1e-7)
    for p in (tapered,uniform):
        if not np.array_equal((p!=cv).any((1,2)),mass>0.):
            raise ValueError('Taper changed the edited recipient mask')
    return uniform,tapered


def main():
    out,private = study.OUT,study.PRIVATE
    with (private/'late_taper.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rp,fp = out/'readout.json',out/'freeze.json'
        r,f = read(rp),read(fp)
        if r['freeze']!=sha(fp) or r['a_labels_opened'] or r['independent_confirmation']:
            raise ValueError('Require fitting-only readout')
        reg = dict(parent=artifact(rp),code=artifact(Path(__file__)),weights=WEIGHT,
                   same_recipient_and_total_displacement=True,original_2pct=True,independent_confirmation=False)
        once(out/'late_taper_registration.json',stable(reg))
        _,donor,_ = base.register()
        data = base.fit_population(lambda **v:None)
        scorer = prior.prior.parent.readout
        rows,peak = [],{}
        for cell in f['cells']:
            ident = cell['identity']
            hi,h = base.subset(data,ident['held'])
            seed = ident['seed']
            cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
            bank = base.expanded(cv,raw)
            with np.load(checked(cell['actions']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],h['ids'])
                a = {k:z[k] for k in PARENT}
            path = {base.CV:cv,base.OLS:ols}
            for key in PARENT:
                p = prior.apply(cv,bank,a[key])
                u,t = redistribute(cv,p)
                path.update({key:p,key+'_uniform':u,key+'_taper':t})
                sums = peak.setdefault(key,np.zeros(3))
                sums[0] += (a[key]>=0).sum()
                sums[1:] += [np.linalg.norm(x-cv,axis=-1).max(1).sum() for x in (u,t)]
            for rec in np.unique(h['recording']):
                ix = h['recording']==rec
                q,y,v = (h[k][ix] for k in ('query','target','valid'))
                c0 = cv[ix]
                p = {k:v[ix] for k,v in path.items()}
                z = scorer.score_recording(p,{},y,v,q,str(rec),easy_cut=cell['cut'][0],hard_cut=cell['cut'][1])
                _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
                comps = {f'{k}_taper_vs_uniform':(k+'_uniform',k+'_taper') for k in PARENT}
                comps.update({f'{k}_{tag}_vs_{s}':(b,k+'_'+tag) for k in PARENT for tag in ('uniform','taper')
                              for s,b in (('cv',base.CV),('ols4',base.OLS))})
                terms = {k:prior.study.motion.full_masses(p[b],p[a],c0,y,v,1./n[inv]) for k,(b,a) in comps.items()}
                source, = np.unique(h['sites'][ix])
                rows.append(dict(z,source=str(source),seed=seed,comparisons=terms))
        summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
        if summary['views']!=72 or any(p[0]!=51614 for p in peak.values()):
            raise ValueError('Expected original matched recipients')
        once(out/'late_taper.json',stable(dict(registration=artifact(out/'late_taper_registration.json'),summary=summary,
            mean_peak_pixel={k:dict(zip(('uniform','taper'),(v[1:]/v[0]).tolist(),strict=True)) for k,v in peak.items()},
            original_2pct=True,independent_confirmation=False,a_labels_opened=False)))


if __name__=='__main__':
    main()
