"""Compare bounded path ranking against equal-peak shrinkage on the same recipients."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_recipient_path as study

exp,np,base,prior = study.exp,study.np,study.base,study.prior
ARMS = ('cost_radius','soft_radius')


def radius_control(cv,old,new,selected):
    r0,r1 = (np.linalg.norm(x-cv,axis=-1).max(1) for x in (old,new))
    if (r1[selected]<=0.).any() or (r1[selected]>r0[selected]+1e-9).any():
        raise ValueError('Invalid within-row radius match')
    f = np.ones(len(cv))
    f[selected] = r1[selected]/r0[selected]
    z = cv+f[:,None,None]*(old-cv)
    np.testing.assert_allclose(np.linalg.norm(z-cv,axis=-1).max(1),r1,rtol=1e-6,atol=1e-8)
    return z


def collect(cells,d,donor):
    scorer = prior.prior.parent.readout
    rows,changes = [],{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = {k:z[k] for k in ('frozen','cost_old','cost_pair','soft_old','soft_pair')}
        paths = {k:exp.slate.path_for(cv,b,z) for k,z in a.items()}
        paths.update({base.CV:cv,base.OLS:ols})
        for k in ('cost','soft'):
            l,r = f'{k}_old',f'{k}_pair'
            at = a[l]>=0
            np.testing.assert_array_equal(at,a[r]>=0)
            paths[f'{k}_radius'] = radius_control(cv,paths[l],paths[r],at)
            r0,r1 = (np.linalg.norm(paths[name]-cv,axis=-1).max(1)[at] for name in (l,r))
            changes.setdefault(k,np.zeros(4))
            changes[k] += [(a[l][at]!=a[r][at]).sum(),at.sum(),r0.sum(),r1.sum()]
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            local = {k:p[ix] for k,p in paths.items()}
            w = exp.weight(q)
            z = scorer.score_recording(local,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            pairs = dict(cost_shape=('cost_radius','cost_pair'),soft_shape=('soft_radius','soft_pair'),
                         shrink_cost=('cost_old','cost_radius'),shrink_soft=('soft_old','soft_radius'),
                         full_vs_frozen=('frozen','soft_pair'))
            terms = {k:prior.study.motion.full_masses(local[l],local[r],cv[ix],y,v,w)
                     for k,(l,r) in pairs.items()}
            site, = np.unique(h['sites'][ix])
            rows.append(dict(z,source=str(site),seed=seed,comparisons=terms))
    return rows,changes


def main():
    f,r = [study.OUT/k for k in ('freeze.json','readout.json')]
    freeze,result = exp.read(f),exp.read(r)
    if (result['freeze']!=exp.sha(f) or result['a_labels_opened']
            or result['independent_confirmation'] or not result['original_2pct']):
        raise ValueError('Require checked development-only recipient study')
    _,donor,_ = base.register()
    data = base.fit_population(lambda **v:None)
    rows,counts = collect(freeze['cells'],data,donor)
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    changed = {k:dict(actions_changed=int(m[0]),matched_count=int(m[1]),sum_radius_ratio=float(m[3]/m[2]))
               for k,m in counts.items()}
    exp.once(study.OUT/'radius_control.json',exp.stable(dict(parent=exp.artifact(r),code=exp.artifact(Path(__file__)),
        summary=summary,changed=changed,diagnostic_only=True,a_labels_opened=False,independent_confirmation=False,
        original_2pct=True)))
    print({k:v['completion_lower_pp'] for k,v in summary['effects'].items()},flush=True)


if __name__=='__main__':
    main()
