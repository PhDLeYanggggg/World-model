"""Bound fixed-recipient utility even for arbitrary paths inside each peak."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_fixed_recipient_support as study

old,base,prior,np,model = study.old,study.base,study.prior,study.np,study.model
ALPHAS = (.25,.5,.75,1.,None)


def bound(c,o,anchor,selected,y,v,q,fraction):
    clean = np.where(v[...,None],y,0.)
    known_cv = np.linalg.norm(c-clean,axis=-1)
    known_ols = np.linalg.norm(o-clean,axis=-1)
    gap = np.linalg.norm(c-o,axis=-1)
    scale = model.peak(anchor-c)
    if fraction is None:
        r = np.where(selected,np.inf,0.)
    else:
        r = np.where(selected,scale*fraction,0.)
    best_known = np.maximum(0.,known_cv-r[:,None])
    best_missing = np.maximum(0.,gap-r[:,None])
    w = old.weight(q)
    ref = w@np.where(v,known_cv,0.).sum(1)
    if ref<=0.:
        return None
    best_cv = w@np.where(v,known_cv-best_known,0.).sum(1)
    best_ols = w@np.where(v,known_ols-best_known,-best_missing).sum(1)
    return (100.*np.asarray([best_cv,best_ols])/ref).tolist()


def main():
    f,r = [old.OUT/k for k in ('freeze.json','readout.json')]
    a,b = map(old.read,(f,r))
    if (b['freeze']!=old.sha(f) or not a['before_held_scoring'] or b['a_labels_opened']
            or b['independent_confirmation'] or not b['original_2pct']):
        raise ValueError('Require development-only frozen recipients')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    out = {str(k):[] for k in ALPHAS}
    for cell in a['cells']:
        i = cell['identity']
        hi,h = base.subset(d,i['held'])
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,i['seed'],d).items()})
        bank = base.expanded(cv,raw)
        with np.load(old.checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            chosen = z['frozen']
        p = old.slate.path_for(cv,bank,chosen)
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            args = (cv[ix],ols[ix],p[ix],chosen[ix]>=0,*(h[k][ix] for k in ('target','valid','query')))
            for k in ALPHAS:
                out[str(k)].append(bound(*args,k))
    summary = {}
    for k,z in out.items():
        if len(z)!=72:
            raise ValueError('Incomplete development roster')
        x = np.asarray([v for v in z if v is not None])
        summary[k] = dict(scored=len(x),undefined=len(z)-len(x),
            ols4_nonpositive=int((x[:,1]<=0.).sum()),cv_nonpositive=int((x[:,0]<=0.).sum()),
            median_upper_pp=np.median(x,axis=0).tolist())
    old.once(study.OUT/'arbitrary_envelope.json',old.stable(dict(parent=old.artifact(r),code=old.artifact(Path(__file__)),
        summary=summary,a_labels_opened=False,original_2pct=True,independent_confirmation=False,
        ignores_recording_risk=True,allows_independent_path_per_step=True,labels_hindsight_only=True)))
    print(summary,flush=True)


if __name__=='__main__':
    main()
