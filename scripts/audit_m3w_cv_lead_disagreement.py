"""Attribute the paired completion-width penalty to shape or recipient changes."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_lead_risk as study

np,base,prior = study.np,study.base,study.prior
read,sha,artifact,once,stable,checked = study.read,study.sha,study.artifact,study.once,study.stable,study.checked
PAIRS = dict(lead_matched=('cond_flat_matched','cond_lead_matched'),
             tail_matched=('cond_lead_matched','tail_lead_matched'))


def main():
    out = study.OUT
    rp,fp = out/'readout.json',out/'freeze.json'
    r,f = read(rp),read(fp)
    if r['freeze']!=sha(fp) or r['a_labels_opened'] or r['independent_confirmation']:
        raise ValueError('Require fitting-only readout')
    _,donor,_ = base.register()
    data = base.fit_population(lambda **v:None)
    masses = {}
    for c in f['cells']:
        i = c['identity']
        hi,h = base.subset(data,i['held'])
        seed = i['seed']
        cv,_,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
        bank = base.expanded(cv,raw)
        y,v,q = (h[k] for k in ('target','valid','query'))
        y = np.where(v[...,None],y,0.)
        _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
        w = 1./n[inv]
        ref = np.where(v,np.linalg.norm(cv-y,axis=-1),0.).sum(1)
        with np.load(checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = {k:z[k] for k in {k for pair in PAIRS.values() for k in pair}}
        for k,(left,right) in PAIRS.items():
            p,z = [prior.apply(cv,bank,a[k0]) for k0 in (left,right)]
            known = np.where(v,np.linalg.norm(p-y,axis=-1)-np.linalg.norm(z-y,axis=-1),0.)
            width = np.where(v,0.,np.linalg.norm(p-z,axis=-1))
            changed = (a[left]>=0)^(a[right]>=0)
            shape = (a[left]>=0)&(a[right]>=0)
            for s in set(h['sites']):
                ix = h['sites']==s
                mass = np.r_[w[ix]@ref[ix],w[ix]@known[ix].sum(1),
                    (w[ix]@width[ix]).reshape(3,4).sum(1),
                    w[ix&changed]@width[ix&changed].sum(1),w[ix&shape]@width[ix&shape].sum(1)]
                key = (k,str(s),seed)
                masses[key] = masses.get(key,np.zeros(7))+mass
    result = {}
    sources = sorted(set(s for _,s,_ in masses))
    for k in PAIRS:
        p = np.array([masses[k,s,t] for s in sources for t in base.SEEDS])
        t = (100*p[:,1:]/p[:,:1]).mean(0)
        lower = t[0]-t[1:4].sum()
        if not np.isclose(lower,r['summary']['effects'][k]['completion_lower_pp']['point'],rtol=0.,atol=1e-9):
            raise ValueError('Missing-step decomposition differs from registered score')
        np.testing.assert_allclose(t[1:4].sum(),t[4:].sum(),atol=1e-9)
        result[k] = dict(known_pp=float(t[0]),paired_lower_pp=float(lower),
            debit_by_lead_block_pp=t[1:4].tolist(),debit_recipient_change_pp=float(t[4]),
            debit_shared_recipient_shape_pp=float(t[5]))
    once(out/'missing_disagreement.json',stable(dict(parent=artifact(rp),code=artifact(Path(__file__)),
        summary=result,lead_blocks=[[1,4],[5,8],[9,12]],original_2pct=True,independent_confirmation=False)))
    print(result)


if __name__=='__main__':
    main()
