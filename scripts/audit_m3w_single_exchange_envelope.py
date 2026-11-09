"""Test whether a more powerful single-query action can close the target-domain gap."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cross_domain_pass_transition as exp
from src.evaluation import m3w_single_exchange_envelope as model

run, np = exp.run, exp.np


def main():
    key,*_ = exp.register()
    fp,rp = (exp.OUT/k for k in ('freeze.json','readout.json'))
    f,r = map(run.read,(fp,rp))
    if (f['registration']!=key or r['freeze']!=run.sha(fp) or not f['before_scoring'] or not r['original_2pct']
            or r['a_labels_opened'] or r['confirmation_access'] or r['independent_confirmation']):
        raise ValueError('Use only the admitted fitting-source development result')
    rows = []
    for c,(meta,d,cv,ols,anchor) in zip(f['moving'],exp.moving_cells(lambda **kw:None),strict=True):
        if any(c[k]!=meta[k] for k in ('fold','seed','cut')):
            raise ValueError('Moving rotation changed')
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['anchor'],anchor)
        for rec in np.unique(d['rec']):
            ix = d['rec']==rec
            q,y,v = (d[k][ix] for k in ('query','y','valid'))
            c0,o0 = cv[ix],ols[ix]
            w = exp.moving.old.weight(q)
            base,swap,full = model.bound(c0,np.stack((c0,o0)),anchor[ix],y,v,
                exp.moving.prior.query_groups(q),w)
            denom = w @ np.where(v,np.linalg.norm(c0-y,axis=-1),0.).sum(1)
            if denom<=0.:
                raise ValueError('Require observed CV error mass')
            s, = np.unique(d['site'][ix])
            rows.append(dict(site=str(s),seed=meta['seed'],recording=str(rec),base_pp=(100.*base/denom).tolist(),
                single_swap_upper_pp=(100.*swap/denom).tolist(),whole_query_upper_pp=(100.*full/denom).tolist()))
    summary = {}
    for k in ('base_pp','single_swap_upper_pp','whole_query_upper_pp'):
        a = np.array([z[k] for z in rows])
        summary[k] = dict(views=len(a),either_nonpositive=int((a.min(1)<=0.).sum()),
                          ols4_nonpositive=int((a[:,1]<=0.).sum()),median=np.median(a,axis=0).tolist())
    run.once(exp.OUT/'relaxed_envelope.json',run.stable(dict(parent=run.artifact(rp),
        code=[run.artifact(Path(__file__)),run.artifact(Path(model.__file__))],rows=rows,summary=summary,
        after_readout_diagnostic=True,risk_ignored_in_bound=True,a_labels_opened=False,confirmation_access=False)))
    print(summary,flush=True)


if __name__ == '__main__':
    main()
