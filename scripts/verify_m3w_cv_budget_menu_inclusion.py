"""Carry independently checked witnesses across mathematically nested conditions."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_budget_menu_factorial as study

exp,np,base,prior = study.exp,study.np,study.base,study.prior
OUT,PRIVATE = study.OUT,study.PRIVATE
EDGES = (('narrow_exact','full_exact'),('narrow_exact','narrow_relaxed'),
         ('full_exact','full_relaxed'),('narrow_relaxed','full_relaxed'))


def propagate(results,verify):
    out = {k:dict(z) for k,z in results.items()}
    for low,high in EDGES:
        if out[low]['status']!='verified' or out[high]['status']=='verified':
            continue
        if out[high]['status']!='unresolved':
            raise ValueError('Negative certificate contradicts a nested witness')
        source = out[low]
        if not verify(source['actions'],high)['verified']:
            raise ValueError('Nested witness did not pass independent replay')
        out[high] = dict(source,inherited_from=low,upper_pp=out[high]['upper_pp'])
    return out


def main():
    path = OUT/'readout.json'
    r = exp.read(path)
    if (r['registration']!=exp.sha(OUT/'registration_v2.json') or r['a_labels_opened']
            or r['independent_confirmation'] or not r['original_2pct']):
        raise ValueError('Need the unchanged development capacity receipt')
    bind = exp.sha(path)
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    f = exp.read(exp.OUT/'freeze.json')
    rows,added = [],Counter()
    for i,c in enumerate(f['cells']):
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
        bank = base.expanded(cv,raw)
        with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            anchor = z['frozen']
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            a = anchor[ix]
            data = (cv[ix],ols[ix],bank[ix],*(h[k][ix] for k in ('target','valid','query')),*c['cut'])
            original = exp.read(PRIVATE/f'cell_v2_{i}_{rec}.json')
            if original['registration']!=r['registration']:
                raise ValueError('Changed original capacity checkpoint')
            updated = propagate(original['results'],lambda actions,k:study.check(actions,data,a,k))
            added.update(k for k,v in updated.items() if 'inherited_from' in v)
            exp.once(PRIVATE/f'nested_{i}_{rec}.json',exp.stable(dict(parent=bind,results=updated)))
            site, = np.unique(h['sites'][ix])
            rows.append(dict(updated,source=str(site),rows=int(ix.sum())))
    result = study.summarize(rows)
    exp.once(OUT/'nested_verification.json',exp.stable(dict(parent=exp.artifact(path),
        code=exp.artifact(Path(__file__)),**result,verified_by_inclusion=dict(added),
        a_labels_opened=False,independent_confirmation=False,original_2pct=True,labels_hindsight_only=True,
        submission_ready=False)))
    print(result,flush=True)


if __name__=='__main__':
    main()
