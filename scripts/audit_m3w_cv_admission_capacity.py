"""Hindsight withholding capacity of the same frozen forced nominees."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_admission as study
from scripts import audit_m3w_cv_joint_feasibility as solver

np, read = study.np, study.read


def test_pool(cv, ols, paths, labels, valid, q, cut):
    # Keep the one nominee supplied by the frozen score and optionally remove it.
    bank = paths[:, None]
    design = solver.model.coefficients(cv, ols, bank, labels, valid, q, cut[0])
    out = {}
    for mode in ('cv_only', 'dual'):
        current = design if mode == 'dual' else dict(design, gain=design['gain'][:, [0,0]], offset=np.zeros(2))
        chosen, status = solver.model.solve(current, study.CAP, seconds=1.5)
        r = status['status']
        if chosen is not None:
            act = np.column_stack((design['row'][chosen], design['arm'][chosen]))
            w = solver.verify(act, cv, ols, bank, labels, valid, q, *cut, study.CAP)
            if (w['original_risk']['supported'] and w['edits'] == w['original_risk']['selected']
                    and w['gain_pp'] is not None and (min(w['gain_pp']) if mode == 'dual' else w['gain_pp'][0]) > 1e-6):
                r = 'verified'
        out[mode] = r
    return out


def main():
    score, f = study.OUT/'readout.json', study.OUT/'freeze.json'
    r, freeze = read(score), read(f)
    if r['freeze'] != study.sha(f) or r['a_labels_opened'] or r['independent_confirmation']:
        raise ValueError('Require frozen fitting-only actions')
    d = study.fit_population(lambda **_: None)
    oof = read(study.prior.donor.OUT/'freeze.json')
    rows = []
    for c in freeze['cells']:
        ident = c['identity']
        ix, held = study.subset(d, ident['held'])
        pred = study.prior.read_donor(oof, ident['seed'], d)
        cv, ols, paths = study.prior.bank(held['history'], {k:v[ix] for k,v in pred.items()})
        with np.load(study.checked(c['actions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], held['ids'])
            nominees = study.prior.apply(cv,study.expanded(cv,paths),z['forced'])
        for name in np.unique(held['recording']):
            at = held['recording'] == name
            rows.append(test_pool(cv[at],ols[at],nominees[at],held['target'][at],held['valid'][at],held['query'][at],c['cut']))
    if len(rows) != 72: raise ValueError('Expected all held recording-seed views')
    fields = sorted({v for x in rows for v in x.values()})
    result = {k: {m:sum(x[k] == m for x in rows) for m in fields} for k in ('cv_only','dual')}
    study.once(study.OUT/'withholding_capacity.json',study.stable(dict(parent=study.artifact(score),
        code=study.artifact(Path(__file__)),views=len(rows),status=result,post_hoc=True,same_fixed_nominees=True,
        a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
    print(result)


if __name__=='__main__':
    main()
