"""Separate early ranking, single nomination, and OLS4 availability bottlenecks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_admission as study
from scripts import audit_m3w_cv_joint_feasibility as solver

np, torch, read = study.np, study.torch, study.read


def assess(cv, ols, bank, y, valid, q, cut, inherited=None):
    d = solver.model.coefficients(cv, ols, bank, y, valid, q, cut[0])
    take, r = solver.model.solve(d, study.CAP, seconds=1.5)
    act = np.column_stack((d['row'][take], d['arm'][take])) if take is not None else None
    status = r['status']
    if act is not None and solver.verify(act, cv, ols, bank, y, valid, q, *cut, study.CAP)['verified']:
        return 'verified', act
    if inherited is not None:
        if status != 'unresolved' or not solver.verify(inherited, cv, ols, bank, y, valid, q, *cut, study.CAP)['verified']:
            raise ValueError('Expansion contradicts the verified subset witness')
        return 'verified', inherited
    return status, None


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    f, score, previous = [study.OUT/n for n in ('freeze.json','readout.json','withholding_capacity.json')]
    r, freeze, old = read(score), read(f), read(previous)
    if r['freeze'] != study.sha(f) or old['parent']['sha256'] != study.sha(score) or r['a_labels_opened']:
        raise ValueError('Require verified fitting-only parent')
    d = study.fit_population(lambda **_: None)
    oof = read(study.prior.donor.OUT/'freeze.json')
    rows = []
    for cell in freeze['cells']:
        ident = cell['identity']
        ix, held = study.subset(d, ident['held'])
        p = study.prior.read_donor(oof, ident['seed'], d)
        cv, ols, paths = study.prior.bank(held['history'], {k:v[ix] for k,v in p.items()})
        bank = study.expanded(cv,paths)
        past, _ = study.prior.study.motion.features(held['history'][:1], 'prefix')
        net = study.prior.model.head(past.shape[1]+25,8).eval()
        state = torch.load(study.checked(cell['fits'][0]['artifact']),weights_only=True,map_location='cpu')
        if state['identity'] != dict(ident,head='risk') or state['step'] != study.STEPS:
            raise ValueError('Changed fixed nomination model')
        net.load_state_dict(state['net'])
        nom = study.nominate(net,cv,bank,held['history'])
        with np.load(study.checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],held['ids'])
            forced = z['forced']
            np.testing.assert_array_equal(forced,study.matched(nom,study.prior.query_groups(held['query'])))
        reduced = study.prior.apply(cv,bank,forced)
        unpruned = study.prior.apply(cv,bank,nom[0])
        for rec in np.unique(held['recording']):
            at = held['recording'] == rec
            args = [held[k][at] for k in ('target','valid','query')] + [cell['cut']]
            a, p0 = assess(cv[at],ols[at],reduced[at,None],*args)
            b, p1 = assess(cv[at],ols[at],unpruned[at,None],*args,inherited=p0)
            c, _ = assess(cv[at],ols[at],np.stack((unpruned[at],ols[at]),axis=1),*args,inherited=p1)
            rows.append(dict(forced_pool=a,unpruned=b,ols4_addback=c))
        print(dict(fold=ident['fold'],seed=ident['seed'],completed=len(rows)),flush=True)
    if len(rows) != 72: raise ValueError('Fitting roster changed')
    fields = sorted({v for x in rows for v in x.values()})
    result = {k:{m:sum(x[k] == m for x in rows) for m in fields} for k in rows[0]}
    confirmed = {k:sum(v[k]=='verified' and v[b] not in ('verified','unresolved') for v in rows)
                 for b,k in (('forced_pool','unpruned'),('unpruned','ols4_addback'))}
    study.once(study.OUT/'unpruned_capacity.json',study.stable(dict(parent=study.artifact(score),
        withholding=study.artifact(previous),code=study.artifact(Path(__file__)),status=result,
        confirmed_new_after_each_expansion=confirmed,views=len(rows),post_hoc=True,cap=study.CAP,
        original_2pct=True,independent_confirmation=False,a_labels_opened=False)))
    print(result,confirmed)


if __name__=='__main__':
    main()
