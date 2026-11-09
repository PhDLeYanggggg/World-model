"""Separate observed risk failures from failures added by missing outcomes."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_signed_coverage as study

np,base,prior = study.np,study.base,study.prior
read,sha,artifact,once,stable,checked = study.read,study.sha,study.artifact,study.once,study.stable,study.checked


def flags(r):
    known_all,known_easy,easy_net = (r[k] for k in ('known_all_risk','known_easy_risk','known_easy_net_degradation'))
    known_pass = all(v is not None and v<=.02 for v in (known_all,known_easy,easy_net))
    return dict(views=1,original_pass=int(r['supported']),known_only_pass=int(known_pass),
        known_all_over=int(known_all is not None and known_all>.02),
        known_easy_over=int(known_easy is not None and known_easy>.02),
        known_easy_net_over=int(easy_net is not None and easy_net>.02),
        undefined_selected_easy=int(known_easy is None),
        missing_debit_flips_known_pass=int(known_pass and not r['supported']),
        edited_wholly_unknown_rows=int(r['unknown_selected']))


def main():
    out = study.OUT
    score,fp = out/'readout.json',out/'freeze.json'
    r,f = read(score),read(fp)
    if r['freeze']!=sha(fp) or r['a_labels_opened'] or r['independent_confirmation']:
        raise ValueError('Require completed fitting-source readout')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    counts = {k:Counter() for k in study.FACTORIAL}
    scorer = prior.prior.parent.readout
    for cell in f['cells']:
        ident = cell['identity']
        hi,h = base.subset(d,ident['held'])
        cv,_,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
        bank = base.expanded(cv,raw)
        with np.load(checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            acts = {k:z[k] for k in study.FACTORIAL}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            c = cv[ix]
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            paths = {base.CV:c,**{k:prior.apply(c,bank[ix],a[ix]) for k,a in acts.items()}}
            z = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=cell['cut'][0],hard_cut=cell['cut'][1])
            for key,agg in counts.items():
                agg.update(flags(z['metrics'][key]['risk']))
    for k,v in counts.items():
        if (v['views'],v['original_pass'])!=(72,r['summary']['risk'][k]['supported_views']):
            raise ValueError('Original risk counts changed')
    once(out/'risk_failures.json',stable(dict(parent=artifact(score),code=artifact(Path(__file__)),
        arms={k:dict(v) for k,v in counts.items()},known_only_diagnostic_not_a_replacement=True,
        original_2pct=True,a_labels_opened=False,independent_confirmation=False)))


if __name__=='__main__':
    main()
