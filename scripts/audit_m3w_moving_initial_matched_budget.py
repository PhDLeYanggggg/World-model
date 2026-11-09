"""Project frozen initial forecasts onto the old recipients and exact radii."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_moving_dual_record_forecast as m

old,np = m.old,m.np
FAMILIES = ('ols4','completion','diagnostic','ols_only','dual_risk')


def project(cv, anchor, candidate):
    peak = m.exchange.model.peak
    ref = peak(anchor-cv)
    delta = candidate-cv
    radii = peak(delta)
    use = (ref>0.)&(radii>=ref)
    p = anchor.copy()
    p[use] = cv[use] + (ref[use]/radii[use])[:,None,None]*delta[use]
    np.testing.assert_allclose(peak(p-cv),ref,rtol=1e-8,atol=1e-8)
    np.testing.assert_array_equal((p!=cv).any((1,2)),ref>0.)
    return p,int(use.sum()),int((p!=anchor).any((1,2)).sum())


def main():
    key,_,_,_,ref = m.register()
    fp,rp = (m.OUT/k for k in ('freeze.json','readout.json'))
    f,r = map(old.read,(fp,rp))
    if (f['registration']!=key or r['freeze']!=old.sha(fp) or not f['before_held_scoring']
            or not r['original_2pct'] or any(r[k] for k in ('a_labels_opened','confirmation_access','independent_confirmation'))):
        raise ValueError('Need frozen initial-forecast development predictions')
    d = m.base.fit_population(lambda **kw:None)
    motion = m.prior.study.motion.bank(d['history'])
    names = ('half_anchor',*FAMILIES)
    pairs = m.previous.parent.vs_references(names)
    pairs.update({f'{k}_vs_anchor':('half_anchor',k) for k in FAMILIES})
    rows, proj = [],{k:[0,0] for k in FAMILIES}
    for c in f['cells']:
        i = c['identity']
        ix,h = m.base.subset(d,i['held'])
        cv,ols = motion[ix,0],motion[ix,1]
        with np.load(old.checked(c['paths']),allow_pickle=False) as z:
            np.testing.assert_array_equal(h['ids'],z['ids'])
            anchor = z['half_anchor']
            choices = {k:z[k] for k in FAMILIES if k!='ols4'}
        paths = {m.base.CV:cv,m.base.OLS:ols,'half_anchor':anchor}
        for k,p in dict(choices,ols4=ols).items():
            paths[k],eligible,changed = project(cv,anchor,p)
            proj[k][0] += eligible
            proj[k][1] += changed
        rows.extend(m.previous.parent.rows_for(h,paths,cv,c['cut'],pairs,i))
    sites = {x['recording']:x['source'] for x in rows}
    summary = m.prior.prefix.grouped.grouped_readout(rows,sites,m.base.SEEDS)
    refstats = ref['summary']['risk']['half_anchor']
    if summary['risk']['half_anchor']!=refstats or any(summary['risk'][k]['switches']!=refstats['switches'] for k in names):
        raise ValueError('Match failed to retain original edit cohort')
    dual = {k:sum(z['metrics'][k]['risk']['supported'] and all(z['comparisons'][f'{k}_vs_{b}']['known_delta']
             > z['comparisons'][f'{k}_vs_{b}']['unknown_radius'] for b in ('cv','ols4')) for z in rows) for k in names}
    old.once(m.OUT/'matched_directions.json',old.stable(dict(parent=old.artifact(rp),code=old.artifact(Path(__file__)),
        summary=summary,dual_pass=dual,eligible_and_changed=proj,after_readout_diagnostic=True,original_2pct=True,
        no_amplification=True,a_labels_opened=False,confirmation_access=False,independent_confirmation=False)))
    print(dict(dual=dual,changed=proj),flush=True)


if __name__ == '__main__':
    main()
