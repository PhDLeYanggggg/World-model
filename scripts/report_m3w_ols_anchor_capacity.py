"""Separate the effect of a small learned gate from the capacity of its directions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_ols_anchor_denoising as study
from src.evaluation import m3w_ols_replacement_amplitude as amplitude


def main():
    np,run = study.np,study.run
    _,jobs = study.register()
    frozen = study.prev.receipt(study.PUBLIC,'train')
    report = study.prev.receipt(study.PUBLIC,'replay')
    d = study.data(jobs,lambda **_:None)
    queries = study.old.fitting_layout(jobs[0]['identity'],d)
    _,paths,_,_,_,_ = study.model.design(d['geometry'][:,:16].reshape(-1,8,2))
    _,src = np.unique(d['sites'],return_inverse=True); w = np.empty(len(src))
    for q in queries:
        w[q] = 1./len(q)
    y,valid = d['target'],d['valid']; known = valid.any(1)
    cv = np.linalg.norm(paths['cv']-np.where(valid[...,None],y,0.),axis=2)
    e = np.where(valid,cv,0.).sum(1)/valid.sum(1).clip(1)
    weight = 100*w/np.bincount(src,weights=w*e)[src]
    draws = np.random.default_rng(73107).integers(9,size=(3000,9))
    metrics = {}
    for a in ('fixed_anchor','fixed_slope'):
        terms = amplitude.components(paths['ols4'],paths[a],y,valid)

        def interval(v):
            return study.old.metric.intervals(np.bincount(src,weights=weight*v)[:,None],draws)

        def common(z):
            return tuple(float(weight@v/9) for v in amplitude.value_slope(terms,z))

        full,_ = common(1.)
        np.testing.assert_allclose(full,report['contrasts'][f'{a}_vs_ols4']['lower']['point'],atol=1e-10)
        alpha,g,s = amplitude.maximize(common)
        row,upper,_ = amplitude.maximize(lambda z:amplitude.value_slope(terms,z))
        metrics[a] = dict(derivative_at_zero_pp=float(s),best_common_amplitude=float(alpha),
            best_common_gain_pp=float(g),rowwise_hindsight_pp=interval(upper),
            positive_rowwise_amplitude_count=int((row > 0.).sum()),whole_fitting_roster=len(y))
    with np.load(run.checked(frozen['coefficients']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids'])
        quantiles = {a:dict(quantiles=np.quantile(f,[.5,.95,.99]).tolist(),at_least_one_percent=int((f >= .01).sum()),
                           row_seed_views=f.size) for a,f in zip(study.model.FITS,z['factors'].transpose(1,0,2),strict=True)}
    run.once(study.PUBLIC/'capacity.json',run.stable(dict(analysis=run.artifact(study.PUBLIC/'analysis.json'),
        directions=metrics,learned_coefficients=quantiles,source_groups=9,posthoc_fitting_only=True,known_rows=int(known.sum()),
        hindsight_is_not_policy=True,external_access=False,development_targets=False,changed_policy=False)))


if __name__ == '__main__':
    main()
