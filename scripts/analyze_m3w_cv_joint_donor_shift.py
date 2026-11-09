"""Compare causal donor and A proposal magnitudes without opening A targets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_joint_selector as study

np = study.np


def stats(history, paths, sites):
    cv, ols, p = study.bank(history, paths)
    scale = study.model.scale(history)
    out = {}
    for name, values in (('speed_pixel_per_step', np.linalg.norm(np.diff(history, axis=1), axis=-1).mean(1)),
                         *[(key, np.linalg.norm(v-cv, axis=-1).max(1)/scale)
                           for key, v in (('ols4_relative_peak', ols), ('eqmotion_relative_peak', p[:, -2]),
                                          ('pct_relative_peak', p[:, -1]))]):
        cells = [np.quantile(values[sites == src], [.1, .5, .9, .99]) for src in sorted(set(sites))]
        out[name] = np.mean(cells, axis=0).tolist()
    return out


def main():
    d = study.read(study.donor.OUT / 'freeze.json')
    refs = study.prior.parent.register()
    if d['a_labels_accessed'] or d['held_source_draws']:
        raise ValueError('Require source-excluded donor forecasts')
    _, jobs = study.study.parent.parent.register()
    fit = study.donor.prior.prior.fitting_only(jobs[0]['identity'], lambda **_: None)
    h = fit['geometry'][:, :16].reshape(-1, 8, 2)
    a, _ = study.a_population()
    rows = []
    for seed in study.SEEDS:
        p = study.read_donor(d, seed, fit)
        q = study.prior.parent.read_banks(refs, seed, a['global_ids'], a['sites'])
        rows.append(dict(seed=seed, donor=stats(h, p, fit['sites']), exposed_a=stats(a['history'], q, a['sites'])))
    def mean(role):
        return {k: np.mean([v[role][k] for v in rows], axis=0).tolist() for k in rows[0][role]}
    out = dict(donor=mean('donor'), exposed_a=mean('exposed_a'))
    study.once(study.OUT / 'donor_shift.json', study.stable(dict(donor_freeze=study.artifact(study.donor.OUT/'freeze.json'),
        a_forecast_receipt=study.artifact(study.prior.parent.REG), quantiles=[.1, .5, .9, .99],
        aggregation='average source quantiles then seeds', no_a_labels=True, summary=out, independent_confirmation=False)))
    print(out)


if __name__ == '__main__':
    main()
