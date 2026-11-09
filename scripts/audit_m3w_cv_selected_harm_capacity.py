"""Separate the fixed-path and count bottlenecks using exact lower-score maxima."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_selected_harm as study

old, base, prior, np = study.old, study.base, study.prior, study.np
ARMS = ('one_path_exact', 'analytic_exact', 'full_exact', 'full_at_most', 'full_free')


def maxima(cv, ols, bank, one, choice, anchor, y, v, q, analytic_count):
    w = old.weight(q)
    groups = prior.query_groups(q)
    mass, offset, one_gain, _ = study.candidate_terms(cv, ols, one[:, None], y, v)
    ref = w @ mass
    if ref <= 0.:
        return None
    optimum = np.full((len(cv), 2), -np.inf)
    analytic = optimum.copy()
    for k in range(bank.shape[1]):
        for alpha in study.parent.model.AMPLITUDES:
            path = cv + alpha * (bank[:, k] - cv)
            _, _, gain, _ = study.candidate_terms(cv, ols, path[:, None], y, v)
            z = np.where((path != cv).any((1, 2))[:, None], gain[:, 0], -np.inf)
            np.maximum(optimum, z, out=optimum)
            if k < analytic_count:
                np.maximum(analytic, z, out=analytic)
    full = np.where(np.isfinite(optimum[:, 0]), 0, -1)
    # The analytic arm can have fewer legal recipients than a count inherited
    # from the expanded bank. Negative infinity records that infeasibility.
    av = np.zeros(2)
    for group in groups:
        k = (anchor[group] >= 0).sum()
        if k:
            av += np.sort(analytic[group] * w[group, None], axis=0)[-k:].sum(0)
    vals = np.stack((study.model.upper_at_count(one_gain[:, 0], choice, anchor, groups, w),
        av, study.model.upper_at_count(optimum, full, anchor, groups, w),
        study.model.upper_at_count(optimum.clip(0.), full, anchor, groups, w),
        w @ optimum.clip(0.)))
    upper = 100. * (vals + [0., w @ offset]) / ref
    if ((upper[2] + 1e-8 < upper[:2]).any() or (np.diff(upper[2:], axis=0) < -1e-8).any()):
        raise ValueError('Nested upper envelope declined')
    return upper


def main():
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, readout = map(old.read, (f, r))
    if (readout['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or not readout['original_2pct']
            or readout['a_labels_opened'] or readout['independent_confirmation']):
        raise ValueError('Require frozen development-only results')
    _, donor, _ = base.register()
    d = base.fit_population(lambda **_: None)
    result = []
    for cell in freeze['cells']:
        ident = cell['identity']
        hi, h = base.subset(d, ident['held'])
        cv, ols, bank = prior.bank(h['history'],
            {k: v[hi] for k, v in prior.read_donor(donor, ident['seed'], d).items()})
        n = study.parent.analytic(h['history'][:1])[2].shape[1]
        with np.load(old.checked(cell['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            choice, anchor = z['proposal'], z['anchor']
        one = study.parent.model.apply(cv, bank, choice)
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            x = maxima(cv[at], ols[at], bank[at], one[at], choice[at], anchor[at], y, v, q, n)
            if x is not None:
                result.append(x)
        print(dict(fold=ident['fold'], seed=ident['seed'], views=len(result)), flush=True)
    z = np.stack(result)
    if z.shape != (72, len(ARMS), 2):
        raise ValueError('Incomplete held development roster')
    out = {}
    for i, arm in enumerate(ARMS):
        out[arm] = {k: dict(nonpositive=int((z[:, i, j] <= 0.).sum()),
            count_infeasible=int(np.isneginf(z[:, i, j]).sum()),
            median_upper_pp=float(np.median(z[np.isfinite(z[:, i, j]), i, j])))
            for j, k in enumerate(('cv', 'ols4'))}
    if any(out['one_path_exact'][k]['nonpositive'] != readout['matched_count_one_path_excluded'][k]
           for k in ('cv', 'ols4')):
        raise ValueError('One-path upper bound did not reproduce')
    old.once(study.OUT / 'capacity.json', old.stable(dict(parent=old.artifact(r), code=old.artifact(Path(__file__)),
        views=72, summary=out, label_aware_upper_only=True, ignores_risk=True, independent_confirmation=False,
        a_labels_opened=False, original_2pct=True)))
    print(out, flush=True)


if __name__ == '__main__':
    main()
