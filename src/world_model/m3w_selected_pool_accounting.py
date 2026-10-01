"""Outcome-only accounting for already frozen, nested causal decisions."""
import math

import numpy as np

MOMENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')
BUDGET = .02


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def difference(left, right):
    return None if left is None or right is None else float(left - right)


def pool(y, p, q, env, take):
    known = np.isfinite(y).all(1)
    ix = take & known
    truth, original, adjusted = y[ix].sum(0), p[ix].sum(0), q[ix].sum(0)
    envelope = float(env[ix].sum())
    out = dict(rows=int(take.sum()), known=int(ix.sum()),
               unknown=int((take & ~known).sum()), envelope=envelope,
               unknown_envelope=float(env[take & ~known].sum()),
               truth=truth.tolist(), original=original.tolist(), adjusted=adjusted.tolist())
    for prefix, h, r in (('all', 1, 2), ('easy', 4, 3)):
        out[prefix] = dict(
            observed_risk=ratio(truth[h], truth[r]),
            predicted_risk=ratio(adjusted[h], adjusted[r]),
            original_predicted_risk=ratio(original[h], original[r]),
            harm_underestimate=ratio(truth[h] - adjusted[h], envelope),
            reference_overestimate=ratio(adjusted[r] - truth[r], envelope),
            signed_bias=ratio(truth[h] - adjusted[h] + BUDGET*(adjusted[r] - truth[r]), envelope))
    return out


def audit_sums(y, p, q, env, mask, result):
    """Independent scalar reduction of native moments, including missing counts."""
    ids = [i for i, selected in enumerate(mask) if selected and np.isfinite(y[i]).all()]
    for name, arr in (('truth', y), ('original', p), ('adjusted', q)):
        for j in range(5):
            expected = math.fsum(float(arr[i, j]) for i in ids)
            assert math.isclose(expected, result[name][j], rel_tol=1e-10, abs_tol=1e-9)
    assert result['rows'] == int(mask.sum())
    assert result['known'] == len(ids) and result['unknown'] == int(mask.sum()) - len(ids)
    assert math.isclose(math.fsum(float(env[i]) for i in ids), result['envelope'], rel_tol=1e-10, abs_tol=1e-9)
    return 18


def account(y, p, q, env, raw, kept, *, audit=False):
    y, p, q, env, raw, kept = map(np.asarray, (y, p, q, env, raw, kept))
    if (y.shape != p.shape or p.shape != q.shape or y.shape != (len(env), 5)
            or raw.shape != env.shape or kept.shape != env.shape
            or raw.dtype != bool or kept.dtype != bool or (kept & ~raw).any()
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or not np.isfinite(env).all() or (env < 0).any() or (p < 0).any() or (q < 0).any()):
        raise ValueError('Aligned nonnegative moments and nested Boolean decisions required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known] < 0).any():
        raise ValueError('Known nonnegative moments or wholly unknown outcome rows required')
    masks = dict(raw=raw, kept=kept, removed=raw & ~kept)
    out = {k: pool(y, p, q, env, mask) for k, mask in masks.items()}
    checks = 0
    for key in ('truth', 'original', 'adjusted'):
        np.testing.assert_allclose(out['raw'][key], np.array(out['kept'][key])+out['removed'][key], rtol=1e-10, atol=1e-9)
        checks += 5
    for key in ('rows', 'known', 'unknown', 'envelope', 'unknown_envelope'):
        assert math.isclose(out['raw'][key], out['kept'][key]+out['removed'][key], rel_tol=1e-10, abs_tol=1e-9)
        checks += 1
    for prefix, h, r in (('all', 1, 2), ('easy', 4, 3)):
        total, retain, remove = [out[k]['truth'] for k in ('raw', 'kept', 'removed')]
        observed_delta = difference(out['kept'][prefix]['observed_risk'], out['raw'][prefix]['observed_risk'])
        mass_identity = ratio(retain[h]*remove[r] - remove[h]*retain[r], retain[r]*total[r])
        assert (observed_delta is None) == (mass_identity is None)
        if observed_delta is not None:
            assert math.isclose(observed_delta, mass_identity, rel_tol=1e-9, abs_tol=1e-10)
        out[prefix+'_contrast'] = dict(
            risk_delta=observed_delta, mass_identity=mass_identity,
            harm_retained=ratio(retain[h], total[h]), reference_retained=ratio(retain[r], total[r]),
            benefit_retained=ratio(retain[0], total[0]),
            signed_bias_shift=difference(out['kept'][prefix]['signed_bias'], out['raw'][prefix]['signed_bias']),
            new_known_violation=bool(observed_delta is not None and total[h]/total[r] <= BUDGET and retain[h]/retain[r] > BUDGET))
        checks += 1
    if audit:
        checks += sum(audit_sums(y, p, q, env, mask, out[k]) for k, mask in masks.items())
    out['scalar_and_identity_checks'] = checks
    return out
