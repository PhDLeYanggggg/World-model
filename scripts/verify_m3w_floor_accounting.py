"""Alternate error-sum reductions for the frozen-floor diagnostic."""
import numpy as np
from src.evaluation.m3w_opportunity_diagnosis import REASONS


def verify_ledger(cv, floor, neural, reasons, sites, mask, result, cfg):
    selected = reasons == 5
    old = np.where(selected, neural, cv)
    new = np.where(selected, neural, floor)
    oracle = np.minimum(floor, neural)
    use = mask & np.isfinite(cv)
    by_component = {k: [] for k in result['summary']}
    assert result['indexed_rows'] == int(mask.sum())
    assert result['supported_rows'] == int(use.sum())
    for site, expected in result['by_scene'].items():
        pop = mask & (sites == site)
        rows = np.flatnonzero(pop & use)
        d, n, c, s = floor[rows], neural[rows], cv[rows], selected[rows]
        better, worse = n < d, n > d
        parts = dict(
            oracle_gain=(d-oracle[rows]).sum(),
            captured_gain=(d[s & better]-n[s & better]).sum(),
            missed_gain=(d[~s & better]-n[~s & better]).sum(),
            selected_harm=(n[s & worse]-d[s & worse]).sum(),
            fallback_regression=(c[~s & (c > d)]-d[~s & (c > d)]).sum(),
            fallback_relief=(d[~s & (c < d)]-c[~s & (c < d)]).sum(),
            original_gain=(d-old[rows]).sum(), rebased_gain=(d-new[rows]).sum(),
            rebase_advantage=(old[rows]-new[rows]).sum(),
            signed_oracle_deficit=(old[rows]-oracle[rows]).sum(),
            union_oracle_gain=(d-np.minimum(c, oracle[rows])).sum())
        for i, name in enumerate(REASONS[:-1]):
            chosen = (reasons[rows] == i) & better
            parts['missed_'+name] = (d[chosen]-n[chosen]).sum()
        assert expected['indexed_rows'] == int(pop.sum())
        assert expected['supported_rows'] == len(rows)
        np.testing.assert_allclose(d.sum(), expected['floor_error_sum'], rtol=1e-10, atol=1e-8)
        for key, value in parts.items():
            np.testing.assert_allclose(value, expected['sums'][key], rtol=1e-10, atol=1e-8)
            p = float(100*value/d.sum()) if d.sum() > 0 else None
            by_component[key].append(p)
            if p is None:
                assert expected['contributions_percent'][key] is None
            else:
                np.testing.assert_allclose(p, expected['contributions_percent'][key], rtol=1e-10, atol=1e-8)
    for key, values in by_component.items():
        expected = result['summary'][key]
        if any(v is None for v in values):
            assert expected == dict(equal_locality=None, ci95=None)
            continue
        x = np.asarray(values)
        draws = np.random.default_rng(cfg['bootstrap_seed']).integers(
            0, len(x), size=(cfg['bootstrap_resamples'], len(x)))
        np.testing.assert_allclose(x.mean(), expected['equal_locality'], rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(np.quantile(x[draws].mean(1), [.025, .975]), expected['ci95'],
                                   rtol=1e-10, atol=1e-8)
