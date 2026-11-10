import numpy as np
import pytest

from src.evaluation.m3w_relative_forecast_readout import evaluate
from src.evaluation.m3w_sparse_pair_bounds import compare


def test_relative_readout_agrees_with_single_query_reference_and_keeps_missing_queries():
    rng = np.random.default_rng(17)
    counts = np.array([1, 3, 2, 4])
    a, b, y = rng.normal(size=(3, counts.sum(), 12, 2))
    valid = rng.random(y.shape[:2]) > .3
    valid[-4:] = False
    y[~valid] = np.nan
    result = evaluate(dict(a=a, b=b), y, valid, counts, {'a_vs_b': ('a', 'b')}, block=2)
    offs = np.r_[0, np.cumsum(counts)]
    refs = [compare(a[i:j], b[i:j], y[i:j], valid[i:j]) for i, j in zip(offs[1:-1], offs[2:])]
    for k in ('lower', 'upper'):
        assert result['contrasts']['a_vs_b'][k] == pytest.approx(np.mean([r[k] for r in refs]))
    assert result['contrasts']['a_vs_b']['observed'] == pytest.approx(np.mean([r['available'] for r in refs[:2]]))
    assert result['support']['pair_queries'] == 3
    assert result['support']['observed_queries'] == 2
    assert result['support']['observed_pair_steps'] == sum(r['observed_pair_steps'] for r in refs)


def test_common_translation_has_no_relative_error():
    y = np.zeros((3, 12, 2))
    a = y + np.arange(12)[None, :, None]
    out = evaluate(dict(a=a, b=y), y, np.ones((3, 12), bool), np.array([3]), {'a_vs_b': ('a', 'b')})
    assert out['absolute']['a']['relative_ade'] == 0
    assert out['contrasts']['a_vs_b'] == dict(observed=0., observed_contribution=0., lower=0., upper=0.)
