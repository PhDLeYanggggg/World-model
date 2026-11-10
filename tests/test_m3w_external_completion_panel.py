import numpy as np
import pytest

from scripts.audit_m3w_external_completion_panel import DRAW_SEED, DRAWS, summarize_songdo


def record(x, counts=3):
    draws = np.random.default_rng(DRAW_SEED).integers(len(x), size=(DRAWS, len(x)))
    return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                component_points=x.tolist(), status='estimable', estimable_components=len(x),
                defined_seed_views=counts, required_seed_views=3)


def test_optimistic_endpoint_is_componentwise_and_bootstrap_is_paired():
    observed, lower = np.array([-.2, .1, .3]), np.array([-.4, .1, .1])
    r = summarize_songdo(dict(known_pp=record(observed), lower_pp=record(lower)), 3, 3)
    upper = record(2*observed-lower)
    assert r['upper'] == upper['point'] and r['upper_ci95'] == upper['ci95']


def test_does_not_subtract_receipts_with_different_support():
    y = record(np.array([0., .1, .3]))
    with pytest.raises(ValueError, match='entire frozen roster'):
        summarize_songdo(dict(known_pp=y, lower_pp=dict(y, defined_seed_views=2)), 3, 3)
