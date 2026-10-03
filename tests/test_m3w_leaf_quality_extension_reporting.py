import numpy as np
from src.world_model import m3w_leaf_quality_extension as api
from scripts.verify_m3w_leaf_quality_extension import aggregate, compare
from test_m3w_cost_support_diagnostic import fitted_fixture


def rows(supported):
    state, x, e, y, q, sites, recordings, frames = fitted_fixture()[0]
    pred = np.tile([1., 0., 10., 10., 0.], (len(y), 1))
    predictions = {k: pred for k in api.CONTROLS+('extended',)}
    result, _ = api.evaluate(state, predictions, y, e, np.ones(len(y), bool),
        np.full(len(y), supported), sites, recordings, frames, np.arange(len(y)))
    return [dict(source=site, result=result) for site in ('a', 'a', 'b')]


def test_independent_aggregation_and_no_identity_promotion():
    rr = rows(True)
    cfg = dict(bootstrap_resamples=101, bootstrap_seed=9)
    summary = api.summarize(rr, cfg)
    assert compare(summary, aggregate(rr, cfg)) > 100
    assert not summary['advance_to_transfer']
    assert len(summary['failure_reasons']) >= 12
    assert summary['extended_minus_cost_signed_MSE']['localities'] == 2


def test_empty_risk_is_undefined_not_a_safety_success():
    rr = rows(False)
    cfg = dict(bootstrap_resamples=101, bootstrap_seed=9)
    summary = api.summarize(rr, cfg)
    assert compare(summary, aggregate(rr, cfg)) > 100
    assert summary['extended']['worst_easy_upper'] is None
    assert not summary['all_heads_easy_supported_within_budget']
    assert not summary['advance_to_transfer']
