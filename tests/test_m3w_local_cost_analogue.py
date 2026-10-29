import numpy as np
import pytest

from src.evaluation import m3w_local_cost_analogue as api
from tests.test_m3w_selection_conditioned_analogue import fixture


def test_quadratic_solution_and_boundary_optima():
    f = np.array([[.1, .2], [.8, .6]])
    w = np.array([1., 9.])
    out = api.solve(f, w)
    np.testing.assert_allclose(out, [.73, .56])
    grid = np.linspace(0, 1, 1001)
    loss = (w[:, None, None]*(grid[None, :, None]-f[:, None])**2).sum(0)
    assert np.all((w[:, None]*(out-f)**2).sum(0) <= loss.min(0)+1e-12)
    np.testing.assert_array_equal(api.solve(np.array([[-2., 3.]]), np.ones(1)), [0., 1.])


def inputs():
    b, q = fixture()
    b['scale'] = np.ones(len(q))*2
    p = b['labels'].copy()
    p[:, 2:4] += .3
    return b, q, p


def test_inherited_neighbors_and_predictions_exact():
    b, q, p = inputs()
    old = api.previous.retrieve(b, q, selected=True, k=4)
    out = api.retrieve(b, q, p, k=4)
    np.testing.assert_array_equal(out['predictions']['selected_nearest32'], old['predictions']['nearest32'])
    for new, prior in zip(out['references'], old['references']):
        for k in ('donor_sha256', 'query_sha256', 'mean_sha256', 'std_sha256', 'neighbor_sha256'):
            assert new[k] == prior[k]
    assert out['supported'].all()


def test_query_future_and_unknownness_cannot_change_own_prediction():
    b, q, p = inputs(); a = api.retrieve(b, q, p, k=4)
    b['labels'][:20] = np.nan
    c = api.retrieve(b, q, p, k=4)
    for arm in api.MODES:
        np.testing.assert_array_equal(a['predictions'][arm][:10], c['predictions'][arm][:10])
    assert c['supported'][:10].all()


def test_unselected_labels_cannot_change_any_prediction():
    b, q, p = inputs(); a = api.retrieve(b, q, p, k=4)
    b['labels'][~q] = np.nan
    c = api.retrieve(b, q, p, k=4)
    for arm in api.MODES:
        np.testing.assert_array_equal(a['predictions'][arm], c['predictions'][arm])


def test_reference_cross_term_uses_donor_predictions_only():
    b, q, p = inputs(); p[:, 2:4] = b['labels'][:, 2:4]
    a = api.retrieve(b, q, p, k=4)
    np.testing.assert_allclose(a['predictions']['local_signed32'], a['predictions']['weighted_harm32'])
    p[:20, 2:4] += 100
    c = api.retrieve(b, q, p, k=4)
    np.testing.assert_array_equal(a['predictions']['local_signed32'][:10], c['predictions']['local_signed32'][:10])


def test_units_rescale_without_changing_fitted_fractions():
    b, q, p = inputs(); a = api.retrieve(b, q, p, k=4)
    b['envelope'] *= 7; b['labels'] *= 7; b['scale'] *= 7; p *= 7
    c = api.retrieve(b, q, p, k=4)
    for arm in api.MODES:
        np.testing.assert_allclose(a['predictions'][arm]*7, c['predictions'][arm], atol=1e-12)


def test_source_counts_do_not_depend_on_query_future():
    b, q, p = inputs(); a = api.retrieve(b, q, p, k=4)
    b['labels'][:20] = np.nan
    c = api.retrieve(b, q, p, k=4)
    assert a['references'][0]['known_donors_by_source'] == c['references'][0]['known_donors_by_source']


def test_insufficient_support_is_not_zero_risk():
    b, q, p = inputs(); b['labels'][20:] = np.nan
    out = api.retrieve(b, q, p, k=32)
    assert len(out['indices']) == 30 and not out['supported'].any()
    for arm in api.MODES: assert np.isnan(out['predictions'][arm]).all()


def test_invalid_causal_scale_is_rejected():
    b, q, p = inputs(); b['scale'][0] = 0
    with pytest.raises(ValueError, match='scale'): api.retrieve(b, q, p, k=4)
