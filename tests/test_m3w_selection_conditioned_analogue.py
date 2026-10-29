import json
import numpy as np
import pytest
from src.evaluation import m3w_selection_conditioned_analogue as api
from src.evaluation import m3w_selected_analogue as prior


def fixture():
    rng = np.random.default_rng(4); n = 60
    recs = np.array([json.dumps(['a', str(i//20)], separators=(',', ':')) for i in range(n)])
    y = np.column_stack((np.zeros(n), np.linspace(0, .8, n), np.ones(n), np.ones(n), np.linspace(0, .4, n)))
    b = dict(x=rng.normal(size=(n, 4)), envelope=np.ones(n), labels=y,
        recordings=recs, sources=np.repeat('a', n), eligible=np.ones(n, bool))
    q = np.arange(n) % 2 == 0
    return b, q


def test_full_bank_matches_frozen_algorithm_exactly():
    b, q = fixture()
    a = api.retrieve(b, q, selected=False, k=4)
    old = prior.retrieve(b['x'], b['envelope'], b['labels'], b['recordings'], b['eligible'], q, k=4)
    for mode, oldmode in zip(api.MODES, prior.MODES):
        np.testing.assert_array_equal(a['predictions'][mode], old['predictions'][oldmode])
    for new, ref in zip(a['references'], old['references']):
        for key in ref: assert new[key] == ref[key]


def test_selection_changes_labels_not_normalization():
    b, q = fixture()
    full = api.retrieve(b, q, selected=False, k=4)
    sel = api.retrieve(b, q, selected=True, k=4)
    for a, c in zip(full['references'], sel['references']):
        assert a['mean_sha256'] == c['mean_sha256'] and a['std_sha256'] == c['std_sha256']
        assert a['causal_donors'] == c['causal_donors']
        assert c['unselected_donors'] == 0 and c['same_recording_donors'] == 0
        assert c['known_donors'] == 20


def test_query_future_cannot_change_own_prediction():
    b, q = fixture(); original = api.retrieve(b, q, selected=True, k=4)
    b['labels'][:20] = np.nan
    new = api.retrieve(b, q, selected=True, k=4)
    for m in api.MODES:
        np.testing.assert_array_equal(original['predictions'][m][:10], new['predictions'][m][:10])
    assert new['supported'][:10].all()


def test_unselected_labels_do_not_supervise_selected_bank():
    b, q = fixture(); a = api.retrieve(b, q, selected=True, k=4)
    b['labels'][~q, 1] = 1; b['labels'][~q, 4] = 1
    c = api.retrieve(b, q, selected=True, k=4)
    for m in api.MODES: np.testing.assert_array_equal(a['predictions'][m], c['predictions'][m])


def test_unsupported_queries_kept_and_unknown_not_zero_risk():
    b, q = fixture(); b['labels'][20:] = np.nan
    a = api.retrieve(b, q, selected=True, k=32)
    assert len(a['indices']) == 30 and not a['supported'].any()
    assert np.isnan(a['predictions']['nearest32']).all()


def test_event_magnitude_accounting_is_not_inference():
    b, q = fixture(); a = api.retrieve(b, q, selected=True, k=4)
    d = api.decomposition(b['labels'][q], a['predictions']['nearest32'],
                          a['probability'], a['severity'], b['envelope'][q], 2., a['supported'])
    for t in d.values():
        assert t['known'] == 30
        assert t['events'] == t['positive_queries_with_positive_donors'] + t['positive_queries_without_positive_donors']
        assert t['harm_SSE'] == pytest.approx(t['event_harm_SSE']+t['non_event_harm_SSE'])
    np.testing.assert_allclose(a['probability']*a['severity'], a['predictions']['nearest32'])


def test_invalid_policy_mask_rejected():
    b, q = fixture(); b['eligible'][0] = False
    with pytest.raises(ValueError, match='policy mask'): api.retrieve(b, q, selected=True, k=4)
