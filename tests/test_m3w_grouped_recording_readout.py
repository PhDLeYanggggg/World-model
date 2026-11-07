import pytest

from src.evaluation.m3w_grouped_recording_readout import grouped_readout


def row(rec, group, seed, ref, change):
    return dict(recording=rec, source=group, seed=seed,
        comparisons={'p': dict(reference=ref, known_delta=change, unknown_radius=0.)},
        metrics={'fallback': dict(risk=dict(rows=20, selected=0, supported=False,
            known_easy_net_degradation=None, completion_easy_risk_upper=None,
            completion_all_risk_upper=None), p95_known_ADE_pixel=3.)})


def test_group_and_seed_weighting_not_row_pooling():
    rows = [row('a', 'x', 17, 100., 0.), row('a', 'x', 29, 100., 0.),
            row('b', 'y', 17, 1., .2), row('b', 'y', 29, 1., .4)]
    a = grouped_readout(rows, dict(a='x', b='y'), [17, 29])
    assert a['effects']['p']['known_pp']['point'] == 15.
    assert a['risk']['fallback']['supported_views'] == 0
    assert a['risk']['fallback']['undefined_easy_views'] == 4
    assert a['risk']['fallback']['switch_rate'] == 0.


def test_missing_group_denominator_is_not_silently_dropped():
    rows = [row('a', 'x', 17, 10., 1.), row('b', 'y', 17, 0., 0.)]
    a = grouped_readout(rows, dict(a='x', b='y'), [17])
    assert a['effects']['p']['completion_lower_pp'] == dict(
        point=None, interval95=None, missing_source_seed_cells=1)


def test_rejects_duplicate_or_mismatched_roster():
    a = row('a', 'x', 17, 10., 1.)
    with pytest.raises(ValueError, match='Missing or duplicate'):
        grouped_readout([a, a], dict(a='x', b='y'), [17])
    with pytest.raises(ValueError, match='Source'):
        grouped_readout([a], dict(a='y'), [17])
