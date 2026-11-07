import pytest

from scripts import build_m3w_ledger_reproduction as bundle
from src.evaluation import causal_ledger_portable as portable


def test_rebuild_retains_utility_risk_and_negative_credit():
    files, got = bundle.make_files()
    assert got['comparisons']['allocation/ledger_vs_query'][0] > 0
    assert got['risk']['bank/full_ledger']['passes'] == 6
    assert got['risk']['allocation/ledger_matched']['passes'] == 3
    assert got['credit']['realized'][0] < 0 and got['credit']['realized'][1] < 0
    assert b'moving_source_component' not in files['evidence.json']


def test_dropping_an_undefined_view_cannot_reproduce():
    e = bundle.make_evidence(); views = e['policies']['credit/donor_discount']['recording_views']
    views.pop(next(i for i,v in enumerate(views) if v[2] is None))
    with pytest.raises(ValueError,match='cannot be dropped'):
        portable.reconstruct(e)
