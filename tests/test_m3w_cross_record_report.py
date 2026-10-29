import copy

import pytest
from scripts import report_m3w_cross_record_policy as api
from tests.test_m3w_cross_record_readout import example


def test_mass_checks_reject_changed_counts():
    d=example();assert api.validate_result(d)>0
    changed=copy.deepcopy(d);changed['policies']['cross_record_meta']['selected_count']+=1
    with pytest.raises((ValueError,AssertionError)):api.validate_result(changed)


def test_source_bootstrap_does_not_multiply_samples_by_seed_or_context():
    rows=[dict(source=s,result=example()) for s in ('a','b','c')]
    a=api.source_bootstrap(rows);b=api.source_bootstrap(rows*6)
    assert a == b and a['source_clusters'] == 3 and a['draws'] == 3000
    assert not a['row_bootstrap'] and not a['independent_confirmation']


def test_all_negative_source_contrast_stays_negative():
    d=example()
    for c in d['comparisons'].values():
        for mode in ('full','matched'):
            c[mode]['lower_pp']=-2.;c[mode]['upper_pp']=-1.
    out=api.source_bootstrap([dict(source=s,result=d) for s in ('a','b','c')])
    for pair in out['comparisons'].values():
        for c in pair.values():assert c['outer_bootstrap_completion_envelope_pp'] == [-2.,-1.]
