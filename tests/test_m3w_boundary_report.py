import numpy as np
import pytest
from scripts.run_m3w_boundary_diagnostic import strict_mean, verify_result
from scripts.manage_m3w_boundary_diagnostic import compare_tree
from tests.test_m3w_boundary_diagnostic import packet
from src.world_model.m3w_boundary_diagnostic import diagnostic
from scripts.run_m3w_boundary_diagnostic import aggregate
from scripts.report_m3w_boundary_diagnostic import verify_summary


def test_undefined_mean_stays_undefined():
    r=strict_mean([1.,None])
    assert r['mean'] is None and r['conditional_mean']==1. and r['undefined']==1


def test_cross_platform_tolerance_does_not_allow_material_difference():
    compare_tree({'a':[1.]},{'a':[1.+1e-11]})
    with pytest.raises(AssertionError):compare_tree({'a':[1.]},{'a':[1.001]})
    with pytest.raises(AssertionError):compare_tree(None,0.)


def test_independent_accounting_catches_corrupt_native_sum():
    a,m,c=packet();r=diagnostic(a,m,c)
    r['arms']['affine']['scopes']['matched']['all']['true_harm']+=1
    with pytest.raises(AssertionError):verify_result(r)


def summary_fixture():
    a,m,c=packet();r=diagnostic(a,m,c)
    import copy
    rows=[]
    for phase,count in [('fitting',72),('internal_transfer',216)]:
        for i in range(count):
            item=copy.deepcopy(r);item['meta']['phase']=phase
            item['meta']['site']=f'site{i%12}'
            item['meta']['view']=f'{phase}_{i}'
            rows.append(item)
    return aggregate(rows,c)


def test_summary_reductions_verify_and_reject_changed_mean():
    s=summary_fixture()
    assert verify_summary(s)>0
    z=s['phases']['fitting']['arms']['affine']['matched']['all']
    z['equal_site']['realized_risk_ratio']['conditional_mean']=123.
    with pytest.raises(AssertionError):verify_summary(s)


def test_summary_rejects_hidden_unknown_count():
    s=summary_fixture()
    s['phases']['internal_transfer']['unknown_rows']+=1
    with pytest.raises(AssertionError):verify_summary(s)
