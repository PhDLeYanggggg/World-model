import copy
import pytest

from scripts.report_m3w_record_balanced_policy import validate_result, run
from tests.test_m3w_direct_policy import fixture


def value():
    packets,_=fixture();args=next(iter(packets.values()));n=len(args[0]);np=run.readout.np
    masks=dict(origin=np.ones(n,bool),cost_sensitive=np.zeros(n,bool),record_balanced=np.arange(n)%2==0)
    return run.readout.evaluate(args,masks,{k:np.arange(n,dtype=float) for k in masks})


def test_record_arithmetic_and_empty_matched_semantics():
    result=value()
    assert validate_result(result)==3*(1+len(result['recordings']))+4


def test_missing_record_mass_fails_validation():
    result=value();record=next(iter(result['recordings']))
    result['recordings'][record]['record_balanced']['known_rows']=0
    with pytest.raises((ValueError,AssertionError)):validate_result(result)


def test_wrong_denominator_rejected():
    result=copy.deepcopy(value());result['full_known_reference_mass']*=2
    with pytest.raises((ValueError,AssertionError)):validate_result(result)
