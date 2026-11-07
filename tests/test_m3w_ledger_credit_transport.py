import numpy as np
import pytest

from src.world_model import m3w_ledger_credit_transport as m


def test_credit_audit_nets_actual_deficits_on_predicted_earning_queries():
    predicted = np.array([[-2.,-1.],[-2.,1.],[1.,-1.]])
    actual = np.array([[-1.,2.],[3.,1.],[-2.,-1.]])
    out = m.earned_credit(predicted,actual)
    assert out['predicted'] == [4.,2.] and out['realized'] == [-2.,-1.]
    assert out['positive_observed_on_credit'] == [1,1]


def test_transport_fit_keeps_held_source_out_and_retains_source_unit():
    donors = {str(i):[dict(predicted=[10.,4.],realized=[float(i),2.])] for i in (1,3,7,9)}
    assert m.calibrate(donors,'x')['alpha'] == [.1,.5]
    with pytest.raises(ValueError,match='Held source'):
        m.calibrate(donors,'1')


def test_partial_transport_is_between_no_carry_and_full_for_synthetic_prefix():
    groups = [np.array([0,1]),np.array([2,3])]; counts = [1,1]
    value = np.array([[1.,0.],[0.,0.],[8.,1.],[0.,0.]])
    a = np.array([[-2.,0.],[0.,0.],[1.5,0.],[0.,0.]])
    ok = np.ones_like(a,bool); parent = np.array([0,-1,1,-1])
    c0,_ = m.stream(groups,value,a,a,ok,counts,parent,[0.,0.])
    ch,_ = m.stream(groups,value,a,a,ok,counts,parent,[.5,.5])
    c1,_ = m.stream(groups,value,a,a,ok,counts,parent,[1.,1.])
    assert c0[2] == ch[2] == 1 and c1[2] == 0
