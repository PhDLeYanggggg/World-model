import numpy as np
import pytest
import torch

from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model.m3w_flight_shape_risk import penalty, quantities
from scripts.audit_m3w_opendd_fixed_radius_feasibility import minimum_ade


def test_budget_hinges_reproduce_original_scoring_with_unknown_debit():
    cv = torch.zeros(3, 12, 2, dtype=torch.float64)
    y = cv.clone()
    y[0, :, 0] = .1
    y[1, :, 0] = 1.
    y[2] = torch.nan
    old = cv.clone()
    old[:, :, 0] = .005
    pred = -old
    valid = torch.isfinite(y).all(2)
    easy = torch.tensor([True, False, False])
    p = policy(pred.numpy(),cv.numpy(),cv.numpy(),y.numpy(),valid.numpy(),['f']*3,
               easy_cut=.2,hard_cut=.8)['recordings']['f']
    a, b, signed, debit = quantities(pred,old,y,valid,torch.tensor([.1,1.,0.],dtype=cv.dtype),
                                    torch.ones(3,dtype=cv.dtype),torch.full((3,),.005,dtype=cv.dtype))
    observed = penalty(a,b,signed,debit,easy,1.1,.1,.1,'flight_budget')
    expected = sum(max(0.,p[k]-.02) for k in ('completion_all_risk_upper',
        'completion_easy_risk_upper','known_easy_net_degradation'))
    assert observed.item() == pytest.approx(expected)
    assert debit.item() == pytest.approx(.005)


def test_noncanceling_detects_balanced_harm_increments():
    new, old = torch.tensor([.01, .0]), torch.tensor([.0, .01])
    easy = torch.ones(2,dtype=torch.bool)
    args = (new,old,new,torch.tensor(0.),easy,1.,1.,1.)
    assert penalty(*args,'flight_relative').item()==0.
    assert penalty(*args,'flight_noncancel').item()==pytest.approx(.01)


def test_missing_evaluation_mass_does_not_divide_by_zero():
    a = torch.zeros(3,requires_grad=True)
    p = penalty(a,a,a,torch.tensor(.05),torch.zeros(3,dtype=torch.bool),0.,0.,0.,'flight_relative')
    p.backward()
    assert p.item()==0 and torch.isfinite(a.grad).all()


def test_fixed_radius_bound_includes_required_observed_witness():
    residual = np.zeros((3,2,2))
    residual[:,:,0] = [[.2,.3],[.2,np.nan],[1.2,.3]]
    valid = np.isfinite(residual).all(2)
    result = minimum_ade(residual,valid,np.ones(3))
    np.testing.assert_allclose(result,[.35,0.,.1])
