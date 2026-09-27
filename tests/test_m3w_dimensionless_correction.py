import torch
from src.world_model.m3w_agent_track_context import AgentTrackSourceForecaster
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget
from src.world_model.m3w_european_source_forecast import baseline_torch
from scripts.probe_m3w_motion_unit_sensitivity import rescale_geometry
from test_m3w_agent_track_context import geometry


def test_matching_parameters_initial_floor_and_finite_gradients():
    torch.manual_seed(17); old=AgentTrackSourceForecaster(1,width=8,heads=2)
    torch.manual_seed(17); new=DimensionlessAgentTrackSourceForecaster(1,width=8,heads=2)
    assert old.state_dict().keys()==new.state_dict().keys()
    for k,v in old.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
    x=pack_geometry(geometry())
    torch.testing.assert_close(new(x),baseline_torch(x['history'],1),rtol=0,atol=0)
    ((new(x)-1)**2).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())


def test_nonzero_predictions_scale_when_conditioning_clamp_inactive():
    torch.manual_seed(29); model=DimensionlessAgentTrackSourceForecaster(1,width=8,heads=2).eval()
    torch.nn.init.normal_(model.predictor.predictor.predictor.output[-1].weight,std=.1)
    x=geometry()
    with torch.no_grad():
        original=model(pack_geometry(x))
        for factor in (.25,.5,2.,4.):
            changed=model(pack_geometry(rescale_geometry(x,factor)))/factor
            torch.testing.assert_close(original,changed,rtol=1e-5,atol=1e-4)


def test_bound_missing_neighbors_and_stationary_floor_preserved():
    torch.manual_seed(43); model=DimensionlessAgentTrackSourceForecaster(1,width=8,heads=2).eval()
    with torch.no_grad(): model.predictor.predictor.predictor.output[-1].bias.fill_(100.)
    x=pack_geometry(geometry()); x['baseline']=baseline_torch(x['history'],1)
    with torch.no_grad(): p=model(x)
    assert (torch.linalg.vector_norm(p-x['baseline'],dim=-1)<=past_motion_budget(x)+1e-4).all()
    x['neighbor_mask'][:]=False; x['neighbors'][:]=float('nan'); x['history'][...,:2]=0
    with torch.no_grad(): p=model(x)
    assert torch.isfinite(p).all() and not p.any()
