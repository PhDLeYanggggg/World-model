import numpy as np
import torch
from scripts.probe_m3w_motion_unit_sensitivity import rescale_geometry
from src.world_model.m3w_partial_context import PartialContextSourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry
from test_m3w_agent_track_context import geometry


def test_rescaling_leaves_past_time_masks_and_requested_horizon_unchanged():
    a=pack_geometry(geometry()); b=pack_geometry(rescale_geometry(geometry(),2))
    for key in ('history_mask','neighbor_mask','prediction_time','request_mask'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    torch.testing.assert_close(a['history'][...,2],b['history'][...,2],rtol=0,atol=0)
    torch.testing.assert_close(a['neighbors'][...,2],b['neighbors'][...,2],rtol=0,atol=0)


def test_current_bounded_wrapper_is_not_scale_equivariant_after_nonzero_output():
    torch.manual_seed(17); model=PartialContextSourceForecaster(1,width=8,heads=2).eval()
    with torch.no_grad(): model.predictor.predictor.predictor.output[-1].bias.fill_(.01)
    x=geometry()
    with torch.no_grad():
        original=model(pack_geometry(x)); changed=model(pack_geometry(rescale_geometry(x,2)))/2
    assert np.isfinite(changed.numpy()).all()
    assert not torch.allclose(original,changed,rtol=1e-5,atol=1e-4)
