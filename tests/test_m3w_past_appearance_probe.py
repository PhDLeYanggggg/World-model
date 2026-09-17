import numpy as np
import pytest
import torch

from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe, pixel_delta_to_native, fit_probe


def inputs():
    torch.manual_seed(17)
    return {'geometry':torch.randn(6,32), 'images':torch.randn(6,8,3,32,32),
            'mask':torch.ones(6,8), 'image_xy':torch.zeros(6,2,dtype=torch.float64),
            'homography':torch.eye(3,dtype=torch.float64).expand(6,-1,-1),
            'target_native':torch.randn(6,12,2,dtype=torch.float64)*0.01,
            'parent_scale':torch.ones(6,dtype=torch.float64), 'start':torch.ones(6)}


def test_projection_zero_exact_and_axis_order_has_finite_gradients():
    delta = torch.zeros(2,12,2,requires_grad=True)
    xy = torch.tensor([[2.,5.],[7.,11.]])
    h = torch.eye(3).expand(2,-1,-1)
    result = pixel_delta_to_native(delta,xy,h)
    assert torch.equal(result,torch.zeros_like(result))
    (result.sum()).backward()
    assert torch.isfinite(delta.grad).all()
    d = torch.ones(2,12,2); d[...,1] = 2
    np.testing.assert_array_equal(pixel_delta_to_native(d,xy,h)[0,0], [192.,96.])


def test_geometry_and_current_arms_do_not_see_disallowed_images():
    data = inputs(); model = PastAppearanceProbe()
    with torch.no_grad():
        model.trajectory.weight.fill_(.01)
    changed = data['images'].clone(); changed[:,:7] += 100
    a = model(data['geometry'],data['images'],data['mask'],'current_rgb')
    b = model(data['geometry'],changed,data['mask'],'current_rgb')
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    a = model(data['geometry'],data['images'],data['mask'],'geometry')
    b = model(data['geometry'],changed*5,data['mask'],'geometry')
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    with pytest.raises(ValueError):
        model(data['geometry'],data['images'][:,:7],data['mask'],'past_rgb')


def test_missing_visual_mask_removes_pixels():
    data = inputs(); model = PastAppearanceProbe(); mask = torch.zeros_like(data['mask'])
    a = model(data['geometry'],data['images'],mask,'past_rgb')
    b = model(data['geometry'],data['images']*3,mask,'past_rgb')
    assert torch.equal(a[1],b[1])


def test_resume_matches_uninterrupted_real_updates(tmp_path):
    torch.set_num_threads(2)
    data = inputs()
    config = {'lr':.001,'weight_decay':.0001,'updates':6,'batch_size':4,
              'distance_epsilon':.001,'loss_numeric_scale':.001,'start_loss_weight':.1,
              'gradient_clip':1.,'checkpoint_every':2}
    identity = {'seed':17,'arm':'past_rgb'}
    torch.manual_seed(31); full = PastAppearanceProbe()
    result = fit_probe(full,data,np.arange(4),config,identity,tmp_path/'full.pt')
    torch.manual_seed(31); split = PastAppearanceProbe()
    fit_probe(split,data,np.arange(4),config,identity,tmp_path/'split.pt',stop_at=2)
    resumed = PastAppearanceProbe()
    after = fit_probe(resumed,data,np.arange(4),config,identity,tmp_path/'split.pt')
    assert after['complete'] and result['losses'] == after['losses']
    for k,v in full.state_dict().items():
        assert torch.equal(v,resumed.state_dict()[k])
    cached = fit_probe(resumed,data,np.arange(4),config,identity,tmp_path/'split.pt')
    assert cached['new_updates'] == 0
