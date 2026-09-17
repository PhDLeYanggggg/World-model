import numpy as np
import pytest
import torch

from src.world_model.m3w_past_frame import rotate_features, past_frame, frame_prediction, fit_frame
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, geometry_features
from src.world_model.m3w_objective_alignment import fit_objective, geometry_prediction


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def fixture_features():
    x = np.zeros((4, 581), dtype=np.float64)
    x[0, 24:26] = [1., 1.]
    n = x[:, 38:166].reshape(4, 8, 8, 2)
    m = x[:, 230:294].reshape(4, 8, 8)
    t = x[:, 166:230].reshape(4, 8, 8)
    n[1, 0, -2:] = [[1., 0.], [1., 2.]]
    m[1, 0, -2:] = 1.; t[1, 0, -2] = -.1
    n[2, 0, -1] = [-1., 2.]; m[2, 0, -1] = 1.
    return x


def test_anchor_hierarchy_and_no_direction_fallback():
    x = fixture_features()
    q, valid, kinds = past_frame(x)
    assert kinds.tolist() == ['ego_past_velocity', 'neighbor_past_velocity', 'neighbor_position', 'no_anchor']
    assert valid.tolist() == [True, True, True, False]
    for i, vector in enumerate(([1, 1], [0, 2], [-1, 2])):
        np.testing.assert_allclose(np.asarray(vector) @ q[i], [np.linalg.norm(vector), 0], atol=1e-12)
    np.testing.assert_array_equal(x, fixture_features())


def test_reexpression_commutes_with_canonical_frame_and_preserves_scalars():
    x = fixture_features()
    x[:, 511:] = np.arange(70)
    q, valid, kinds = past_frame(x)
    canonical = rotate_features(x, q)
    r = np.broadcast_to(rotation(.731), (len(x), 2, 2))
    rotated = rotate_features(x, r)
    q2, valid2, kinds2 = past_frame(rotated)
    np.testing.assert_array_equal(valid, valid2)
    np.testing.assert_array_equal(kinds, kinds2)
    np.testing.assert_allclose(rotate_features(rotated, q2)[valid], canonical[valid], atol=1e-11)
    scalar = [*range(16,24), *range(166,308), *range(476,511)]
    np.testing.assert_array_equal(rotated[:, scalar], x[:, scalar])
    np.testing.assert_allclose(rotate_features(rotated, r.transpose(0,2,1)), x, atol=1e-11)


def test_rotation_layout_matches_structured_geometry_builder():
    rng = np.random.default_rng(2)
    data = dict(history_xy=rng.normal(size=(8,2)), history_frame_offsets=np.arange(-7,1),
        history_velocity=rng.normal(size=(7,2)), neighbor_xy=rng.normal(size=(8,8,2)),
        neighbor_mask=np.ones((8,8), bool), neighbor_frame_offsets=np.tile(np.arange(-7,1), (8,1)),
        causal_features=np.arange(14), baseline_rollouts=rng.normal(size=(7,12,2)),
        prediction_frame_offsets=np.arange(1,13))
    x = np.concatenate((geometry_features(data), np.zeros(105)))[None]
    r = rotation(.42)
    expected = dict(data)
    for key in ('history_xy','history_velocity','neighbor_xy','baseline_rollouts'):
        expected[key] = data[key] @ r
    np.testing.assert_allclose(rotate_features(x, r[None])[0,:476], geometry_features(expected), atol=3e-7)


def test_prediction_identity_zero_init_and_no_anchor_guard():
    torch.manual_seed(17)
    model = OfflineVisualForecast(581)
    x, obs, b = torch.randn(4,581), torch.ones(4,8), torch.randn(4,12,2)
    q = torch.eye(2).repeat(4,1,1); valid = torch.tensor([True,True,True,False])
    assert torch.equal(frame_prediction(model,x,obs,b,q,valid), b)
    torch.nn.init.normal_(model.output.weight)
    actual = frame_prediction(model,x,obs,b,q,valid)
    torch.testing.assert_close(actual[:3], geometry_prediction(model,x,obs,b)[:3], rtol=0, atol=0)
    assert torch.equal(actual[3], b[3])


def test_training_identity_matches_control_and_resume(tmp_path):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    x, obs, b, y = torch.randn(20,16), torch.ones(20,8), torch.zeros(20,12,2), torch.randn(20,12,2)
    q, valid = torch.eye(2).repeat(20,1,1), torch.ones(20, dtype=torch.bool)
    batch = lambda ids:(x[ids],obs[ids],b[ids],y[ids])
    fb = lambda ids:(*batch(ids),q[ids],valid[ids])
    cfg = dict(updates=80,batch_size=8,learning_rate=.0003,weight_decay=.0001,checkpoint_every=40)
    ids, identity = torch.arange(20), dict(test=True)
    torch.manual_seed(17); old = OfflineVisualForecast(16)
    fit_objective(old,batch,ids,np.repeat([0,1],10),arm='row_log',config=cfg,seed=17,
        identity=identity,checkpoint=tmp_path/'old.pt',heartbeat=lambda v:None)
    torch.manual_seed(17); new = OfflineVisualForecast(16)
    fit_frame(new,fb,ids,config=cfg,seed=17,identity=identity,checkpoint=tmp_path/'new.pt',heartbeat=lambda v:None,stop_at=40)
    fit_frame(new,fb,ids,config=cfg,seed=17,identity=identity,checkpoint=tmp_path/'new.pt',heartbeat=lambda v:None)
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key],new.state_dict()[key])
    with pytest.raises(ValueError, match='identity'):
        fit_frame(new,fb,ids,config=cfg,seed=17,identity=dict(changed=True),checkpoint=tmp_path/'new.pt',heartbeat=lambda v:None)


def test_bad_inputs_rejected():
    x = fixture_features()
    with pytest.raises(ValueError):
        past_frame(x[:,:500])
    with pytest.raises(ValueError):
        rotate_features(x,np.zeros((4,2,2)))
    x[0,230] = .5
    with pytest.raises(ValueError):
        past_frame(x)
