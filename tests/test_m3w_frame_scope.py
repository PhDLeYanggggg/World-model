import numpy as np

from src.data_unification.m3w_causal_recordings import causal_coordinate_transform


def test_moving_source_pipeline_already_removes_global_rotation():
    history = np.column_stack((np.arange(8), np.arange(8)**2/20))
    frames = np.arange(8)*10
    angle = .71
    r = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    old = causal_coordinate_transform(history,frames,120)
    rotated = causal_coordinate_transform(history@r,frames,120)
    a = (history-old['origin_xy'])@old['rotation']/old['scale']
    b = (history@r-rotated['origin_xy'])@rotated['rotation']/rotated['scale']
    np.testing.assert_allclose(a,b,atol=1e-12)


def test_static_ego_has_no_direction_to_canonicalize_neighbor_context():
    history = np.zeros((8,2))
    frames = np.arange(8)*10
    r = np.array([[0.,-1.],[1.,0.]])
    old = causal_coordinate_transform(history,frames,120)
    rotated = causal_coordinate_transform(history@r,frames,120)
    np.testing.assert_array_equal(old['rotation'],np.eye(2))
    np.testing.assert_array_equal(rotated['rotation'],np.eye(2))
    neighbor = np.array([1.,2.])
    assert not np.array_equal(neighbor@old['rotation'],neighbor@r@rotated['rotation'])
