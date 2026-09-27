import numpy as np
from src.world_model.m3w_observation_quality import masked_neighbors, repair_geometry
from src.world_model.m3w_temporal_support import temporal_features
from src.world_model.m3w_european_source_forecast import pack_scene


def test_partial_mask_has_defined_temporal_features():
    h = np.zeros((3, 8, 2)); h[0, :, 0] = np.arange(8)
    h[1, :, 0] = np.arange(8)+1; h[2, :, 1] = 10
    valid = np.ones((3, 8), bool); valid[1, :6] = False
    h[~valid] = 0
    s = dict(history_xy=h, history_valid=valid, target_eligible=valid.all(1),
             agent_id=np.array([1, 2, 3]), baseline_cv=np.zeros((3, 12, 2)))
    legacy, _ = pack_scene(s)
    repaired = repair_geometry(legacy, masked_neighbors(s))
    old_h, old_n = temporal_features(legacy, np.ones(2))
    new_h, new_n = temporal_features(repaired, np.ones(2))
    np.testing.assert_array_equal(new_h, old_h)
    assert np.isfinite(new_n).all() and not np.array_equal(new_n, old_n)
    assert repaired[:, 230:294].sum() == 20


def test_nearest_eight_and_ties_do_not_depend_on_agent_order():
    h = np.zeros((11, 8, 2)); h[:, :, 0] = np.arange(11)[:, None]
    valid = np.ones((11, 8), bool); valid[1:, :4] = False
    s = dict(history_xy=h, history_valid=valid, target_eligible=valid.all(1), agent_id=np.arange(11))
    n = masked_neighbors(s)
    np.testing.assert_array_equal(n['agent_ids'], [np.arange(1, 9)])
    rev = masked_neighbors({k:v[::-1] for k,v in s.items()})
    for key in ('xy', 'valid', 'agent_ids', 'partial_neighbors'):
        np.testing.assert_array_equal(n[key], rev[key])
