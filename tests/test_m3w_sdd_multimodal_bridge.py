import numpy as np
import pytest

from src.world_model.m3w_sdd_multimodal_bridge import visual_model_arrays, SDDStepImageStore
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter
from src.world_model.m3w_sdd_past_images import SDDPastImageStore


def example():
    rows = np.array([[0, f, 0, f+2, 2, f, 0, 0, 1] for f in range(30)], float)
    adapter = SDDStepAdapter(rows, np.array(['Pedestrian']*30), 's/v', 1)
    inputs = adapter.get_inputs(0)
    images = dict(source_frames=np.arange(8), query_frame=7, state_mask=np.ones(8, bool),
                  rgb_retained=np.full((8, 3, 32, 32), 127, np.uint8),
                  retained_count=np.full((8, 32, 32), 9, np.uint8), block_area=9)
    return inputs, images


def test_model_input_whitelist_and_shapes():
    inputs, images = example()
    bundle = visual_model_arrays(inputs, images)
    assert set(bundle) == {'geometry', 'rgb', 'coverage', 'baseline'}
    assert bundle['geometry'].shape == (476,)
    assert bundle['baseline'].shape == (12, 2)
    assert bundle['rgb'].shape == (8, 3, 32, 32)
    assert bundle['coverage'].shape == (8, 1, 32, 32)
    assert bundle['coverage'].min() == 1
    inputs['future_endpoint'] = np.ones(2)*1e6
    images['future_target'] = np.ones((12, 2))*1e6
    for key, value in visual_model_arrays(inputs, images).items():
        np.testing.assert_array_equal(value, bundle[key])


@pytest.mark.parametrize('field', ['source_frames', 'state_mask', 'retained_count'])
def test_join_rejects_misalignment_or_invalid_support(field):
    inputs, images = example()
    images[field] = images[field].copy()
    images[field].flat[-1] = {'source_frames': 8, 'state_mask': False, 'retained_count': 10}[field]
    with pytest.raises(ValueError):
        visual_model_arrays(inputs, images)


def test_sparse_cache_missing_is_not_unsupported_image(monkeypatch):
    store = SDDStepImageStore.__new__(SDDStepImageStore)
    store.block_area = 9
    monkeypatch.setattr(SDDPastImageStore, 'inputs', lambda *a, **k: {
        'annotation_present_mask': np.array([True]*7+[False])})
    with pytest.raises(ValueError, match='Uncached'):
        store.inputs(query_frame=20, agent_id=0)
    with pytest.raises(ValueError, match='eight'):
        store.inputs(query_frame=20, agent_id=0, length=16)
