import copy
import pytest
from src.world_model import m3w_policy_region_cost as api
from tests.test_m3w_policy_region_cost import regions_for
from tests.test_m3w_scene_context_cost import scene_sample, SETTINGS


def test_changed_training_region_centroid_cannot_silently_resume(tmp_path):
    *_, data = scene_sample(); regions = regions_for(data, 'physical_neural')
    kw = dict(data=data, arm='physical_neural', settings=SETTINGS, seed=17,
        identity={'fixture':True}, path=tmp_path/'checkpoint.gz', region_weight=1.)
    api.fit(**kw, regions=regions, stop_at=2)
    changed = copy.deepcopy(regions); changed['target'][0, 4] += .1
    with pytest.raises(AssertionError):
        api.fit(**kw, regions=changed, resume=True)
