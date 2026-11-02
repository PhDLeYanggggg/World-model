import copy
import numpy as np
import pytest
import torch

from scripts.run_m3w_selection_capacity import width_only_settings,assert_common_training,initial_prediction
from src.world_model import m3w_selection_cardinal_cost as api
from tests.test_m3w_selection_controller import data


def test_only_width_may_change():
    base = dict(width=64,learning_rate=.001,minimum_updates=2000)
    wide = dict(base,width=128)
    width_only_settings(base,wide)
    with pytest.raises(ValueError): width_only_settings(base,dict(wide,learning_rate=.002))


def test_capacity_preserves_initial_outputs_draws_and_exact_resume(tmp_path):
    old_threads = torch.get_num_threads(); torch.set_num_threads(4)
    try:
        d = data(); settings = dict(width=64,learning_rate=.001,query_batch_size=2,minimum_updates=4,
            largest_source_cycles=1,gradient_clip=5.,heartbeat_every=2,checkpoint_every=2)
        common = dict(arm='mean_absolute',seed=17,identity={'test':True})
        base = api.fit(d,settings=settings,path=tmp_path/'base.gz',**common)
        wide_settings = dict(settings,width=128)
        api.fit(d,settings=wide_settings,path=tmp_path/'resume.gz',stop_at=2,**common)
        wide = api.fit(d,settings=wide_settings,path=tmp_path/'resume.gz',resume=True,**common)
        direct = api.fit(d,settings=wide_settings,path=tmp_path/'direct.gz',**common)
        for key in wide:
            if key != 'seconds': api.base.exact(wide[key],direct[key])
        assert_common_training(base,wide)
        np.testing.assert_array_equal(initial_prediction(base,d['causal']),initial_prediction(wide,d['causal']))
        assert base['parameters'] == 29253 and wide['parameters'] == 58501
        broken = copy.deepcopy(wide); broken['draw_hash'] = 'different'
        with pytest.raises(AssertionError): assert_common_training(base,broken)
        with pytest.raises(ValueError): initial_prediction(wide,dict(d['causal'],future_endpoint=np.zeros((8,2))))
    finally:
        torch.set_num_threads(old_threads)
