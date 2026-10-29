import numpy as np
import torch
from src.evaluation.m3w_multiaction_record_inference import predict
from src.world_model import m3w_multiaction_record_policy as original


def state(arm):
    model = original.RecordActionPolicy(8, 17)
    return dict(arm=arm, settings={'width':8}, seed=17, model=model.state_dict())


def test_multi_action_inference_exactly_matches_frozen_function():
    torch.set_num_threads(1)
    f = np.random.default_rng(4).normal(size=(13, 7, 12))
    anchor = np.zeros((13, 7), bool); anchor[::2, 2] = True
    allowed = np.ones_like(anchor)
    for arm in ('physical_only', 'physical_neural'):
        s = state(arm)
        old = original.predict(s, f, anchor, allowed, arm=arm, chunk_size=4)
        new = predict(s, f, anchor, allowed, arm=arm, chunk_size=4)
        for a,b in zip(old, new):
            np.testing.assert_array_equal(a,b)


def test_single_neural_action_preserves_candidate_axis_and_legality():
    torch.set_num_threads(1); s = state('neural_only')
    f = np.zeros((5, 7, 12)); anchor = np.zeros((5, 7), bool); anchor[:3, 6] = True
    allowed = np.ones_like(anchor); allowed[1, 6] = False
    selected, ranks = predict(s, f, anchor, allowed, arm='neural_only', chunk_size=2)
    assert selected.tolist() == [6,-1,6,-1,-1]
    assert ranks.tolist() == [1.,0.,1.,0.,0.]
