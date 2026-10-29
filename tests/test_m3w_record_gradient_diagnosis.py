import copy
import numpy as np
import torch
from src.world_model import m3w_multiaction_record_policy as policy
from src.evaluation.m3w_record_gradient_diagnosis import gradient_components, reference_consistency
from tests.test_m3w_multiaction_record_policy import data_fixture


def test_decomposition_is_exact_without_changing_weights_or_duals():
    torch.set_num_threads(1); data = data_fixture()
    model = policy.RecordActionPolicy(8, 17)
    state = dict(settings={'width':8, 'chunk_size':4, 'gradient_clip':5., 'dual_maximum':100.},
        seed=17, model=copy.deepcopy(model.state_dict()), dual=torch.ones((2, 3)))
    old = copy.deepcopy(state)
    result = gradient_components(state, data)
    policy.exact(state, old)
    assert result['no_optimizer_step']
    assert result['decomposition_relative_error'] < 2e-4
    assert result['selected_rows'] == 12


def test_candidate_reference_variation_is_not_hidden_as_ground_truth():
    y = np.tile([1., 0., 2., 1., 0.], (3, 7, 1)); y[-1] = np.nan
    p = np.tile([1., 0., 2., 1., 0.], (3, 7, 1)); p[:, -1, 2] = 4
    result = reference_consistency(p, y, np.array([6, -1, 6]))
    assert result['reference']['rows_over_one_percent_spread'] == 3
    assert result['known_selected']['rows'] == 1
    assert result['known_selected']['predicted_over_actual'][2] == 2.
