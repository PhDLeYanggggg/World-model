import numpy as np
import torch

from scripts.diagnose_m3w_visual_actor_branch import moments


def test_branch_diagnosis_reports_zero_response_and_input_clipping():
    state = dict(image_mean=torch.zeros(46), image_std=torch.ones(46))
    state['image_encoder.0.weight'] = torch.ones(32, 46)
    state['image_encoder.2.weight'] = torch.zeros(8, 32)
    x = np.full((3, 46), 10, np.float32)
    result = moments(x, state, np.array([0, 2]))
    assert result['rows'] == 2
    assert result['input_element_clipped_fraction'] == 1
    assert result['rows_with_any_clipped_input_fraction'] == 1
    assert result['offset_max_absolute'] == result['offset_mean_absolute'] == 0
    assert result['branch_saturated_fraction'] == 0
    assert result['final_weight_norm'] == 0
