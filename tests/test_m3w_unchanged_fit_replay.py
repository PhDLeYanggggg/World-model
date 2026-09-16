from copy import deepcopy

import pytest
import torch

from scripts.verify_m3w_unchanged_fit_replay import compare_training_states


def example():
    return {'identity': {'protocol_sha256': 'one', 'settings': {'steps': 2}},
        'step': 2, 'losses': [3., 2.], 'model': {'w': torch.tensor([1., 2.])},
        'order': torch.tensor([0, 1]), 'cursor': 1, 'sampler_rng': torch.tensor([2])}


def test_protocol_change_does_not_imply_changed_weights():
    a, b = example(), example()
    b['identity']['protocol_sha256'] = 'two'
    assert compare_training_states(a, b)['model_state_exactly_equal']


def test_numeric_difference_is_reported_not_hidden():
    a, b = example(), example()
    b['model']['w'][0] += .5
    result = compare_training_states(a, b)
    assert not result['model_state_exactly_equal']
    assert result['max_model_state_absolute_difference'] == .5


def test_changed_training_or_incomplete_checkpoint_refused():
    a = example()
    for key in ('identity', 'step'):
        b = deepcopy(a)
        if key == 'identity':
            b['identity']['settings']['seed'] = 3
        else:
            b['step'] = 1
        with pytest.raises(ValueError, match='complete fits'):
            compare_training_states(a, b)
