import copy

import pytest
import torch

from scripts.verify_m3w_producer_crossfit_pilot import compare


def states():
    a = dict(model=torch.tensor([1.]), sampler_rng=torch.tensor([2]),
             seconds=2., losses=[dict(step=1, loss=2.), dict(step=20, loss=1.5), dict(step=40, loss=1.)])
    b = copy.deepcopy(a)
    b['seconds'] = 3.
    b['losses'].pop(1)
    return a, b


def test_extra_stop_log_not_optimizer_difference():
    a, b = states()
    assert compare(a, b) == [20]


@pytest.mark.parametrize('key', ['model', 'sampler_rng'])
def test_scientific_state_difference_still_fails(key):
    a, b = states()
    a[key] += 1
    with pytest.raises(ValueError):
        compare(a, b)


def test_changed_common_loss_fails():
    a, b = states()
    b['losses'][-1]['loss'] = 1.01
    with pytest.raises(ValueError):
        compare(a, b)
