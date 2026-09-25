import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.m3w_symmetric_utility import (
    complete_budget, validate_config, verify_matched_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def test_registration_changes_only_utility_objective():
    old = json.loads((ROOT/'configs/m3w_european_protected_motion_v1.json').read_text())
    new = json.loads((ROOT/'configs/m3w_european_symmetric_utility_v1.json').read_text())
    validate_config(new, old)
    for key, value in [('predicted_risk_budget', .03), ('deployment_promotion', True),
                       ('bootstrap_resamples', 2000), ('independent_reserved_readout', True)]:
        bad = dict(new, **{key:value})
        with pytest.raises(ValueError, match='Simultaneous'):
            validate_config(bad, old)


def checkpoint_pair():
    state = dict(arm='underharm4', settings=dict(steps=2, batch_size=2), seed=17, step=2,
        preprocess=dict(mean=np.zeros(2), std=np.ones(2), constant=np.ones(2),
            weights=np.array([.5, .5, 0]), known=np.array([True, True, False]), cost_scale=1.),
        draws=np.array([2, 2, 0]), sampler_rng=np.array([1, 2]))
    new = copy.deepcopy(state)
    new['arm'] = 'mse'
    return new, state


def test_matched_draws_and_support_are_enforced():
    new, old = checkpoint_pair()
    verify_matched_checkpoint(new, old)
    new['draws'] = np.array([1, 2, 1])
    with pytest.raises(AssertionError):
        verify_matched_checkpoint(new, old)


@pytest.mark.parametrize('field', ['constant', 'weights', 'mean', 'std'])
def test_preprocessing_and_initialization_cannot_change(field):
    new, old = checkpoint_pair()
    new['preprocess'][field][0] += .1
    with pytest.raises(AssertionError):
        verify_matched_checkpoint(new, old)


def test_complete_matrix_before_readout():
    heads = {str(i):dict(fit=dict(complete=True, step=2000, unknown_rows_sampled=0)) for i in range(18)}
    assert complete_budget(heads) == 36000
    with pytest.raises(ValueError):
        complete_budget(dict(list(heads.items())[:17]))
    heads['0']['fit']['step'] = 100
    with pytest.raises(ValueError):
        complete_budget(heads)
