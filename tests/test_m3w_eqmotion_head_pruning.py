import os

import pytest
import torch

from test_m3w_eqmotion_adapter import architecture, batch, source, threads
from src.world_model.m3w_supervised_intervention import build_forecaster


@pytest.mark.parametrize('head', [0, 7, 19])
def test_pruned_fixed_head_matches_predictions_and_trainable_gradients(source, head):
    config = {**architecture(source), 'fixed_head': head}
    full = build_forecaster(config)
    pruned = build_forecaster({**config, 'prune_unused_heads': True})
    pruned.load_state_dict(full.state_dict())
    inputs = batch()
    a, b = full(inputs), pruned(inputs)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, param in full.named_parameters():
        if param.requires_grad:
            actual = dict(pruned.named_parameters())[name]
            assert (param.grad is None) == (actual.grad is None)
            if param.grad is not None:
                torch.testing.assert_close(param.grad, actual.grad, atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get('M3W_TEST_MPS') != '1', reason='Explicit Metal access required')
def test_pruning_mps_matches_full_core(source):
    config = architecture(source)
    full = build_forecaster(config).to('mps')
    pruned = build_forecaster({**config, 'prune_unused_heads': True}).to('mps')
    pruned.load_state_dict(full.state_dict())
    inputs = {k: v.to('mps') for k, v in batch().items()}
    a, b = full(inputs), pruned(inputs)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, p in full.named_parameters():
        q = dict(pruned.named_parameters())[name]
        if p.requires_grad and p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
