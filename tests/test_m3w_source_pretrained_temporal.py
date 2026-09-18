import numpy as np
import pytest
import torch

from src.world_model.m3w_source_pretrained_temporal import (
    ARMS, TemporalSourceDynamics, embedding_lookup, normalized_embedding,
)
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate

torch.set_num_threads(4)


def features():
    g = torch.randn(3, 480)
    e = torch.randn(3, 8, 512)
    c = torch.ones(3, 8)
    return g, e, c


def test_zero_initialization_and_bounded_outputs():
    x = features()
    for arm in ARMS:
        model = TemporalSourceDynamics(arm)
        assert torch.count_nonzero(model(*x)) == 0
        with torch.no_grad(): model.head[-1].bias.fill_(1000)
        assert torch.all(torch.linalg.vector_norm(model(*x), dim=-1) < 1)


def test_arms_isolate_observed_sequence_information():
    g, e, c = features()
    changed = e.clone(); changed[:, :-1] += 100
    for arm in ARMS:
        torch.manual_seed(17); model = TemporalSourceDynamics(arm)
        with torch.no_grad(): model.head[-1].weight.normal_()
        same = torch.equal(model(g, e, c), model(g, changed, c))
        assert same == (arm != 'sequence')
    model = TemporalSourceDynamics('geometry')
    with torch.no_grad(): model.head[-1].weight.normal_()
    torch.testing.assert_close(model(g, e, c), model(g, e+100, c), rtol=0, atol=0)


def test_embedding_normalization_uses_no_batch_statistics():
    e = torch.randn(4, 512); c = torch.tensor([1., .5, 0., 1.])
    output = normalized_embedding(e, c)
    assert torch.count_nonzero(output[2]) == 0
    torch.testing.assert_close(output[:2], normalized_embedding(e[:2], c[:2]), rtol=0, atol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(output[[0, 1, 3]], dim=-1), torch.ones(3))
    with pytest.raises(ValueError): normalized_embedding(e, torch.full((4,), 2.))


def test_lookup_rejects_main_outer_and_unknown():
    allowed = np.array([11, 13, 18]); rows = np.arange(24).reshape(3, 8)
    np.testing.assert_array_equal(embedding_lookup([18, 11], allowed, rows), rows[[2, 0]])
    for ids in ([0], [12], [19]):
        with pytest.raises(ValueError): embedding_lookup(ids, allowed, rows)


def test_exact_checkpoint_resume_for_temporal_head(tmp_path):
    torch.manual_seed(12)
    g, e, c = features(); y = torch.randn(3, 12, 2)*.1
    ids = np.arange(3); weights = np.full(3, 1/3)
    def inputs(query):
        n = len(query)
        return ((g[query], e[query], c[query]),
                (torch.ones(n), torch.eye(2).expand(n, 2, 2), torch.ones(n, dtype=torch.bool)))
    config = dict(start_step=2, updates=6, batch_size=2, learning_rate=.0003,
                  minimum_lr_ratio=.01, weight_decay=.0001, checkpoint_every=2)
    def run(path, limit=None):
        torch.manual_seed(17); model = TemporalSourceDynamics('sequence')
        result = fit_motion_candidate(model, inputs, lambda query:y[query], ids, weights,
            scale=1., seed=17, config=config, identity={'fixture': True}, checkpoint=path,
            heartbeat=lambda **kw:None, suppress_zero_targets=False, stop_at=limit)
        return model, result
    full, _ = run(tmp_path/'full.pt')
    run(tmp_path/'resume.pt', 2); resumed, result = run(tmp_path/'resume.pt')
    assert result['new_updates'] == 4
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    _, complete = run(tmp_path/'resume.pt')
    assert complete['new_updates'] == 0
