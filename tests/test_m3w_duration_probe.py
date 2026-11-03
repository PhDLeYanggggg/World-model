import numpy as np
import pytest
import torch

from src.evaluation.m3w_duration_probe import extended_settings, require_exact_prefix
from src.world_model import m3w_interaction_feature_cost as api


def test_duration_changes_only_coverage_and_keeps_exact_prefix_and_resume(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 7, 508)).astype(np.float32)
    causal = dict(x=x, envelope=np.ones((4, 7)), scales=np.ones(4, np.float32),
        allowed=np.ones((4, 7), bool), groups=[np.array([0, 1]), np.array([2, 3])],
        source_groups=[np.array([0]), np.array([1])], sites=np.array(['G0','G0','G1','G1']), ids=np.arange(4))
    paths = np.broadcast_to(np.arange(12)[None, None, :, None], (4, 7, 12, 2)).copy().astype(float)
    paths[:, 6] += .1
    data = api.base.training_data(causal, paths, paths[:, 0]+.2, np.ones((4, 12), bool), 0, 1.)
    settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=4,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    long = extended_settings(settings)
    assert settings['minimum_updates'] == 4
    assert {k for k in settings if settings[k] != long[k]} == {'minimum_updates', 'largest_source_cycles'}
    kwargs = dict(arm='interaction64', seed=17, identity={'probe': True})
    old = api.fit(data, settings=settings, path=tmp_path/'old.pt.gz', **kwargs)
    prefix = api.fit(data, settings=long, path=tmp_path/'long.pt.gz', stop_at=old['budget'], **kwargs)
    require_exact_prefix(old, prefix)
    recovered = api.fit(data, settings=long, path=tmp_path/'long.pt.gz', resume=True, **kwargs)
    direct = api.fit(data, settings=long, path=tmp_path/'direct.pt.gz', **kwargs)
    for key in direct:
        if key != 'seconds':
            api.base.base.exact(recovered[key], direct[key])
    with pytest.raises(ValueError):
        extended_settings(settings, 8)
