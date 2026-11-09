import numpy as np
import torch

from scripts import build_m3w_songdo_matched_proposals as run


def test_cached_actor_embeddings_reproduce_direct_pair_models():
    x = torch.randn(8, 45)
    i, j = np.triu_indices(len(x), 1)
    r = torch.randn(len(i), 6)
    models = {a: [run.learn.model.PairValue(a == 'joint').eval() for _ in range(2)]
              for a in ('additive', 'joint')}
    a, p = run.logits(models, x, i, j, r)
    with torch.inference_mode():
        expected = {k: np.mean([n(x[i], x[j], r).numpy().astype(float) for n in v], axis=0)
                    for k, v in models.items()}
    np.testing.assert_allclose(.5*(a[i]+a[j]), expected['additive'], atol=1e-7)
    np.testing.assert_allclose(p, expected['joint'], atol=1e-7)
    assert run.learn.model.merit(p).argmax() == run.learn.model.merit(expected['joint']).argmax()


def test_independent_and_joint_use_same_distinct_actor_roster():
    row = np.array([0, 0, 1, 2])
    family = np.array([0, 1, 0, 1])
    i, j = np.triu_indices(len(row), 1)
    keep = row[i] != row[j]
    i, j = i[keep], j[keep]
    a = np.zeros((len(row), 4))
    a[:, 2:] = np.array([[4], [5], [3], [1]])
    z = np.zeros((len(i), 4))
    z[-1, 2:] = 10
    out = run.ranked_pairs(row, family, i, j, a, z)
    assert tuple(out) == run.ARMS
    assert set(row[out['independent'][0]]) == {0, 1}
    assert set(row[out['joint'][0]]) == {1, 2}
    assert len(set(family[out['same_family'][0]])) == 1
