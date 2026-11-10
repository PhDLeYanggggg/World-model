import numpy as np

from src.world_model.m3w_vehicle_angular_pair import CAUSAL, candidates, choices


def test_angular_bank_keeps_each_actors_each_step_norm():
    rng = np.random.default_rng(57)
    cv = rng.normal(size=(2, 12, 2))
    original = cv+rng.normal(size=cv.shape)
    b = candidates(cv, original)
    np.testing.assert_array_equal(b[:, 0], original)
    r = np.linalg.norm(original-cv, axis=-1)
    np.testing.assert_allclose(np.linalg.norm(b-cv[:, None], axis=-1), np.repeat(r[:, None], 13, axis=1))


def test_forced_causal_choices_have_same_rotation_count_and_exact_objectives():
    rng = np.random.default_rng(7)
    cv, anchor, ols = rng.normal(size=(3, 2, 12, 2))
    bank = candidates(cv, anchor)
    result = choices(bank, ols)
    assert tuple(result) == CAUSAL
    assert all(min(v) >= 1 and max(v) <= 12 for v in result.values())
    assert len(set(result['shared'])) == 1

    def loss(i, j, pair):
        err = bank[[0, 1], [i, j]]-ols
        return np.linalg.norm(err, axis=-1).mean(1).sum()+pair*np.linalg.norm(err[0]-err[1], axis=-1).mean()

    for k, add in (('independent', 0), ('coupled', 1)):
        assert loss(*result[k], add) <= min(loss(i, j, add) for i in range(1, 13) for j in range(1, 13))+1e-12
