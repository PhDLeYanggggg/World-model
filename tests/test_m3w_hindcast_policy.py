import numpy as np
import pytest

from src.world_model import m3w_hindcast_policy as method


def test_backtest_translation_rotation_and_constant_velocity():
    h = np.cumsum(np.random.default_rng(7).normal(size=(10, 8, 2)), axis=1)
    v = method.backtests(h)
    np.testing.assert_allclose(v, method.backtests(h[..., ::-1]+33), atol=1e-13)
    assert v.shape == (10, 4, 4)
    straight = np.tile(np.arange(8)[None, :, None], (2, 1, 2))
    np.testing.assert_array_equal(method.backtests(straight), 0)
    with pytest.raises(ValueError, match="eight"):
        method.backtests(h[:, :-1])


def test_temporal_agreement_requires_positive_origin_and_legal_path():
    x = np.ones((3, 4, 4))
    x[1, 0] = -0.1
    actions = method.menu(x, np.array([True, True, False]))
    np.testing.assert_array_equal(actions["four_origins_3"], [3, 0, 0])
    np.testing.assert_array_equal(actions["three_origins_2"], [2, 2, 0])


def test_group_selection_never_uses_the_held_out_result():
    scores = {k: {"A": dict(passed=0, recordings=1, gain_pp=0.),
                  "B": dict(passed=0, recordings=1, gain_pp=0.)} for k in method.MENU}
    scores["all_1"]["A"].update(passed=1)
    scores["all_2"]["B"].update(passed=1, gain_pp=100)
    scores["all_3"]["A"].update(passed=1, gain_pp=-5)
    scores["all_3"]["B"].update(passed=1, gain_pp=-5)
    assert method.select(scores, ["A", "B"]) == "all_3"
    assert method.select(scores, ["A"]) == "all_1"
