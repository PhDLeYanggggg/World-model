import pytest

from scripts.verify_m3w_dut_population import COUNT_KEYS, compare_counts, recount


def test_targets_do_not_require_future_and_types_are_separate():
    counts = recount([(frame, agent) for frame in range(20) for agent in (0, 1)])
    assert counts["queries"] == 13
    assert counts["targets"] == counts["visible"] == 26
    assert counts["complete"] == 2
    assert counts["partial_targets"] == 22
    assert counts["zero_future_targets"] == 2
    assert counts["known_steps"] == 156
    assert counts["complete_pedestrian_targets"] == counts["complete_vehicle_targets"] == 1


def test_broken_history_is_not_filled_from_future():
    counts = recount([(f, 0) for f in range(20) if f != 6])
    assert counts["targets"] == 6
    assert counts["visible"] == 13
    assert counts["unknown_cv_context"] == 1
    assert counts["complete"] == 0


def test_duplicate_raw_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        recount([(0, 2), (0, 2)])


def test_each_model_view_requires_all_population_counts():
    expected = recount([(f, 0) for f in range(20)])
    receipt = dict(queries=13, next_query=13,
                   stats={"view_a": {k: expected[k] for k in COUNT_KEYS},
                          "view_b": {k: expected[k] for k in COUNT_KEYS}})
    compare_counts(expected, receipt)
    receipt["stats"]["view_b"]["targets"] -= 1
    with pytest.raises(ValueError, match="targets"):
        compare_counts(expected, receipt)


def test_incomplete_receipt_rejected():
    expected = recount([(f, 0) for f in range(20)])
    with pytest.raises(ValueError, match="Incomplete"):
        compare_counts(expected, dict(queries=13, next_query=12, stats={}))
