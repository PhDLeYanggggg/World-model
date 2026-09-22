import numpy as np
import pytest

from scripts.audit_m3w_harm_concentration import mass_concentration, overlap_components, region_summary


def test_mass_groups_before_ranking_and_deterministic_ties():
    out = mass_concentration([2, 2, 1, 3], ["a", "a", "b", "c"])
    assert out["groups"] == 3 and out["top1_share"] == .5
    assert out["groups_for_half_mass"] == 1
    assert out["inverse_herfindahl"] == 64 / 26
    tied = mass_concentration([1, 1], ["b", "a"])
    assert tied["top5"][0]["key"] == "a"


def test_zero_mass_and_empty_population_do_not_claim_concentration():
    for out in (mass_concentration([0, 0], ["a", "b"]), mass_concentration([], [])):
        assert out["top1_share"] is None and out["inverse_herfindahl"] is None
        assert out["groups_for_half_mass"] == 0 and not out["top5"]


@pytest.mark.parametrize("mass", [[-1], [np.inf], [np.nan]])
def test_invalid_mass_rejected(mass):
    with pytest.raises(ValueError): mass_concentration(mass, ["a"])


def test_overlap_connected_components_preserve_track_and_boundary():
    ids, c = overlap_components(np.array(["b", "a", "a", "a", "a"]),
        np.array([0, 500, 0, 228, 229]), np.ones(5, bool), raw_span=228)
    np.testing.assert_array_equal(ids, [2, 3, 4, 1, 0])
    np.testing.assert_array_equal(c, [0, 0, 0, 1, 2])


def test_overlap_is_support_connectivity_not_count_of_adjacent_rows():
    ids, c = overlap_components(np.array(["a"] * 4), np.array([0, 200, 400, 800]),
                               np.array([1, 1, 1, 0], bool), raw_span=228)
    np.testing.assert_array_equal(ids, [0, 1, 2]); np.testing.assert_array_equal(c, [0, 0, 0])


def test_duplicate_query_and_float_frame_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        overlap_components(np.array(["a", "a"]), np.array([0, 0]), np.ones(2, bool), raw_span=228)
    with pytest.raises(ValueError):
        overlap_components(np.array(["a"]), np.array([.5]), np.ones(1, bool), raw_span=228)


def test_partial_and_unknown_outcomes_remain_explicit():
    out = region_summary([1, 2, np.nan, 4], [3, 1, np.nan, 4], np.ones(4, bool),
        np.array([0, 1, 0, 1], bool), np.array(["a"]*4), np.array(["a:1"]*4), np.arange(4), 228)
    assert out["selected"] == 4 and out["supported"] == 3
    assert out["partial_supported"] == 1 and out["unknown"] == 1
    assert out["harmful"] == out["beneficial"] == out["tied"] == 1
    assert out["gross_harm_sum"] == out["partial_supported_harm_sum"] == 2
    assert out["complete_harm_sum"] == 0 and out["net_error_reduction_sum"] == -1


@pytest.mark.parametrize("candidate,complete", [([np.inf], [False]), ([np.nan], [True]), ([-1], [True])])
def test_invalid_error_support_rejected(candidate, complete):
    with pytest.raises(ValueError):
        region_summary([1], candidate, np.ones(1, bool), np.array(complete, bool),
                       np.array(["a"]), np.array(["a:1"]), np.array([0]), 228)
