import numpy as np
import torch

from scripts.audit_m3w_one_edit_guard import guard
from scripts.run_m3w_one_edit_stopping import donor_select, plans, stopping


def test_stopping_never_reads_beyond_the_first_accepted_window():
    def score():
        yield .2
        yield .6
        raise AssertionError("Future window was read")

    assert stopping(score(), 0, .6) == 1


def test_no_last_window_fallback_and_no_edited_count_increase():
    windows = [(np.array([11, 12]), np.array([0, 2]), np.array([.3, .7]))]
    got = plans(windows, {"immediate": (0, 0.), "fixed": (2, 0.), "high": (0, .8), "ok": (0, .7)})
    assert got == {"immediate": {0: (11, 0, 0)}, "fixed": {}, "high": {}, "ok": {0: (12, 2, 1)}}


def test_donor_choice_uses_macro_rates_and_frozen_grid_tie_order():
    score = lambda n, total, gain=0.: dict(dual_positive=n, views=total, vs_cv=gain)
    rows = [{"threshold_0": score(2, 2), "threshold_1": score(0, 2)},
            {"threshold_0": score(0, 100), "threshold_1": score(100, 100)}]
    best, values = donor_select(rows, "threshold")
    assert best == "threshold_0" and values["threshold_0"] == values["threshold_1"]
    rows[1]["threshold_1"]["vs_cv"] = .1
    assert donor_select(rows, "threshold")[0] == "threshold_1"


def test_guard_requires_both_donor_outcomes_not_only_pass_rate():
    grid = {"threshold_0": [.1, .2], "threshold_1": [.2, .19], "threshold_2": [.11, .21]}
    assert guard(grid) == ("threshold_2", ["threshold_0", "threshold_2"])
    del grid["threshold_2"]
    assert guard(grid) == ("threshold_0", ["threshold_0"])
