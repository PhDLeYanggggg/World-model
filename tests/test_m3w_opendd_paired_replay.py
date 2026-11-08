from copy import deepcopy

import pytest

from src.reproduction import m3w_opendd_paired_replay as replay


def packet():
    cell = {"effects": {"matched": 2.}, "counts": {"control": {"edits": 4, "views": 5, "risk_pass": 2}}}
    return dict(independent_confirmation=False, risk_cap=.02,
                studies={"control": [[deepcopy(cell) for _ in range(3)] for _ in range(7)]})


def test_recompute_and_check_contrast_and_counts():
    p = packet()
    z = replay.recompute(p)
    assert z["control"]["counts"]["control"]["edits"] == 84
    assert z["control"]["effects"]["matched"]["interval95"] == [2., 2.]
    expected = deepcopy(z)
    expected["control"]["counts"]["control"]["edits"] += 1
    with pytest.raises(ValueError, match="counts"):
        replay.check(z, expected)


def test_reject_confirmation_and_missing_seed():
    p = packet()
    p["independent_confirmation"] = True
    with pytest.raises(ValueError, match="development"):
        replay.recompute(p)
    p["independent_confirmation"] = False
    p["studies"]["control"][1].pop()
    with pytest.raises(ValueError, match="three paired seeds"):
        replay.recompute(p)
