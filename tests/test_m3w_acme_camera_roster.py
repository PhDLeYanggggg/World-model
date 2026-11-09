import pytest

from scripts.audit_m3w_acme_roster_ceiling import support_counts
from scripts.run_m3w_acme_camera_roster import OBS, PRED, STRIDE, roster_counts, scheduled_starts


@pytest.mark.parametrize("frame_count,factor", [(23349, 1), (23370, 6)])
def test_schedule_is_disjoint_and_fits_camera_clock(frame_count, factor):
    starts = scheduled_starts(frame_count, factor)
    assert len(starts) == 12
    assert starts[0] == 0
    assert (starts[-1] + (OBS + PRED - 1) * STRIDE) * factor < frame_count
    assert all(b - a >= (OBS + PRED) * STRIDE for a, b in zip(starts, starts[1:]))
    with pytest.raises(ValueError, match="Insufficient"):
        scheduled_starts(12, factor)


def test_complete_pair_does_not_certify_whole_roster():
    starts = set(range(32, 80, 4))
    online = [{0, 1, 2, 3, 4}] * OBS
    pairs = [{(0, "a"), (1, "b"), (2, "c"), (3, "d")}] * OBS
    targets = {"a": starts, "b": starts, "c": {32}, "d": {28}}
    result = roster_counts(online, pairs, targets, 0)
    assert result == dict(online=5, stable=4, complete=2, partial=1, absent=1,
                          unmatched=1, multiagent=True, complete_pair=True,
                          whole_multiagent_roster=False)
    assert roster_counts([{0, 1}] * OBS, [{(0, "a"), (1, "b")}] * OBS,
                         targets, 0)["whole_multiagent_roster"]


def test_identity_swap_not_stable_even_with_full_label_presence():
    online = [{0, 1}] * OBS
    matches = [{(0, "a"), (1, "b")}] * (OBS - 1) + [{(1, "a"), (0, "b")}]
    ticks = set(range(32, 80, 4))
    result = roster_counts(online, matches, {"a": ticks, "b": ticks}, 0)
    assert result["online"] == 2 and result["stable"] == 0
    assert not result["complete_pair"]
    with pytest.raises(ValueError, match="one-to-one"):
        roster_counts(online, [{(0, "a"), (0, "b")}] * OBS, {}, 0)


def test_vacuous_roster_does_not_pass():
    result = roster_counts([set()] * OBS, [set()] * OBS, {}, 0)
    assert not result["multiagent"] and not result["whole_multiagent_roster"]


def test_reference_and_detection_limits_are_not_confused():
    past, future = {0, 1}, {2, 3}
    labels = {"a": past | future, "b": past | future, "c": past, "d": {0, 2, 3}}
    result = support_counts(past, future, labels, {"a", "c"})
    assert result["reference_past"] == 3
    assert result["reference_complete_pair"] and result["framewise_past_pair"]
    assert not result["framewise_complete_pair"]
    with pytest.raises(ValueError, match="every observed tick"):
        support_counts(past, future, labels, {"d"})
