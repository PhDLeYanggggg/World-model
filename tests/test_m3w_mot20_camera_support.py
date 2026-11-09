import json

import numpy as np
import pytest

from scripts import run_m3w_mot20_camera_support as pilot
from scripts.evaluate_m3w_mot20_camera_support import zero_based_references
from scripts.replay_m3w_mot20_motion_diagnostic import json_counts
from scripts.run_m3w_mot20_camera_support import (
    LENGTH, OBS, PRED, check_clock, frame_id, match, parse_gt, starts, window_support,
)
from scripts.score_m3w_mot20_motion_diagnostic import summarize


def test_camera_schedule_never_uses_labels_and_has_no_overlap():
    slots = [{frame_id(start, step) for step in range(OBS + PRED)} for start in starts()]
    assert len(slots) == 13 and min(slots[0]) == 1 and max(slots[-1]) <= LENGTH
    assert all(max(a) < min(b) for a, b in zip(slots, slots[1:]))
    ini = "[Sequence]\nname=MOT20-02\nimDir=img1\nimExt=.jpg\nframeRate=25\nseqLength=2782\nimWidth=1920\nimHeight=1080"
    check_clock(ini)
    with pytest.raises(ValueError, match="clock"):
        check_clock(ini.replace("=25", "=30"))


def test_mot_reference_is_one_based_xywh_and_deduplicates_nothing():
    raw = b"1,5,4,6,10,20,1,1,0.9\n1,6,14,7,2,5,0,7,1\n"
    table = parse_gt(raw)
    np.testing.assert_array_equal(table[1][0, 2:6], [4, 6, 14, 26])
    np.testing.assert_array_equal(zero_based_references(raw)[1][0, 2:6], [3, 5, 13, 25])
    with pytest.raises(ValueError, match="duplicate"):
        parse_gt(raw + raw)
    with pytest.raises(ValueError, match="Malformed"):
        parse_gt(raw.replace(b"1,5", b"0,5"))


def test_stable_distractor_is_not_a_live_filter_or_an_unmatched_pass():
    data = np.array([[0, 0, 0, 4, 4, .8, 1], [1, 6, 0, 10, 4, .8, 1],
                     [2, 12, 0, 16, 4, .8, 1]])
    targets = np.array([[1, 5, 0, 0, 4, 4, 1, 1, 1], [1, 6, 6, 0, 10, 4, 1, 1, 1],
                        [1, 7, 12, 0, 16, 4, 0, 7, 1]])
    assert match(data, targets) == ({(0, 5), (1, 6)}, {(2, 7)})
    frames = [{"rows": data, "camera_frame": frame_id(1, k)} for k in range(OBS)]
    refs = {f["camera_frame"]: targets for f in frames}
    ticks = set(frame_id(1, k) for k in range(OBS + PRED))
    stats = window_support(frames, refs, {5: ticks, 6: ticks})
    assert stats["strict"]["complete_pair"] and not stats["strict"]["whole_multiagent_roster"]
    assert stats["ignore_only_diagnostic"]["whole_multiagent_roster"]
    assert stats["ignored"] == 1
    refs[frames[-1]["camera_frame"]] = targets[:2]
    assert window_support(frames, refs, {5: ticks, 6: ticks})["ignored"] == 0


def test_subthreshold_and_empty_matches_do_not_create_support():
    row = np.array([[0, 0, 0, 10, 10, .9, 1]])
    label = np.array([[1, 5, 5, 5, 15, 15, 1, 1, 1]])
    assert match(row, label) == (set(), set())
    assert match(row, label[:0]) == (set(), set())


def test_motion_summary_weights_windows_and_does_not_mistake_oracle_for_model():
    first = np.tile([[1., 2.], [3., 4.]], (6, 1, 1))
    second = np.array([[[4., 6.], [2., 1.]]])
    assert summarize([first, second])["status"] == "not_run_insufficient_complete_support"
    report = summarize([first, second, second])
    assert report["complete_histories"] == 8
    assert report["equal_window_mean_ade_fde"]["CV"] == [3., 14. / 3]
    assert report["per_target_endpoint_oracle_fde_px"] == 4. / 3
    assert report["ols4_lower_fde_windows"] == 2
    assert json.loads(json.dumps(json_counts(report))) == report


def test_range_transport_requires_partial_response_and_pinned_snapshot(monkeypatch):
    def reply(status="206 Partial Content", etag=pilot.ARCHIVE_VERSION):
        return (f"HTTP/1.1 {status}\r\nETag: {etag}\r\n"
                "Content-Range: bytes 0-2/5\r\n\r\nabc").encode()

    with pilot.MOTRangeArchive(pilot.URL, 5, byte_limit=4) as stream:
        monkeypatch.setattr(pilot.subprocess, "check_output", lambda _: reply())
        assert stream._fetch(0, 2) == b"abc" and stream.total == 3
        with pytest.raises(ValueError, match="budget"):
            stream._fetch(0, 2)
    for bad in (reply("200 OK"), reply(etag='"wrong"')):
        monkeypatch.setattr(pilot.subprocess, "check_output", lambda _: bad)
        with pilot.MOTRangeArchive(pilot.URL, 5) as stream:
            with pytest.raises(ValueError, match="response changed"):
                stream._fetch(0, 2)


def test_future_dropout_changes_scoreability_but_not_observed_support():
    data = np.array([[0, 0, 0, 4, 4, .8, 1], [1, 6, 0, 10, 4, .8, 1]])
    targets = np.array([[1, 5, 0, 0, 4, 4, 1, 1, 1], [1, 6, 6, 0, 10, 4, 1, 1, 1]])
    past = {frame_id(1, k) for k in range(OBS)}
    all_ticks = {frame_id(1, k) for k in range(OBS + PRED)}
    frames = [{"rows": data, "camera_frame": t} for t in sorted(past)]
    references = {t: targets for t in past}
    full = window_support(frames, references, {5: all_ticks, 6: all_ticks})["strict"]
    truncated = window_support(frames, references, {5: all_ticks, 6: past})["strict"]
    assert full["stable"] == truncated["stable"] == full["online"] == truncated["online"] == 2
    assert full["complete"] == 2 and truncated["complete"] == 1
