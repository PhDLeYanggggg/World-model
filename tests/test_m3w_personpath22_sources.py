import pytest

from scripts.audit_m3w_personpath22_sources import grid_support, source_groups, split_overlaps
from scripts.compare_m3w_personpath22_box_references import point_support
from scripts.run_m3w_personpath22_camera_probe import choose, coverage, schedule


def annotation(ticks_by_id):
    return {
        "metadata": {"data_path": "sample.mp4", "fps": 25, "number_of_frames": 200},
        "entities": [{"id": i, "blob": {"frame_idx": k * 5}, "time": 200 * k}
                     for i, ticks in ticks_by_id.items() for k in ticks],
    }


def test_clock_grid_keeps_missing_future_in_past_roster():
    full = list(range(0, 40, 2))
    row = grid_support(annotation({0: full, 1: full, 2: full[:8]}), "sample.mp4")
    assert row["clock_ok"]
    assert row["counts"] == dict(camera_grid_queries=1, reference_past=3, reference_complete=2,
                                 reference_pair_queries=1, complete_pair_queries=1,
                                 full_reference_roster_queries=0)
    data = annotation({0: full})
    data["entities"][0].update(blob={"frame_idx": 1}, time=40)
    rejected = grid_support(data, "sample.mp4")
    assert not rejected["clock_ok"] and rejected["off_grid_rows"] == 1
    assert rejected["clock_mismatch_rows"] == 0 and rejected["counts"] is None


def test_repeated_identity_at_same_native_frame_is_not_deduplicated():
    with pytest.raises(ValueError, match="repeated"):
        grid_support(annotation({0: [0, 0]}), "sample.mp4")


def test_detect_same_original_and_camera_across_official_roles():
    text = '''
virat_id_to_uid_and_urlid = {}
external_video_id_mapping = {
    "Meva": {"2018.time.school.G420_0-50.mp4": "a", "2018.time.school.G420_50-100.mp4": "b",
             "2018.other.school.G420_0-50.mp4": "c"},
    "PathTrack": {"_abcd_e_4_11": "x", "_abcd_e_12_22": "y"},
    "Virat": {k: v[0] for k,v in virat_id_to_uid_and_urlid.items()},
}
raise RuntimeError("do not execute this script")
'''
    result = split_overlaps(source_groups(text), {"train": ["a", "x"], "test": ["b", "c", "y"]})
    assert [x["source"] for x in result["original"]] == ["Meva", "PathTrack"]
    assert result["camera"] == [{"source": "Meva", "key": "G420", "train": ["a"], "test": ["b", "c"]}]


def test_camera_selection_does_not_inspect_reference_future_counts():
    rows = {uid: dict(clock_ok=True, in_video_zip=True, fps=24., frames=604, counts=values)
            for uid, values in (("b", {"future": 100}), ("a", {"future": 0}))}
    assert choose({"training_support": rows}) == "a"
    frames = schedule(24., 604)
    assert len(frames) == 3 and all(len(q) == 20 for q in frames)
    assert all(a[-1] < b[0] for a, b in zip(frames, frames[1:]))


def test_matcher_does_not_fill_missing_future_or_remove_online_track():
    ticks = list(range(0, 200, 10))
    refs = {t: [("a", (0, 0, 10, 10)), ("b", (20, 0, 30, 10))] for t in ticks[:8]}
    frames = [{"camera_frame": t, "rows": [[0, 0, 0, 10, 10, .9, 1], [1, 20, 0, 30, 10, .9, 1],
                                          [2, 40, 0, 50, 10, .9, 1]]} for t in ticks[:8]]
    result = coverage(frames, ticks[8:], refs, {"a": ticks, "b": ticks[:9]})
    assert result["online"] == 3 and result["unmatched"] == 1
    assert result["complete"] == 1 and result["partial"] == 1 and not result["complete_pair"]


def test_framewise_support_can_survive_identity_switch_but_not_label_dropout():
    ticks = list(range(20))
    ref = {t: [("a", (0, 0, 10, 10)), ("b", (20, 0, 30, 10))] for t in ticks[:8]}
    frames = [{"camera_frame": t, "rows": [[t % 2, 0, 0, 10, 10, .9, 1],
                                          [1 - t % 2, 20, 0, 30, 10, .9, 1]]} for t in ticks[:8]]
    id_ticks = {"a": set(ticks), "b": set(ticks[:8])}
    result = point_support(frames, ticks[8:], ref, id_ticks)
    assert result["framewise_pairs"] == 16 and result["complete_without_online_identity"] == 1
    assert coverage(frames, ticks[8:], ref, id_ticks)["stable"] == 0
