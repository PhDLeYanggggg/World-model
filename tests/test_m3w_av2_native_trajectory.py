import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import run_m3w_av2_native_trajectory as av2


def history_rows(agent, y=0., timesteps=av2.OBS, **extra):
    return [dict(track_id=agent, timestep=t, city="PIT", observed=True,
                 position_x=float(t), position_y=y, **extra) for t in timesteps]


def test_roster_is_past_only_deterministic_and_does_not_use_categories():
    rows = history_rows("incomplete", timesteps=av2.OBS[:-1])
    for k in range(10):
        rows += history_rows(str(k), float(k), object_category=k % 4, velocity_x=10000.)
    city, eligible, ids, positions = av2.past_roster(rows)
    assert (city, eligible, len(ids)) == ("PIT", 10, av2.MAX_ROSTER)
    assert "incomplete" not in ids
    for row in rows:
        row["object_category"] = -1
        row["velocity_x"] = -10000.
    again = av2.past_roster(rows[::-1])
    np.testing.assert_array_equal(again[2], ids)
    np.testing.assert_array_equal(again[3], positions)
    with pytest.raises(ValueError, match="Non-observed"):
        av2.past_roster(rows + history_rows("future", timesteps=av2.FUT[:1]))
    with pytest.raises(ValueError, match="Duplicate"):
        av2.past_roster(rows + [rows[0]])


def test_future_missing_steps_remain_masked_on_the_frozen_roster(tmp_path):
    rows = history_rows("a", object_type="pedestrian") + history_rows("b", 1., object_type="vehicle")
    _, _, ids, h = av2.past_roster(rows)
    future = history_rows("a", timesteps=(50, 52, 61), object_type="pedestrian")
    for row in future:
        row["observed"] = False
    # A future-only actor cannot enter scoring or alter the roster.
    future += history_rows("new", timesteps=av2.FUT, object_type="vehicle")
    path = tmp_path/"native.parquet"
    pq.write_table(pa.Table.from_pylist(rows+future), path)
    y, mask, types = av2.targets(path, ids, h)
    assert types.tolist() == ["pedestrian", "vehicle"]
    assert np.flatnonzero(mask[0]).tolist() == [0, 2, 11]
    assert not mask[1].any()
    np.testing.assert_allclose(y[0, mask[0], 0], [1., 3., 12.])
    np.testing.assert_array_equal(y[~mask], 0.)


def test_train_uri_matches_scenario_id_only():
    a = "0000b0f9-99f9-4a1f-a231-5be9e4c523f7"
    b = "0000b6ab-e100-4f6b-aee8-b520b57c0530"
    assert av2.KEY.fullmatch(f"{av2.PREFIX}{a}/scenario_{a}.parquet")[1] == a
    assert av2.KEY.fullmatch(f"{av2.PREFIX}{a}/scenario_{b}.parquet") is None
    assert av2.KEY.fullmatch(f"{av2.PREFIX}{a}/log_map_archive_{a}.json") is None
    assert av2.KEY.fullmatch(f"{av2.PREFIX.replace('train', 'test')}{a}/scenario_{a}.parquet") is None


def test_city_interval_averages_seeds_and_scenarios_before_city_resampling():
    rows = [dict(city="A", clip="a", value=x) for x in [1., 3., 5.]]
    rows += [dict(city="A", clip="b", value=9.), dict(city="B", clip="c", value=20.)]
    got = av2.interval(rows, lambda row: row["value"])
    assert got["cities"] == 2
    assert got["resampling_unit"] == "city"
    assert got["site_mean"] == 13.
    assert av2.interval(rows, lambda row: None) is None
