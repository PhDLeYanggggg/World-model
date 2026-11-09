import numpy as np
import pytest

from scripts.run_m3w_okutama_past_camera import roster, summarize
from src.world_model.m3w_causal_camera_motion import camera_maps, compare, transport


def test_camera_transport_recovers_agent_motion_in_last_image_plane():
    p = np.repeat(np.eye(3)[None], 7, axis=0)
    p[:, 0, 2] = 5.
    h = np.zeros((2, 8, 2))
    h[:, :, 0] = 7.*np.arange(8)
    h[:, :, 1] = np.arange(2)[:, None]*20.
    obs, future = camera_maps(p, [640., 360.])
    stable = transport(h, obs)
    np.testing.assert_allclose(np.diff(stable[..., 0], axis=1), 2.)
    out = compare(h, p, [640., 360.], [True]*7)
    t = np.arange(1., 13.)
    np.testing.assert_allclose(out["raw_cv"][:, :, 0], np.tile(7.*t, (2, 1)))
    np.testing.assert_allclose(out["stable_cv"][:, :, 0], np.tile(2.*t, (2, 1)))
    np.testing.assert_allclose(future[:, 0, 2], 5.*np.cumsum(.9**np.arange(12)))
    np.testing.assert_allclose(out["camera_cv"][:, :, 0],
                               np.tile(2.*t+5.*np.cumsum(.9**np.arange(12)), (2, 1)))


def test_incomplete_fit_has_exact_raw_fallback():
    rng = np.random.default_rng(2)
    h = rng.normal(size=(2, 8, 2))
    out = compare(h, None, None, [True]*6+[False])
    for method in ("stable_cv", "camera_cv"):
        np.testing.assert_array_equal(out[method], out["raw_cv"])
    for method in ("stable_ols4", "camera_ols4"):
        np.testing.assert_array_equal(out[method], out["raw_ols4"])
    with pytest.raises(ValueError, match="Seven accepted"):
        camera_maps(np.broadcast_to(np.diag([-1., 1., 1.]), (7, 3, 3)), [0., 0.])


def test_roster_reads_only_past_starts(tmp_path, monkeypatch):
    path = tmp_path/"past.npz"
    np.savez(path, start=[0, 200, 200, 400, 400], future=np.arange(11))
    monkeypatch.setattr("scripts.run_m3w_okutama_past_camera.study.checked", lambda x: path)
    result = roster({"clips": {"2.1.9": {"nongenerated_10": {}}}})
    assert result["2.1.9"]["past"] == list(range(200, 280, 10))
    assert result["2.1.9"]["scenario"] == "1.9"


def test_camera_views_are_averaged_within_scenario_before_bootstrap():
    from src.world_model.m3w_causal_camera_motion import ARMS

    def row(group, gain, pair):
        return dict(scenario=group, arms={arm: [1., 2., gain, 3., pair, 0.] for arm in ARMS})
    result = summarize([row("a", 0., 4.), row("a", 4., 0.), row("b", 10., None)])
    gain = result["camera_cv"]["lower_gain_vs_matched_raw_pp"]
    known = result["camera_cv"]["pair_gain_known_px"]
    assert gain["site_mean"] == 6. and gain["scenario_groups"] == 2
    assert known["site_mean"] == 2. and known["scenario_groups"] == 1
