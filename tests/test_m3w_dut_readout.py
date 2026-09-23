import numpy as np
import pytest

from src.evaluation.m3w_dut_readout import PrefixRecording, score_arrays, empty_statistics, accumulate, fixed_baselines
from src.world_model.m3w_frozen_policy_chain import decide


def recording():
    points = np.array([[t, a, a*3.+.2*t, .03*t] for a in range(3) for t in range(24)
                       if not (a == 1 and t >= 10) and not (a == 2 and t < 7)])
    return points, np.lexsort((points[:, 1], points[:, 0]))


def test_past_population_retains_no_future_and_short_context():
    p, order = recording()
    r = PrefixRecording(p, order, "fixture")
    assert r.queries.tolist() == list(range(7, 24))
    s = r.inputs(7)
    assert s.agent_ids.tolist() == [0, 1, 2]
    assert s.target_ids.tolist() == [0, 1]
    y, m = r.labels(7, s.target_ids)
    assert m[0].all() and m[1].sum() == 2
    assert r.inputs(23).target_ids.tolist() == [0, 2]
    assert not r.labels(23, np.array([0, 2]))[1].any()


def test_future_mutation_does_not_change_input_or_membership():
    p, order = recording()
    a = PrefixRecording(p, order, "fixture").inputs(7)
    changed = p.copy(); changed[changed[:, 0] > 7, 2:] += 1e6
    b = PrefixRecording(changed, order, "fixture").inputs(7)
    for key in ("geometry", "target_ids", "agent_ids", "baseline", "cv_valid"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_missing_label_bound_contains_actual_future_gain():
    rng = np.random.default_rng(17)
    b, p, y = (rng.normal(size=(13, 12, 2)) for _ in range(3))
    m = rng.random((13, 12)) > .4
    s = score_arrays(b, p, y, m, np.ones(13), easy_cut=.1, hard_cut=.5)
    actual = (np.linalg.norm(b-y, axis=2)-np.linalg.norm(p-y, axis=2)).mean(1)
    assert np.all(s["gain_lower"] <= actual+1e-12)
    assert np.all(actual <= s["gain_upper"]+1e-12)
    y[~m] = np.nan
    ss = score_arrays(b, p, y, m, np.ones(13), easy_cut=.1, hard_cut=.5)
    np.testing.assert_array_equal(s["gain_lower"], ss["gain_lower"])


def test_floor_has_zero_gain_interval_even_with_unknown_future():
    b = np.ones((2, 12, 2)); y = np.full_like(b, np.nan); m = np.zeros((2, 12), bool)
    s = score_arrays(b, b, y, m, np.ones(2), easy_cut=.1, hard_cut=.5)
    assert not s["complete"].any() and not s["easy"].any()
    assert not s["gain_lower"].any() and not s["gain_upper"].any()


def test_native_and_normalized_metrics_and_zero_easy_are_distinct():
    b = np.zeros((3, 12, 2)); y = b.copy(); y[0, :, 0] = 2; y[1, :, 0] = .01
    p = b.copy(); p[2, :, 0] = 1
    s = score_arrays(b, p, y, np.ones((3, 12), bool), np.array([100., 1., 1.]), easy_cut=.03, hard_cut=.1)
    assert s["easy"].tolist() == [True, True, True]
    assert s["zero"].tolist() == [False, False, True]
    assert s["ade"].tolist() == pytest.approx([2., .01, 1.])
    assert s["normalized_ade"].tolist() == pytest.approx([.02, .01, 1.])


@pytest.mark.parametrize("bad", ["prediction", "scale", "mask"])
def test_invalid_score_inputs_refused(bad):
    b = np.zeros((1, 12, 2)); p = b.copy(); m = np.ones((1, 12), bool); scale = np.ones(1)
    if bad == "prediction": p[:] = np.nan
    elif bad == "scale": scale[:] = 0
    else: m = m.astype(float)
    with pytest.raises(ValueError):
        score_arrays(b, p, b, m, scale, easy_cut=.1, hard_cut=.5)


def test_full_scoring_counts_missing_cases_and_seven_causal_controls():
    p, order = recording(); r = PrefixRecording(p, order, "fixture"); scene = r.inputs(7)
    baseline = scene.geometry[:, 332:356].reshape(-1, 12, 2)
    out = decide(scene, baseline, np.zeros((2, 2)), cost_scale=1.)
    y, mask = r.labels(7, scene.target_ids)
    agent_table = {str(a): {"agent_type":"pedestrian"} for a in range(3)}
    stats = accumulate(empty_statistics(), scene, out, y, mask, agent_table,
        easy_cut=.02349, hard_cut=.1679, controls=fixed_baselines(r, scene))
    assert stats["targets"] == 2 and stats["complete"] == 1
    assert stats["visible"] == 3 and stats["unknown_cv_context"] == 1
    assert len(stats["baseline_controls"]) == 7
    assert stats["arms"]["floor"]["gain_lower_sum"] == 0
    assert stats["arms"]["half_joint"]["slices"]["all"]["n"] == 1


def test_empty_targets_are_reported_not_dropped():
    p = np.array([[t, t, t*.3, 0.] for t in range(9)])
    r = PrefixRecording(p, np.arange(9), "fixture"); s = r.inputs(7)
    out = decide(s, np.empty((0, 12, 2)), np.empty((0, 2)), cost_scale=1.)
    y, m = r.labels(7, s.target_ids)
    result = accumulate(empty_statistics(), s, out, y, m, {}, easy_cut=.1, hard_cut=.5)
    assert result["queries"] == 1 and result["targets"] == 0
    assert result["visible"] == 1 and result["complete"] == 0
