import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from src.world_model.m3w_temporal_intervention import candidates, selections, policy_arm, second_difference


def test_endpoints_and_exact_distance_matching():
    rng = np.random.default_rng(413)
    b, p = rng.normal(size=(2, 100, 12, 2))
    q, alpha = candidates(b, p)
    np.testing.assert_array_equal(q['ramp'][:, 0], b[:, 0])
    np.testing.assert_allclose(q['ramp'][:, -1], p[:, -1], atol=1e-14)
    dr = np.linalg.norm(q['ramp']-b, axis=-1).mean(1)
    du = np.linalg.norm(q['uniform']-b, axis=-1).mean(1)
    np.testing.assert_allclose(dr, du, rtol=1e-12, atol=1e-14)
    assert np.all((alpha>=0) & (alpha<=1))


def test_zero_and_first_step_only_disagreement():
    b = np.zeros((3, 12, 2)); p = b.copy(); p[1, 0] = 1; p[2, -1] = 1
    q, alpha = candidates(b, p)
    for a in q:
        np.testing.assert_array_equal(q[a][:2], b[:2])
    np.testing.assert_array_equal(alpha, [0, 0, 1])


def test_rotation_translation_scale_equivariance():
    rng = np.random.default_rng(12); b, p = rng.normal(size=(2, 4, 12, 2))
    rot = np.array([[0., -1.], [1., 0.]])
    q, a = candidates(b, p)
    other, oa = candidates(13*(b@rot)+42, 13*(p@rot)+42)
    np.testing.assert_allclose(a, oa, atol=1e-14)
    for arm in q:
        np.testing.assert_allclose(other[arm], 13*(q[arm]@rot)+42, atol=1e-12)


@pytest.mark.parametrize('bad', [np.zeros((2, 8, 2)), np.full((2, 12, 2), np.nan)])
def test_invalid_forecasts_rejected(bad):
    with pytest.raises(ValueError):
        candidates(bad, bad)


def test_no_target_or_future_mask_interface():
    assert list(inspect.signature(candidates).parameters) == ['baseline', 'neural']
    assert list(inspect.signature(selections).parameters) == ['ramp', 'uniform', 'past', 'distance', 'ids']


def test_crossover_counts_and_past_stop():
    r = np.array([[1., .01], [1., .01], [1., 1.], [1., 1.]])
    u = r[::-1].copy(); past = np.zeros((4, 8, 2)); past[:, -1, 0] = 1
    d = np.ones(4); ids = np.arange(4); past[0] = 0
    out = selections(r, u, past, d, ids)
    assert out['ramp_strict'].sum() == out['uniform_matched'].sum() == 1
    np.testing.assert_array_equal(out['ramp_at_uniform'], out['uniform_strict'])
    np.testing.assert_array_equal(out['uniform_at_ramp'], out['ramp_strict'])
    assert not any(v[0] for v in out.values())
    for name in out:
        assert policy_arm(name) in ('ramp', 'uniform')
    with pytest.raises(ValueError):
        policy_arm('posthoc_winner')


def test_smoothness_has_no_seconds_or_zero_denominator():
    past = np.zeros((2, 8, 2)); past[:, :, 0] = np.arange(-7, 1)
    future = np.zeros((2, 12, 2)); future[:, :, 0] = np.arange(1, 13)
    np.testing.assert_array_equal(second_difference(past, future, np.ones(2)), np.zeros(2))
    future[1, 3, 0] += 4
    assert second_difference(past, future, np.ones(2))[1]>0


def test_frozen_protocol():
    from scripts.run_m3w_temporal_intervention import validate_config
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/m3w_temporal_intervention_v1.json').read_text())
    parent = json.loads((root/'configs/m3w_prefix_cost_v1.json').read_text())
    validate_config(cfg, parent)
    for k in ('threshold_search', 'closed_role_readout', 'deployment'):
        with pytest.raises(ValueError):
            validate_config(dict(cfg, **{k:True}), parent)


def test_verifier_summary_shape_and_corrupted_interval():
    from scripts.verify_m3w_temporal_intervention import verify_contrasts
    from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
    sites = ['a', 'b']; a, b = [2., 3.], [1., 1.]
    def summary(g):
        return {'ADE': {'by_scene':{s:{'gain_percent':x} for s,x in zip(sites,g)}}}
    r = dict(summaries=dict(ramp_strict=summary(a), uniform_strict=summary(b)),
             contrasts=dict(uniform_strict=paired_scene_contrast(a,b)))
    verify_contrasts(r, {}, sites)
    r['contrasts']['uniform_strict']['ci95_pp'][0] += 1
    with pytest.raises(AssertionError):
        verify_contrasts(r, {}, sites)


def test_separate_convex_formula_preserves_coincidence():
    from scripts.verify_m3w_temporal_intervention import manual_candidates
    b = np.random.default_rng(21).normal(size=(20, 12, 2))
    p = b.copy(); p[10:, 0] += 1
    expected, alpha = manual_candidates(b, p)
    actual, aa = candidates(b, p)
    for arm in actual:
        np.testing.assert_array_equal(expected[arm], b)
        np.testing.assert_array_equal(actual[arm], b)
    np.testing.assert_array_equal(alpha, aa)
