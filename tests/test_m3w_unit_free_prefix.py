import numpy as np
import pytest

from src.data_unification.m3w_unit_free_prefix import (
    UnitFreePrefixAdapter, unit_free_cost_features, require_unit_free_checkpoint, SCHEMA,
)
from src.data_unification.m3w_causal_recordings import restore_scene_rollouts


def prefix():
    return np.array([[t, a, .002*t*(a+1), .03*a] for a in range(3) for t in range(8)])


@pytest.mark.parametrize("factor", [.01, 1., 100.])
def test_scale_translation_equivalence_and_inverse(factor):
    p = prefix()
    r = UnitFreePrefixAdapter(p, query_frame=7, recording_id="synthetic")
    g, s = r.geometry_batch()
    changed = p.copy()
    changed[:, 2:] = changed[:, 2:]*factor + [13., -9.]
    other = UnitFreePrefixAdapter(changed, query_frame=7, recording_id="synthetic")
    h, u = other.geometry_batch()
    np.testing.assert_allclose(g, h, rtol=1e-5, atol=1e-5)
    b = g[:, 332:356].reshape(-1, 12, 2)
    x, d, same = unit_free_cost_features(g, b + .125)
    y, e, _ = unit_free_cost_features(h, h[:, 332:356].reshape(-1, 12, 2)+.125)
    np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d, e, atol=1e-6)
    assert not same.any() and x.shape == (3, 355)
    restored = restore_scene_rollouts(s, {a["agent_id"]: v for a, v in zip(s["agents"], b)})
    raw = r.restore_outer(restored["xy_dataset_local"])
    expected = np.array([[[.002*(7+t)*(a+1), .03*a] for t in range(1, 13)] for a in range(3)])
    np.testing.assert_allclose(raw, expected, atol=1e-7)


def test_future_rows_rejected_not_silently_normalized():
    p = prefix()
    with pytest.raises(ValueError, match="Future"):
        UnitFreePrefixAdapter(np.r_[p, [[8, 0, 1e10, 1e10]]], query_frame=7, recording_id="test")


def test_degenerate_prefix_refused_but_stationary_agent_context_retained():
    p = prefix()
    p[:, 2:] = 0
    with pytest.raises(ValueError, match="Zero-extent"):
        UnitFreePrefixAdapter(p, query_frame=7, recording_id="test")
    p[p[:, 1] == 2, 2] = 1
    g, scene = UnitFreePrefixAdapter(p, query_frame=7, recording_id="test").geometry_batch()
    assert len(scene["agents"]) == 3 and np.isfinite(g).all()


def test_no_future_targets_or_old_checkpoint_contract():
    reader = UnitFreePrefixAdapter(prefix(), query_frame=7, recording_id="test")
    with pytest.raises(PermissionError):
        reader.get_labels(0)
    with pytest.raises(ValueError, match="incompatible"):
        require_unit_free_checkpoint({"feature_width": 356})
    require_unit_free_checkpoint(dict(feature_schema=SCHEMA, feature_width=355, target_unit="past_normalized"))


def test_incomplete_current_agent_not_removed_by_normalization():
    p = prefix()
    p = p[~((p[:, 1] == 2) & (p[:, 0] < 6))]
    g, scene = UnitFreePrefixAdapter(p, query_frame=7, recording_id="test").geometry_batch()
    assert len(g) == 2 and len(scene["observed_agent_ids"]) == 3
    assert scene["agents"][0]["inputs"]["neighbor_mask"][1].sum() == 2
