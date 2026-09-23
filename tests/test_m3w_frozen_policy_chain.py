from dataclasses import replace
from types import SimpleNamespace
import itertools

import numpy as np
import pytest

from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.world_model.m3w_frozen_policy_chain import SceneInput, scene_from_prefix, decide


def prefix(short=False):
    points = [[t, agent, t*.4 + agent*2., .1*t + agent*.3]
              for agent in range(4) for t in range(8) if not (short and agent == 3 and t < 7)]
    return ExternalPrefixAdapter(np.asarray(points), query_frame=7, recording_id="synthetic")


def test_input_identity_and_incomplete_visible_context():
    scene = scene_from_prefix(prefix(short=True))
    scene.validate()
    assert scene.agent_ids.tolist() == [0, 1, 2, 3]
    assert scene.target_ids.tolist() == [0, 1, 2]
    assert scene.cv_valid.tolist() == [True, True, True, False]
    assert scene.geometry.shape == (3, 476)
    with pytest.raises(TypeError):
        SceneInput(**scene.__dict__, future_target=np.zeros((3, 12, 2)))
    with pytest.raises(ValueError):
        replace(scene, target_ids=np.array([0, 0, 2])).validate()


def test_no_target_history_does_not_invent_supported_forecasts():
    adapter = ExternalPrefixAdapter(np.array([[7, 1, 2., 3.]]), query_frame=7, recording_id="short")
    scene = scene_from_prefix(adapter)
    out = decide(scene, np.empty((0, 12, 2)), np.empty((0, 2)), cost_scale=2.)
    assert out["reason"] == "no_supported_neural_targets"
    assert not out["forecast_valid"].any()
    assert not any(v.any() for v in out["choices"].values())


def test_unsigned_unsorted_ids_cannot_bypass_identity_order_check():
    scene = scene_from_prefix(prefix())
    with pytest.raises(ValueError, match="IDs"):
        replace(scene, agent_ids=np.array([3, 2, 1, 0], dtype=np.uint64)).validate()


@pytest.mark.parametrize("change", ["future_time", "rotation", "origin", "scale", "cv", "frame_step", "mislabelled_stride"])
def test_incompatible_geometry_is_rejected(change):
    scene = scene_from_prefix(prefix())
    if change == "future_time":
        g = scene.geometry.copy(); g[:, 23] = 1
        scene = replace(scene, geometry=g)
    elif change == "rotation":
        scene = replace(scene, rotations=np.zeros_like(scene.rotations))
    elif change == "origin":
        scene = replace(scene, origins=scene.origins+1)
    elif change == "scale":
        scene = replace(scene, scales=-scene.scales)
    elif change == "cv":
        scene = replace(scene, baseline=scene.baseline+10)
    elif change == "frame_step":
        scene = replace(scene, frame_step=0)
    else:
        scene = replace(scene, frame_step=12)
    with pytest.raises(ValueError):
        scene.validate()


def test_inherited_stop_and_harm_guards_and_matching():
    scene = scene_from_prefix(prefix())
    b = scene.geometry[:, 332:356].reshape(-1, 12, 2)
    candidate = b+.03
    costs = np.tile([.1, .002], (4, 1))
    out = decide(scene, candidate, costs, cost_scale=2.)
    assert out["eligible"].all()
    for arm in ("half_independent", "half_unary", "half_joint"):
        assert out["choices"][arm].sum() == 2
    assert out["budgets"]["half"]["matched"]
    costs[0, 1] = .02
    out = decide(scene, candidate, costs, cost_scale=2.)
    assert not out["eligible"][0]
    stationary = np.asarray([[t, 0, 0., 0.] for t in range(8)])
    stopped = scene_from_prefix(ExternalPrefixAdapter(stationary, query_frame=7, recording_id="stop"))
    out = decide(stopped, np.ones((1, 12, 2)), np.array([[.1, 0.]]), cost_scale=2.)
    assert not out["eligible"].any()


def test_solver_failure_falls_back_and_is_not_reported_as_matched(monkeypatch):
    monkeypatch.setattr("src.world_model.m3w_interaction_controls.milp", lambda **kw: SimpleNamespace(success=False))
    scene = scene_from_prefix(prefix())
    b = scene.geometry[:, 332:356].reshape(-1, 12, 2)
    out = decide(scene, b+.03, np.tile([.1, .002], (4, 1)), cost_scale=2.)
    assert out["choices"]["half_independent"].sum() == 2
    assert not out["choices"]["half_joint"].any()
    assert not out["budgets"]["half"]["matched"]
    assert not out["budgets"]["half"]["controls"]["joint"]["solver_optimal"]


def test_bound_invalid_output_falls_back_but_other_targets_remain_visible():
    scene = scene_from_prefix(prefix())
    b = scene.geometry[:, 332:356].reshape(-1, 12, 2)
    candidate = b+.01
    candidate[0] = np.nan
    costs = np.tile([.01, 0.], (4, 1)); costs[1] = [1e8, 0]
    out = decide(scene, candidate, costs, cost_scale=2.)
    assert out["model_output_supported"].tolist() == [False, False, True, True]
    assert not out["eligible"][:2].any()
    np.testing.assert_array_equal(out["candidate"][:2], out["baseline"][:2])
    assert len(out["agent_ids"]) == 4 and out["forecast_valid"].all()


def test_joint_choices_agree_with_exhaustive_objective():
    scene = scene_from_prefix(prefix())
    b = scene.geometry[:, 332:356].reshape(-1, 12, 2)
    candidate = b+np.array([.02, -.04])
    costs = np.array([[.05, .001], [.03, .002], [.04, .001], [.06, .002]])
    out = decide(scene, candidate, costs, cost_scale=2.)
    from src.world_model.m3w_native_joint_controls import make_problem
    from src.world_model.m3w_interaction_controls import decompose_pair_objective, _describe
    p, _ = make_problem(baseline=out["baseline"], candidate=out["candidate"], current=scene.current,
        forecast_valid=scene.cv_valid, target_mask=np.ones(4, bool), eligible=out["eligible"],
        benefit=costs[:, 0], harm=costs[:, 1], scale=2., past_target_scales=scene.scales)
    p.max_interventions = 2; p.max_mean_predicted_harm = out["budgets"]["half"]["predicted_harm_budget"]
    parts = decompose_pair_objective(p); feasible = []
    for pair in itertools.combinations(range(4), 2):
        bits = np.zeros(4, bool); bits[list(pair)] = True
        result = _describe(p, bits, parts)
        if result["predicted_constraints_satisfied"]:
            feasible.append(result["full_objective"])
    selected = _describe(p, out["choices"]["half_joint"], parts)["full_objective"]
    assert selected == pytest.approx(min(feasible), abs=1e-9)
