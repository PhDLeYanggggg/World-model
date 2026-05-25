import numpy as np

from src import stage41_joint_route_conditioned_world_state as jr


def test_as_ft_pred_keeps_required_trajectory_keys():
    pred = {
        "waypoint_delta": np.zeros((2, 4, 2)),
        "traj_risk": np.zeros(2),
        "interaction": np.zeros(2),
        "occupancy": np.zeros(2),
        "physical": np.ones(2),
        "route_logits": np.zeros((2, 6)),
    }
    out = jr._as_ft_pred(pred)
    assert sorted(out) == ["interaction", "occupancy", "physical", "traj_risk", "waypoint_delta"]


def test_aux_metrics_has_route_and_binary_heads():
    pred = {
        "route_logits": np.asarray([[3, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0]], dtype=float),
        "physical": np.asarray([0.1, 0.9]),
        "interaction": np.asarray([0.2, 0.8]),
        "occupancy": np.asarray([0.3, 0.7]),
    }
    labels = {
        "route": np.asarray([0, 1]),
        "physical": np.asarray([False, True]),
        "interaction": np.asarray([False, True]),
        "occupancy": np.asarray([False, True]),
        "domain": np.asarray(["a", "a"]),
    }
    out = jr._aux_metrics(pred, labels)
    assert out["route"]["top1"] == 1.0
    assert out["physical_challenge"]["auroc"] == 1.0
