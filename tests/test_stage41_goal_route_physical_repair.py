import numpy as np

from src import stage41_goal_route_physical_repair as gr


def test_angle_between_signed():
    a = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    b = np.asarray([[0.0, 1.0], [0.0, -1.0]])
    out = gr._angle_between(a, b)
    assert out[0] > 0
    assert out[1] < 0


def test_route_metrics_beats_majority_when_predictions_are_correct():
    labels = {"route": np.asarray([0, 1, 1, 2]), "domain": np.asarray(["a", "a", "b", "b"])}
    logits = np.full((4, len(gr.ROUTE_NAMES)), -5.0)
    logits[np.arange(4), labels["route"]] = 5.0
    out = gr._route_metrics({"route_logits": logits}, labels)
    assert out["top1"] == 1.0
    assert out["top1"] > out["majority_top1"]


def test_class_weights_are_finite_for_missing_classes():
    weights = gr._class_weights(np.asarray([0, 0, 1, 1, 1]))
    assert weights.shape == (len(gr.ROUTE_NAMES),)
    assert np.isfinite(weights).all()
