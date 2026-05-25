import numpy as np

from src import stage41_teacher_guided_proposal as tgp


def test_policy_switch_teacher_and_gain_mode():
    pred = {
        "teacher_prob": np.asarray([0.9, 0.9, 0.2]),
        "gain": np.asarray([0.5, -0.1, 0.5]),
        "harm": np.asarray([0.1, 0.1, 0.1]),
        "uncertainty": np.asarray([0.1, 0.1, 0.1]),
    }
    switch = tgp._policy_switch(
        pred,
        {"mode": "teacher_and_gain", "teacher_min": 0.5, "gain_min": 0.0, "harm_max": 0.5, "uncertainty_max": 0.5},
    )
    assert switch.tolist() == [True, False, False]


def test_policy_switch_teacher_or_gain_mode():
    pred = {
        "teacher_prob": np.asarray([0.9, 0.1, 0.1]),
        "gain": np.asarray([-0.1, 0.5, 0.5]),
        "harm": np.asarray([0.1, 0.1, 0.9]),
        "uncertainty": np.asarray([0.1, 0.1, 0.1]),
    }
    switch = tgp._policy_switch(
        pred,
        {"mode": "teacher_or_gain", "teacher_min": 0.5, "gain_min": 0.0, "harm_max": 0.5, "uncertainty_max": 0.5},
    )
    assert switch.tolist() == [True, True, False]


def test_selected_residual_trial_handles_missing_report(monkeypatch):
    monkeypatch.setattr(tgp, "read_json", lambda *_args, **_kwargs: {})
    ckpt, clip = tgp._selected_residual_trial()
    assert ckpt is None
    assert clip is None
