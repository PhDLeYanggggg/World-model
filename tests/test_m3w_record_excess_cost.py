import numpy as np
import pytest
import torch
from src.world_model.m3w_record_excess_cost import record_excess, training_packet


def data(records=(0, 0), sources=(0,), actions=(6, 6), scales=(1., 1.)):
    n = len(actions)
    return dict(teacher_actions=np.array(actions, np.int64), record_codes=np.array(records, np.int64),
        record_source_codes=np.array(sources, np.int64), causal=dict(scales=np.array(scales, np.float32)),
        truth_by_arm=dict(bound_aware=torch.zeros(n, 7, 5)), preprocess=dict(rms=np.ones(5, np.float32)))


def test_record_loss_groups_signed_errors_instead_of_row_squares():
    p = torch.zeros(2, 7, 5)
    p[0, 6, 1] = 1
    p[1, 6, 1] = -1
    assert record_excess(p, data(), np.arange(2))['record_excess'] == 0
    assert record_excess(p, data(records=(0, 1), sources=(0, 0)), np.arange(2))['record_excess'] > 0


def test_reference_and_harm_gradients_follow_original_ratio_free_budget():
    p = torch.zeros(2, 7, 5, requires_grad=True)
    d = data()
    d['truth_by_arm']['bound_aware'][:, 6, 1] = 1
    r = record_excess(p, d, np.arange(2))
    r['record_excess'].backward()
    assert (p.grad[:, 6, 1] < 0).all()
    assert (p.grad[:, 6, 2] > 0).all()
    assert r['teacher_selected_rows'] == 2


def test_inactive_teacher_rows_have_no_gradient_or_fake_support():
    p = torch.ones(2, 7, 5, requires_grad=True)
    r = record_excess(p, data(actions=(-1, -1)), np.arange(2))
    assert r['active_records'] == 0 and r['record_excess'] == 0
    r['record_excess'].backward()
    assert torch.count_nonzero(p.grad) == 0


def test_scale_weighting_and_equal_source_averaging_are_explicit():
    p = torch.zeros(3, 7, 5)
    p[:, 6, 1] = torch.tensor([1., 1., 3.])
    d = data(records=(0, 1, 2), sources=(0, 0, 1), actions=(6, 6, 6), scales=(1., 2., 1.))
    got = record_excess(p, d, np.arange(3))['record_excess']
    assert float(got) == pytest.approx(2.5 / 1.0004)


def test_loss_rejects_relaxed_risk_budget():
    with pytest.raises(ValueError):
        record_excess(torch.zeros(2, 7, 5), data(), np.arange(2), budget=.03)


def test_loss_metadata_never_changes_causal_input_schema():
    c = dict(x=np.zeros((2, 7, 428)), envelope=np.ones((2, 7)), scales=np.ones(2),
        allowed=np.ones((2, 7), bool), groups=[np.arange(2)], source_groups=[np.array([0])],
        sites=np.array(['s', 's']), ids=np.arange(2))
    learning = dict(causal=c)
    out = training_packet(learning, np.array([6, -1]), np.array(['r', 'r']))
    assert out['causal'] is c
    assert 'teacher_actions' not in c and 'record_codes' not in c
    with pytest.raises(ValueError):
        training_packet(dict(causal=dict(c, future_endpoint=np.zeros((2, 2)))), np.array([6, 6]), np.array(['r', 'r']))
    with pytest.raises(ValueError):
        training_packet(dict(causal=dict(c, sites=np.array(['s', 't']))), np.array([6, 6]), np.array(['r', 'r']))
