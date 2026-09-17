import numpy as np
import pytest
import torch

from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter, masked_future_ade


def source(n=70):
    rows, labels = [], []
    for agent in (0, 1):
        for frame in range(n):
            x, y = frame+agent*10, agent*2
            rows.append([agent, x, y, x+4, y+6, frame, 0, frame % 7 == 0, 1])
            labels.append('Pedestrian' if agent == 0 else 'Biker')
    return np.asarray(rows, float), np.asarray(labels)


def test_inputs_match_registered_shape_and_label_tail_is_not_filtered():
    rows, labels = source(30)
    a = SDDStepAdapter(rows, labels, 'scene/video0', 1)
    assert len(a) == 23
    assert a.get_geometry(0).shape == (476,)
    assert not a.get_labels(len(a)-1)['future_label_mask'].any()
    assert a.get_inputs(0)['neighbor_mask'][0].all()
    assert a.get_past_provenance(0)['generated'].all()
    np.testing.assert_array_equal(a.get_inputs(0)['prediction_frame_offsets'], np.arange(1, 13))


def test_future_mutation_and_truncation_preserve_query_and_all_features():
    rows, labels = source(70)
    a = SDDStepAdapter(rows, labels, 'scene/video0', 2)
    key = a.identity(5); q = key['frame_id']; before = a.get_inputs(5)
    altered = rows.copy(); future = rows[:, 5] > q
    altered[future, 1:5] += 999; altered[future, 6:9] = 1
    for r, l in ((altered, labels), (rows[~future], labels[~future])):
        b = SDDStepAdapter(r, l, 'scene/video0', 2)
        matched = [i for i in range(len(b)) if b.identity(i) == key]
        assert len(matched) == 1
        after = b.get_inputs(matched[0])
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])


def test_missing_past_rejected_occlusion_retained_and_future_lost_masked():
    rows, labels = source(30)
    rows[(rows[:, 0] == 0) & (rows[:, 5] == 10), 6] = 1
    a = SDDStepAdapter(rows, labels, 'scene/video0', 1)
    frames = [a.identity(i)['frame_id'] for i in range(len(a))]
    assert not set(range(10, 18)) & set(frames)
    i = frames.index(7); label = a.get_labels(i)
    assert not label['future_label_mask'][2] and label['future_lost'][2]
    assert a.get_past_provenance(i)['occluded'].any()


def test_stride_mask_and_schema_fail_closed():
    rows, labels = source()
    for stride in (0, True, 1.2):
        with pytest.raises(ValueError):
            SDDStepAdapter(rows, labels, 'scene/video0', stride)
    with pytest.raises(ValueError):
        SDDStepAdapter(rows, labels, 'scene/video0', 1, data_role='supervised_training')
    with pytest.raises(ValueError):
        SDDStepAdapter(np.r_[rows, rows[:1]], np.r_[labels, labels[:1]], 'scene/video0', 1)


def test_masked_loss_ignores_unsupported_labels_without_zero_error_claim():
    p = torch.ones(2, 12, 2, requires_grad=True)
    target = torch.full_like(p, float('nan')); target[0, :4] = 0
    mask = torch.zeros(2, 12, dtype=torch.bool); mask[0, :4] = True
    ade, supported = masked_future_ade(p, target, mask)
    assert supported.tolist() == [True, False]
    ade[supported].mean().backward()
    assert not p.grad[~mask].any()
    assert p.grad[mask].any()
