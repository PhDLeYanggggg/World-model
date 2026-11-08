import numpy as np
import pytest
import torch

from src.world_model import m3w_opendd_record_teacher as m
from src.world_model.m3w_opendd_slate_queue import Packet


def fixture():
    rows = tuple(np.array(r, dtype=int) for r in ([], [0], [1], [0, 1]))
    return Packet(rows, np.eye(4), np.array([0, 1, 3]), np.array([0, 1, 1, 2]), 1, 1.)


def test_teacher_and_local_labels_share_count_and_candidates():
    p = fixture()
    restricted = m.at_count(p, 1)
    assert restricted.release == 1
    assert [r.tolist() for r in restricted.rows] == [[], [0], [1]]
    x, z = m.labels(p, 1, np.array([1]), np.array([.9, .5]))
    assert z == [0, 1, 2]
    np.testing.assert_array_equal(x[:2], p.features[[1, 2]])
    assert not x[2:].any()
    net = m.network(4)
    for teacher in (0, 1):
        loss = m.loss(net, torch.from_numpy(x[None]), torch.tensor([z]), teacher)
        assert torch.isfinite(loss)
        loss.backward()


def test_invalid_witness_not_turned_into_supervision():
    with pytest.raises(ValueError, match="one slate"):
        m.labels(fixture(), 1, np.array([0, 1]), np.ones(2))


def test_single_addition_respects_other_edits_and_both_gains():
    from scripts.audit_m3w_opendd_one_extra_edit import assess

    cv = np.zeros((2, 12, 2))
    y = cv.copy()
    y[..., 0] = 1.
    path = y.copy()
    path[1, :, 0] = -1.
    mask = np.ones((2, 12), dtype=bool)
    status, witness = assess(cv, cv, path, y, mask, cv, 2.)
    assert status == "qualified" and witness["1.0"][0] == 0
    assert assess(cv, y, path, y, mask, cv, 2.)[0] == "not_dual_positive"
    base = cv.copy()
    base[1] = path[1]
    assert assess(cv, cv, path, y, mask, base, 2.)[0] == "risk_infeasible"
