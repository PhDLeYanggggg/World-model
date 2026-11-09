import numpy as np
import pytest
import torch

from src.world_model import m3w_av2_observability as model


def test_context_reads_only_observed_steps_and_captures_extra_fragments():
    steps = tuple(range(42, 50))
    h = np.stack((np.arange(8.)*.2, np.zeros(8)), axis=1)[None]
    row = lambda k, t, p: dict(track_id=k, timestep=t, position_x=p[0], position_y=p[1], observed=True)
    own = [row("self", t, p) for t, p in zip(steps, h[0], strict=True)]
    other = [row("partial", t, [0.5, 0.1]) for t in steps[2:6]]
    m, c, partial = model.past_context(own+other, ["self"], h, steps)
    m0, c0, no_partial = model.past_context(own, ["self"], h, steps)
    np.testing.assert_array_equal(m, m0)
    assert m.shape[-1] == model.MOTION_WIDTH and c.shape[-1] == model.CONTEXT_WIDTH
    assert partial == 1 and no_partial == 0 and c[0, 4] > c0[0, 4]
    np.testing.assert_array_equal(c, model.past_context(list(reversed(own+other)), ["self"], h, steps)[1])
    with pytest.raises(ValueError, match="Out-of-window"):
        model.past_context(own+[row("partial", 50, [1., 0.])], ["self"], h, steps)
    with pytest.raises(ValueError, match="frozen roster"):
        model.past_context(own, ["self"], h+1., steps)


def test_missing_coverage_and_easy_support_are_distinct_labels():
    cv = np.zeros((3, 12, 2))
    y = np.full_like(cv, .04)
    v = np.zeros(cv.shape[:2], dtype=bool)
    v[1, :6], v[2] = True, True
    a, supported = model.labels(cv, y, v, [0, 1], .08)
    np.testing.assert_array_equal(a, [[0., 0., 0.], [.5, 1., 1.]])
    assert supported
    assert not model.labels(cv, y, v, [0], .08)[1]


def test_context_broadcasts_shared_scale_to_multiple_recipients():
    h = np.zeros((3, 8, 2))
    h[:, :, 0] = np.arange(8)
    h[:, :, 1] = np.arange(3)[:, None]
    rows = [dict(track_id=i, timestep=t, observed=True, position_x=p[0], position_y=p[1])
            for i, tr in enumerate(h) for t, p in enumerate(tr)]
    m, c, _ = model.past_context(rows, np.arange(3), h, tuple(range(8)))
    assert m.shape == (3, model.MOTION_WIDTH) and c.shape == (3, model.CONTEXT_WIDTH)
    np.testing.assert_array_equal(m[:, 4], m[0, 4])


def test_factored_output_cannot_exceed_predicted_support():
    z = torch.tensor([[4., -5., 9.], [-4., 5., 9.]])
    direct, factored = model.scores(z)
    assert direct[0] > direct[1] and factored[0] < factored[1]
    assert (factored <= z[:, 1].sigmoid()).all()


def test_gate_invariant_to_edit_recipient_order():
    gate = model.Gate(5).eval()
    a = torch.randn(2, 2, model.CONTEXT_WIDTH+model.MOTION_WIDTH)
    x = torch.randn(2, 5)
    q1, v1 = gate(x, a)
    q2, v2 = gate(x, a.flip(1))
    torch.testing.assert_close(q1, q2)
    torch.testing.assert_close(v1, v2.flip(1))
