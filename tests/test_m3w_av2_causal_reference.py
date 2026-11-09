import numpy as np

from scripts import run_m3w_av2_causal_reference as run
from src.world_model import m3w_av2_causal_reference as model
from src.world_model import m3w_av2_recipient_reassignment as assignment


def test_history_proxy_is_translation_invariant_and_keeps_ordered_slots():
    h = np.array([[.1, .2], [.2, -.15], [-.15, .3]])[:, None]*np.arange(8)[None, :, None]
    c, o, _, _ = model.history.motion.rollout(h, 12)
    p = np.repeat(c[:, None], 2, axis=1)
    p[:, 0, :, 0] += .15
    p[:, 1, :, 1] += .12
    opts = assignment.assignments(len(c))
    d = dict(h=h, cv=c, ols4=o)
    a = model.select(d, p, opts, .1, 0, 1)
    b = model.select(dict(d, h=h+[10., 7.]), p, opts, .1, 0, 1)
    assert a == b
    assert a[:2] == (0, 1)
    assert all(opts[k, 0] != opts[k, 1] for k in a)


def test_exact_opportunity_requires_both_arms_eligible():
    cv = np.zeros((2, 12, 2))
    y = cv.copy()
    y[0, :, 0] = 2.
    v = np.ones(y.shape[:-1], dtype=bool)
    p = np.stack([y*.8, y, cv])
    ok = np.array([True, True, False])
    result = run.opportunities(p, ok, (0, 1, 2), dict(cv=cv), y, v)
    assert result == ([True, False, False], True, True)
    assert run.opportunities(p, [False, True, False], (0,), dict(cv=cv), y, v) == ([False], False, False)
