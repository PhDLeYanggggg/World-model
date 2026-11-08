import numpy as np

from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_opendd_record_support as m


def test_first_edit_from_later_query_is_invisible():
    q = np.array([0, 1, 1, 3])
    x = np.arange(8).reshape(4, 2)
    g, e, s = np.arange(4.), np.arange(4.)/4., np.ones(4)
    rows = np.array([1, 2])
    np.testing.assert_array_equal(m.prefix(rows, q, 3, x, g, e, s), m.prefix(rows, q, None, x, g, e, s))
    np.testing.assert_array_equal(m.prefix(rows, q, 2, x, g, e, s)[2:5], [1., 1., 0.])
    assert m.prefix(rows, q, 0, x, g, e, s)[4] == np.float32(np.log(2.))


def test_record_support_label_reproduces_original_policy_with_missing_rows():
    rng = np.random.default_rng(27)
    true_risk = []
    for _ in range(100):
        cv = np.zeros((6, 12, 2))
        y = rng.normal(size=cv.shape)
        valid = rng.random(cv.shape[:2]) > .12
        valid[-1] = False
        large = np.where(valid[..., None], y*.1, .05)
        small = np.where(valid[..., None], y*.05, .01)
        small[3] *= -1
        ols = np.full_like(cv, .05)
        easy_cut = 1.3

        def parts(p):
            a = moments(cv, p, y, valid, easy_cut)
            gain = lambda ref: np.where(valid, np.linalg.norm(ref-y, axis=-1)-np.linalg.norm(p-y, axis=-1),
                                        -np.linalg.norm(ref-p, axis=-1)).sum(1)
            offset = np.where(valid, np.linalg.norm(ols-y, axis=-1)-np.linalg.norm(cv-y, axis=-1),
                              -np.linalg.norm(ols-cv, axis=-1)).sum(1)
            g = np.c_[gain(cv), gain(ols)-offset]
            z = np.column_stack([a[k] for k in ("all_excess", "easy_excess", "easy_signed")])
            return g, z, a["ref_easy"], a["base"], offset

        rows = np.arange(1, len(cv))
        g, z, e, b, off = parts(small)
        a = tuple(v[0] for v in parts(large)[:4])
        safe, hit = m.support_targets(g[rows], z[rows], e[rows], b[rows], e.sum(), a, off.sum())
        for i, j in enumerate(rows):
            p = cv.copy()
            p[0], p[j] = large[0], small[j]
            rec = policy(p, cv, cv, y, valid, ["r"]*len(p), easy_cut=easy_cut, hard_cut=2.)["recordings"]["r"]
            true_risk.append(rec["supported"])
            assert safe[i] == rec["supported"]
            lower = [np.where(valid, np.linalg.norm(ref-y, axis=-1)-np.linalg.norm(p-y, axis=-1),
                              -np.linalg.norm(ref-p, axis=-1)).sum() for ref in (cv, ols)]
            assert hit[i] == (rec["supported"] and min(lower) > 0.)
    assert any(true_risk) and not all(true_risk)


def test_zero_easy_denominator_does_not_pass_and_guard_retains_anchor():
    gain = np.ones((3, 2))
    excess = -np.ones((3, 3))
    e, b = np.zeros(3), np.ones(3)
    assert not m.support_targets(gain, excess, e, b, 0.)[0].any()
    rows = np.array([11, 42, 9])
    scores = np.array([[2., 2., 2.], [3., 1., 9.], [1., 3., 9.]])
    assert m.choose(rows, 11, scores, guard=True) == 11
    scores[2, 0] = 3.
    assert m.choose(rows, 11, scores, guard=True) == 9
