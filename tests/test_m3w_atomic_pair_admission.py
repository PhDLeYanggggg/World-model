import numpy as np
import torch

from scripts import run_m3w_opendd_atomic_pair_admission as run
from scripts import audit_m3w_atomic_pair_calibration as audit
from src.world_model.m3w_atomic_pair_admission import PairAdmission


def test_held_calibration_and_fitting_sites_never_overlap():
    for held in run.SITES:
        cal, donors = run.roles(held)
        assert cal != held
        assert set(donors) == set(run.SITES) - {held, cal}
        assert len(donors) == 5


def test_model_is_symmetric_in_the_two_actions():
    torch.manual_seed(12)
    model = PairAdmission()
    x, g = torch.randn(5, 2, 45), torch.randn(5, 8)
    torch.testing.assert_close(model(x, g), model(x.flip(1), g))
    torch.testing.assert_close(model(x, g)[:, 0], model(x, g+1.)[:, 0])


def toy():
    cv = np.zeros((3, 12, 2))
    y = np.ones_like(cv)
    good = cv.copy()
    good[:2] = .6
    bad = cv.copy()
    bad[:2] = -.6
    d = dict(cv=cv, ols=cv, y=y, v=np.ones(y.shape[:2], bool),
             rec=np.repeat("r", len(cv)), q=np.zeros(len(cv), int))
    p = dict(cv=cv, forced_add=good, forced_joint=good, joint_probability=np.full(len(cv), .7))
    return d, p, good, bad


def test_atomic_label_requires_support_and_risk_in_both_actions():
    d, p, good, bad = toy()
    batch = np.stack((good, bad))
    v = run.labels(dict(c=d["cv"], ols=d["ols"]), batch, d["y"], d["v"], 2.)
    np.testing.assert_array_equal(v, [[1., 1.], [0., 1.]])
    v = run.labels(dict(c=d["cv"], ols=d["ols"]), batch, d["y"], d["v"], .2)
    np.testing.assert_array_equal(v[:, 0], [0., 0.])


def test_no_release_is_not_a_pass_and_calibration_checks_both_policies():
    d, p, _, bad = toy()
    threshold, rows = run.choose_threshold(p, d, (2., 3.))
    assert threshold == .7
    assert not rows[-1]["eligible"]
    assert rows[-1]["edited_arm_views"] == 0
    p["forced_add"] = bad
    threshold, rows = run.choose_threshold(p, d, (2., 3.))
    assert threshold is None
    assert not any(r["eligible"] for r in rows)
    for path in run.release(p, None).values():
        np.testing.assert_array_equal(path, d["cv"])


def test_exact_calibration_counts_tied_scores_as_one_policy():
    d, p, _, _ = toy()
    p["q"] = d["q"]
    result = audit.exact_thresholds(p, d, (2., 3.))
    assert result == dict(nonempty_breakpoints=1, feasible_breakpoints=1, max_feasible_edits=4)
