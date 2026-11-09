import numpy as np
import pytest
import torch

from scripts import run_m3w_av2_selected_veto as run


def test_inner_ranker_excludes_recipient_calibration_and_held():
    split = dict(donors=["a", "b", "c", "d"], calibration="e", held="f")
    assert run.ranker_donors(split, "b") == ["a", "c", "d"]
    with pytest.raises(ValueError):
        run.ranker_donors(split, "f")
    with pytest.raises(ValueError):
        run.ranker_donors(dict(split, donors=["a", "b", "e", "d"]), "b")


def test_independent_veto_cannot_change_fixed_choice(monkeypatch):
    x, ctx = torch.zeros(3, 2), torch.zeros(3, 21)
    opts, row = torch.tensor([[0, 1], [1, 2]]), torch.arange(3)
    monkeypatch.setattr(run.prior, "features", lambda *_: (x, ctx, opts, row))
    cv = np.zeros((3, 12, 2))
    m = dict(unit=np.ones(3), row=np.arange(3), col=np.zeros(3, dtype=int),
             opts=opts.numpy(), bank=np.ones((3, 1, 12, 2)))

    def fixed(*_):
        return torch.zeros(3, 5), torch.zeros(3, 3)

    def rank_risk(*_):
        return None, torch.tensor([[-1.]*3, [1.]*3])

    def veto(*_):
        return None, torch.tensor([[5.]*3])

    path, ix, scores, _ = run.rank({"selected": veto}, rank_risk, fixed, dict(cv=cv), m, (1., 2.))
    assert ix == 0 and scores["selected"] < scores["self"]
    np.testing.assert_array_equal(path[2], cv[2])
