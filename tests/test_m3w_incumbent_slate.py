import pytest
import torch

from src.world_model import m3w_incumbent_slate as model


def test_incumbent_label_reconstruction_and_consistency():
    passed = torch.tensor([True, False, True, False])
    event = torch.tensor([1, 0, 2, 1])
    assert torch.equal(model.incumbent_targets(passed, event, [2, 2]), torch.tensor([True, False]))
    with pytest.raises(ValueError, match='Inconsistent'):
        model.incumbent_targets(passed, event, [4])


def test_empty_donor_preserves_an_empty_boolean_target():
    y = model.incumbent_targets(torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.long), [])
    assert y.shape == (0,) and y.dtype == torch.bool


def test_anchor_contains_only_shared_past_context():
    x = torch.arange(34.).reshape(1, 2, 17)
    z = model.with_incumbent(x)
    assert z.shape == (1, 3, 18)
    assert torch.count_nonzero(z[0, 0, :10]) == 0
    assert torch.equal(z[0, 0, 10:17], x[0, 0, 10:])
    assert z[0, 0, -1] == 1
    assert torch.count_nonzero(z[0, 1:, -1]) == 0
    assert torch.equal(z[0, 1:, :-1], x[0])


def test_padding_never_changes_loss_or_gradient():
    z = torch.tensor([[.2, .1, -.1, 99.]], requires_grad=True)
    y = torch.tensor([[True, False, True]])
    mask = torch.tensor([[True, True, False]])
    anchor = torch.tensor([False])
    loss = model.objective(z, y, mask, anchor, pairwise=True)
    changed = z.detach().clone()
    changed[0, -1] = -200.
    assert loss.item() == pytest.approx(model.objective(changed, y, mask, anchor, pairwise=True).item())
    loss.backward()
    assert z.grad[0, -1] == 0
    assert z.grad[0, 1] < 0


def test_no_pair_penalty_for_homogeneous_outcomes():
    z = torch.tensor([[1., .3, .8]])
    y = torch.ones((1, 2), dtype=torch.bool)
    a = torch.ones(1, dtype=torch.bool)
    assert model.objective(z, y, y, a, pairwise=True) == model.objective(z, y, y, a, pairwise=False)


def test_optional_tie_keeps_incumbent_but_forced_still_exchanges():
    assert model.choices(torch.tensor([1., 1., 0.])) == {'forced': 1, 'optional': 0}
    assert model.choices(torch.tensor([0., 1., 2.])) == {'forced': 2, 'optional': 2}
    with pytest.raises(ValueError, match='finite'):
        model.choices(torch.tensor([0., float('nan')]))


def test_primary_requires_both_incumbent_and_same_mask_control(monkeypatch):
    from scripts import run_m3w_opendd_incumbent_slate as runner

    names = (*runner.FIELDS, 'rescued_full_grid', 'destroyed_full_grid')
    def row():
        arms = {arm: dict.fromkeys(names, 1.) for arm in runner.ARMS}
        for arm, count in [('incumbent', 2.), ('aligned', 1.),
                           ('pair_optional', 3.), ('pair_uniform_mask', 4.)]:
            arms[arm].update(original=count, full_grid=count)
        return dict(counts=dict(views=10, edits=4, eligible=2),
                    switches=dict.fromkeys(runner.POLICIES, 2), arms=arms)

    groups = {str(i): [row()] for i in range(7)}
    reference = dict(counts=dict(views=70, edits=28, switched=14), arms={
        k: {f: 7*groups['0'][0]['arms'][k][f] for f in runner.prior.FIELDS}
        for k in ('incumbent', 'aligned')})
    monkeypatch.setattr(runner.run, 'read', lambda _: dict(summary=reference))
    result = runner.summarize(groups)
    assert not result['primary_supported']
    assert result['full_grid_contrasts']['pair_optional_minus_incumbent']['ci95_pp'] == [10., 10.]
    assert result['full_grid_contrasts']['pair_optional_minus_pair_uniform_mask']['ci95_pp'] == [-10., -10.]


def test_score_diagnostic_keeps_empty_support_unestimable():
    from scripts.audit_m3w_incumbent_slate import probabilities

    assert probabilities([]) == dict(n=0, status='not_estimable')
    result = probabilities([(0., False), (1., True)])
    assert result['n'] == 2 and result['brier'] == result['ece'] == 0.
    assert result['bins'][-1]['bin'] == 9
    wrong = probabilities([(1., False)])
    assert wrong['brier'] == wrong['ece'] == 1.
