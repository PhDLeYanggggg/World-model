import numpy as np
import pytest
import torch

from src.world_model import m3w_advantage_factorial as m
from src.world_model import m3w_onpolicy_donors as roles


def test_normalization_uses_only_valid_training_options_not_labels():
    x = torch.tensor([[[1., 0.], [3., 0.], [999., 999.]]])
    valid = torch.tensor([[True, True, False]])
    a = m.normalization([(x, torch.zeros_like(valid), valid)])
    b = m.normalization([(x, torch.ones_like(valid), valid)])
    assert a == b and a['options'] == 2
    assert a['mean'] == [2., 0.]
    assert np.allclose(a['scale'], [1., .1])


def test_relative_loss_ignores_common_score_offset_and_padding():
    s = torch.tensor([[0., .5, 99.], [1., 0., -99.]], requires_grad=True)
    y = torch.tensor([[False, True, True], [True, False, False]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = m.objective(s, y, mask, relative=True)
    assert torch.allclose(loss, m.objective(s+4., y, mask, relative=True))
    loss.backward()
    assert torch.isfinite(s.grad).all() and (s.grad[:, 2] == 0).all()
    assert m.objective(torch.zeros_like(s), y, mask, relative=True) == 1.


def test_keep_ties_and_invalid_options_cannot_win():
    scores = torch.tensor([[0., 0., 99.], [-1., 0., 99.]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    assert m.choose(scores, valid).tolist() == [0, 1]
    valid[0, 0] = False
    with pytest.raises(ValueError):
        m.choose(scores, valid)


def test_inner_selection_has_fixed_ties_and_nonpositive_keep():
    values = dict.fromkeys(m.ARMS, 0.)
    assert m.select_arm(values) == 'keep'
    values['linear_advantage'] = values['mlp_advantage'] = .01
    assert m.select_arm(values) == 'linear_advantage'
    with pytest.raises(ValueError):
        m.select_arm({'mlp_absolute': 1.})


def test_nested_teacher_excludes_outer_validation_donor_and_target():
    sites = tuple('abcdefg')
    for target in roles.fit_roster(sites, ('a', 'b', 'c')):
        fit = roles.fit_roster(sites, ('a', 'b', 'c', target))
        assert len(fit) == 3 and not set(fit) & {'a', 'b', 'c', target}
    with pytest.raises(ValueError, match='exposed'):
        roles.verify_role(dict(omitted=['a', 'c'], fit=list('bdefg')), sites, ('a', 'b', 'c'))


@pytest.mark.parametrize('arm', m.ARMS)
def test_models_train_with_finite_loss_and_restore_buffers(arm):
    x = torch.randn(3, 4, 5)
    y = torch.rand(3, 4) > .5
    valid = torch.ones_like(y)
    norm = m.normalization([(x, y, valid)])
    net = m.Selector(arm, norm)
    loss = m.objective(net(x), y, valid, relative=arm.endswith('advantage'))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    clone = m.Selector(arm, norm)
    clone.load_state_dict(net.state_dict())
    assert torch.equal(clone(x), net(x))


def test_summary_keeps_zero_controls_and_does_not_promote_zero_gain():
    from scripts import run_m3w_opendd_advantage_factorial as runner
    arms = {k: dict.fromkeys(runner.FIELDS, 0.) for k in runner.RESULT_ARMS}
    row = dict(seed=17, counts=dict(views=10, edits=2, eligible=1),
               switches=dict.fromkeys(runner.POLICIES, 0), arms=arms)
    result = runner.summary({str(i): [row] for i in range(7)})
    assert result['arms']['incumbent']['risk_violation'] == 0
    assert result['contrasts']['inner_selected_minus_incumbent']['ci95_pp'] == [0., 0.]
    assert not result['primary_supported']


def test_no_exchange_recordings_have_zero_gain_not_a_safety_pass():
    from scripts.run_m3w_opendd_advantage_factorial import recording_gain
    v = dict(slates=0, rescued=0, destroyed=0, gain_rate=None)
    r = recording_gain(v, dict(recordings=3, slates=0, no_exchange=3))
    assert r['lower'] == r['upper'] == 0. and r['exchange_slates'] == 0
    assert 'risk_pass' not in r and 'full_grid_pass' not in r


def test_unknown_solver_rows_are_bounded_not_counted_as_zero_gain():
    from scripts.run_m3w_opendd_advantage_factorial import recording_gain
    r = recording_gain(dict(slates=4, rescued=2, destroyed=1),
                       dict(recordings=10, slates=4, no_exchange=5))
    assert r['unresolved_recordings'] == 1
    assert r['lower'] == 0. and r['upper'] == .2


def test_relative_diagnostic_preserves_harm_and_is_not_a_probability():
    from scripts.diagnose_m3w_advantage_factorial import observations, compact
    scores = torch.tensor([[0., 2., 99.]])
    y = torch.tensor([[True, False, True]])
    valid = torch.tensor([[True, True, False]])
    result = compact(observations(lambda x: x, 'linear_advantage', scores, y, valid),
                     'linear_advantage')
    assert result['destructions'] == 1 and result['rescues'] == 0
    assert result['observed_gain_mean'] == -1. and result['predicted_gain_mean'] > 0
    assert result['score_semantics'] == 'tanh_score_difference_not_a_probability'
    assert 'uncalibrated_sigmoid_diagnostics' not in result


def test_absolute_diagnostic_reports_chosen_outcome_without_certifying_score():
    from scripts.diagnose_m3w_advantage_factorial import observations, compact
    scores = torch.tensor([[0., 2.]])
    y = torch.tensor([[False, True]])
    valid = torch.ones_like(y)
    result = compact(observations(lambda x: x, 'mlp_absolute', scores, y, valid),
                     'mlp_absolute')
    assert result['rescues'] == 1 and result['observed_gain_mean'] == 1.
    assert result['uncalibrated_sigmoid_diagnostics']['chosen']['positive_rate'] == 1.
