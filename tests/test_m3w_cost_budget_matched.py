from argparse import Namespace
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.run_m3w_cost_budget_matched import validate_config, validate_args, check_matched
from src.evaluation.m3w_cost_budget_matched_eval import contribution_gate
from src.world_model.m3w_bounded_cost_head import build, loss_value
from src.world_model.m3w_tempered_cost_head import loss_value as tempered_loss


def test_fixed_control_configuration():
    c = json.loads(Path('configs/m3w_cost_budget_matched_v1.json').read_text())
    p = json.loads(Path('configs/m3w_cost_capacity_v1.json').read_text())
    t = json.loads(Path('configs/m3w_tempered_cost_v1.json').read_text())
    validate_config(c, p, t)
    for key, value in (('training', c['training'] | {'width':256}), ('reference_arm', 'winner'),
            ('primary_contrasts', ['tempered_minus_fraction']), ('closed_role_readout', True),
            ('threshold_search', True), ('policies', ['strict_stop'])):
        with pytest.raises(ValueError):
            validate_config(c | {key:value}, p, t)


@pytest.mark.parametrize('overrides', [dict(stop_at=100), dict(view='a', arm='native', stop_at=12001),
    dict(evaluate=True, verify=True), dict(verify=True, resume=True), dict(audit_only=True, arm='native')])
def test_phase_mixing_rejected(overrides):
    values = dict(audit_only=False, evaluate=False, verify=False, resume=False, view=None, arm=None, stop_at=None)
    with pytest.raises(ValueError):
        validate_args(Namespace(**(values | overrides)))


def test_common_forward_initial_state_and_distinct_objectives():
    native, fraction = build(5, 8, 17), build(5, 8, 17)
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    distance = torch.tensor([0., 1., 10.])
    a, b = native(x, distance, 'bounded_native'), fraction(x, distance, 'bounded_fraction')
    assert torch.equal(a, b) and torch.equal(a[0], torch.zeros(2))
    score, target, d = torch.tensor([[1., 0.], [10., 0.]]), torch.zeros(2, 2), torch.tensor([1., 10.])
    assert float(loss_value(score, target, d, 'bounded_native')) == 25.25
    assert float(tempered_loss(score, target, d)) == 2.75
    assert float(loss_value(score, target, d, 'bounded_fraction')) == .5


def test_joint_contribution_requires_both_controls_and_all_seeds():
    r = dict(seeds={'17':dict(zero_CV_harmed=0, ADE={'equal_scene_gain_percent':3.},
        subsets={'positive_easy':{'equal_scene_gain_percent':-1.}})})
    c = {k:{'ci95_pp':[.1, .5]} for k in ('tempered_minus_native', 'tempered_minus_fraction')}
    assert all(contribution_gate(r, c).values())
    c['tempered_minus_native']['ci95_pp'][0] = -.01
    assert not all(contribution_gate(r, c).values())
    c['tempered_minus_native']['ci95_pp'][0] = .1
    r['seeds']['29'] = dict(zero_CV_harmed=1, ADE={'equal_scene_gain_percent':4.},
        subsets={'positive_easy':{'equal_scene_gain_percent':-2.1}})
    g = contribution_gate(r, c)
    assert not g['exact_zero'] and not g['easy']


def test_matched_sampler_and_preprocess_not_just_step_count():
    settings = json.loads(Path('configs/m3w_cost_budget_matched_v1.json').read_text())['training']
    pr = dict(mean=np.zeros(2), std=np.ones(2), known=np.array([True, False]), weights=np.array([1., 0.]),
              constant=np.zeros(2), cost_scale=1., hard_cut=2., positive_easy_cut=.2)
    cp = dict(settings=settings, step=12000, seed=17, arm='bounded_native', preprocess=pr,
              draws=np.array([12000*256, 0]))
    ref = copy.deepcopy(cp)
    check_matched(cp, ref, {'training':settings}, 17, 'native')
    bad = copy.deepcopy(cp)
    bad['draws'] -= np.array([1, -1])
    with pytest.raises(AssertionError):
        check_matched(bad, ref, {'training':settings}, 17, 'native')
    bad = copy.deepcopy(cp)
    bad['preprocess']['mean'][0] = 1
    with pytest.raises(AssertionError):
        check_matched(bad, ref, {'training':settings}, 17, 'native')
