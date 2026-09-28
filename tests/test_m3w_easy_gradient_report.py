import pytest
import torch
from src.world_model.m3w_easy_gradient_diagnostic import geometry
from scripts.report_m3w_easy_gradient_diagnostic import check_scalar_geometry, distribution
from scripts.verify_m3w_easy_gradient_diagnostic import check_pairs
import itertools


def test_scalar_geometry_consistency_and_tampering():
    row = geometry(dict(marginal=torch.tensor([1., 2.]), occurrence=torch.tensor([-3., .5]), conditional=torch.tensor([.1, .2])))
    check_scalar_geometry(row)
    row['risk_projection']['supervised'] += .1
    with pytest.raises(AssertionError):
        check_scalar_geometry(row)


def test_undefined_is_not_zero_conflict():
    out = distribution([None, -1., 1.])
    assert out['undefined'] == 1 and out['defined'] == 2 and out['negative'] == 1
    assert distribution([None])['median'] is None


def test_group_identity_and_role_exclusion():
    groups = []
    for i in range(108):
        sites = [str((j+i) % 12) for j in range(12)]
        roles = dict(producer_sites=sites[:4], controller_sites=sites[4:8],
                     training_sites=sites[8:10], held_sites=sites[10:])
        rows = [dict(arm=arm, point=point, batch=batch, sampled_rows=32,
                     losses={'marginal': 1.}, gradients={})
                for arm, point, batch in itertools.product(
                    ('marginal', 'supervised'), ('initial', 'final'), range(4))]
        groups.append(dict(group=str(i), parameter_updates=0, held_outcomes_used=False,
                           identity=dict(roles=roles), rows=rows))
    assert check_pairs(groups)['initial_pair_comparisons'] == 432
    groups[0]['identity']['roles']['held_sites'][0] = groups[0]['identity']['roles']['training_sites'][0]
    with pytest.raises(AssertionError):
        check_pairs(groups)
