from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.run_m3w_cost_capacity import validate_args,validate_config
from src.evaluation.m3w_cost_capacity_eval import factorial_contrasts


def args(**kw):
    return Namespace(**(dict(audit_only=False,evaluate=False,verify=False,resume=False,view=None,width=None,stop_at=None)|kw))


@pytest.mark.parametrize('kw',[dict(stop_at=100),dict(view='a',width='narrow',stop_at=100),
    dict(view='a',width='wide',stop_at=12001),dict(evaluate=True,resume=True),dict(evaluate=True,verify=True),
    dict(audit_only=True,width='narrow')])
def test_illegal_phases(kw):
    with pytest.raises(ValueError): validate_args(args(**kw))


def test_fixed_matrix_and_legal_pilot():
    validate_args(args(view='coupa_seed17',width='narrow',stop_at=3100))
    validate_args(args(view='coupa_seed17',width='wide',stop_at=100))
    cfg=json.loads(Path('configs/m3w_cost_capacity_v1.json').read_text())
    p=json.loads(Path('configs/m3w_tempered_cost_v1.json').read_text()); validate_config(cfg,p)
    for k,v in (('widths',[64,256]),('primary_arm','best_result'),('closed_role_readout',True),('loss_exponent',2)):
        with pytest.raises(ValueError): validate_config(cfg|{k:v},p)


def test_factorial_interaction_is_difference_of_differences():
    x=dict(narrow_short=[1.,2.,3.,4.],narrow_long=[2.,3.,4.,5.],
           wide_short=[3.,4.,5.,6.],wide_long=[5.,6.,7.,8.])
    r=factorial_contrasts(x)
    assert r['capacity_short']['mean_gain_difference_pp']==2
    assert r['duration_narrow']['mean_gain_difference_pp']==1
    assert r['interaction']['mean_gain_difference_pp']==1
    np.testing.assert_array_equal(r['interaction']['ci95_pp'],[1.,1.])
