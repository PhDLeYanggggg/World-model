from argparse import Namespace
import json
from pathlib import Path
import pytest
from scripts.run_m3w_conditional_cost import validate_config,validate_args
from src.evaluation.m3w_conditional_cost_eval import gate


def test_fixed_comparison_and_no_threshold_search():
    c=json.loads(Path('configs/m3w_conditional_cost_v1.json').read_text())
    p=json.loads(Path('configs/m3w_cost_budget_matched_v1.json').read_text())
    validate_config(c,p)
    for k,v in [('region_multiplier',8),('loss_exponent',2),('region_reference','winner'),
        ('primary_reference','fraction_strict_stop'),('threshold_search',True),
        ('training',c['training']|{'steps':24000}),('closed_role_readout',True)]:
        with pytest.raises(ValueError):validate_config(c|{k:v},p)


@pytest.mark.parametrize('change',[dict(stop_at=100),dict(view='a',stop_at=12001),
    dict(evaluate=True,verify=True),dict(verify=True,resume=True)])
def test_distinct_execution_phases(change):
    fields=dict(audit_only=False,evaluate=False,verify=False,resume=False,view=None,stop_at=None)
    with pytest.raises(ValueError):validate_args(Namespace(**(fields|change)))


def test_joint_gate_does_not_hide_scene_easy_failure():
    s=dict(seeds={'17':dict(zero_CV_harmed=0,ADE={'equal_scene_gain_percent':4},
        subsets={'positive_easy':{'equal_scene_gain_percent':1,'by_scene':{'a':{'gain_percent':-3}}}})})
    g=gate(s,{'ci95_pp':[.1,1]})
    assert g['aggregate_easy'] and not g['each_scene_seed_easy'] and not all(g.values())
    s['seeds']['17']['subsets']['positive_easy']['by_scene']['a']['gain_percent']=-1
    assert all(gate(s,{'ci95_pp':[.1,1]}).values())
    assert not all(gate(s,{'ci95_pp':[-.1,1]}).values())
