import json
import numpy as np
import pytest
from scripts import run_m3w_selected_set_calibration as run


def packet():
    p = np.tile([.4,.002,2,2,.002],(6,1))
    z = dict(p=p,y=np.tile([0,.015,1,1,.015],(6,1)),env=np.ones(6),
             moving=np.ones(6,bool),support=np.ones(6,bool),recordings=np.repeat(['a','b','c'],2))
    old, actions = run.api.parent.calibrate(**dict(p=z['p'],y=z['y'],env=z['env'],
        moving=z['moving'],support=z['support'],recordings=z['recordings']))
    meta = dict(source='source',identity={'source':'source'},
        array_hashes={k:run.api.action_hash(v) for k,v in z.items()},
        parent_oof_action_hashes={k:run.api.action_hash(v) for k,v in actions.items()},
        parent_source=old['source'],parent_final=old['final'])
    z['meta_json'] = np.array(json.dumps(meta))
    return z


def test_real_parent_reproduction_and_exact_replay():
    cfg = dict(round_cap=8,empirical_quantile=.9,bootstrap_seed=20261002,bootstrap_resamples=3000)
    z = packet();a = run.compute(z,cfg)
    assert a == run.compute(z,cfg)
    report = run.summary([a],cfg)
    assert not report['transfer_evaluated'] and not report['deployment_promoted']
    assert report['comparisons']['joint_source_oof']['parent']['complete_support_pass'] == 0
    assert report['comparisons']['joint_source_oof']['utility_change_percent_full_reference']['mean'] == 0


def test_corrupt_input_hash_rejected():
    z = packet();z['p'][0,0] += .01
    with pytest.raises(AssertionError): run.compute(z,dict(round_cap=8,empirical_quantile=.9))


def test_locality_cluster_bootstrap_not_row_weighted():
    cfg = dict(bootstrap_seed=3,bootstrap_resamples=3000)
    a = run.interval([('a',1.)]*100+[('b',3.)],cfg)
    assert a['mean'] == 2 and a['locality_count'] == 2
    b = run.interval([('a',None),('b',3.)],cfg)
    assert b['undefined_views'] == 1 and b['strict_all_views_mean'] is None


def test_immutable_output_refuses_silent_overwrite(tmp_path):
    p = tmp_path/'result.json'
    run.once(p,{'v':1});run.once(p,{'v':1})
    with pytest.raises(AssertionError):run.once(p,{'v':2})
