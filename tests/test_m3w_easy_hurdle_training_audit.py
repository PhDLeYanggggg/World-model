import pytest
import torch
from scripts.verify_m3w_easy_hurdle_portable_training import check_state


def fixture():
    identity=dict(roles=dict(training_sites=['a','b']))
    cfg=dict(head_training=dict(steps=2000,query_batch_size=32))
    s=dict(identity=identity,arm='supervised',settings=cfg['head_training'],step=2000,
        unknown_rows_sampled=0,query_draws=64000,norm=dict(training_sites=['a','b']),
        probability_calibration_certificate=False,model={'w':torch.ones(2)},
        trace=[dict(step=k,monitor=dict(loss=.1)) for k in (0,2000)])
    return s,identity,cfg


def test_checkpoint_audit_accepts_exact_completed_fit():
    check_state(*fixture(),'supervised')


@pytest.mark.parametrize('problem',['step','nan','sampling','source','unknown'])
def test_checkpoint_audit_refuses_invalid_fit(problem):
    s,identity,cfg=fixture()
    if problem=='step':s['step']=100
    if problem=='nan':s['model']['w'][0]=float('nan')
    if problem=='sampling':s['query_draws']=0
    if problem=='source':s['norm']['training_sites']=['a','held']
    if problem=='unknown':s['unknown_rows_sampled']=1
    with pytest.raises(AssertionError):check_state(s,identity,cfg,'supervised')
