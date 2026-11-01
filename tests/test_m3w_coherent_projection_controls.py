import copy
import numpy as np
import pytest
from scripts import run_m3w_coherent_cost_projection as run


def fixture():
    rng=np.random.default_rng(17);p={a:rng.uniform(.1,1,(4,7,5)) for a in run.api.ARMS}
    choices={a:np.array([0,2,-1,6],np.int64) for a in ('locked_joint','learned_joint')}
    ranks={}
    for prefix,arm in zip(('locked','learned'),run.api.ARMS):
        c=choices[prefix+'_joint'];at=np.flatnonzero(c>=0);r=np.zeros(4);r[at]=p[arm][at,c[at],0]-p[arm][at,c[at],1];ranks[prefix+'_joint']=r
    hashes={a:run.api.parent.digest(v) for a,v in p.items()}
    saved=dict(choices=choices,ranks=ranks,prediction_sha256=hashes)
    old=dict(prediction_sha256=hashes,action_sha256={a:run.api.parent.digest(c) for a,c in choices.items()})
    return p,saved,old


def test_original_native_predictions_actions_and_matching_ranks_remain_exact():
    p,saved,old=fixture();c,r=run.frozen_controls(p,saved,old)
    for name in saved['choices']:
        np.testing.assert_array_equal(c['raw_'+name],saved['choices'][name])
        np.testing.assert_array_equal(r['raw_'+name],saved['ranks'][name])


def test_runtime_helper_is_not_shadowed_by_a_local_resource_receipt():
    assert 'runtime' not in run.evaluate.__code__.co_varnames
    assert 'runtime' in run.evaluate.__code__.co_names


@pytest.mark.parametrize('change',('prediction','action','rank','future_field'))
def test_old_controls_cannot_be_silently_recomputed_changed_or_label_augmented(change):
    p,saved,old=copy.deepcopy(fixture())
    if change=='prediction':p['learned_references'][0,0,0]+=1
    elif change=='action':saved['choices']['learned_joint'][0]=-1
    elif change=='rank':saved['ranks']['learned_joint'][0]+=1
    else:p['future_endpoint']=np.zeros((4,2))
    with pytest.raises((ValueError,AssertionError)):run.frozen_controls(p,saved,old)
