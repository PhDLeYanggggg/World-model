import numpy as np
import pytest

from src.world_model.m3w_prefix_cost_targets import causal_prefix_disagreement, supervised_prefix_costs


def test_prefix_harm_reverses_without_changing_full_horizon_label():
    b=np.zeros((1,12,2)); p=np.zeros_like(b); y=np.zeros_like(b)
    p[:,:,0]=2; y[:,1:,0]=10
    out=supervised_prefix_costs(b,p,y,np.ones((1,12),bool),np.ones(1))
    np.testing.assert_array_equal(out['costs'][0,0],[0,2])
    np.testing.assert_allclose(out['costs'][0,-1],[(11*2-2)/12,0])
    np.testing.assert_array_equal(out['disagreement'],np.full((1,12),2.))


def test_future_target_change_cannot_change_causal_disagreement():
    rng=np.random.default_rng(17)
    b,p,y=[rng.normal(size=(3,12,2)) for _ in range(3)]
    a=supervised_prefix_costs(b,p,y,np.ones((3,12),bool),np.ones(3))
    c=supervised_prefix_costs(b,p,y+100,np.ones((3,12),bool),np.ones(3))
    np.testing.assert_array_equal(a['disagreement'],c['disagreement'])
    assert not np.array_equal(a['costs'],c['costs'])


def test_gap_invalidates_all_later_prefix_labels_without_imputation():
    b=np.zeros((1,12,2)); p=np.ones_like(b); y=np.zeros_like(b)
    mask=np.ones((1,12),bool); mask[:,2]=False; y[:,2]=np.nan
    out=supervised_prefix_costs(b,p,y,mask,np.ones(1))
    np.testing.assert_array_equal(out['available'][0],np.arange(12)<2)
    assert np.isnan(out['costs'][0,2:]).all()


def test_identical_predictions_have_exact_zero_supported_costs():
    rng=np.random.default_rng(29); b=rng.normal(size=(2,12,2)); y=rng.normal(size=b.shape)
    out=supervised_prefix_costs(b,b,y,np.ones((2,12),bool),np.ones(2))
    assert not out['costs'].any() and not out['disagreement'].any()


def test_triangle_bound_and_terminal_scalar_identity():
    rng=np.random.default_rng(43)
    b,p,y=[rng.normal(size=(100,12,2)) for _ in range(3)]; s=rng.uniform(.1,100,100)
    out=supervised_prefix_costs(b,p,y,np.ones((100,12),bool),s)
    assert np.all(out['costs'].sum(-1)<=out['disagreement']+1e-10)
    gain=(np.linalg.norm(b-y,axis=-1)-np.linalg.norm(p-y,axis=-1)).mean(1)*s
    np.testing.assert_allclose(out['costs'][:,-1,0],np.maximum(gain,0),atol=1e-10)
    np.testing.assert_allclose(out['costs'][:,-1,1],np.maximum(-gain,0),atol=1e-10)


@pytest.mark.parametrize('scale',[[0],[-1],[np.inf]])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError): causal_prefix_disagreement(np.zeros((1,12,2)),np.zeros((1,12,2)),scale)


def test_valid_label_nan_and_wrong_horizon_rejected():
    with pytest.raises(ValueError):
        supervised_prefix_costs(np.zeros((1,12,2)),np.zeros((1,12,2)),np.full((1,12,2),np.nan),np.ones((1,12),bool),[1])
    with pytest.raises(ValueError): causal_prefix_disagreement(np.zeros((1,11,2)),np.zeros((1,11,2)),[1])
