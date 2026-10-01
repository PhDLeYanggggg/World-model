import pytest
import numpy as np
from scripts.report_m3w_source_checkpoint import locality_interval, query_average_squared_error


def test_equal_site_weight_not_row_weight():
    z=locality_interval([('a',1.)]*20+[('b',3.)],draws=3000)
    assert z['mean']==2 and z['localities']==2 and z['CI95']==[1.,3.]


def test_undefined_source_is_not_silently_excluded():
    z=locality_interval([('a',1.),('b',None)])
    assert z['mean'] is None and z['CI95'] is None and z['undefined_localities']==['b']
    with pytest.raises(ValueError):locality_interval([])


def test_constant_paired_difference():
    z=locality_interval([(str(i),-.1) for i in range(12)])
    assert z['mean']==pytest.approx(-.1)
    assert z['CI95']==pytest.approx([-.1,-.1])


def test_query_loss_balances_queries_and_excludes_unknown_labels():
    target=np.tile([0.,0.,1.,0.,0.],(5,1));prediction=target.copy()
    prediction[:,0]=[1.,1.,1.,3.,1e9];target[-1]=np.nan
    score=query_average_squared_error(prediction,target,1.,np.ones(3),np.array(['r']*5),np.array([1,1,1,2,3]))
    assert score==pytest.approx((1+9)/6)
    with pytest.raises(ValueError):
        query_average_squared_error(prediction,target*np.nan,1.,np.ones(3),np.array(['r']*5),np.arange(5))
