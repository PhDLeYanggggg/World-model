import numpy as np
from scripts.run_m3w_european_harm_tail_crossfit import fold_inputs


def test_held_locality_labels_and_features_cannot_change_fit_definition():
    x=np.arange(32,dtype=float).reshape(8,4); env=np.ones(8)*20
    sites=np.repeat(['a','b','c','d'],2); cv=np.arange(1,9,dtype=float)
    y=np.column_stack((cv,cv/2,cv*0,cv*0))
    a=fold_inputs(x,env,y,cv,sites,'d')
    x2=x.copy(); x2[6:]=100000.; y2=y.copy(); y2[6:]=50000.; cv2=cv.copy(); cv2[6:]=900000
    b=fold_inputs(x2,env,y2,cv2,sites,'d')
    for key in ('mean','std','weights','known'):
        np.testing.assert_array_equal(a[2][key],b[2][key])
    assert a[3]==b[3] and a[2]['cost_scale']==b[2]['cost_scale']
    np.testing.assert_array_equal(a[4],b[4])
    assert 'd' not in a[2]['training_sites']
