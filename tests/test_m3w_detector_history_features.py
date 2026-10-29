import inspect

import numpy as np
import pytest

from src.world_model import m3w_detector_history_features as model


def raw():
    dtype=[('agent','i8'),('frame','i8'),('x_min','f8'),('y_min','f8'),('x_max','f8'),('y_max','f8'),('confidence','f8')]
    rows=np.zeros(20,dtype=dtype);rows['agent']=1;rows['frame']=np.arange(20)*12
    rows['x_min']=np.arange(20);rows['y_min']=np.arange(20)*2
    rows['x_max']=rows['x_min']+10;rows['y_max']=rows['y_min']+20
    rows['confidence']=np.linspace(.5,.9,20)
    boxes=np.stack([rows[k][:8] for k in ('x_min','y_min','x_max','y_max')],-1)[None]
    return rows,boxes


def test_exact_past_keys_and_future_truncation_poison_invariance():
    rows,expected=raw();args=np.array([1]),np.array([84]),expected
    a,c,frames=model.lookup_history(rows,*args)
    b,d,_=model.lookup_history(rows[:8],*args)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(c,d)
    rows['confidence'][8:]=0.;rows['x_max'][8:]+=1000
    e,f,_=model.lookup_history(rows,*args)
    np.testing.assert_array_equal(a,e);np.testing.assert_array_equal(c,f)
    assert frames.max()==84 and not any('future' in k or 'label' in k for k in inspect.signature(model.features).parameters)


def test_invalid_alignment_and_duplicate_are_not_silently_imputed():
    rows,boxes=raw()
    with pytest.raises(AssertionError):model.lookup_history(rows,np.array([1]),np.array([84]),boxes+1)
    with pytest.raises(ValueError):model.lookup_history(rows[:7],np.array([1]),np.array([84]),boxes)
    rows['frame'][1]=rows['frame'][0]
    with pytest.raises(ValueError):model.lookup_history(rows,np.array([1]),np.array([84]),boxes)


def test_history_schema_current_control_and_translation_invariance():
    rows,boxes=raw();confidence=rows['confidence'][:8][None]
    a=model.features(boxes,confidence);b=model.features(boxes+np.array([100,200,100,200]),confidence)
    np.testing.assert_array_equal(a,b);assert a.shape==(1,64) and len(model.NAMES)==64
    current=model.features(boxes,confidence,'current')
    np.testing.assert_array_equal(current[:,16:24],np.repeat(confidence[:,-1:],8,axis=1).astype(np.float32))
    assert not np.array_equal(a,current)
    assert not model.features(boxes,confidence,'disabled').any()


def test_zero_area_boxes_are_retained_with_finite_features():
    boxes=np.zeros((2,8,4));confidence=np.zeros((2,8))
    assert np.isfinite(model.features(boxes,confidence)).all()
    with pytest.raises(ValueError):model.features(boxes,confidence+2)


def test_key_noise_control_is_reproducible_without_outcomes():
    keys=np.array([1,2,3]);a=model.noise_features(keys,keys+2,keys+5)
    np.testing.assert_array_equal(a,model.noise_features(keys,keys+2,keys+5))
    assert a.shape==(3,64) and (np.abs(a)<=1).all() and not np.array_equal(a[0],a[1])
