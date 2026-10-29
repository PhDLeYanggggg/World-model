import copy
import numpy as np
import pytest
from src.evaluation import m3w_pooled_analogue as api


def bank():
    # Same local recording ID in two sources must not collapse into one recording.
    n=80;x=np.arange(n*3,dtype=float).reshape(n,3)/(n*3)
    return dict(x=x,envelope=np.ones(n),labels=np.column_stack((np.zeros(n),np.linspace(0,.5,n),
        np.ones(n),np.ones(n),np.linspace(0,.2,n))),
        recordings=np.repeat(['["a","1"]','["b","1"]'],40),
        eligible=np.ones(n,bool),sources=np.repeat(['a','b'],40))


def test_pooled_reference_excludes_query_record_and_keeps_other_source():
    b=bank();q=np.zeros(80,bool);q[:4]=True
    out=api.retrieve(b,q)
    assert out['supported'].all()
    r=out['references'][0]
    assert r['known_donors_by_source']=={'a':0,'b':40}
    changed=copy.deepcopy(b);changed['labels'][:40]=np.nan
    replay=api.retrieve(changed,q)
    for arm in api.ARMS[2:]:np.testing.assert_array_equal(out['predictions'][arm],replay['predictions'][arm])
    assert out['references']==replay['references']


def test_no_donor_source_labels_does_not_silently_mean_zero_risk():
    b=bank();b['labels'][40:]=np.nan;q=np.zeros(80,bool);q[:4]=True
    out=api.retrieve(b,q)
    assert not out['supported'].any() and len(out['indices'])==4
    assert np.isnan(out['predictions']['pooled_nearest32']).all()


def test_harm_fraction_preserves_units_under_source_specific_scaling():
    b=bank();q=np.zeros(80,bool);q[:4]=True;original=api.retrieve(b,q)
    scaled=copy.deepcopy(b);scaled['labels'][40:]*=100;scaled['envelope'][40:]*=100
    other=api.retrieve(scaled,q)
    for a in api.ARMS[2:]:np.testing.assert_allclose(original['predictions'][a],other['predictions'][a])


def test_pool_requires_registered_four_sources():
    with pytest.raises(ValueError):api.pool({})
