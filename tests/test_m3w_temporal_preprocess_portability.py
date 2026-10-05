import numpy as np
import pytest
from scripts.audit_m3w_temporal_preprocess_portability import compare


def test_measures_roundoff_without_admitting_it_as_equivalent():
    a=dict(mean=np.array([1.,2.]),scale=1.,site='train',known_rows=5)
    b=dict(a,mean=np.array([np.nextafter(1.,2.),2.]),scale=np.nextafter(1.,2.))
    rows={r['field']:r for r in compare(a,b)}
    assert not rows['mean']['exact'] and rows['mean']['changed']==1
    assert rows['scale']['max_absolute_difference']==np.finfo(float).eps
    assert rows['site']['exact'] and rows['known_rows']['exact']
    assert not any('passed' in row for row in rows.values())


@pytest.mark.parametrize('bad',[dict(x=np.ones(2)),dict(x=np.ones(1,dtype=np.float32)),dict(y=np.ones(1)),dict(x=np.array([np.nan]))])
def test_semantic_or_nonfinite_difference_is_not_hidden(bad):
    with pytest.raises(ValueError):compare(dict(x=np.ones(1)),bad)
