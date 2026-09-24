import numpy as np
import pytest

from scripts.audit_m3w_ht21_cadence import cadence_support


@pytest.mark.parametrize('stride',[1,5,10])
def test_all_anchor_phases_without_future_eligibility(stride):
    rows=np.array([[f,1,float(f),float(f),2.,2.,1,1,1.] for f in range(1,241)])
    result=cadence_support(rows,240,stride)
    assert result['past_eligible']==240-7*stride
    assert result['complete_future12']==240-19*stride
    assert result['no_future12']==stride
    assert result['partial_future12']==11*stride


def test_sparse_track_has_no_invented_cadence():
    rows=np.array([[f,1,float(f),float(f),2.,2.,1,1,1.] for f in range(1,241,5)])
    assert cadence_support(rows,240,1)['past_eligible']==0
    assert cadence_support(rows,240,5)['past_eligible']==41


@pytest.mark.parametrize('stride',[0,-1,1.5])
def test_invalid_stride(stride):
    with pytest.raises(ValueError):cadence_support(np.zeros((0,9)),240,stride)
