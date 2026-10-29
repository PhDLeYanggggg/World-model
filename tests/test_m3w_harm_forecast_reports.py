import numpy as np
import pytest
from scripts.report_m3w_dimensionless_harm_forecast import metrics
from scripts.report_m3w_reference_harm_forecast import write_text


def test_matched_forecast_metric_arithmetic():
    new=np.array([1.,2.,np.nan]);old=np.array([2.,2.,np.nan]);ref=np.array([2.,2.,np.nan])
    cv=np.array([3.,3.,np.nan]);r=metrics(new,old,ref,cv)
    assert r['new_gain_CV']==50.
    assert r['new_gain_selected']==25.
    assert r['paired_improvement_pp_CV']==pytest.approx(100/6)
    assert r['supported_rows']==2 and r['unknown_rows']==1


def test_incomplete_zero_reference_no_fake_gain():
    z=np.array([0.]);n=np.array([1.])
    r=metrics(n,n,z,z)
    assert r['new_gain_CV'] is None and r['new_zero_CV_harmed']==1
    with pytest.raises(ValueError):metrics(z,z,np.array([np.nan]),z)


def test_report_replay_does_not_overwrite_changed_results(tmp_path):
    path=tmp_path/'report.md'
    write_text(path,'matched\n')
    write_text(path,'matched\n')
    with pytest.raises(ValueError):write_text(path,'changed\n')
    assert path.read_text()=='matched\n'
