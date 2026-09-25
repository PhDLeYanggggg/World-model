import numpy as np
import pytest

from scripts import report_m3w_european_symmetric_utility as report


def metric(value=None):
    return dict(indexed_rows=3,supported_rows=2,unknown_rows=1,
        equal_scene_gain_percent=value,worst_scene_gain_percent=value,
        scene_bootstrap_ci95=None,by_scene={'source':dict(rows=2)})


def test_summary_retains_negative_and_undefined_contrasts(monkeypatch):
    monkeypatch.setattr(report,'sha',lambda path:'fixture_hash')
    m = metric(-1.)
    description = dict(ADE_vs_CV=m,zero_CV=dict(rows=0,harmed_rows=0),calibrated_safety=False)
    r = dict(result_source='fixture',source_rows=3,joint_rows=2,
        policies={'all_views_retained':dict(full=description,joint_population={'joint':description},
            comparisons={'undefined':metric()},queries={'total':1})},
        neural_vs_damping={'view':{'pointwise':m}},symmetric_vs_asymmetric={'view':{'pointwise':metric()}},
        cost_prediction={},decision_changes={})
    out = report.summary(r)
    assert out['neural_vs_damping']['view']['pointwise']['equal_scene_gain_percent']==-1.
    assert out['symmetric_vs_asymmetric']['view']['pointwise']['equal_scene_gain_percent'] is None
    assert out['policies']['all_views_retained']['full']['ADE_vs_CV']['unknown_rows']==1
    assert out['policies']['all_views_retained']['full']['zero_CV']['rows']==0
    assert out['deployment_changed'] is False and out['calibrated_safety'] is False


def test_array_artifact_must_match_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    path=tmp_path/'costs.npz'
    np.savez(path,ids=np.arange(3),costs=np.ones((3,2)))
    a=dict(path='costs.npz',sha256=report.sha(path))
    assert report.load_artifact(a)['costs'].shape==(3,2)
    np.savez(path,ids=np.arange(3),costs=np.zeros((3,2)))
    with pytest.raises(ValueError,match='Artifact changed'):
        report.load_artifact(a)
