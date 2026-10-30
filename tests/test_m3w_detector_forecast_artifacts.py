import json
import pytest
from scripts import report_m3w_detector_forecaster as report


@pytest.mark.parametrize('reference',[0.,1.])
def test_report_support_and_tails_separate_repeated_views_and_undefined_labels(tmp_path,monkeypatch,reference):
    monkeypatch.setattr(report,'ROOT',tmp_path);refs=[]
    for kind in ('single','complement'):
        for i in range(9):
            for arm in report.run.api.REPRESENTATIONS:
                metrics=dict(all=dict(ADE=dict(indexed=4,known=3),FDE=dict(known=2)),complete_future=dict(ADE=dict(indexed=2)))
                record=dict(reference_ADE=dict(mean=reference,p95=reference),native_ADE=dict(history=dict(mean=1.03,p95=1.05)))
                row=dict(trial=kind+str(i),kind=kind,representation=arm,result=dict(rows=4,
                    complete_future_rows=2,partial_future_rows=1,missing_future_rows=1,
                    stationary_observed_rows=0,stationary_known_nonzero_error_rows={a:0 for a in report.run.api.REPRESENTATIONS},
                    sites={'site0':dict(metrics=metrics,records={'r0':record})}))
                path=tmp_path/(kind+str(i)+'_'+arm+'.json');path.write_text(json.dumps(row))
                refs.append(dict(path=path.name,sha256=report.run.sha(path)))
    profile=report.support_profile(dict(groups=refs));p=profile['single']
    assert p['indexed_row_views']==36 and p['unique_queries']==dict(indexed=4,complete_future=2,partial_future=1,missing_future=1)
    assert p['unique_recordings']==1
    assert p['known_record_mean_tail_diagnostic']['mean_worse_by_over2percent']==(9 if reference else 0)
    assert p['known_record_mean_tail_diagnostic']['p95_worse_by_over2percent']==(9 if reference else 0)
    assert p['known_record_mean_tail_diagnostic']['undefined_mean_views']==(0 if reference else 9)
    assert p['known_record_mean_tail_diagnostic']['descriptive_not_approved_positive_harm_policy_gate']
    with pytest.raises(ValueError):report.support_profile(dict(groups=refs[:-1]))


def test_supplementary_paired_contrast_keeps_full_roster_and_missing_endpoint():
    new=dict(by_locality={str(i):2. for i in range(12)})
    old=dict(by_locality={str(i):0. for i in range(12)})
    cfg=dict(bootstrap_seed=17,bootstrap_resamples=32)
    out=report.paired_summary_difference(new,old,cfg)
    assert out['point']==2. and out['CI95']==[2.,2.]
    old['by_locality']['0']=None
    assert report.paired_summary_difference(new,old,cfg)['CI95'] is None
    del old['by_locality']['0']
    with pytest.raises(ValueError):report.paired_summary_difference(new,old,cfg)


def test_later_exact_replay_preserves_first_artifact_runtime_but_not_changed_science(tmp_path,monkeypatch):
    monkeypatch.setattr(report.run,'PUBLIC',tmp_path)
    first=dict(summary_sha256='fixed',replay=dict(pid=1,seconds=2.))
    assert report.freeze_evidence(first)==first
    later=dict(summary_sha256='fixed',replay=dict(pid=3,seconds=4.))
    assert report.freeze_evidence(later)==first
    with pytest.raises(ValueError):report.freeze_evidence(dict(later,summary_sha256='changed'))
