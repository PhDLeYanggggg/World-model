import copy
import inspect

import numpy as np
import pytest

from src.world_model import m3w_easy_harm_deviance_readout as api
from scripts import verify_m3w_easy_harm_readout as scalar


def fixture():
    y = np.array([[2,0,10,10,0], [0,1,10,10,1], [0,1,0,0,1],
                  [np.nan]*5, [1,0,4,0,0], [0,0,5,5,0.]])
    p = np.tile([1., .01, 10., 10., .01], (6,1))
    pred = {k:p.copy() for k in api.ARMS}
    pred[api.TARGET][[1,2],1] = 1.
    pred[api.TARGET][4,0] = 2.
    return (dict(scale=2.,rms=np.arange(1.,9)),pred,y,np.full(6,3.),
        np.ones(6,bool),np.ones(6,bool),np.array(['a']*6),
        np.array([3,3,3,4,4,4]),np.array([1,1,2,1,1,2]),np.array([9,1,8,2,7,6]))


def test_six_arms_full_and_query_matched_with_unknown_actions_retained():
    r,a = api.evaluate(*fixture())
    assert scalar.verify(*fixture(),r,a) > 500
    assert len(r['policies']) == 16
    assert r['policies'][api.TARGET]['selected_unknown'] == 1
    assert r['policies'][api.TARGET]['known_easy_positive_risk'] == 0.
    assert r['policies'][api.TARGET]['easy_selected_risk_upper'] > .02
    assert not r['policies'][api.TARGET]['finite_completion_supported']
    assert r['pairs']['original_full']['known_difference_mass'] == 2.
    assert r['pairs']['original_full']['unknown_exchanged_envelope_mass'] == 0.
    assert r['contrasts']['original_full_paired_lower_percent'] == pytest.approx(200/29)
    assert r['recording_results']['3'][api.TARGET]['selected_known_harm_mass'] == 0
    assert r['diagnostics']['original']['observed_moments'] == [3.,2.,29.,25.,2.]
    assert r['diagnostics'][api.TARGET]['predicted_observed_easy_harm_ratio'] is None


def test_no_label_or_availability_in_inference_actions():
    args=list(fixture()); _,original=api.evaluate(*args)
    args[2]=np.full_like(args[2],np.nan)
    result,unknown=api.evaluate(*args)
    for name in original:
        np.testing.assert_array_equal(original[name],unknown[name])
    assert result['contrasts']['original_all_MSE'] is None
    assert result['policies'][api.TARGET]['easy_selected_risk_upper'] is None
    assert 'targets' not in inspect.signature(api.decisions).parameters


@pytest.mark.parametrize('missing',api.ARMS)
def test_missing_control_fails_closed(missing):
    args=list(fixture()); del args[1][missing]
    with pytest.raises(ValueError): api.evaluate(*args)


def test_matching_uses_recording_frame_and_stable_ids_not_future_rank():
    args=list(fixture()); args[1][api.TARGET][:,1]=1.
    args[1][api.TARGET][[0,4],1]=.01
    result,actions=api.evaluate(*args)
    assert scalar.verify(*args,result,actions) > 500
    assert set(np.flatnonzero(actions['original_matched_easy_deviance'])) == {1,3}
    assert set(np.flatnonzero(actions['easy_deviance_matched_original'])) == {0,4}


def grid():
    result,_=api.evaluate(*fixture())
    rows=[dict(group=f'c{c}_s{s}',source=f's{s}',head_seed=seed,result=copy.deepcopy(result))
          for s in range(12) for c in range(2) for seed in (17,29,43)]
    expected=[{k:r[k] for k in ('group','source','head_seed')} for r in rows]
    cfg=dict(source_heads=72,neural_fits=144,head_seeds=[17,29,43],risk_budget=.02,
             bootstrap_resamples=3000,bootstrap_seed=20261006)
    return rows,expected,cfg


def test_favorable_paired_utility_does_not_override_risk_failure():
    rows,expected,cfg=grid()
    for r in rows:
        r['result']['contrasts']={k:1. for k in r['result']['contrasts']}
    out=api.summarize(rows,expected,cfg)
    assert scalar.verify_summary(rows,cfg,out) > 300
    assert not out['advance_to_transfer_design']
    assert 'selected_easy_risk_violated_easy_deviance' in out['failure_reasons']
    assert out['safety']['easy_deviance']['easy_upper_violations'] == 72


def test_only_prespecified_primary_contrasts_are_advancement_tests():
    rows,expected,cfg=grid()
    for row in rows:
        result=row['result']
        result['contrasts']={k:1. if '_paired_lower_' in k else 100. for k in result['contrasts']}
        for name in [api.TARGET]+[api.TARGET+'_matched_'+k for k in api.PRIMARY_COMPARATORS]:
            result['policies'][name].update(known_easy_positive_risk=.005,
                easy_selected_risk_upper=.01,finite_completion_supported=True)
    out=api.summarize(rows,expected,cfg)
    assert scalar.verify_summary(rows,cfg,out) > 300
    assert out['advance_to_transfer_design']
    assert out['deployment_changed'] is False and not out['independent_confirmation']
    assert len(out['primary_contrasts']) == 4
    assert len(out['contrasts']['quadratic_full_paired_lower_percent']['localities']) == 12
    rows[0]['result']['contrasts']['quadratic_matched_paired_lower_percent']=None
    assert not api.summarize(rows,expected,cfg)['advance_to_transfer_design']


@pytest.mark.parametrize('mutation',['missing','duplicate','budget','seed'])
def test_no_partial_grid_or_budget_substitution(mutation):
    rows,expected,cfg=grid()
    if mutation=='missing': rows.pop()
    elif mutation=='duplicate': rows[-1]=rows[0]
    elif mutation=='budget': cfg['risk_budget']=.03
    else: rows[0]['head_seed']=1
    with pytest.raises(ValueError): api.summarize(rows,expected,cfg)


@pytest.mark.parametrize('corruption',['risk','tail','action','pair'])
def test_scalar_checks_detect_corrupt_result(corruption):
    result,actions=api.evaluate(*fixture())
    if corruption=='risk': result['policies'][api.TARGET]['easy_selected_risk_upper']=0.
    elif corruption=='tail': result['diagnostics']['original']['observed_moments'][4]=0.
    elif corruption=='action': actions[api.TARGET][1]=True
    else: result['pairs']['original_full']['lower_mass']=99.
    with pytest.raises(AssertionError): scalar.verify(*fixture(),result,actions)
