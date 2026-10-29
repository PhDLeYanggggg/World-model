import numpy as np
import pytest
from src.world_model import m3w_cost_head_census as api
from tests.test_m3w_easy_harm_deviance_training import data, settings
from tests.test_m3w_task_risk_training import CONFIG


def test_unknown_outcomes_stay_selected_and_bounded():
    y = np.array([[0.,.01,1.,1.,.01],[np.nan]*5])
    prediction = np.array([[.1,0.,1.,1.,0.]]*2)
    row = api.cohort(prediction,y,np.ones(2,bool),np.ones(2))
    assert row['selected_count'] == 2 and row['selected_unknown'] == 1
    assert row['known_easy_risk'] == .01 and row['easy_selected_risk_upper'] == 1.01
    assert not row['finite_completion_supported']


def test_sampler_and_full_census_bind_to_frozen_TRAIN(tmp_path):
    args = list(data())
    args[0] = np.random.default_rng(41).normal(size=(24,380))
    args[2] = np.array([[.3,0.,1.,1.,0.] if i%2 else [0.,.2,1.,1.,.2] for i in range(24)])
    args[3] = np.stack([np.tile([r[1]-r[0],r[2]],(12,1)) for r in args[2]])
    args[2][-1] = np.nan; args[3][-1] = np.nan
    args[-1] = api.train.core.preprocess(args[0],args[1],args[2],*args[4:7],training_site='site')
    eligible = api.moving_from_features(args[0])
    _, support = api.train.parent.forest.causal_inputs(args[0],args[1],args[-1]); eligible &= support
    state = api.train.fit(*args, eligible=eligible, arm='task_risk', settings=settings(),
        objective_config=CONFIG, seed=17, identity={'test':True}, experiment_sha256='test',
        path=tmp_path/'model.pt.gz', heartbeat=lambda **_:None)
    coverage = api.sampling_coverage(state,args)
    assert coverage['sampler_exact'] and coverage['known_rows'] == 23
    row = api.census(state,args)
    assert row['full_TRAIN_rows'] == 24 and row['inference_replay_exact']
    assert not row['inference_outcome_filter']
    wrong = list(args); wrong[0] = args[0].copy(); wrong[0][0,0] += 1
    with pytest.raises(ValueError): api.census(state,wrong)


def test_empty_selection_not_safe_and_zero_harm_ratio_undefined():
    y = np.array([[.1,0.,1.,1.,0.]])
    row = api.cohort(y,y,np.zeros(1,bool),np.ones(1))
    assert row['known_easy_risk'] is None
    assert row['predicted_observed_easy_harm_ratio'] is None
    assert not row['finite_completion_supported']
