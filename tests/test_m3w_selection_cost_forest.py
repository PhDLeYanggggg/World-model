import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_selection_cost_forest as api


def example():
    rng = np.random.default_rng(5); n, d = 8, 4
    c = dict(x=rng.normal(size=(n,7,d)).astype(np.float32), envelope=np.full((n,7), 3.),
        scales=np.ones(n), allowed=np.ones((n,7), bool), groups=[np.arange(3),np.arange(3,7),np.array([7])],
        source_groups=[np.arange(3)], sites=np.repeat('source',n), ids=np.arange(n))
    c['allowed'][0, 2:] = False
    target = rng.uniform(.02,1,(n,7,5)).astype(np.float32)
    return dict(causal=c, preprocess=dict(mean=np.zeros(d),std=np.ones(d),clip=8.,rms=np.ones(5)),
                truth=torch.from_numpy(target), known=np.ones(n,bool))


def test_augmented_targets_match_original_absolute_cost_objective():
    data = example(); c = data['causal']; counts = np.array([4,5,7])
    matrix = api.training_matrix(data, counts, 'row_forest')
    rows, actions = np.nonzero(c['allowed']); pred = data['truth'].numpy()+.1
    augmented = api.augmented_targets(pred[rows,actions], data['preprocess']['rms'])
    measured = np.average(np.square(augmented-matrix['y']).mean(1), weights=matrix['weight'])
    per_row = np.square(pred-data['truth'].numpy()).mean(-1)
    gain = ((pred[:,:,0]-pred[:,:,1])-(data['truth'][:,:,0]-data['truth'][:,:,1]).numpy())/np.sqrt(2)
    per_row += .5*np.square(gain)
    per_row = (per_row*c['allowed']).sum(1)/c['allowed'].sum(1)
    expected = sum(counts[q]*per_row[ix].mean() for q,ix in enumerate(c['groups']))/counts.sum()
    assert measured == pytest.approx(expected, abs=1e-7)


def test_query_context_is_past_only_and_singleton_zero():
    data = example(); c = data['causal']
    z, env, context = api.encoded(c, data['preprocess'], 'context_forest')
    np.testing.assert_allclose(context[0,:,:-1], z[1:3].mean(0), atol=1e-7)
    assert not context[7].any()
    c['future_endpoint'] = np.ones((8,2))
    with pytest.raises(ValueError, match='past-only'): api.encoded(c, data['preprocess'], 'context_forest')


def test_unknown_training_rows_are_not_supervised_and_counts_need_complete_coverage():
    data = example(); data['known'][2] = False; data['truth'][2] = float('nan')
    matrix = api.training_matrix(data, np.array([4,5,7]), 'row_forest')
    assert matrix['positive_query_rows'] == 7 and np.isfinite(matrix['y']).all()
    with pytest.raises(ValueError, match='multiplicities'):
        api.training_matrix(data, np.array([4,0,7]), 'row_forest')


def test_exact_forest_resume_and_candidate_invariant_references(tmp_path):
    data = example(); matrix = api.training_matrix(data, np.array([4,5,7]), 'context_forest')
    settings = dict(trees=4,max_depth=4,min_samples_leaf=1,max_features=.5,fit_threads=2,checkpoint_every=2)
    args = dict(preprocess=data['preprocess'],arm='context_forest',settings=settings,seed=17,identity={'test':True})
    api.fit(matrix, **args, path=tmp_path/'a.gz', stop_at=2)
    resumed = api.fit(matrix, **args, path=tmp_path/'a.gz', resume=True)
    direct = api.fit(matrix, **args, path=tmp_path/'b.gz')
    api.exact_forest(resumed, direct)
    pred = api.predict(resumed, data['causal'])
    assert np.isfinite(pred).all() and (pred >= 0).all()
    np.testing.assert_array_equal(pred[:,:,2],np.repeat(pred[:,0,2,None],7,axis=1))
    np.testing.assert_array_equal(pred[:,:,3],np.repeat(pred[:,0,3,None],7,axis=1))
    assert (pred[:,:,4] <= pred[:,:,1]).all()
    changed = copy.deepcopy(matrix); changed['input_hashes']['weights'] = 'changed'
    with pytest.raises(AssertionError): api.fit(changed, **args, path=tmp_path/'a.gz', resume=True)


def test_positive_development_effect_cannot_override_original_risk_and_missing_roster():
    from scripts.run_m3w_selection_cost_forest import summarize
    rows = []
    for fold in range(3):
        for seed in (17,29,43):
            for site in range(6):
                view = dict(supported=False,known_all_risk=.03,known_easy_risk=.05,
                            completion_easy_risk_upper=.06,known_easy_net_degradation=.01)
                policies = {a:dict(known_ADE_gain_floor_percent=2.,recordings={'r':view})
                            for a in ('mean_absolute',*api.ARMS)}
                effects = {a:dict(known_pp=1.,completion_lower_pp=.5,completion_upper_pp=1.5) for a in api.ARMS}
                rows.append(dict(trial=f'single{fold}_seed{seed}',site=str(site),policies=policies,effects=effects))
    cfg = dict(bootstrap_resamples=3000,bootstrap_seed=20261102)
    out = summarize(rows,cfg)
    assert out['primary_pass'] and not out['original_record_risk_pass'] and not out['candidate_pass']
    assert not out['deployment_changed']
    with pytest.raises(ValueError): summarize(rows[:-1],cfg)
