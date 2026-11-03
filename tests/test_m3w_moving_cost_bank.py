import numpy as np
import pytest

from src.data_unification.m3w_moving_cost_bank import aligned_rows, bank, job_designs
from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model import m3w_shared_motion_cost as api


def fixture():
    h = np.stack([np.column_stack((np.arange(8.)*(i+1), np.arange(8.)*.5+i)) for i in range(6)])
    g, _, context = encode_scene(h, np.ones((6, 8), bool), np.arange(6))
    data = dict(sites=np.array(['a', 'a', 'b', 'b', 'c', 'c']), recordings=np.arange(6),
        frames=np.full(6, 7), history=h, origin=h[:, -1], geometry=g, shared_context=context,
        target_eval=np.ones((6, 12, 2))*20, valid=np.ones((6, 12), bool))
    folds = dict(a=0, b=1, c=2, empty=0)
    producers = [dict(family='eqmotion', fold=f, seed=s,
        fit_groups=[g for g, k in folds.items() if k == f], baseline_index=1,
        easy_cut=1., hard_cut=5.) for f in range(3) for s in (17, 29, 43)]
    settings = dict(query_batch_size=3, minimum_updates=2, largest_source_cycles=4)
    return data, job_designs(data, folds, producers, settings)


def test_all_role_combinations_and_future_free_evaluation_bank():
    data, jobs = fixture()
    assert len(jobs) == 18
    assert {j['seed'] for j in jobs} == {17, 29, 43}
    for job in jobs:
        assert not set(job['producer_sites']) & set(job['critic_sites'])
        assert not set(job['producer_sites']) & set(job['evaluation_sites'])
        assert job['fit_scope']['budget'] >= 4
    job = jobs[0]
    ids = np.flatnonzero(~np.isin(data['sites'], job['producer_sites']))
    pred = np.ones((len(ids), 12, 2), np.float32)
    a = bank(data, job, ids, pred)
    assert 'learning' not in a
    data['target_eval'][:] = -999999
    data['valid'][:] = False
    b = bank(data, job, ids, pred)
    np.testing.assert_array_equal(a['causal']['x'], b['causal']['x'])
    assert not a['causal']['allowed'][:, :6].any()
    assert set(a['sites']) == set(job['evaluation_sites'])
    with pytest.raises(ValueError, match='disjoint'):
        bank(data, dict(job, producer_sites=job['evaluation_sites']), ids, pred)


def test_alignment_never_silently_reorders_or_drops_rows():
    np.testing.assert_array_equal(aligned_rows(np.array([1, 3, 5]), np.array([3, 5])), [1, 2])
    for wanted in (np.array([1, 7]), np.array([1, 1]), np.array([5, 3]), np.array([0])):
        with pytest.raises(ValueError):
            aligned_rows(np.array([1, 3, 5]), wanted)


def test_no_implicit_unequal_group_sampler():
    data, _ = fixture()
    folds = dict(a=0, b=1, c=2, empty=0)
    with pytest.raises(ValueError, match='three-seed'):
        job_designs(data, folds, [], dict(query_batch_size=3, minimum_updates=2, largest_source_cycles=4))
    data['sites'][1] = 'd'
    folds['d'] = 0
    with pytest.raises(ValueError, match='Equal source sampling'):
        job_designs(data, folds, [], dict(query_batch_size=3, minimum_updates=2, largest_source_cycles=4))


def test_matched_heads_resume_exactly_without_held_preprocessing(tmp_path):
    data, jobs = fixture()
    job = jobs[0]
    ids = np.flatnonzero(~np.isin(data['sites'], job['producer_sites']))
    pred = np.ones((len(ids), 12, 2), np.float32)
    fitting = bank(data, job, ids, pred, training=True)
    learned = fitting['learning']
    assert learned['preprocess']['source_names'] == job['critic_sites']
    settings = dict(width=8, learning_rate=.001, query_batch_size=3,
        minimum_updates=4, largest_source_cycles=1, gradient_clip=5.,
        heartbeat_every=1, checkpoint_every=1)
    states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz')
        api.fit(learned, arm=arm, seed=17, settings=settings, identity=dict(job=job),
            path=path, stop_at=2)
        resumed = api.fit(learned, arm=arm, seed=17, settings=settings, identity=dict(job=job),
            path=path, resume=True)
        direct = api.fit(learned, arm=arm, seed=17, settings=settings, identity=dict(job=job),
            path=tmp_path/(arm+'_direct.pt.gz'))
        api.base.base.exact({k:v for k,v in resumed.items() if k != 'seconds'},
            {k:v for k,v in direct.items() if k != 'seconds'})
        states.append(resumed)
    for key in ('initial_model', 'parameters', 'budget', 'sampler', 'draw_hash',
                'fit_ids', 'preprocess', 'torch_rng'):
        api.base.base.exact(states[0][key], states[1][key])
