import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from src.world_model import m3w_source_error_priority as api
from tests.test_m3w_kinematic_risk import fixture


def prepared():
    d = fixture()
    d['preprocess']['source_names'] = sorted(set(d['causal']['sites']))
    return api.teacher_data(d)


def test_circular_matched_discovery_and_source_provenance():
    sites = np.array(['a', 'a', 'b', 'c'])
    predictions = {n: np.full((4, 7, 2), i, float) for i, n in enumerate('abc')}
    provenance = {n: sorted(set('abc')-{n}) for n in 'abc'}
    pair, mapping = api.discovery_pair(sites, predictions, provenance)
    assert pair['held_source_errors'][:, 0, 0].tolist() == [0, 0, 1, 2]
    assert pair['in_source_errors'][:, 0, 0].tolist() == [1, 1, 2, 0]
    assert [r['rows'] for r in mapping] == [2, 1, 1]
    provenance['a'] = ['a', 'b']
    with pytest.raises(ValueError, match='exclude exactly'):
        api.discovery_pair(sites, predictions, provenance)


def test_source_subset_preserves_complete_queries_and_original_ids():
    c = fixture()['causal']; c['sites'] = np.array(['a', 'a', 'b', 'b'])
    selected, ix = api.source_subset(c, {'b'})
    assert ix.tolist() == [2, 3]
    np.testing.assert_array_equal(selected['ids'], c['ids'][ix])
    assert [v.tolist() for v in selected['groups']] == [[0, 1]]
    assert [v.tolist() for v in selected['source_groups']] == [[0]]
    c['sites'] = np.array(['a', 'b', 'a', 'b'])
    with pytest.raises(ValueError, match='split a causal query'):
        api.source_subset(c, {'b'})


def test_own_normalization_causal_schema_and_unknown_priority():
    d = prepared(); before = d['extra_preprocess']['mean'].copy()
    d['truth_by_arm']['bound_aware'][:] = 123
    np.testing.assert_array_equal(api.prepare_basis(d)['extra_preprocess']['mean'], before)
    d['observed_future'][0] = False
    d['truth_by_arm']['bound_aware'][1, 6, 4] = 1000
    a = api.prioritize(d, np.full((4, 7, 2), -1.))
    assert a['priority_mask'][1, 6, 1] and not a['priority_mask'][0].any()
    assert not (a['priority_mask'] & ~torch.from_numpy(d['causal']['allowed'])[..., None]).any()
    d['preprocess']['source_names'] = ['held']
    with pytest.raises(ValueError, match='Own-source'):
        api.prepare_basis(d)
    d = prepared(); d['causal']['future_endpoint'] = np.zeros((4, 2))
    with pytest.raises(ValueError, match='Own-source'):
        api.prepare_basis(d)


def test_real_scratch_resume_and_matched_student_updates(tmp_path):
    torch.set_num_threads(1); d = prepared()
    settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=4,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    warm = api.scratch(settings, 17)
    assert warm['step'] == 0
    kw = dict(arm='continued_mse', settings=settings, identity={'scratch': True})
    api.parent.fit(d, warm, **kw, path=tmp_path/'resume.pt.gz', stop_at=2)
    a = api.parent.fit(d, warm, **kw, path=tmp_path/'resume.pt.gz', resume=True)
    b = api.parent.fit(d, warm, **kw, path=tmp_path/'direct.pt.gz')
    api.kin.parent.parent.router.exact_state(a, b)
    np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
    states = []
    for i, arm in enumerate(api.ARMS):
        chosen = api.prioritize(d, np.full((4, 7, 2), -1.+2*i))
        states.append(api.parent.fit(chosen, a, arm='false_safe_priority', settings=settings,
            identity={'arm': arm}, path=tmp_path/(arm+'.pt.gz')))
    api.assert_student_pair(*states)
    before = api.predict(states[0], d['causal'])
    d['truth_by_arm']['bound_aware'][:] = -999
    np.testing.assert_array_equal(before, api.predict(states[0], d['causal']))
    with pytest.raises(ValueError, match='causal schema'):
        api.predict(a, dict(d['causal'], future_endpoint=np.zeros((4, 2))))


def test_full_preprocessing_exposure_rejected():
    d = prepared(); c = d['causal']
    s = dict(fit_ids=np.array([999]), source_sites=['other'], preprocess=dict(source_names=['other']),
        extra_preprocess=dict(source_names=['other']))
    api.assert_excluded(s, c, sorted(set(c['sites'])))
    s['extra_preprocess']['source_names'].append(c['sites'][0])
    with pytest.raises(ValueError, match='Held source'):
        api.assert_excluded(s, c, sorted(set(c['sites'])))


def test_prospective_config_and_outcome_order():
    from scripts import run_m3w_moving_source_error_priority as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['teacher_heads'] == 8 and cfg['student_heads'] == 6
    assert cfg['risk_budget'] == cfg['easy_net_budget'] == .02
    assert not any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification'))
    tree = ast.parse(Path(run.__file__).read_text())
    f = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert 'api.scratch(' in f['teacher_fit']
    assert "training=True" in f['teacher_packet'] and 'student_data' not in f['teacher_packet']
    assert f['evaluate'].index('api.kin.parent.choose(') < f['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 6" in f['evaluate']
