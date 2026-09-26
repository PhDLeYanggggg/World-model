import copy
import json
import numpy as np
import pytest
import torch
from src.world_model import m3w_aux_trajectory as m
from src.world_model import m3w_strong_cap_auxiliary as strong
from tests.test_m3w_strong_cap_auxiliary import data


@pytest.mark.parametrize('arm', strong.ARMS)
def test_instrumentation_and_segmented_resume_are_numerically_exact(tmp_path, arm):
    torch.set_num_threads(4)
    x, y, easy, event, sites, env, pr, cfg = data()
    kw = dict(arm=arm, seed=17, settings=cfg, identity={'run': 'fixture'}, heartbeat=lambda **_: None)
    strong.fit(x, y, easy, event, sites, 'outside', env, pr, directory=tmp_path/'reference', **kw)
    reference = strong.restore(tmp_path/'reference')[1]
    spec = m.severity_spec(y, env, pr['weights'], [.5, .9, .99])
    z, target = m.tensors(x, y, pr)
    for step in (2, 4, cfg['steps']):
        model, _ = strong.fit(x, y, easy, event, sites, 'outside', env, pr, directory=tmp_path/'replay',
                             resume=step != 2, stop_at=step, **kw)
        state = strong.restore(tmp_path/'replay')[1]
        before = copy.deepcopy(model.state_dict())
        result = m.measure(model, state, z, target, y, env, sites, spec, event)
        m.gradients(model, state, z, target, env, event, [state['fixed_ids']])
        assert set(result) == set(sites)
        m.exact_tree(before, model.state_dict())
        m.snapshot(state, tmp_path/f'{step}.pt'); m.snapshot(state, tmp_path/f'{step}.pt')
    m.match_final(strong.restore(tmp_path/'replay')[1], reference)


def test_severity_partition_zero_unknown_and_ties():
    y = np.zeros((8, 4)); y[:, 3] = [0, 1, 1, 2, 10, 20, 100, np.nan]
    y[-1] = np.nan; env = np.ones(8); w = np.ones(8)/7; w[-1] = 0
    spec = m.severity_spec(y, env, w, [.5, .9, .99])
    masks = m.strata(y, env, spec)
    union = sum(masks[k].astype(int) for k in ('severity_0_50', 'severity_50_90', 'severity_90_99', 'severity_99_100'))
    np.testing.assert_array_equal(union, masks['positive_easy_harm'].astype(int))
    assert masks['zero_easy_harm'].sum() == 1 and not any(mask[-1] for mask in masks.values())
    bad = w.copy(); bad[-1] = 1
    with pytest.raises(ValueError): m.severity_spec(y, env, bad, [.5, .9, .99])


def test_snapshot_changed_state_rejected(tmp_path):
    state = dict(model={'w': torch.ones(3)}, optimizer={}, step=200, settings={}, seed=17, arm='cost_only', identity={})
    m.snapshot(state, tmp_path/'model.pt')
    state['model']['w'][0] = 2
    with pytest.raises(AssertionError): m.snapshot(state, tmp_path/'model.pt')


def test_original_extra_pilot_trace_only():
    keys = ('model', 'optimizer', 'initial_model', 'draws', 'sampler_rng', 'torch_rng',
            'loss_scales', 'fixed_ids', 'preprocess', 'auxiliary_target')
    state = {k: {} for k in keys}
    state.update(settings={'heartbeat_every': 200}, seed=17, arm='cost_only', step=2000,
                 trace=[{'step': s, 'loss': 1.} for s in [0, 1, *range(200, 2001, 200)]])
    old = copy.deepcopy(state); old['trace'].insert(2, {'step': 100, 'loss': 2.})
    assert m.match_final(state, old)['original_extra_pilot_steps'] == [100]
    old['trace'][2]['step'] = 101
    with pytest.raises(AssertionError): m.match_final(state, old)
    old = copy.deepcopy(state); old['trace'][2]['loss'] += 1
    with pytest.raises(AssertionError): m.match_final(state, old)


def test_complete_aggregate_and_missing_strata_are_json_safe():
    from scripts.report_m3w_european_aux_trajectory import aggregate
    cfg = dict(arms=list(strong.ARMS), seeds=[17, 29, 43], bootstrap_seed=4, bootstrap_resamples=3000)
    rows = []
    for seed in cfg['seeds']:
        for outer in ('a', 'b', 'c', 'd'):
            for arm in cfg['arms']:
                for step in (200, 600):
                    value = 1. if arm == 'cap_aux' else 2.
                    rows.append(dict(identity=dict(pair='full', producer=1, controller=2, seed=seed,
                        excluded_locality=outer, arm=arm), step=step,
                        localities={s: {'all': dict(cost4=value, easy_harm=value),
                            'empty': dict(cost4=None, easy_harm=None)} for s in ('a', 'b', 'c', 'd') if s != outer},
                        gradients=[{'actual_training': {m: dict(cosine=None, norm_ratio=None)
                            for m in ('cost4_shared', 'easy_harm_shared')}}]))
    result = aggregate(rows, cfg); json.dumps(result, allow_nan=False)
    assert all(r['point'] == 50. for r in result['comparisons'] if r['stratum'] == 'all')
    assert all(r['status'] == 'not_estimable' for r in result['comparisons'] if r['stratum'] == 'empty')
    assert all(r['positive_steps'] == [200, 600] for r in result['timelines'] if r['stratum'] == 'all')
    with pytest.raises(AssertionError): aggregate(rows+rows[:1], cfg)
