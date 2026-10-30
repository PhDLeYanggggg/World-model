import copy
import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_active_bank_cost as api
from tests.test_m3w_detector_cost import data as old_data


def packets_and_data():
    torch.set_num_threads(1)
    base = old_data()
    rng = np.random.default_rng(17)
    n = len(base['truth'])
    extra = rng.normal(size=(n, 64))
    sites = np.array(['a'] * 10 + ['b'] * 10)
    original = api.detector.training_data(base, extra, sites, 'history')
    x = base['z'].numpy()[..., :380].copy()
    scene = base['z'].numpy()[..., 380:].copy()
    truth = base['truth'].numpy().astype(float)
    truth[::4] = np.nan
    envelope = np.ones((n, 7))
    bounds, known = api.completion.completion_endpoints(truth, envelope)
    original.update(known=torch.from_numpy(known), truth=torch.tensor(np.nan_to_num(truth), dtype=torch.float32),
                    bound_truth=torch.tensor(bounds, dtype=torch.float32), groups=[np.arange(10), np.arange(10, 20)],
                    draw_pool=dict(rows=n, known_rows=int(known.sum()), unknown_rows=int((~known).sum())))
    rollouts = np.ones((n, 7, 12, 2), np.float32)
    ade = np.where(known[:, None], np.ones((n, 7)), np.nan)
    all_packets = dict(x=x, scene=scene, y=truth, envelope=envelope, scales=np.ones(n),
        sites=sites, recordings=np.array(['r0'] * n), frames=np.arange(n), ids=np.arange(n),
        allowed=np.ones((n, 7), bool), rollouts=rollouts, floor_rollout=np.zeros((n, 12, 2), np.float32),
        ade=ade, fde=ade.copy(), floor_ade=ade[:, 0], floor_fde=ade[:, 0], cv_ade=ade[:, 0],
        hard=np.zeros(n, bool), args=('old',))
    packets = {}
    changed = {}
    for name in ('a', 'b'):
        at = sites == name
        packet = {k: v[at] if isinstance(v, np.ndarray) else v for k, v in all_packets.items()}
        fields = dict(x=packet['x'][:, 6].copy(), envelope=packet['envelope'][:, 6].copy(),
                      rollout=packet['rollouts'][:, 6].copy(), scene=packet['scene'][:, 6].copy())
        packets[name] = packet
        changed[name] = api.replace_packet(packet, fields, causal_eligible=np.ones(at.sum(), bool),
            labels=packet['y'][:, 6], ade=packet['ade'][:, 6], fde=packet['fde'][:, 6])
    original['input_hashes'] = {key: api.concatenated_fingerprint([packets[n][key] for n in sorted(packets)])
        for key in ('x', 'envelope', 'y', 'allowed')}
    original['input_hashes'].update(scene_causal_features=api.fingerprint(scene), completion_endpoints=api.fingerprint(bounds))
    return original, packets, changed


def test_chunked_replacement_fingerprint_matches_materialized_and_never_changes_physical_columns():
    rng = np.random.default_rng(18)
    base = rng.normal(size=(23, 7, 4)).astype(np.float32)
    new = rng.normal(size=(23, 4)).astype(np.float32)
    before = base.copy()
    wrapped = api.ReplacedActionArray(base, new)
    out = np.asarray(wrapped)
    np.testing.assert_array_equal(out[:, :6], before[:, :6])
    np.testing.assert_array_equal(out[:, 6], new)
    np.testing.assert_array_equal(base, before)
    assert api.concatenated_fingerprint([wrapped], chunk=3) == api.fingerprint(out)
    assert api.concatenated_fingerprint([out[:9], out[9:]], chunk=2) == api.fingerprint(out)
    with pytest.raises(ValueError):
        wrapped[0]


def test_neutral_replacement_keeps_features_preprocess_unknowns_and_original_sampler():
    original, old, changed = packets_and_data()
    data = api.frozen_training_data(original, old, changed)
    np.testing.assert_array_equal(data['z'][np.arange(20)].numpy(), original['z'][np.arange(20)].numpy())
    assert data['preprocess'] is original['preprocess'] and data['groups'] is original['groups']
    api.completion.base.exact(data['truth'], original['truth'])
    api.completion.base.exact(data['input_hashes'], original['input_hashes'])
    assert 'args' not in changed['a'] and np.isnan(changed['a']['y'][::4]).any()
    assert 'target' not in inspect.signature(api.causal_neural_fields).parameters
    with pytest.raises(TypeError):
        api.causal_neural_fields(future_endpoint=np.ones((2, 2)))


def test_neural_change_does_not_refit_normalization_or_remove_alias_rows_from_draw_pool():
    original, old, changed = packets_and_data()
    for packet in changed.values():
        packet['x'].replacement[:] += 1000
        packet['scene'].replacement[:] -= 1000
        packet['allowed'][:, 6] = False
    data = api.frozen_training_data(original, old, changed)
    assert data['preprocess'] is original['preprocess'] and data['groups'] is original['groups']
    torch.testing.assert_close(data['z'][np.arange(20)][:, :6], original['z'][np.arange(20)][:, :6], rtol=0, atol=0)
    assert not torch.equal(data['z'][np.arange(20)][:, 6], original['z'][np.arange(20)][:, 6])
    assert data['draw_pool'] == original['draw_pool']
    assert data['input_hashes']['x'] != original['input_hashes']['x']
    changed['a']['floor_ade'] = changed['a']['floor_ade'].copy()
    changed['a']['floor_ade'][1] += 1
    with pytest.raises(AssertionError):
        api.frozen_training_data(original, old, changed)


def test_replaced_label_reference_and_future_support_are_checked():
    _, packets, changed = packets_and_data()
    p = packets['a']
    fields = {k: v for k, v in dict(x=changed['a']['x'].replacement,
        scene=changed['a']['scene'].replacement, envelope=p['envelope'][:, 6], rollout=p['rollouts'][:, 6]).items()}
    labels = p['y'][:, 6].copy()
    labels[1, 2] += 1
    with pytest.raises(AssertionError):
        api.replace_packet(p, fields, causal_eligible=np.ones(10, bool), labels=labels, ade=p['ade'][:, 6], fde=p['fde'][:, 6])
    labels[0] = 0
    with pytest.raises(ValueError):
        api.replace_packet(p, fields, causal_eligible=np.ones(10, bool), labels=labels, ade=p['ade'][:, 6], fde=p['fde'][:, 6])


def test_real_torch_bank_recovery_common_design_and_inactive_action_control(tmp_path):
    original, old, changed = packets_and_data()
    new = api.frozen_training_data(original, old, changed)
    settings = dict(width=64, steps=6, batch_size=8, learning_rate=.001,
        underharm_weight=4., ranking_weight=.5, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    states = []
    for variant in api.VARIANTS:
        data = new if variant == 'new_neural' else original
        kw = dict(variant=variant, settings=settings, seed=17, identity=dict(fixture=True))
        direct = api.fit(data, **kw, path=tmp_path / (variant + '.gz'))
        api.fit(data, **kw, path=tmp_path / (variant + '_resumed.gz'), stop_at=3)
        resumed = api.fit(data, **kw, path=tmp_path / (variant + '_resumed.gz'), resume=True)
        api.assert_scientific_equal(direct, resumed)
        states.append(direct)
    for state in states[1:]:
        api.assert_common_design(states[0], state)
    api.assert_scientific_equal(states[1], states[2])
    for packet in changed.values():
        packet['x'].replacement += 999
    altered = api.frozen_training_data(original, old, changed)
    physical = api.fit(altered, variant='physical_only', settings=settings, seed=17, identity={}, path=tmp_path/'inactive.gz')
    for field in ('model', 'optimizer', 'trace', 'draw_hash'):
        api.completion.base.exact(states[0][field], physical[field])
