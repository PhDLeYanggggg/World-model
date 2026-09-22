from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from scripts.run_m3w_eqmotion_nested import specification, check_checkpoint, read_prediction
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_eqmotion_adapter import ROOT, SPEC, verified_source
from test_m3w_native_nested import population


def design():
    reg = dict(sites=['a', 'b', 'c', 'd'], seeds=[17])
    return specification(reg, population(), {'frozen': True}, ['a', 'b'], 17)


def test_identity_binds_family_both_exclusions_and_loss_factors():
    fold, ti, key = design()
    assert key == 'a__b_seed17'
    assert ti['producer']['family'] == 'eqmotion_fixed_head'
    assert ti['producer']['training_sites'] == ['c', 'd']
    assert ti['producer']['excluded_sites'] == ['a', 'b']
    assert ti['producer']['parents'] == []
    assert not fold['factors'][:8].any()
    poisoned = population(); poisoned['target'][:8] = 1e10
    poisoned['valid'][:8] = False; poisoned['scale'][:8] = -1
    newfold, newti, _ = specification(dict(sites=list('abcd')), poisoned, {'frozen': True}, ['b', 'a'], 17)
    assert ti == newti
    np.testing.assert_array_equal(fold['factors'], newfold['factors'])


def test_checkpoint_rejects_family_settings_draws_and_held_exposure():
    fold, ti, _ = design(); settings = dict(steps=4, batch_size=4)
    cp = dict(identity=ti, settings=settings, seed=17, train_ids=fold['train_ids'],
        factors=fold['factors'], step=4, draws=np.array([0]*8+[2]*8))
    check_checkpoint(cp, deepcopy(cp), fold, ti, settings)
    for field, value in [('seed', 29), ('identity', {}), ('settings', {})]:
        bad = deepcopy(cp); bad[field] = value
        with pytest.raises(ValueError):
            check_checkpoint(bad, cp, fold, ti, settings)
    bad = deepcopy(cp); bad['draws'][0] = 1; bad['draws'][8] -= 1
    with pytest.raises(AssertionError):
        check_checkpoint(bad, cp, fold, ti, settings)
    with pytest.raises(ValueError):
        check_checkpoint(bad, deepcopy(bad), fold, ti, settings)


@pytest.mark.parametrize('kind', ['labels', 'misaligned_ids', 'nonfinite', 'wrong_grid'])
def test_inference_archives_cannot_hide_labels_or_change_alignment(tmp_path, kind):
    ids = np.arange(2)
    arrays = dict(ids=ids, prediction=np.zeros((2, 12, 2), np.float32))
    good = tmp_path/'good.npz'; write_arrays(good, arrays)
    np.testing.assert_array_equal(read_prediction(good, ids), arrays['prediction'])
    if kind == 'labels':
        arrays['future_endpoint'] = np.zeros((2, 2))
    elif kind == 'misaligned_ids':
        arrays['ids'] = ids[::-1]
    elif kind == 'nonfinite':
        arrays['prediction'][0, 0, 0] = np.nan
    else:
        arrays['prediction'] = arrays['prediction'][:, :-1]
    path = tmp_path/'bad.npz'; write_arrays(path, arrays)
    with pytest.raises((AssertionError, ValueError)):
        read_prediction(path, ids)


def test_actual_pair_eqmotion_resume_matches_and_labels_are_not_inputs(tmp_path):
    source = ROOT/json.loads(SPEC.read_text())['destination']
    if not (source/'eth_ucy/model_t.py').is_file():
        pytest.skip('Pinned optional author source absent; no test downloads')
    verified_source(source); torch.set_num_threads(2)
    data = population(); fold, ti, _ = design()
    architecture = dict(family='eqmotion_fixed_head', history_steps=8, prediction_steps=12,
        hidden_nf=8, channels=8, layers=2, fixed_head=0, prune_unused_heads=True,
        input_conditioning='observed_joint_max_norm')
    settings = dict(steps=4, batch_size=4, learning_rate=.0003, minimum_lr_ratio=.01,
        weight_decay=.0001, gradient_clip=5, checkpoint_every=2, heartbeat_every=1)
    def fit(path, resume=False, stop_at=None):
        torch.manual_seed(17); model = build_forecaster(architecture)
        report = fit_trial(model, data, fold, seed=17, settings=settings, identity=ti,
            directory=path, resume=resume, stop_at=stop_at, heartbeat=lambda **_:None)
        return model, report
    full, _ = fit(tmp_path/'full')
    fit(tmp_path/'resume', stop_at=2)
    resumed, result = fit(tmp_path/'resume', resume=True)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    assert result['held_rows_sampled'] == 0 and result['new_updates'] == 2
    before = predict(resumed, data, fold['held_ids'])
    data['target'][:] = 1e10; data['valid'][:] = False
    np.testing.assert_array_equal(before, predict(resumed, data, fold['held_ids']))
