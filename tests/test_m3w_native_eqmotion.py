from copy import deepcopy
import numpy as np
import pytest
import torch
from scripts.run_m3w_native_eqmotion import check_matching, trial_identity
from src.world_model.m3w_native_forecast import fold_design, fit_trial, predict
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_eqmotion_adapter import ROOT, SPEC, verified_source
import json
from test_m3w_native_forecast import population


def test_draw_matching_rejects_hidden_exposure_and_budget_drift():
    data = population()
    fold = fold_design(data, 'c', 'native_coordinate')
    cp = dict(train_ids=fold['train_ids'], factors=fold['factors'], step=4,
              draws=np.array([2]*8+[0]*4))
    check_matching(cp, deepcopy(cp), fold, 4, 4)
    altered = deepcopy(cp)
    altered['draws'][0] -= 1
    altered['draws'][8] += 1
    with pytest.raises(AssertionError):
        check_matching(altered, cp, fold, 4, 4)
    with pytest.raises(ValueError):
        check_matching(altered, deepcopy(altered), fold, 4, 4)


def test_training_identity_changes_with_role_membership():
    fold = fold_design(population(), 'c', 'native_coordinate')
    view = dict(fold=fold, site='c', seed=17)
    a = trial_identity({'frozen': True}, 'c_seed17', view)
    view['fold'] = deepcopy(fold)
    view['fold']['train_ids'] = fold['train_ids'][:-1]
    b = trial_identity({'frozen': True}, 'c_seed17', view)
    assert a['train_ids_sha256'] != b['train_ids_sha256']


def test_real_eqmotion_native_objective_resume_and_future_invariance(tmp_path):
    source = ROOT/json.loads(SPEC.read_text())['destination']
    if not (source/'eth_ucy/model_t.py').is_file():
        pytest.skip('Optional pinned author source absent; tests never download it')
    verified_source(source)
    torch.set_num_threads(2)
    data = population()
    fold = fold_design(data, 'c', 'native_coordinate')
    architecture = dict(family='eqmotion_fixed_head', history_steps=8, prediction_steps=12,
        hidden_nf=8, channels=8, layers=2, fixed_head=0, prune_unused_heads=True,
        input_conditioning='observed_joint_max_norm')
    settings = dict(steps=4, batch_size=4, learning_rate=.0003, minimum_lr_ratio=.01,
        weight_decay=.0001, gradient_clip=5., checkpoint_every=2, heartbeat_every=1)
    def run(path, resume=False, stop_at=None):
        torch.manual_seed(17)
        m = build_forecaster(architecture)
        fit = fit_trial(m, data, fold, seed=17, settings=settings, identity={'synthetic': True},
            directory=path, resume=resume, stop_at=stop_at, heartbeat=lambda **kw: None)
        return m, fit
    full, report = run(tmp_path/'full')
    run(tmp_path/'resume', stop_at=2)
    resumed, result = run(tmp_path/'resume', resume=True)
    assert report['held_rows_sampled'] == 0 and result['new_updates'] == 2
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    before = predict(resumed, data, fold['held_ids'])
    data['target'][:] = 1e10
    data['valid'][:] = False
    np.testing.assert_array_equal(before, predict(resumed, data, fold['held_ids']))
    assert np.isfinite(before).all()
