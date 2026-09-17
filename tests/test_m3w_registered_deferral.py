from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_m3w_development_evaluation import fitted_fixture, BaselineCopy, GEOMETRY
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_registered_deferral import (
    registered_spec, fit_registered, load_registered, matched_rows, infer_scene,
)
from src.world_model.m3w_cost_sensitive_deferral import make_oof_deferral_rows, DeferralHead
from src.world_model.m3w_supervised_intervention import ContractForecastDataset, load_verified_forecaster


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def fixture(tmp_path):
    def development_only(protocol):
        protocol['study_design'] = 'development_only'
        protocol['scope'] = 'exploratory'
        protocol['risk']['delta'] = None
        protocol['risk']['risks'] = []
        for name, role in protocol['assignments'].items():
            if role in ('calibration', 'confirmation'):
                protocol['assignments'][name] = 'excluded'
    contract, plan = fitted_fixture(tmp_path, protocol_updates=development_only)
    registration = {'schema_version': 1, 'parent_protocol_sha256': contract.digest,
                    'scope': 'development_diagnostic_only', 'eligible_for_selection': False, 'bindings': {},
                    'variants': {'linear': {'cost_bound': 1., 'width': 0, 'fit_settings': {
                        'steps': 8, 'batch_size': 8, 'learning_rate': .01, 'checkpoint_every': 2, 'heartbeat_every': 2}}}}
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(registration))
    reference = {'path': path.name, 'sha256': file_digest(path), 'variant': 'linear'}
    baseline = plan['candidates'][0]['baseline']
    data = ContractForecastDataset(contract, ['b'], purpose='fit', baseline_name=baseline)
    model = load_verified_forecaster(contract, 'fit_a', device='cpu')
    group = make_oof_deferral_rows(contract, 'fit_a', data, model, batch_size=8, device='cpu')
    return contract, reference, group, data


def test_registration_does_not_mutate_parent_and_resume_exact(tmp_path):
    contract, ref, group, _ = fixture(tmp_path)
    original = deepcopy(contract.protocol)
    full = fit_registered(contract, [group], reference=ref, seed=1, output=tmp_path/'full')
    part = fit_registered(contract, [group], reference=ref, seed=1, output=tmp_path/'resume', stop_after=3)
    with pytest.raises(ValueError, match='budget'):
        load_registered(contract, part)
    done = fit_registered(contract, [group], reference=ref, seed=1, output=tmp_path/'resume', resume=True)
    assert done['losses'] == full['losses']
    left, right = load_registered(contract, full), load_registered(contract, done)
    for k in left.state_dict():
        torch.testing.assert_close(left.state_dict()[k], right.state_dict()[k], atol=0, rtol=0)
    assert contract.protocol == original and 'comparators' not in original
    changed = deepcopy(group)
    changed['targets'][0, 0] += .1
    with pytest.raises(ValueError, match='identity changed'):
        fit_registered(contract, [changed], reference=ref, seed=1, output=tmp_path/'resume', resume=True)


@pytest.mark.parametrize('mutation', ['file', 'parent', 'eligible', 'scope', 'budget'])
def test_registration_tampering_refused(tmp_path, mutation):
    contract, ref, _, _ = fixture(tmp_path)
    path = tmp_path / ref['path']
    value = json.loads(path.read_text())
    if mutation == 'parent':
        value['parent_protocol_sha256'] = 'wrong'
    elif mutation == 'eligible':
        value['eligible_for_selection'] = True
    elif mutation == 'scope':
        value['scope'] = 'confirmation'
    elif mutation == 'budget':
        value['variants']['linear']['fit_settings']['steps'] = 0
    path.write_text(json.dumps(value) + ' ')
    if mutation != 'file':
        ref['sha256'] = file_digest(path)
    with pytest.raises(ValueError):
        registered_spec(contract, ref)


def test_matching_rejects_changed_identity_or_forecast():
    row = {'agent_id': 1, 'baseline_ade': 2., 'arms': {'floor': {'ade': 2., 'fde': 3.},
           'uncontrolled': {'ade': 1., 'fde': 2.}}}
    fresh = deepcopy(row)
    fresh['arms']['deferral_linear'] = {'ade': 1.}
    assert 'deferral_linear' in matched_rows([row], [fresh])[0]['arms']
    fresh['agent_id'] = 2
    with pytest.raises(ValueError, match='identity'):
        matched_rows([row], [fresh])
    fresh['agent_id'] = 1
    fresh['arms']['uncontrolled']['ade'] = 1.01
    with pytest.raises(ValueError, match='scoring'):
        matched_rows([row], [fresh])


def test_inference_accepts_no_future_and_future_change_cannot_change_decisions(tmp_path):
    contract, ref, group, data = fixture(tmp_path)
    result = fit_registered(contract, [group], reference=ref, seed=1, output=tmp_path/'head')
    head = load_registered(contract, result)
    scene = data.readers[0].get_scene_inputs(100, 120)
    first = infer_scene(scene, BaselineCopy(), {'deferral_linear': head}, baseline=data.baseline_name, geometry=GEOMETRY)
    points = np.array(data.readers[0].points)
    points[points[:, 0] > 100, 2:] += 1000
    data.readers[0].points = points
    second = infer_scene(data.readers[0].get_scene_inputs(100, 120), BaselineCopy(),
                         {'deferral_linear': head}, baseline=data.baseline_name, geometry=GEOMETRY)
    for arm in first['arms']:
        np.testing.assert_array_equal(first['arms'][arm]['prediction'], second['arms'][arm]['prediction'])
