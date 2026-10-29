import copy
import json
import pytest
from scripts.report_m3w_feature_cost_evidence import calibration_differences, render
from scripts import report_m3w_feature_cost_evidence as module


def group():
    return dict(scope='fit', context='context', source='source', seed=17,
        result=dict(rows=10, known_rows=8, calibration={arm:dict(normalized_moment_MSE=[1., 2., 3., 4., 5.])
            for arm in ('physical_only', 'physical_neural', 'neural_only')}))


def test_moment_pairing_rejects_changed_support_and_keeps_sign():
    old = group(); new = copy.deepcopy(old)
    new['result']['calibration']['physical_neural']['normalized_moment_MSE'][1] = 1.5
    result = calibration_differences(old, new)
    assert result['result']['physical_neural']['harm'] == -.5
    assert result['result']['physical_neural']['benefit'] == 0
    new['result']['known_rows'] = 9
    with pytest.raises(ValueError):
        calibration_differences(old, new)


def test_export_separates_controller_seeds_and_calibration_from_claims():
    text = render(dict(policies=[], training={}, inner_held_mixed_minus_physical={},
        calibration_MSE_new_minus_original={}))
    assert 'three newly trained full forecasting' in text
    assert 'Empty risk is not_estimable' in text
    assert 'SOURCE inner-held was exposed' in text
    assert 'not independent confirmation' in text
    assert 'neither neural utility nor a safety certificate' in text


def test_export_accepts_unchanged_legacy_individual_groups_without_repacking(tmp_path, monkeypatch):
    folder = tmp_path/'public'/'legacy'; (folder/'groups').mkdir(parents=True)
    path = folder/'groups'/'view.json'; path.write_text(json.dumps(group()))
    (folder/'complete.json').write_text(json.dumps(dict(groups=[
        dict(path=str(path.relative_to(tmp_path)), sha256=module.previous.sha(path))])))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.previous, 'PUBLIC', tmp_path/'public')
    assert module.aggregate_groups('legacy')['view.json'] == group()
    path.write_text('{}')
    with pytest.raises(ValueError):
        module.aggregate_groups('legacy')
