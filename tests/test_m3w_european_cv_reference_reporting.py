import json

import pytest

from scripts.report_m3w_european_cv_reference import (
    ci, observed_safety_label, require_verification, sha, value,
)


def receipts(path):
    result = {'training': {'head_a': {}, 'head_b': {}}}
    (path/'analysis.json').write_text(json.dumps(result))
    digest = sha(path/'analysis.json')
    verification = {'analysis_sha256': digest, 'metrics_recomputed': True}
    replay = {'analysis_sha256': digest, 'all_passed': True,
              'checks': [{'trial': name, 'exact': True, 'rows': 4,
                          'max_difference': 0.0} for name in result['training']]}
    return result, verification, replay


def write_receipts(path, verification, replay):
    (path/'verification.json').write_text(json.dumps(verification))
    (path/'checkpoint_replay.json').write_text(json.dumps(replay))


def test_complete_exact_replay_is_accepted(tmp_path):
    result, verification, replay = receipts(tmp_path)
    write_receipts(tmp_path, verification, replay)
    require_verification(tmp_path, result)


@pytest.mark.parametrize('defect', [
    'failed_metrics', 'failed_replay', 'missing_head', 'duplicate_head',
    'approximate_replay', 'empty_support', 'different_analysis',
])
def test_report_rejects_incomplete_evidence(tmp_path, defect):
    result, verification, replay = receipts(tmp_path)
    if defect == 'failed_metrics':
        verification['metrics_recomputed'] = False
    elif defect == 'failed_replay':
        replay['all_passed'] = False
    elif defect == 'missing_head':
        replay['checks'].pop()
    elif defect == 'duplicate_head':
        replay['checks'][1] = dict(replay['checks'][0])
    elif defect == 'approximate_replay':
        replay['checks'][0]['exact'] = False
        replay['checks'][0]['max_difference'] = 1e-6
    elif defect == 'empty_support':
        replay['checks'][0]['rows'] = 0
    elif defect == 'different_analysis':
        replay['analysis_sha256'] = 'changed'
    write_receipts(tmp_path, verification, replay)
    with pytest.raises(ValueError):
        require_verification(tmp_path, result)


def test_missing_zero_support_is_not_reported_as_safety_success():
    metric = {'zero_CV': {'rows': 0}, 'safety_observed_pass': True}
    assert observed_safety_label(metric) == 'zero-event support absent; not certified'


def test_undefined_locality_contrast_is_not_replaced_by_zero():
    metric = {'equal_scene_gain_percent': None, 'scene_bootstrap_ci95': None}
    assert value(metric) == 'undefined'
    assert ci(metric) == 'undefined'
