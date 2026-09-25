import json
from pathlib import Path


def test_fixed_budget_all_directions_and_common_anchors():
    c = json.loads(Path('configs/m3w_european_fixed_producer_roles_v1.json').read_text())
    assert c['groups'] == 3*2*3*2 == 36
    assert c['new_neural_heads'] == 36*2*2 == 144
    assert c['neural_updates'] == 144*2000
    assert c['views'] == 36*5 and c['ridge_fits'] == 72
    assert c['ridge_alpha'] == .01 and c['risk_budget'] == .02
    assert c['seeds'] == [17, 29, 43] and c['bootstrap_resamples'] == 3000
    assert c['arms'] == ['producer_matched', 'oof_control']
    assert not any(c[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit',
        'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
