import json
import pytest
from scripts import report_m3w_conditional_moment as report


def test_registered_conditional_cost_control_and_role_contract():
    reg = json.loads(report.run.REG.read_text())
    cfg = json.loads(report.run.CONFIG.read_text())
    assert cfg['training']['largest_source_cycles'] == 4
    assert cfg['parameters_per_head'] == 31381
    assert cfg['risk_budget'] == .02
    assert cfg['occurrence_loss_weight'] == cfg['conditional_loss_weight'] == 1.
    assert reg['frozen_ranking'] == 'current_bank_final_TDEF'
    assert reg['both_training_query_losses'] == 'fixed_original_causal_initial_choices'
    assert reg['source_role'] == 'exposed_SOURCE_fitting_resubstitution_only'
    assert not reg['independent_roles_opened'] and not cfg['deployment_changed']


def test_full_conditional_public_reconstruction_when_available():
    if not (report.run.PUBLIC/'verification.json').exists():
        pytest.skip('Complete real 72-group inference replay not available yet')
    result = report.build()
    assert result['summary']['groups'] == 72
    assert result['summary']['unique_target_rows'] == 149510
    assert len(result['fits']) == len(result['traces']['cells']) == 36
    assert not result['summary']['independent_confirmation']
    assert not result['summary']['deployment_changed']


def test_private_read_denial_receipt_when_available():
    path = report.run.PUBLIC/'public_rebuild_verification.json'
    if not path.exists():
        pytest.skip('Public-only reconstruction not run yet')
    result = json.loads(path.read_text())
    assert result['private_input_read_attempts'] == 0
    assert result['guard_installed_before_report_imports']
    assert result['summary_sha256'] == report.run.sha(report.run.PUBLIC/'summary.json')
    assert result['report_script_sha256'] == report.run.sha(
        report.ROOT/'scripts/report_m3w_conditional_moment.py')
