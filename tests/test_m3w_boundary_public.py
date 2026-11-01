import json
import pytest
from scripts import report_m3w_boundary_guard as report


def test_training_sampler_and_role_contract_are_fixed():
    reg = json.loads(report.run.REG.read_text())
    cfg = json.loads(report.run.CONFIG.read_text())
    assert cfg['training']['largest_source_cycles'] == 4
    assert cfg['risk_budget'] == .02 and cfg['boundary_loss_weight'] == 1.
    assert reg['frozen_ranking'] == 'complete_current_bank_TDEF_final_scores'
    assert reg['both_training_query_losses'] == 'fixed_original_causal_initial_choices'
    assert not reg['independent_roles_opened'] and not cfg['deployment_changed']


def test_full_public_result_reconstruction_when_available():
    if not (report.run.PUBLIC/'verification.json').exists():
        pytest.skip('Complete real 72-group result not available yet')
    v = report.build()
    assert v['summary']['groups'] == 72
    assert v['summary']['unique_target_rows'] == 149510
    assert not v['summary']['independent_confirmation']
    assert not v['summary']['deployment_changed']


def test_private_read_denial_receipt_when_available():
    p = report.run.PUBLIC/'public_rebuild_verification.json'
    if not p.exists(): pytest.skip('Public-only reconstruction not run yet')
    v = json.loads(p.read_text())
    assert v['private_input_read_attempts'] == 0
    assert v['guard_installed_before_report_imports']
    assert v['summary_sha256'] == report.run.sha(report.run.PUBLIC/'summary.json')
    assert v['report_script_sha256'] == report.run.sha(report.ROOT/'scripts/report_m3w_boundary_guard.py')
