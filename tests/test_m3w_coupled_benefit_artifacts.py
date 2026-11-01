import json
from scripts import report_m3w_coupled_benefit_risk as report


def test_complete_real_critic_readout_replay_and_bundle_are_not_submission_success():
    v = report.build(); p = report.run.PUBLIC; s = v['summary']
    assert len(v['rows']) == v['bundle']['groups'] == s['groups'] == 72
    assert s['unique_target_rows'] == 149510 and s['unique_recordings'] == 74
    assert v['replay']['all72_original_controls_exact'] and v['replay']['all72groups_exact']
    assert s['primary_optimizer_updates'] == 72000 and s['exact_refit_updates'] == 4000
    assert not s['independent_confirmation'] and not s['new_neural_trajectory_training']
    assert not s['deployment_changed'] and not s['submission_ready'] and not s['smc_enabled'] and not s['stage5c_executed']
    assert (p/'readout.md').read_text() == report.render(v)
    evidence = json.loads((p/'evidence.json').read_text())
    assert evidence['summary_sha256'] == report.run.sha(p/'summary.json')
    assert evidence['verification_sha256'] == report.run.sha(p/'verification.json')
    assert evidence['readout_sha256'] == report.run.sha(p/'readout.md')
    assert b'\r' not in (p/'locality_effects.csv').read_bytes()
    assert len((p/'locality_effects.csv').read_text().splitlines()) == 13
