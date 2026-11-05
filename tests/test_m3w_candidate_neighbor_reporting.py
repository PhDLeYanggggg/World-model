import json
from pathlib import Path
from scripts.report_m3w_candidate_neighbors import summarize


def test_report_matches_frozen_primary_and_does_not_call_probe_neural_training():
    path = Path(__file__).resolve().parents[1]/'outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1/analysis.json'
    result = summarize(json.loads(path.read_text()))
    assert result['source_folds'] == 8 and result['fitted_ridge_probes'] == 16
    assert result['own_B_rows'] == 62466 and result['known_proposals'] == 48061
    assert result['unknown_proposals'] == 1274
    assert result['followup_screen_passed_contexts'] == 0
    assert all(r['MSE_improvement_percent'] < 0 for r in result['contexts'])
    assert not result['new_neural_training'] and not result['deployment_changed']
    assert result['original_rules_unchanged'] and not result['identity_certification']
    assert not result['new_V_or_H'] and not result['submission_ready']
