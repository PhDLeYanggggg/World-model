import json
from pathlib import Path
from scripts.report_m3w_moving_channel_specialist import summary


def test_complete_negative_results_retained():
    path=Path(__file__).resolve().parents[1]/'outputs/publication_readiness_2026_09/moving_channel_specialist_v1/analysis.json'
    a=json.loads(path.read_text()); s=summary(a)
    assert s['passing_contexts']==0
    assert s['contexts'][1]['arms']['separate64']['harmful_admissions']==19
    assert s['contexts'][2]['arms']['separate64']['admissions']==0
    assert s['contexts'][0]['arms']['separate64']['cost_reduction_percent']<0
    assert not s['submission_ready'] and not s['safety_certificate']


def test_secondary_is_reported_not_selected():
    path=Path(__file__).resolve().parents[1]/'outputs/publication_readiness_2026_09/moving_channel_specialist_v1/analysis.json'
    s=summary(json.loads(path.read_text()))
    assert all(set(c['arms'])=={'shared32','shared64','separate64'} for c in s['contexts'])
    assert not s['deployment_changed']
