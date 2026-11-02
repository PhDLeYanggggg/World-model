import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_model_selection_inputs_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_registered_input_definitions_and_original_roles_unchanged():
    reg=json.loads((PUBLIC/'registration.json').read_text())
    for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():
        assert sha(ROOT/rel)==digest
    roles=ROOT/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json'
    assert sha(roles)==reg['roles_sha256']
    r=json.loads(roles.read_text())
    members={v['source_member'] for v in reg['members']}
    assert len(members)==28
    for row in r['recordings']:
        assert (row['source_member'] in members)==(row['role']=='model_selection_reserved')
        if row['source_member'] in members:
            assert not row['training_access'] and not row['predictive_readout_access']
    assert not reg['model_selection_forecast_readout_authorized']


def test_complete_raw_reparse_binds_all28_input_records_when_available():
    if not (PUBLIC/'verification.json').exists():pytest.skip('Complete real input conversion not available')
    r=json.loads((PUBLIC/'analysis.json').read_text());v=json.loads((PUBLIC/'verification.json').read_text());c=r['cohort']
    assert v['analysis_sha256']==sha(PUBLIC/'analysis.json')
    assert v['all28_raw_reparsed_all_input_arrays_exact'] and v['parsed_raw_records']==28
    assert c['recordings']==28 and len(c['localities'])==6 and len(c['record_receipts'])==28
    assert c['target_rows']==sum(x['input_statistics']['target_rows'] for x in c['record_receipts'])
    assert c['visible_rows']>=c['target_rows']>0
    assert c['query_frames']==sum(x['input_statistics']['queries'] for x in c['record_receipts'])
    assert all(0<x['input_statistics']['queries']<=256 for x in c['record_receipts'])
    assert c['independent_prefix_query_checks']==84
    assert c['calibration_or_confirmation_members_deserialized']==0
    assert not c['future_target_arrays_constructed'] and not c['forecast_errors_opened']
    assert not c['normalization_fitted'] and not c['original_training_accessor_changed']
    assert v['predictive_generalization_result']=='not_run'


def test_selection_members_disjoint_from_existing_source_inputs():
    reg=json.loads((PUBLIC/'registration.json').read_text())
    roles=json.loads((ROOT/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json').read_text())
    selection={r['source_member'] for r in reg['members']}
    for purpose in ('source_training','risk_calibration_reserved','confirmation_reserved'):
        assert not selection&{r['source_member'] for r in roles['recordings'] if r['role']==purpose}
