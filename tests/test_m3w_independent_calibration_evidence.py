"""Public receipts only; execution evidence is not predictive/risk evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def read(name):
    return json.loads((PUBLIC/name).read_text())


def test_registered_cohort_and_complete_raw_replay():
    inventory = read('input_inventory.json'); build = read('build.json'); replay = read('verify_inputs.json')
    registration = read('registration.json')
    assert inventory['target_rows'] == sum(v['target_rows'] for v in inventory['by_locality'].values())
    assert inventory['visible_rows'] == sum(v['visible_rows'] for v in inventory['by_locality'].values())
    assert inventory['unique_recordings'] == 102 and inventory['locality_groups'] == 12
    assert inventory['query_frames'] <= inventory['unique_recordings']*inventory['per_recording_query_cap']
    assert build['record_receipts'] == replay['record_receipts'] and len(build['record_receipts']) == 102
    roles = registration['roles']
    assert hashlib.sha256((ROOT/roles['path']).read_bytes()).hexdigest() == roles['sha256']
    assert not inventory['future_targets_constructed'] and not inventory['normalizer_fitted']
    assert (inventory['obs'], inventory['pred'], inventory['stride_raw_frames']) == (8, 12, 12)
    assert inventory['coordinate_unit'] == 'local_image_pixel'


def test_all_nine_model_inferences_freshly_replayed():
    inferred, replayed = read('infer.json'), read('verify_inference.json')
    assert inferred['model_record_receipts'] == replayed['model_record_receipts']
    refs = inferred['model_record_receipts']; assert len(refs) == 918
    counts = Counter(Path(r['path']).parent.name for r in refs)
    expected = {f'single{f}_seed{s}' for f in range(3) for s in (17, 29, 43)}
    assert set(counts) == expected and set(counts.values()) == {102}
    inventory = read('inference_inventory.json'); targets = read('input_inventory.json')['target_rows']
    assert inventory['repeated_model_target_predictions'] == 9*targets
    assert inventory['fresh_replay_all918_forecasts_and_costs_exact']
    assert not inventory['new_optimizer_updates'] and not inventory['future_labels_read']
    assert not inventory['confirmation_access'] and not inventory['deployment_changed']


def test_storage_repair_is_execution_only_with_actual_grid_recovery():
    reg = read('allocation_storage_registration_v2.json'); pilot = read('allocation_storage_pilot_v2.json')
    assert reg['numerical_rule_unchanged'] and not reg['model_and_threshold_changes']
    assert pilot['engineering_pass'] and pilot['real_largest_record_resume_and_direct_grid_exact']
    assert pilot['original_pilot_grid_exact'] and not pilot['labels_read']
    assert pilot['storage_registration']['sha256'] == hashlib.sha256(
        (PUBLIC/'allocation_storage_registration_v2.json').read_bytes()).hexdigest()
