"""Publish verified calibration input inventory, without opening target labels."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.build_m3w_model_selection_inputs import sha, write

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def main():
    build = json.loads((PUBLIC/'build.json').read_text())
    replay = json.loads((PUBLIC/'verify_inputs.json').read_text())
    if build['record_receipts'] != replay['record_receipts'] or len(build['record_receipts']) != 102:
        raise ValueError('Complete original cohort/input replay required')
    rows = []
    for ref in build['record_receipts']:
        path = ROOT/ref['path']
        if sha(path) != ref['sha256']:
            raise ValueError('Recording receipt changed')
        rows.append(json.loads(path.read_text()))
    if any(r['future_labels_constructed'] or r['role'] != 'risk_calibration_inputs_only' for r in rows):
        raise ValueError('Input-only receipts required')
    sites = sorted({r['locality_group'] for r in rows})
    counts = {site: dict(recordings=sum(r['locality_group'] == site for r in rows),
        target_rows=sum(r['input_statistics']['target_rows'] for r in rows if r['locality_group'] == site),
        visible_rows=sum(r['input_statistics']['visible_rows'] for r in rows if r['locality_group'] == site)) for site in sites}
    reg = json.loads((PUBLIC/'registration.json').read_text())
    if sha(ROOT/reg['roles']['path']) != reg['roles']['sha256']:
        raise ValueError('Original role manifest changed')
    value = dict(result_source='fresh_run_full_raw_input_bridge_replay', unique_recordings=102, locality_groups=12,
        query_frames=sum(r['input_statistics']['queries'] for r in rows),
        target_rows=sum(r['input_statistics']['target_rows'] for r in rows),
        visible_rows=sum(r['input_statistics']['visible_rows'] for r in rows),
        partial_history_context_rows=sum(r['input_statistics']['partial_history_context_rows'] for r in rows),
        input_prefix_checks=sum(r['prefix_queries_checked'] for r in rows),
        bridge_prefix_checks=sum(r['bridge_prefix_queries_checked'] for r in rows), by_locality=counts,
        registration_sha256=sha(PUBLIC/'registration.json'), original_roles_unchanged=True,
        inventory_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        coordinate_unit='local_image_pixel', obs=8, pred=12, stride_raw_frames=12,
        label_quality='released_detector_tracker_silver_not_human_gold',
        future_targets_constructed=False, calibration_outcomes_opened=False,
        normalizer_fitted=False, confirmation_access=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    write(PUBLIC/'input_inventory.json', value)
    text = '# Calibration Input Inventory\n\n'
    text += (f'Fresh full replay: {value["target_rows"]:,} target rows, {value["visible_rows"]:,} visible rows, '
             f'{value["query_frames"]:,} query frames, 102 recordings/12 locality groups.\n\n')
    text += ('All arrays replay exactly from raw CSVs; input and bridge prefix checks: '
             f'{value["input_prefix_checks"]} and {value["bridge_prefix_checks"]}. '
             'Partial-history neighbors remain; futureless targets are not filtered by labels.\n\n')
    text += ('Obs8/pred12/stride12 raw frames; local image pixels; released detector/tracker silver. '
             'No metric/seconds/human-gold or online-sensor-causality claim. Original role '
             'manifest/accessors unchanged. Future targets have not been constructed by this phase; '
             'confirmation remains closed. Input replay is not predictive or safety success.\n\n')
    text += '|Locality|Recordings|Targets|Visible rows|\n|---|---:|---:|---:|\n'
    text += '\n'.join(f'|{site}|{v["recordings"]}|{v["target_rows"]}|{v["visible_rows"]}|' for site,v in counts.items())+'\n'
    (PUBLIC/'input_inventory.md').write_text(text)
    infer_path, verify_path = PUBLIC/'infer.json', PUBLIC/'verify_inference.json'
    if infer_path.exists() and verify_path.exists():
        inferred, verified = (json.loads(p.read_text()) for p in (infer_path, verify_path))
        refs = inferred['model_record_receipts']
        if refs != verified['model_record_receipts'] or len(refs) != 918:
            raise ValueError('Complete nine-model inference replay required')
        records = []
        for ref in refs:
            path = ROOT/ref['path']
            if sha(path) != ref['sha256']:
                raise ValueError('Inference receipt changed')
            records.append(json.loads(path.read_text()))
        trials = sorted({r['trial'] for r in records})
        if (len(trials) != 9 or any(sum(r['trial'] == t for r in records) != 102 for t in trials)
                or any(r['future_labels_read'] for r in records)):
            raise ValueError('Complete frozen pipelines, no target-label input required')
        by_trial = {t: sum(r['rows'] for r in records if r['trial'] == t) for t in trials}
        if set(by_trial.values()) != {value['target_rows']}:
            raise ValueError('Every original target must appear for each model')
        write(PUBLIC/'inference_inventory.json', dict(result_source='fresh_run_frozen_Torch_inference_and_complete_exact_replay',
            frozen_SOURCE_models_and_normalizers='cached_verified', model_record_views=918, by_trial=by_trial,
            repeated_model_target_predictions=sum(by_trial.values()), unique_target_rows=value['target_rows'],
            batch_size=128, fresh_replay_all918_forecasts_and_costs_exact=True,
            normalization_or_weights_fitted=False, new_optimizer_updates=0,
            future_labels_read=False, calibration_parameter_fitted=False,
            full_grid_allocation_complete=(PUBLIC/'policy.json').exists(),
            calibration_scoring_complete=(PUBLIC/'eval.json').exists(),
            confirmation_access=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    print(json.dumps(value))


if __name__ == '__main__':
    main()
