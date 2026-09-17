"""Count source-level state-change support without admitting a training role."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_state_support import (
    control_provenance, past_features, past_window_indices, track_support, validate_track,
)

CONFIG = ROOT/'configs/m3w_sdd_state_support_audit.json'


def load_source(path):
    names = ['c'+str(i) for i in range(9)]+['label']
    source = np.loadtxt(path, dtype=dict(names=names, formats=[np.float64]*9+['U32']),
                        usecols=range(10), ndmin=1)
    rows = np.column_stack([source[n] for n in names[:9]])
    order = np.lexsort((rows[:, 5], rows[:, 0]))
    return rows[order], np.char.strip(source['label'][order], '"')


def future_input_check(track, stride):
    q, histories = past_window_indices(track, stride)
    if not len(q):
        return False
    cutoff = q[0]
    selected = q <= cutoff
    before = past_features(track, histories[selected])
    altered = track.copy()
    future = altered[:, 5] > cutoff
    if not future.any():
        return False
    altered[future, 1:5] = altered[future, 1:5]*7+1234
    altered[future, 6:9] = 1
    qa, ha = past_window_indices(altered, stride)
    np.testing.assert_array_equal(q[selected], qa[qa <= cutoff])
    after = past_features(altered, ha[qa <= cutoff])
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    truncated = track[track[:, 5] <= cutoff]
    qt, ht = past_window_indices(truncated, stride)
    np.testing.assert_array_equal(q[selected], qt)
    for name, value in past_features(truncated, ht).items():
        np.testing.assert_array_equal(before[name], value)
    return True


def calculate_record(entry, reg, identity, output):
    started = time.monotonic()
    rows, labels = load_source(ROOT/entry['annotations_path'])
    parts = np.split(np.arange(len(rows)), np.flatnonzero(np.diff(rows[:, 0]))+1)
    strata = {str(s):{} for s in reg['raw_annotation_frame_strides']}
    quality = {}
    gap_histogram, visible_control_count_histogram = Counter(), Counter()
    checks = 0
    for t, indices in enumerate(parts):
        track = validate_track(rows[indices])
        types = np.unique(labels[indices])
        if len(types) != 1:
            raise ValueError('One source track changes agent type')
        kind = str(types[0])
        provenance = control_provenance(track)
        generated = track[:, 8] == 1
        visible = track[:, 6] == 0
        bracketed_generated = generated & provenance['bracketed']
        error = provenance['max_box_interpolation_error']
        controls_visible = int((~generated & visible).sum())
        q = quality.setdefault(kind, Counter())
        q.update(dict(rows=len(track), tracks=1, nonlost_rows=int(visible.sum()),
            nonlost_unoccluded_rows=int((visible & (track[:, 7] == 0)).sum()),
            generated_rows=int(generated.sum()), nonlost_generated_rows=int((generated & visible).sum()),
            control_rows=int((~generated).sum()), nonlost_control_rows=controls_visible,
            tracks_with_at_least_20_nonlost_controls=int(controls_visible >= 20),
            generated_bracketed_rows=int(bracketed_generated.sum()),
            generated_unbracketed_rows=int((generated & ~provenance['bracketed']).sum()),
            generated_linear_box_error_le_half_pixel=int((bracketed_generated & (error <= .5+1e-9)).sum()),
            generated_linear_box_error_le_one_pixel=int((bracketed_generated & (error <= 1+1e-9)).sum()),
            generated_linear_box_error_le_two_pixels=int((bracketed_generated & (error <= 2+1e-9)).sum())))
        gap_histogram.update(map(int, provenance['control_gaps']))
        visible_control_count_histogram[controls_visible] += 1
        for stride in reg['raw_annotation_frame_strides']:
            summary = track_support(track, stride, provenance)
            strata[str(stride)].setdefault(kind, Counter()).update(summary)
            if t < 3:
                checks += int(future_input_check(track, stride))
        if t % 100 == 0:
            json_write(output/'heartbeat.json', dict(pid=os.getpid(), updated_unix=time.time(),
                recording=entry['annotation_key'], tracks_done=t+1, tracks_total=len(parts)))
    return dict(identity=identity, recording=entry['annotation_key'], scene=entry['scene_id'],
        annotation_sha256=entry['annotations_sha256'], quality_by_type=quality,
        support_by_stride_and_type=strata,
        control_frame_gap_histogram={str(k):v for k,v in sorted(gap_histogram.items())},
        nonlost_control_count_per_track_histogram={str(k):v for k,v in sorted(visible_control_count_histogram.items())},
        future_mutation_and_truncation_checks=checks, elapsed_seconds=time.monotonic()-started)


def aggregate(records, identity, fresh, reused):
    quality, all_strides, types, scenes = Counter(), {}, {}, {}
    gaps, control_counts = Counter(), Counter()
    summaries = []
    for r in records:
        record_quality, record_support = Counter(), {}
        for kind, q in r['quality_by_type'].items():
            quality.update(q)
            record_quality.update(q)
            types.setdefault(kind, dict(quality=Counter(), support={} ))['quality'].update(q)
        for stride, type_summaries in r['support_by_stride_and_type'].items():
            total = Counter()
            for kind, values in type_summaries.items():
                total.update(values)
                types[kind]['support'].setdefault(stride, Counter()).update(values)
            all_strides.setdefault(stride, Counter()).update(total)
            scenes.setdefault(r['scene'], {}).setdefault(stride, Counter()).update(total)
            record_support[stride] = total
        gaps.update(r['control_frame_gap_histogram'])
        control_counts.update(r['nonlost_control_count_per_track_histogram'])
        summaries.append(dict(recording=r['recording'], quality=record_quality, support=record_support,
                              annotation_sha256=r['annotation_sha256']))
    return dict(identity=identity, result_source='fresh_run_source_support_and_cached_verified_resume',
        newly_calculated_this_invocation=fresh, verified_receipts_reused=reused,
        recordings=len(records), scene_folders=len(scenes), quality=quality,
        support_by_stride=all_strides, by_agent_type=types, by_scene=scenes, by_recording=summaries,
        control_frame_gap_histogram=dict(sorted(gaps.items(), key=lambda pair:int(pair[0]))),
        nonlost_control_count_per_track_histogram=dict(sorted(control_counts.items(), key=lambda pair:int(pair[0]))),
        future_mutation_and_truncation_checks=sum(r['future_mutation_and_truncation_checks'] for r in records),
        summed_source_audit_seconds=sum(r['elapsed_seconds'] for r in records),
        descriptive_labels_only=True, proxy_labels_not_human_gold=True,
        independent_people_or_events_certified=False, controls_not_online_certified=True,
        future_labels_in_inputs=False, training_admitted=False, threshold_or_stride_selected=False,
        development_calibration_confirmation_opened=False, physical_seconds_or_metric_verified=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    reg = json.loads(CONFIG.read_text())
    if reg['training_admitted'] or reg['new_models_or_threshold_selection_allowed'] or reg['primary_or_split_changed']:
        raise ValueError('A source audit must not change the scientific study')
    if file_digest(ROOT/reg['source_manifest']) != reg['source_manifest_sha256']:
        raise ValueError('Changed source manifest')
    manifest = json.loads((ROOT/reg['source_manifest']).read_text())
    entries = [r for r in manifest['records'] if args.recording is None or r['annotation_key'] == args.recording]
    if not entries:
        raise ValueError('No source matched')
    identity = dict(config_sha256=file_digest(CONFIG), numpy_version=np.__version__,
        source_manifest_sha256=reg['source_manifest_sha256'],
        source_definition_sha256=file_digest(ROOT/'external_data/OpenTraj/datasets/SDD/README.md'),
        code_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in (Path(__file__),
            ROOT/'src/world_model/m3w_sdd_state_support.py')})
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    records, fresh, reused, checked = [], 0, 0, []
    for entry in entries:
        if file_digest(ROOT/entry['annotations_path']) != entry['annotations_sha256']:
            raise ValueError('Changed raw annotation')
        path = output/entry['annotation_key']/'receipt.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['identity'] != identity or record['annotation_sha256'] != entry['annotations_sha256']:
                raise ValueError('Changed audit identity; no silent overwrite')
            if args.verify:
                again = calculate_record(entry, reg, identity, output)
                same = all(again[k] == record[k] for k in record if k != 'elapsed_seconds')
                if not same:
                    raise ValueError('Fresh source-support replay differs')
                checked.append(dict(recording=entry['annotation_key'], exact_recomputed=True,
                                    preserved_receipt_sha256=file_digest(path)))
            reused += 1
        else:
            if args.verify:
                raise ValueError('Missing receipt to verify')
            record = calculate_record(entry, reg, identity, output)
            json_write(path, record)
            fresh += 1
        records.append(record)
        print(json.dumps(dict(recording=entry['annotation_key'], completed=len(records), total=len(entries),
                              phase='verified' if args.verify else 'audited')), flush=True)
    suffix = '' if args.recording is None else '_'+args.recording.replace('/', '_')
    if args.verify:
        result = dict(identity=identity, result_source='fresh_run_exact_source_recomputation',
            records=checked, no_new_role_or_model=True, cached_receipts_preserved=True)
        json_write(reports/('verification'+suffix+'.json'), result)
    else:
        result = aggregate(records, identity, fresh, reused)
        result['complete_inventory'] = len(records) == len(manifest['records'])
        name = 'audit' if fresh else 'resume'
        json_write(reports/(name+suffix+'.json'), result)
    json_write(output/'heartbeat.json', dict(pid=os.getpid(), updated_unix=time.time(),
        phase='complete', recordings=len(records), all_records=len(records) == len(manifest['records'])))


if __name__ == '__main__':
    main()
