"""Prepare past-only diagnostic indices and verify the SDD model interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 Python')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter, masked_future_ade


def same_inputs(left, right):
    if set(left) != set(right):
        raise ValueError('Input schema changed')
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--recording')
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    if reg['data_role'] != 'diagnostic_only' or reg['training_admitted']:
        raise ValueError('This command does not authorize training')
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Bound file changed: '+path)
    links = json.loads((ROOT/reg['source_manifest']).read_text())
    old = json.loads((ROOT/reg['split_manifest']).read_text())
    if links['data_role'] != 'source_audit_only':
        raise ValueError('Expected preserved diagnostic media correspondence')
    for path, digest in links['report_bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Source audit changed')
    videos = {r['scene_id']+'/'+r['video_id']: r for r in old['video_reports']}
    chosen = [e for e in links['records'] if videos[e['annotation_key']]['split_id'] == 'train']
    if len(chosen) != 40:
        raise ValueError('Original 40-video train roster changed')
    if args.recording and args.recording not in {e['annotation_key'] for e in chosen}:
        raise ValueError('Only original train recordings may be inspected')
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    identity = dict(registration_sha256=file_digest(args.registration),
                    torch_version=torch.__version__, numpy_version=np.__version__)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(17); model = OfflineVisualForecast(476).eval()
    fresh = reused = 0
    summaries = []
    for entry in chosen:
        key = entry['annotation_key']
        if args.recording and args.recording != key:
            continue
        path = ROOT/entry['annotations_path']
        if str(path.relative_to(ROOT)) != videos[key]['annotation_path'] or file_digest(path) != entry['annotations_sha256']:
            raise ValueError('Source/split correspondence changed: '+key)
        folder = output/key
        rp = folder/'receipt.json'
        if rp.exists():
            receipt = json.loads(rp.read_text())
            if receipt['identity'] != identity or receipt['annotation_sha256'] != entry['annotations_sha256']:
                raise ValueError('Completed recording identity changed')
            for stride, part in receipt['strides'].items():
                if file_digest(folder/f'index_stride{stride}.npy') != part['index_sha256']:
                    raise ValueError('Saved index changed')
            reused += 1; summaries.append(receipt); continue
        start = time.monotonic()
        json_write(output/'heartbeat.json', dict(pid=os.getpid(), state='reading_source',
            recording=key, completed_recordings=fresh+reused))
        rows, labels = load_source(path)
        receipt = dict(identity=identity, recording=key, original_split='train',
            annotation_sha256=entry['annotations_sha256'], source_rows=len(rows),
            data_role='diagnostic_only', training_admitted=False, strides={})
        folder.mkdir(parents=True, exist_ok=True)
        for stride in reg['raw_frame_strides']:
            adapter = SDDStepAdapter(rows, labels, key, stride)
            temporary = folder/f'index_stride{stride}.tmp.npy'
            np.save(temporary, adapter.index, allow_pickle=False)
            index_path = folder/f'index_stride{stride}.npy'; os.replace(temporary, index_path)
            sample = np.unique(np.linspace(0, len(adapter)-1, min(reg['queries_per_recording_stride'], len(adapter)), dtype=int))
            geometry, baseline, targets, masks = [], [], [], []
            for item in sample:
                inp, lab = adapter.get_inputs(int(item)), adapter.get_labels(int(item))
                if np.any(inp['history_frame_offsets'] > 0) or np.any(inp['neighbor_frame_offsets'][inp['neighbor_mask']] > 0):
                    raise ValueError('Post-query input')
                geometry.append(adapter.get_geometry(int(item))); baseline.append(inp['baseline_rollouts'][1])
                targets.append(lab['future_xy_normalized']); masks.append(lab['future_label_mask'])
            if not len(sample):
                raise ValueError('No indexed diagnostic samples in '+key)
            x, b, y, mask = map(torch.from_numpy, (np.asarray(geometry), np.asarray(baseline), np.asarray(targets), np.asarray(masks)))
            if not torch.isfinite(x).all() or not torch.isfinite(b).all():
                raise ValueError('Nonfinite past-only features/baselines')
            with torch.no_grad():
                pred = model(x, torch.zeros(len(x), 8, 3, 32, 32), torch.zeros(len(x), 8, 1, 32, 32), b, 'geometry')
                torch.testing.assert_close(pred, b, rtol=0, atol=0)
                ade, supported = masked_future_ade(pred, y, mask)
                if not torch.isfinite(ade[supported]).all():
                    raise ValueError('Nonfinite masked interface result')
            checks = 0
            for item in np.unique([sample[0], sample[-1]]):
                query = adapter.identity(int(item)); cutoff = query['frame_id']
                before = adapter.get_inputs(int(item)); future = rows[:, 5] > cutoff
                altered = rows.copy(); altered[future, 1:5] += 1234; altered[future, 6:9] = 1
                for changed_rows, changed_labels in ((altered, labels), (rows[~future], labels[~future])):
                    other = SDDStepAdapter(changed_rows, changed_labels, key, stride)
                    current = other.points[other.index['current_row']]
                    match = np.flatnonzero((current[:, 0] == cutoff) & (current[:, 1] == query['agent_id']))
                    if len(match) != 1:
                        raise ValueError('Future-only mutation changed query membership')
                    same_inputs(before, other.get_inputs(int(match[0]))); checks += 1
            receipt['strides'][str(stride)] = dict(past_only_queries=len(adapter),
                indexed_local_agents=len(np.unique(adapter.points[adapter.index['current_row'], 1])),
                index_sha256=file_digest(index_path), index_bytes=index_path.stat().st_size,
                sampled_queries=len(sample), sampled_complete_labels=int(mask.all(1).sum()),
                sampled_partial_labels=int((mask.any(1) & ~mask.all(1)).sum()),
                sampled_no_labels=int((~mask.any(1)).sum()), future_mutation_truncation_checks=checks,
                feature_width=x.shape[1], zero_initialized_forward_matches_CV=True,
                geometry_only_no_image_training=True, optimizer_updates=0)
            json_write(output/'heartbeat.json', dict(pid=os.getpid(), state='building', recording=key,
                stride=stride, completed_recordings=fresh+reused, elapsed_seconds=time.monotonic()-start))
        receipt['seconds'] = time.monotonic()-start
        json_write(rp, receipt); summaries.append(receipt); fresh += 1
        print(json.dumps(dict(recording=key, status='fresh_run_bridge_verified', seconds=receipt['seconds'])), flush=True)
    if args.recording:
        print(json.dumps(dict(pilot_complete=True, fresh=fresh, reused=reused)), flush=True)
        return
    report = dict(identity=identity, result_source='fresh_run',
        provenance=dict(adapter_indices_interface_checks='fresh_run',
                        source_manifest_and_original_split='cached_verified',
                        new_auxiliary_training='not_run',
                        training_reason='Await separate auxiliary source-role and sampling contract'),
        recordings=summaries, original_train_recordings=len(summaries), original_val_test_opened=False,
        geometry_schema_width=476, history_steps=8, prediction_steps=12,
        strides=reg['raw_frame_strides'], no_training_stride_selected=True,
        total_private_index_bytes=sum(s['index_bytes'] for r in summaries for s in r['strides'].values()),
        optimizer_updates=0, new_model_training=False, predictive_gain_measured=False,
        source_role='diagnostic_only', new_auxiliary_training_protocol_pending=True,
        stage5c_executed=False, smc_enabled=False)
    if (reports/'report.json').exists():
        if json.loads((reports/'report.json').read_text()) != report:
            raise ValueError('Completed diagnostic report changed')
    else:
        json_write(reports/'report.json', report)
    json_write(output/'heartbeat.json', dict(pid=os.getpid(), state='complete', fresh_recordings=fresh,
        reused_recordings=reused, optimizer_updates=0, report_sha256=file_digest(reports/'report.json')))
    print(json.dumps(dict(complete=True, recordings=len(summaries), fresh=fresh, reused=reused,
        indexed={str(k):sum(r['strides'][str(k)]['past_only_queries'] for r in summaries) for k in reg['raw_frame_strides']})), flush=True)


if __name__ == '__main__':
    main()
