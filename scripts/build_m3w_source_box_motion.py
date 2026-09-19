"""Source-only observed box-motion cache; no target access in extraction."""
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config, context, FeatureCorpus
from scripts.run_m3w_source_crossfit import save_arrays, immutable_json, array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_temporal_information import verify_history_keys, describe
from src.world_model.m3w_source_box_motion import pair_features, MOTION_NAMES, QUALITY_NAMES
from src.world_model.m3w_offline_visual_data import json_write
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    runtime = ROOT/'data/stage_cvpr2027_experiments/optical_flow_runtime'
    binary = runtime/'cv2/cv2.abi3.so'
    expected = '2a262de6b70d3b6cfa2cc78a76a5d7edd0117eb1fe260dbda3f29d6120c881f0'
    if file_digest(binary) != expected:
        raise ValueError('Bound OpenCV runtime changed')
    sys.path.insert(0, str(runtime)); import cv2
    if cv2.__version__ != '4.13.0': raise ValueError('Bound OpenCV version required')
    cv2.setNumThreads(1)
    reg = load_config(Path('configs/m3w_source_pretrained_temporal_v1.json'))
    data = context(reg)
    cached = FeatureCorpus(data, ROOT/reg['output'], json.loads((ROOT/reg['reports']/'preparation.json').read_text()))
    ids = cached.ids; loc = ids-data.nmain
    manifest = json.loads(data.manifest_path.read_text())
    links_path = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment/diagnostic_media_links.json'
    links = {v['annotation_key']: v for v in json.loads(links_path.read_text())['records']}
    private = ROOT/'data/stage_cvpr2027_experiments/source_box_motion_v1'
    public = ROOT/'outputs/publication_readiness_2026_09/source_box_motion_v1'
    identity = dict(source=data.identity, assignment=data.assignment_hash, ids_sha256=array_hash(ids),
        media_links_sha256=file_digest(links_path), opencv_binary_sha256=expected,
        opencv=cv2.__version__, numpy=np.__version__, code={p:file_digest(ROOT/p) for p in
        ('scripts/build_m3w_source_box_motion.py', 'src/world_model/m3w_source_box_motion.py')})
    immutable_json(private/'extraction_identity.json', identity)
    output = np.empty((len(ids), 7, 19), np.float32); seen = np.zeros(len(ids), bool)
    records = []; new_pairs = 0
    for rid in np.unique(data.record_ids[data.sid[loc]]):
        at = np.flatnonzero(data.record_ids[data.sid[loc]] == rid)
        item = manifest['records'][int(rid)]; folder = data.manifest_path.parent/item['recording']
        arrays = {}
        names = ('query_keys','crop_keys','image_boxes','source_flags','image_rows','rgb','coverage')
        for name in names:
            path = folder/(name+'.npy')
            if file_digest(path) != item['arrays'][name+'.npy']:
                raise ValueError('Changed source array: '+str(path))
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        rows = arrays['image_rows'][data.local_ids[data.sid[loc[at]]]]
        keys = arrays['query_keys'][data.local_ids[data.sid[loc[at]]]]
        verify_history_keys(keys, arrays['crop_keys'], rows)
        pairs, inverse = np.unique(np.stack((rows[:, :-1], rows[:, 1:]), -1).reshape(-1, 2),
                                   axis=0, return_inverse=True)
        geometry = links[item['recording']]['geometry']
        scale = np.asarray(geometry['video_size'])/np.asarray(geometry['annotation_size'])
        part = private/'pairs'/f'record{rid}.npz'; receipt_path = part.with_suffix('.json')
        pair_identity = dict(identity, record=int(rid), pairs_sha256=array_hash(pairs), scale_xy=scale.tolist())
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt['identity'] == pair_identity and receipt['sha256'] == file_digest(part)
            with np.load(part, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['pairs'], pairs); values = saved['features'].copy()
        else:
            start = time.monotonic(); values = np.empty((len(pairs), 19), np.float32)
            for i, pair in enumerate(pairs):
                values[i] = pair_features(arrays['rgb'][pair], arrays['coverage'][pair],
                    arrays['image_boxes'][pair], arrays['source_flags'][pair], scale, cv2)
            save_arrays(part, dict(pairs=pairs, features=values))
            receipt = dict(identity=pair_identity, pairs=len(pairs), seconds=time.monotonic()-start,
                           path=str(part.relative_to(ROOT)), sha256=file_digest(part))
            immutable_json(receipt_path, receipt); new_pairs += len(pairs)
        output[at] = values[inverse].reshape(-1, 7, 19); seen[at] = True
        records.append(receipt)
        beat = dict(pid=os.getpid(), time=time.time(), recording=item['recording'], state='record_complete',
                    rows=len(at), pairs=len(pairs), new_pairs=new_pairs)
        json_write(private/'heartbeat.json', beat); print(json.dumps(beat), flush=True)
    assert seen.all() and np.isfinite(output).all()
    path = private/'feature_store.npz'
    save_arrays(path, dict(ids=ids, features=output))
    names = MOTION_NAMES+QUALITY_NAMES
    report = dict(identity=identity, result_source='fresh_run_flow_on_cached_verified_past_pixels',
        rows=len(ids), query_pairs=len(ids)*7, unique_pairs=sum(r['pairs'] for r in records), records=records,
        summary={name:describe(output[..., i].reshape(-1)) for i, name in enumerate(names)},
        by_site={site:{name:describe(output[data.source_sites[loc] == site, :, i].reshape(-1))
                      for i, name in enumerate(names)} for site in reg['sites']},
        artifact=dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path)),
        feature_names=names, all_rows_retained=True, new_training=0, future_labels_used=False,
        main_outer_rows_scored=0, sensor_asof_certified=False, stage5c_executed=False, smc_enabled=False,
        limitations=['annotation_box_not_segmentation', 'surround_not_verified_camera_motion',
                     '32px_lowpass_crops', 'offline_supplied_annotations', 'four_explored_source_sites'])
    immutable_json(public/'extraction.json', report)
    print(json.dumps(dict(state='complete', rows=len(ids), unique_pairs=report['unique_pairs'], new_pairs=new_pairs)))


if __name__ == '__main__': main()
