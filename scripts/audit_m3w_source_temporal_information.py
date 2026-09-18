"""Post-hoc input-boundary audit, same source-only population, no new fit or forecast."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config, context, FeatureCorpus
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_temporal_information import verify_history_keys, temporal_diagnostics, describe
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(Path('configs/m3w_source_pretrained_temporal_v1.json'))
    previous = ROOT/reg['reports']
    verification = json.loads((previous/'verification.json').read_text())
    for path, sha in verification['artifact_hashes'].items():
        if file_digest(ROOT/path) != sha: raise ValueError('Changed prior artifact: '+path)
    data = context(reg)
    features = FeatureCorpus(data, ROOT/reg['output'], json.loads((previous/'preparation.json').read_text()))
    ids = features.ids; loc = ids-data.nmain
    receipts = json.loads(data.manifest_path.read_text())['records']
    values = {}; seen = np.zeros(len(ids), bool); checked_frames = 0
    for rid in np.unique(data.record_ids[data.sid[loc]]):
        at = np.flatnonzero(data.record_ids[data.sid[loc]] == rid)
        folder = data.manifest_path.parent/receipts[int(rid)]['recording']
        arrays = {name:np.load(folder/(name+'.npy'), mmap_mode='r', allow_pickle=False)
                  for name in ('query_keys','crop_keys','image_boxes')}
        store = data.images[int(rid)]
        rows = store['image_rows'][data.local_ids[data.sid[loc[at]]]]
        keys = arrays['query_keys'][data.local_ids[data.sid[loc[at]]]]
        verify_history_keys(keys, arrays['crop_keys'], rows)
        for begin in range(0, len(at), 128):
            selected = at[begin:begin+128]; image_rows = rows[begin:begin+128]
            metrics = temporal_diagnostics(store['rgb'][image_rows], store['coverage'][image_rows],
                arrays['image_boxes'][image_rows], features.embedding[features.rows[selected]])
            for name, value in metrics.items():
                if name not in values: values[name] = np.empty(len(ids), value.dtype)
                values[name][selected] = value
            seen[selected] = True; checked_frames += image_rows.size
        print(json.dumps(dict(recording=receipts[int(rid)]['recording'], rows=len(at))), flush=True)
    assert seen.all() and checked_frames == len(ids)*8
    private = ROOT/'data/stage_cvpr2027_experiments/source_temporal_information_v1'
    public = ROOT/'outputs/publication_readiness_2026_09/source_temporal_information_v1'
    path = private/'input_diagnostics.npz'
    save_arrays(path, dict(ids=ids, **values))
    summary = {name:describe(value) for name, value in values.items()}
    result = dict(result_source='fresh_run_input_diagnostic_cached_verified_pixels_embeddings',
        post_hoc=True, new_training=0, new_forecasts=0, target_labels_used=False,
        rows=len(ids), checked_history_keys=checked_frames, recordings=29, sites=4,
        previous_verification_sha256=file_digest(previous/'verification.json'),
        data_identity=data.identity, assignment_hash=data.assignment_hash,
        summary=summary, by_site={site:{k:describe(v[data.source_sites[loc] == site]) for k,v in values.items()}
            for site in reg['sites']},
        row_archive=dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path)),
        input_boundary='offline_supplied_annotations_not_sensor_asof',
        main_outer_queries_processed=0, cohort_changed=False, new_deployment=False,
        spatial_caveat='boxes_are_annotation_regions_not_segmentation; pixel_MAE_is_not_motion_or_intent',
        stage5c_executed=False, smc_enabled=False)
    immutable_json(public/'audit.json', result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
