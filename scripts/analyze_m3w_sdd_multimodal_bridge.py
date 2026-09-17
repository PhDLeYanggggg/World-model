"""Describe observed image support and projected actor size, without target labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_multimodal_bridge import SDDStepImageStore


def summary(short_axis, geometric, retained, flags):
    return dict(rows=len(short_axis),
        projected_short_axis_pixels_quantiles=np.quantile(short_axis, [0, .1, .5, .9, 1]).tolist(),
        projected_short_axis_lt_pixels={str(k):int((short_axis < k).sum()) for k in (4, 8, 16)},
        geometric_support_fraction_quantiles=np.quantile(geometric, [0, .1, .5, .9, 1]).tolist(),
        retained_support_fraction_quantiles=np.quantile(retained, [0, .1, .5, .9, 1]).tolist(),
        source_occluded_rows=int(flags[:, 1].sum()), source_generated_rows=int(flags[:, 2].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    path = ROOT/config['reports']/'report.json'
    report = json.loads(path.read_text())
    if report['identity']['config_sha256'] != file_digest(args.config):
        raise ValueError('Extraction config changed')
    parts, per_record = [], []
    for record in report['records']:
        directory = ROOT/config['output']/record['recording']
        if file_digest(directory/'metadata.json') != record['artifacts']['metadata.json']:
            raise ValueError('Image metadata changed')
        store = SDDStepImageStore(directory, observation_mode='offline_annotated')
        boxes = store.arrays['image_boxes']
        block = store.metadata['crop_size']/store.output_size
        axis = (boxes[:, 2:]-boxes[:, :2]).min(1)/block
        geometric = store.arrays['geometric_count'].sum((1, 2))/store.metadata['crop_size']**2
        retained = store.arrays['retained_count'].sum((1, 2))/store.metadata['crop_size']**2
        flags = np.asarray(store.arrays['source_flags'])
        per_record.append(dict(recording=record['recording'], **summary(axis, geometric, retained, flags)))
        parts.append((axis, geometric, retained, flags))
    pooled = [np.concatenate([r[i] for r in parts]) for i in range(4)]
    result = dict(result_source='fresh_run', extraction_report_sha256=file_digest(path),
        analysis_code_sha256=file_digest(Path(__file__)), summary=summary(*pooled),
        per_record=per_record, quantile_probabilities=[0, .1, .5, .9, 1],
        measurement='short_bbox_axis_after_video_resize_and_3x_pooling_not_visible_person_size',
        scope='fixed_past_only_cache_requests_overlapping_not_population_or_independent_samples',
        thresholds_are_descriptive_not_training_or_admission_rules=True,
        semantic_alignment_certified=False, old_eth_ucy_failures_explained=False,
        new_training=False, future_target_arrays_used=False)
    json_write(ROOT/config['reports']/'input_quality.json', result)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
