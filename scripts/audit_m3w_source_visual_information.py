"""Describe past pixel variation without changing rows, labels or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_m3w_source_visual_start import VisualCorpus
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def temporal_pixel_variation(rgb, coverage):
    rgb, coverage = np.asarray(rgb), np.asarray(coverage)
    if (rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.shape[1] < 2
            or coverage.shape != (len(rgb), rgb.shape[1], *rgb.shape[-2:])):
        raise ValueError('Aligned past RGB and coverage arrays required')
    common = (coverage[:, 1:] > 0) & (coverage[:, :-1] > 0)
    difference = np.abs(rgb[:, 1:].astype(float)-rgb[:, :-1].astype(float))
    count = common.sum((2, 3))
    total = np.sum(difference*common[:, :, None], axis=(2, 3, 4))
    mean = np.divide(total, 3*count, out=np.zeros_like(total), where=count > 0)
    return dict(mean_absolute_uint8_difference=mean, common_pixels=count,
                identical_supported_pairs=(total == 0) & (count > 0))


def effective_box_extent(boxes):
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError('Finite image-space boxes required')
    extent = boxes[:, 2:]-boxes[:, :2]
    if np.any(extent < 0):
        raise ValueError('Ordered image-space boxes required')
    return extent/3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    data = VisualCorpus(reg)
    records = []
    for name, ids in [('ETH', np.flatnonzero(data.folds == 0)),
                      ('Hotel', np.flatnonzero(data.folds == 1)),
                      ('SDD_train', np.arange(data.nmain, len(data.y)))]:
        pieces = []
        for start in range(0, len(ids), 128):
            pieces.append(temporal_pixel_variation(*data.raw_images(ids[start:start+128])))
        values = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        available = values['common_pixels'] > 0
        delta = values['mean_absolute_uint8_difference'][available]
        records.append(dict(population=name, rows=len(ids), consecutive_pairs=available.size,
            supported_consecutive_pairs=int(available.sum()),
            mean_uint8_pixel_difference=float(delta.mean()),
            quantile_levels=[0, .25, .5, .75, 1],
            pixel_difference_quantiles=np.quantile(delta, [0, .25, .5, .75, 1]).tolist(),
            identical_supported_pairs=int(values['identical_supported_pairs'].sum()),
            entirely_identical_history_rows=int(values['identical_supported_pairs'].all(1).sum()),
            common_pixel_fraction=float(values['common_pixels'].mean()/(32*32))))
    receipts = json.loads(data.manifest_path.read_text())['records']
    extents = []
    for record in np.unique(data.record_ids[data.sid]):
        selected = data.sid[data.record_ids[data.sid] == record]
        image_rows = np.unique(data.images[int(record)]['image_rows'][data.local_ids[selected]])
        directory = data.manifest_path.parent/receipts[int(record)]['recording']
        boxes = np.load(directory/'image_boxes.npy', mmap_mode='r', allow_pickle=False)[image_rows]
        extents.append(effective_box_extent(boxes))
    extent = np.concatenate(extents)
    box_audit = dict(population='SDD supervised stationary queries, unique past crops only',
        unique_past_crops=len(extent), axes=['width','height'], quantile_levels=[0,.25,.5,.75,1],
        model_pixel_extent_quantiles=np.quantile(extent,[0,.25,.5,.75,1],axis=0).tolist(),
        minimum_axis_below_4_model_pixels=int((extent.min(1)<4).sum()),
        minimum_axis_below_8_model_pixels=int((extent.min(1)<8).sum()),
        boxes_extending_beyond_32_model_pixels=int((extent.max(1)>32).sum()),
        main_box_audit='not_run_no_verified_box_extent_in_current_main_cache',
        annotation_boxes_are_not_visible_body_segmentation=True,used_for_filtering=False)
    result = dict(result_source='fresh_run_descriptive_audit_of_cached_verified_past_pixels',
        registration_sha256=file_digest(args.registration), identity=data.identity,
        crop_original_pixels=[96, 96], model_pixels=[32, 32], records=records, source_box_extent=box_audit,
        future_pixels_used=False, labels_used_in_pixel_statistic=False,
        rows_filtered=False, training_changed=False, independent_confirmation=False,
        interpretation='Pixel changes may be body motion, other agents, lighting, compression or registration; not an intention label.')
    reports = ROOT/reg['reports']
    json_write(reports/'pixel_information_audit.json', result)
    lines = ['# Past Pixel Support and Temporal Variation', '',
        'Descriptive input audit only. No labels or model scores enter these statistics.',
        'Crops reduce a 96 x 96 source-pixel region to 32 x 32; physical scale differs by camera.', '',
        '| Population | Rows | Supported adjacent pairs | Identical pairs | Entirely identical histories | Mean absolute pixel change (0-255) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append(f"| {r['population']} | {r['rows']} | {r['supported_consecutive_pairs']} | {r['identical_supported_pairs']} | {r['entirely_identical_history_rows']} | {r['mean_uint8_pixel_difference']:.6f} |")
    lines += ['', 'Only pixels supported at both adjacent times are compared. Missing pairs are not treated as unchanged.',
        'Nonzero RGB change does not prove informative body-state visibility or correct intent labels.',
        'Compression, lighting, other agents and crop registration can also change pixels. No physical time or scale equivalence is asserted.',
        '', '## Source Box Extent', '',
        f"The {len(extent):,} unique past SDD crops have median annotation-box width/height {np.median(extent[:,0]):.3f}/{np.median(extent[:,1]):.3f} model pixels.",
        f"The smaller axis is below four model pixels in {box_audit['minimum_axis_below_4_model_pixels']:,} crops and below eight in {box_audit['minimum_axis_below_8_model_pixels']:,} crops.",
        f"An axis extends outside the 32-pixel crop in {box_audit['boxes_extending_beyond_32_model_pixels']:,} cases.",
        'These are mapped annotation extents, not visible-body masks or a certified posture-resolution threshold.',
        'Main-site box-size audit is not_run because the current main cache has no verified box extents.', '',
        'This audit neither filters the registered population nor changes training, thresholds or evaluation roles.', '']
    (reports/'pixel_information_audit.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
