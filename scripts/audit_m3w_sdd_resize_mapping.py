"""Verify resized image-coordinate mapping, without fitting a forecast model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_image_coordinates import SDDImageCoordinates
from src.world_model.m3w_sdd_media_audit import annotation_summary, reference_comparison


def draw_comparison(images, rows, mapper, destination, key):
    sheet = Image.new('RGB', (len(images)*420, 950), 'white')
    caption = ImageDraw.Draw(sheet)
    for col, (frame, image) in enumerate(sorted(images.items())):
        current = rows[(rows[:, 5] == frame) & (rows[:, 6] == 0)]
        corrected = mapper.past_boxes(current[:, 1:5], frame_ids=current[:, 5], query_frame=frame)
        for row, boxes in enumerate((current[:, 1:5], corrected)):
            panel = Image.fromarray(image.copy())
            pen = ImageDraw.Draw(panel)
            for box in boxes:
                pen.rectangle(tuple(box), outline=(230, 45, 50), width=2)
            panel.thumbnail((420, 440))
            y = row*475
            sheet.paste(panel, (col*420, y+30))
            label = 'unscaled' if row == 0 else 'reference-to-video mapped'
            caption.text((col*420+4, y+5), f'{key} frame {frame}: {label}', fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main():
    audit_path = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment/audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['complete'] or len(audit['records']) != 60:
        raise ValueError('Full decoder/index audit must complete first')
    parent = json.loads((ROOT/'configs/m3w_offline_visual_forecast.json').read_text())
    sys.path.insert(0, str(ROOT/parent['decoder_path']))
    import av
    output = ROOT/'data/stage_cvpr2027_experiments/sdd_resize_mapping'
    output.mkdir(parents=True, exist_ok=True)
    selected = {'bookstore/video0', 'coupa/video0', 'deathCircle/video2',
                'gates/video0', 'hyang/video7', 'quad/video0'}
    records = []
    for old in audit['records']:
        key = old['identity']['key']
        directory = ROOT/'external_data/StanfordDroneDataset'
        video = directory/'video'/key/'video.mp4'
        annotation = directory/'annotations'/key/'annotations.txt'
        ref_path = directory/'annotations'/key/'reference.jpg'
        for name, path in (('video', video), ('annotations', annotation), ('reference', ref_path)):
            if file_digest(path) != old['identity']['source_hashes'][name]:
                raise ValueError('Source changed since full audit: '+key)
        reference = Image.open(ref_path).convert('RGB')
        rows = np.loadtxt(annotation, usecols=range(9), dtype=float, ndmin=2)
        wanted = old['sampled_frame_indices'] if key in selected else [0]
        images = {}
        with av.open(str(video)) as container:
            container.streams.video[0].codec_context.thread_count = 4
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    image = frame.to_ndarray(format='rgb24')
                    if hashlib.sha256(image.tobytes()).hexdigest() != old['sampled_frame_pixel_sha256'][str(index)]:
                        raise ValueError('Independent sample decode differs: '+key)
                    images[index] = image
                if index >= max(wanted):
                    break
        if set(images) != set(wanted):
            raise ValueError('Selected frame not decoded: '+key)
        height, width = images[0].shape[:2]
        mapper = SDDImageCoordinates(reference.width, reference.height, width, height)
        state_check = annotation_summary(rows, reference.width, reference.height)
        resized_reference = np.asarray(reference.resize((width, height), Image.Resampling.BILINEAR))
        compared = reference_comparison(images[0], resized_reference)
        mapped_boxes = mapper.past_boxes(rows[:, 1:5], frame_ids=rows[:, 5],
                                        query_frame=int(rows[:, 5].max()))
        mapped_in_bounds = ((mapped_boxes[:, 0] >= -1e-8) & (mapped_boxes[:, 1] >= -1e-8)
                            & (mapped_boxes[:, 2] <= width+1e-8) & (mapped_boxes[:, 3] <= height+1e-8))
        restored = mapper.video_boxes_to_annotation(mapped_boxes)
        roundtrip = float(np.abs(restored-rows[:, 1:5]).max())
        current = dict(recording=key, source_hashes=old['identity']['source_hashes'],
            geometry=mapper.metadata(), annotation_space_summary=state_check,
            old_video_space_visible_boxes_outside=old['annotation']['visible_boxes_outside_image'],
            mapped_video_space_visible_boxes_outside=int(((rows[:, 6] == 0) & ~mapped_in_bounds).sum()),
            coordinate_roundtrip_max_absolute_pixel_error=roundtrip,
            first_frame_resized_reference_comparison=compared,
            independently_redecoded_frames=sorted(images), independent_sample_pixel_hashes_match=True,
            visual_review_status='not_reviewed', manual_boxes_not_human_gold=True,
            new_training_source_admitted=False, model_or_future_targets_built=False)
        if key in selected:
            relative = Path('private_visual_checks')/(key.replace('/', '_')+'.png')
            draw_comparison(images, rows, mapper, output/relative, key)
            current.update(private_comparison=str(relative), private_comparison_sha256=file_digest(output/relative))
        records.append(current)
        print(json.dumps(dict(recording=key, scale=mapper.scale_xy.tolist(), roundtrip=roundtrip,
                              reference_correlation=compared['pixel_correlation'])), flush=True)
    result = dict(result_source='fresh_run_coordinate_mapping_and_independent_sample_decode',
        audit_sha256=file_digest(audit_path), decoder_version=av.__version__,
        code_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in
                    (Path(__file__), ROOT/'src/world_model/m3w_sdd_image_coordinates.py')},
        records=records, dimension_mismatches=sum(r['geometry']['annotation_size'] != r['geometry']['video_size'] for r in records),
        independent_sample_frames=sum(len(r['independently_redecoded_frames']) for r in records),
        unscaled_visible_boxes_outside=sum(r['old_video_space_visible_boxes_outside'] for r in records),
        mapped_visible_boxes_outside=sum(r['mapped_video_space_visible_boxes_outside'] for r in records),
        semantic_alignment_of_all_rows_certified=False, coordinate_state_not_changed=True,
        new_training_source_admitted=False, sealed_evaluation_roles_opened=False,
        physical_time_or_metric_certified=False, stage5c_executed=False, smc_enabled=False)
    reports = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment'
    json_write(reports/'resize_mapping.json', result)


if __name__ == '__main__':
    main()
