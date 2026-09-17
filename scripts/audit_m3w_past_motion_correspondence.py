"""Registered fit-only motion correspondence on local past video frames."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_past_motion_correspondence import (
    centered_patch, match_past_patch, select_moving_controls,
)
from scripts.run_m3w_stationary_start_probe import atomic_json


def summarize(rows):
    matched = [r for r in rows if r['status'] == 'matched']
    if not matched:
        return {'requested_pairs': len(rows), 'matched_pairs': 0}
    error = np.array([r['annotation_error_px'] for r in matched])
    zero = np.array([r['zero_shift_error_px'] for r in matched])
    cosine = [r['direction_cosine'] for r in matched if r['direction_cosine'] is not None]
    agent_means = [np.mean([r['annotation_error_px'] for r in matched if r['agent_id'] == agent])
                   for agent in sorted({r['agent_id'] for r in matched})]
    return {'requested_pairs': len(rows), 'matched_pairs': len(matched),
            'support_or_texture_missing': len(rows) - len(matched),
            'mean_annotation_error_px': float(error.mean()),
            'median_annotation_error_px': float(np.median(error)),
            'mean_zero_shift_error_px_same_pairs': float(zero.mean()),
            'matched_error_below_zero_shift_fraction': float(np.mean(error < zero)),
            'within_3px_fraction': float(np.mean(error <= 3)),
            'direction_cosine_mean_nonzero_pairs': float(np.mean(cosine)) if cosine else None,
            'direction_nonzero_pairs': len(cosine),
            'mean_peak_zncc': float(np.mean([r['peak_zncc'] for r in matched])),
            'equal_agent_mean_error_px': float(np.mean(agent_means)),
            'independent_statistical_confirmation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed registered source: ' + name)
    parent = ExperimentContract(json.loads((ROOT/reg['parent_protocol']).read_text()), ROOT)
    if parent.digest != reg['parent_protocol_sha256']:
        raise ValueError('Changed parent protocol')
    sys.path.insert(0, str(ROOT/reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Changed decoder')
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if (not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments')
            or not reports.is_relative_to(ROOT) or output.exists() or reports.exists()):
        raise ValueError('New local ignored data output and new workspace report required')
    output.mkdir(parents=True)
    started = time.monotonic()
    aggregate, local_pairs, local_controls = {}, [], []
    identity = {'registration_sha256': file_digest(args.registration),
                'parent_protocol_sha256': parent.digest, 'decoder': av.__version__}
    atomic_json(output/'run_identity.json', identity)
    for rid in reg['recordings']:
        if parent.protocol['assignments'][rid] != 'fit':
            raise ValueError('Moving diagnostic must remain fit-only')
        reader, _ = parent.open_recording(rid, purpose='fit')
        original = ROOT/reader.metadata['files'][0]['path']
        if file_digest(original) != reader.metadata['files'][0]['sha256']:
            raise ValueError('Annotation source changed')
        h = np.loadtxt(original.parent/'H.txt')
        selected, eligible = select_moving_controls(reader.points, h, rid,
            max_agents=reg['max_agents_per_recording'], min_motion=reg['min_past_net_motion_px'])
        if not selected:
            raise ValueError('No past moving controls: ' + rid)
        wanted = {f for r in selected for f in r['history_frames']}
        images, gray = {}, {}
        directory = output/rid
        directory.mkdir()
        with av.open(str(original.parent/'video.avi')) as container:
            rate = str(container.streams.video[0].average_rate)
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    im = frame.to_image()
                    im.save(directory/f'frame_{index:06d}.png')
                    images[index] = np.asarray(im).copy()
                    gray[index] = np.asarray(im.convert('L'), dtype=float)
                if index % 1000 == 0:
                    atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': 'decode',
                        'recording': rid, 'frame': index, 'elapsed_seconds': time.monotonic()-started})
                if index >= max(wanted):
                    break
        if set(images) != wanted:
            raise ValueError('Missing requested observed images')
        missing_crops, patches = 0, {}
        scene_pairs = []
        for r in selected:
            local_controls.append({'recording_id': rid, **r})
            for j, frame in enumerate(r['history_frames']):
                if frame > r['frame_id']:
                    raise ValueError('Future image request')
                crop = centered_patch(images[frame], r['image_xy'][j], reg['inspection_crop_size'])
                if crop is None:
                    missing_crops += 1
                else:
                    patches[(r['agent_id'], j)] = crop
            for j in range(1, 8):
                earlier, later = r['history_frames'][j-1:j+1]
                for size in reg['template_sizes']:
                    match = match_past_patch(gray[earlier], gray[later], r['image_xy'][j-1],
                        template_size=size, search_radius=reg['search_radius'])
                    record = {'recording_id': rid, 'agent_id': r['agent_id'], 'pair': j,
                              'earlier_frame': earlier, 'later_frame': later,
                              'query_current_frame': r['frame_id'], 'template_size': size, **match}
                    if match['status'] == 'matched':
                        annotated = np.asarray(r['image_xy'][j]) - r['image_xy'][j-1]
                        predicted = np.asarray(match['shift_xy'])
                        denominator = float(np.linalg.norm(annotated)*np.linalg.norm(predicted))
                        record.update(annotation_error_px=float(np.linalg.norm(
                            np.asarray(match['image_xy'])-r['image_xy'][j])),
                            zero_shift_error_px=float(np.linalg.norm(annotated)),
                            direction_cosine=float(annotated@predicted/denominator) if denominator > 1e-8 else None)
                    scene_pairs.append(record)
        # Mark only the projected point; this is not a body box or pose label.
        for page, offset in enumerate(range(0, len(selected), 4)):
            subset = selected[offset:offset+4]
            sheet = Image.new('RGB', (8*136+8, len(subset)*166+30), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 6), f'{rid}: eight observed images; projected point, not a verified body label', fill='black')
            for i, r in enumerate(subset):
                for j, f in enumerate(r['history_frames']):
                    crop = patches.get((r['agent_id'], j))
                    x, y = 8+j*136, 30+i*166
                    if crop is not None:
                        im = Image.fromarray(crop).resize((128, 128), Image.Resampling.NEAREST)
                        mark = ImageDraw.Draw(im)
                        mark.line((59,64,69,64), fill='red', width=1)
                        mark.line((64,59,64,69), fill='red', width=1)
                        sheet.paste(im, (x,y))
                    else:
                        draw.text((x,y+50), 'missing support', fill='black')
                    draw.text((x,y+132), f'id {r["agent_id"]} / f {f}', fill='black')
            sheet.save(directory/f'contacts_{page}.png')
        aggregate[rid] = {'eligible_agents': eligible, 'selected_agents': len(selected),
            'requested_past_images': 8*len(selected), 'unique_frames': len(wanted),
            'missing_centered_crops': missing_crops, 'encoded_rate': rate,
            'all_requests_at_or_before_query': True,
            'methods': {str(s): summarize([r for r in scene_pairs if r['template_size']==s])
                        for s in reg['template_sizes']}}
        local_pairs.extend(scene_pairs)
        atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': 'recording_complete',
            'recording': rid, 'elapsed_seconds': time.monotonic()-started})
        print(json.dumps({'recording': rid, **aggregate[rid]}, allow_nan=False), flush=True)
    atomic_json(output/'local_controls.json', local_controls)
    atomic_json(output/'local_pairs.json', local_pairs)
    result = {'result_source': 'fresh_run_fit_only_past_image_correspondence', **identity,
        'records': aggregate, 'elapsed_seconds': time.monotonic()-started,
        'future_label_api_used': False, 'later_annotation_used_in_matching': False,
        'later_observed_annotation_used_for_correspondence_scoring': True,
        'temporal_offset_fitted': False, 'selected_template_size': None,
        'new_model_training': False, 'primary_protocol_changed': False,
        'development_calibration_confirmation_opened': False,
        'physical_clock_or_full_alignment_certified': False, 'stage5c_executed': False, 'smc_enabled': False}
    reports.mkdir(parents=True)
    atomic_json(reports/'audit.json', result)
    atomic_json(output/'completion.json', {'report_sha256': file_digest(reports/'audit.json'),
        'local_controls_sha256': file_digest(output/'local_controls.json'),
        'local_pairs_sha256': file_digest(output/'local_pairs.json')})
    lines = ['# Moving Past-Frame Correspondence', '',
        'Fresh fit-only input diagnostic, not future forecasting or independent alignment certification.', '',
        '| Source | Template | Matched / pairs | Mean error px | Median error px | Zero-shift mean px | Within 3px | Direction cosine |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for rid, a in aggregate.items():
        for size, m in a['methods'].items():
            lines.append(f'| {rid} | {size} | {m["matched_pairs"]}/{m["requested_pairs"]} | '
                f'{m.get("mean_annotation_error_px",float("nan")):.3f} | '
                f'{m.get("median_annotation_error_px",float("nan")):.3f} | '
                f'{m.get("mean_zero_shift_error_px_same_pairs",float("nan")):.3f} | '
                f'{m.get("within_3px_fraction",float("nan")):.1%} | {m.get("direction_cosine_mean_nonzero_pairs")} |')
    lines += ['', 'Support failures remain counted. Templates/backgrounds can match the wrong person or static texture.',
              'Both fixed sizes are reported; none selected using forecast outcomes. No time offset or seconds conversion.',
              'Only past/current images and annotations; no future targets, new training, Stage5C or SMC.', '']
    (reports/'results.md').write_text('\n'.join(lines))
    atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': 'complete',
        'elapsed_seconds': time.monotonic()-started})


if __name__ == '__main__':
    main()
