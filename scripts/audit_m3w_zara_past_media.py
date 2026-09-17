"""Validate a fixed source-origin adapter and inspect eight past Zara images."""
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
from src.evaluation.m3w_zara_media_lineage import read_vsp, trace_rows, image_coordinates
from src.evaluation.m3w_zara_past_media import (
    project_image_xy, source_origin_anchor, native_to_image_xy,
    select_past_media_controls, validate_past_request,
)
from src.evaluation.m3w_past_motion_correspondence import centered_patch
from scripts.audit_m3w_annotation_clock_geometry import avi_header
from scripts.run_m3w_stationary_start_probe import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report-dir', required=True, type=Path)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Changed source: ' + name)
    previous_reg = json.loads((ROOT / reg['lineage_registration']).read_text())
    for name, digest in previous_reg['bindings'].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Changed original lineage source: ' + name)
    parent = ExperimentContract(json.loads((ROOT / previous_reg['parent_protocol']).read_text()), ROOT)
    if parent.digest != previous_reg['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    sys.path.insert(0, str(ROOT / reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Decoder changed')
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if (not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments')
            or not reports.is_relative_to(ROOT) or output.exists() or reports.exists()):
        raise ValueError('Require new ignored private output and new workspace report')
    output.mkdir(parents=True)
    started, result, local_sha = time.monotonic(), {}, {}
    for rid in previous_reg['recordings']:
        if parent.protocol['assignments'][rid] != 'fit':
            raise ValueError('Fit-only source diagnostic')
        reader, _ = parent.open_recording(rid, purpose='fit')
        directory = ROOT / 'external_data/OpenTraj/datasets/UCY' / rid.removeprefix('ucy_')
        header, points = avi_header(directory / 'video.avi'), reader.points
        trace = trace_rows(points, read_vsp(directory / 'annotation.vsp'), 1)
        if not trace['available'].all():
            raise ValueError('Every source row must be traceable without extrapolation')
        xy = image_coordinates(trace['centered_xy'], header['width'], header['height'])
        matrix = np.loadtxt(directory / 'H.txt')
        if not trace['exact_control'][0]:
            raise ValueError('The origin anchor must be an exact source control')
        origin = (np.zeros(2) if rid == 'ucy_zara01'
                  else source_origin_anchor(points[0, 2:], xy[0], matrix))
        native = project_image_xy(xy, matrix) + origin
        error = np.linalg.norm(native - points[:, 2:], axis=1)
        relative_errors = []
        for agent in np.unique(points[:, 1]):
            ids = np.flatnonzero(points[:, 1] == agent)
            ids = ids[np.argsort(points[ids, 0])]
            relative_errors.extend(np.linalg.norm(np.diff(native[ids], axis=0)
                                                 - np.diff(points[ids, 2:], axis=0), axis=1))
        back_error = np.linalg.norm(native_to_image_xy(points[:, 2:], matrix, origin) - xy, axis=1)
        mapping_pass = bool(np.all(error <= reg['absolute_native_tolerance']))
        if not mapping_pass:
            raise ValueError('One-anchor geometry hypothesis failed; no media requests admitted')
        controls = select_past_media_controls(points, trace, header['width'], header['height'],
                                             max_agents=reg['max_agents_per_recording'])
        for control in controls:
            validate_past_request(control)
            if trace['source_frame'][0] > control['query_source_frame']:
                raise ValueError('Origin anchor after query')
        wanted = {f for c in controls for f in c['history_source_frames']}
        images, local = {}, output / rid
        local.mkdir()
        with av.open(str(directory / 'video.avi')) as container:
            decoder_rate = str(container.streams.video[0].average_rate)
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    images[index] = np.asarray(frame.to_image()).copy()
                if index % 1000 == 0:
                    atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'state': 'decode',
                        'recording': rid, 'frame': index, 'elapsed_seconds': time.monotonic() - started})
                if index >= max(wanted):
                    break
        if set(images) != wanted:
            raise ValueError('Missing requested indexed frames')
        missing, sheets, valid = 0, [], []
        for page, offset in enumerate(range(0, len(controls), 4)):
            subset = controls[offset:offset + 4]
            sheet = Image.new('RGB', (8 * 136 + 8, len(subset) * 168 + 40), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 5), rid + ': source-annotation anchors; not online/physical calibration', fill='black')
            for i, control in enumerate(subset):
                mask = []
                for j, frame in enumerate(control['history_source_frames']):
                    patch = centered_patch(images[frame], control['history_image_xy'][j], reg['crop_size'])
                    x, y = 8 + j * 136, 40 + i * 168
                    mask.append(patch is not None)
                    if patch is None:
                        missing += 1
                        draw.text((x, y + 50), 'out of frame', fill='black')
                    else:
                        im = Image.fromarray(patch).resize((128, 128), Image.Resampling.NEAREST)
                        mark = ImageDraw.Draw(im)
                        mark.line((59, 64, 69, 64), fill='red', width=1)
                        mark.line((64, 59, 64, 69), fill='red', width=1)
                        sheet.paste(im, (x, y))
                    draw.text((x, y + 132), f'id {control["agent_id"]} f {frame}', fill='black')
                valid.append(mask)
            path = local / f'contacts_{page}.png'
            sheet.save(path)
            sheets.append({'path': str(path.relative_to(ROOT)), 'sha256': file_digest(path)})
        atomic_json(local / 'controls.json', controls)
        np.savez(local / 'input_manifest.npz',
                 agent_id=np.array([c['agent_id'] for c in controls]),
                 frames=np.array([c['history_source_frames'] for c in controls]),
                 image_xy=np.array([c['history_image_xy'] for c in controls]),
                 valid_mask=np.array(valid), origin=origin,
                 latest_source_controls=np.array([c['latest_contributing_control_frames'] for c in controls]))
        local_sha[rid] = {'controls': file_digest(local / 'controls.json'),
                         'input_manifest': file_digest(local / 'input_manifest.npz')}
        result[rid] = {'rows': len(points), 'origin_policy': 'zero' if rid == 'ucy_zara01' else 'first_source_row_only',
            'origin_translation_native': origin.tolist(), 'max_native_replay_error': float(error.max()),
            'max_nonanchor_native_replay_error': float(error[1:].max()),
            'max_relative_increment_error': float(max(relative_errors)),
            'max_inverse_image_error_px': float(back_error.max()), 'numeric_lineage_pass': mapping_pass,
            'selected_agents': len(controls), 'requested_past_images': 8 * len(controls),
            'unique_decoded_frames': len(wanted), 'missing_crops': missing,
            'complete_crops': int(np.all(valid, axis=1).sum()),
            'controls_as_of_query_histories': sum(c['source_controls_as_of_query'] for c in controls),
            'header_fps': header['header_fps'], 'decoder_rate': decoder_rate,
            'local_contact_sheets': sheets, 'visual_inspection_completed_by_script': False}
        print(json.dumps({'recording': rid, **result[rid]}), flush=True)
    reports.mkdir(parents=True)
    report = {'result_source': 'fresh_run_fit_only_origin_repair_and_past_media_decode',
        'registration_sha256': file_digest(args.registration), 'parent_protocol_sha256': parent.digest,
        'records': result, 'private_row_cache_hashes': local_sha, 'elapsed_seconds': time.monotonic() - started,
        'new_training': False, 'future_label_api_used': False, 'parent_protocol_modified': False,
        'development_calibration_confirmation_opened': False, 'media_training_admitted': False,
        'capture_clock_or_metric_calibration_verified': False, 'observation_definition_pending': True,
        'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
    atomic_json(reports / 'audit.json', report)
    lines = ['# Zara Source-Origin Repair and Past Images', '',
        'Fresh fit-only source repair, not a forecasting result or physical calibration.', '',
        '| Recording | Rows replayed | Max native error | Max pixel inverse error | Past crops | Complete histories |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for rid, r in result.items():
        lines.append(f'| {rid} | {r["rows"]} | {r["max_native_replay_error"]:.9g} | '
                     f'{r["max_inverse_image_error_px"]:.9g} | '
                     f'{r["requested_past_images"] - r["missing_crops"]}/{r["requested_past_images"]} | '
                     f'{r["complete_crops"]}/{r["selected_agents"]} |')
    lines += ['', 'The Zara02 origin is defined by one exact source control, not all-row regression.',
        'Every remaining row is an audit residual. No future outcome selects rows, lag, matrix or origin.',
        'Frame indices follow the source annotation convention; physical time is not certified.',
        'Private contact sheets require inspection; successful decoding is not verified person localization.',
        'Offline interpolated annotations still carry later-control provenance. Formal observation semantics remain pending.', '']
    (reports / 'audit.md').write_text('\n'.join(lines))
    atomic_json(output / 'completion.json', {'report_sha256': file_digest(reports / 'audit.json'), **local_sha})
    atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'state': 'complete',
                'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
