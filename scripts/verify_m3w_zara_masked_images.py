"""Compare fixed prior inspection histories with the new partial-support reader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_masked_history_images import MaskedHistoryImageStore
from scripts.run_m3w_stationary_start_probe import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    args = p.parse_args()
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if (not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments')
            or not reports.is_relative_to(ROOT) or output.exists() or reports.exists()):
        raise ValueError('New private inspection and public report directories required')
    build = json.loads((args.cache / 'completion.json').read_text())
    parent_report = ROOT / 'outputs/publication_readiness_2026_09/zara_past_media/audit.json'
    prior = json.loads(parent_report.read_text())
    current_report = ROOT / 'outputs/publication_readiness_2026_09/zara_masked_images/report.json'
    if file_digest(current_report) != build['report_sha256']:
        raise ValueError('Build completion changed')
    current = json.loads(current_report.read_text())
    result = {}
    output.mkdir(parents=True)
    for rid in current['records']:
        store = MaskedHistoryImageStore(args.cache / rid, observation_mode='offline_annotation_diagnostic')
        if store.metadata['array_sha256'] != current['records'][rid]['array_sha256']:
            raise ValueError('Wrong registered build cache')
        previous = ROOT / 'data/stage_cvpr2027_experiments/zara_past_media' / rid
        if (file_digest(previous / 'controls.json') != prior['private_row_cache_hashes'][rid]['controls']
                or file_digest(previous / 'input_manifest.npz') != prior['private_row_cache_hashes'][rid]['input_manifest']):
            raise ValueError('Original fixed inspection controls changed')
        controls = json.loads((previous / 'controls.json').read_text())
        with np.load(previous / 'input_manifest.npz', allow_pickle=False) as old:
            old_valid = old['valid_mask'].copy()
        keys, histories = store.arrays['row_keys'], store.arrays['history_rows']
        by_query = {tuple(keys[row[-1]].astype(int)): index for index, row in enumerate(histories)}
        batches = []
        for control in controls:
            index = by_query[(control['query_stored_frame'], control['agent_id'])]
            batch = store.inputs(index)
            if batch['source_frames'].tolist() != control['history_source_frames']:
                raise ValueError('Different fixed past history')
            batches.append(batch)
        fractions = np.stack([b['pixel_coverage'].mean((1, 2, 3)) for b in batches])
        if not np.array_equal(fractions == 1, old_valid):
            raise ValueError('All-or-nothing original support not reproduced')
        recovered = (~old_valid) & (fractions > 0)
        sheets = []
        for page, first in enumerate(range(0, len(batches), 4)):
            subset = batches[first:first+4]
            canvas = Image.new('RGB', (8*136+8, len(subset)*170+36), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), rid + ': checkerboard = unobserved; fractional mask, not filled image', fill='black')
            for i, batch in enumerate(subset):
                for j in range(8):
                    rgb = np.rint(batch['rgb'][j].transpose(1, 2, 0)*255).astype(np.uint8)
                    coverage = batch['pixel_coverage'][j, 0]
                    yy, xx = np.indices(coverage.shape)
                    checker = np.where((xx//4+yy//4) % 2, 190, 225).astype(np.uint8)
                    display = rgb.copy()
                    display[coverage == 0] = np.repeat(checker[..., None], 3, axis=2)[coverage == 0]
                    im = Image.fromarray(display).resize((128, 128), Image.Resampling.NEAREST)
                    x, y = 8+j*136, 36+i*170
                    canvas.paste(im, (x, y))
                    draw.text((x, y+132), f'id {batch["agent_id"]} f {batch["source_frames"][j]}', fill='black')
                    draw.text((x, y+146), f'visible {fractions[first+i,j]:.1%}', fill='black')
            path = output / (rid + f'_masked_{page}.png')
            canvas.save(path)
            sheets.append({'path': str(path.relative_to(ROOT)), 'sha256': file_digest(path)})
        result[rid] = {'fixed_histories': len(controls), 'past_requests': int(old_valid.size),
            'old_full_support_requests': int(old_valid.sum()), 'old_missing_whole_crops': int((~old_valid).sum()),
            'old_missing_now_partial_requests': int(recovered.sum()),
            'minimum_recovered_pixel_fraction': float(fractions[recovered].min()),
            'mean_recovered_pixel_fraction': float(fractions[recovered].mean()),
            'every_prior_past_index_exactly_retained': True, 'old_full_support_mask_reproduced': True,
            'private_sheets': sheets}
    reports.mkdir(parents=True)
    report = {'result_source': 'fresh_run_fixed_prior_histories_partial_mask_verification',
        'build_report_sha256': file_digest(current_report), 'prior_report_sha256': file_digest(parent_report),
        'verifier_sha256': file_digest(Path(__file__)), 'records': result,
        'new_training': False, 'new_independent_scenes': 0, 'future_labels_used': False,
        'visual_inspection_completed_by_script': False, 'formal_training_admitted': False}
    atomic_json(reports / 'verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
