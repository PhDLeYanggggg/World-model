"""Compare registered image searches on matched support, keeping every failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from scripts.run_m3w_stationary_start_probe import atomic_json


def pair_key(row):
    return row['recording_id'], row['agent_id'], row['pair'], row['template_size']


def paired_errors(initial, wider):
    a, b = {pair_key(r): r for r in initial}, {pair_key(r): r for r in wider}
    if len(a) != len(initial) or len(b) != len(wider) or a.keys() != b.keys():
        raise ValueError('Correspondence pair identities differ or repeat')
    keys = [k for k in a if a[k]['status'] == b[k]['status'] == 'matched']
    if not keys:
        return {'requested_pairs': len(a), 'jointly_supported': 0}
    return {'requested_pairs': len(a), 'jointly_supported': len(keys),
            'initial_mean_error_px': float(np.mean([a[k]['annotation_error_px'] for k in keys])),
            'wider_mean_error_px': float(np.mean([b[k]['annotation_error_px'] for k in keys])),
            'initial_missing': sum(r['status'] != 'matched' for r in a.values()),
            'wider_missing': sum(r['status'] != 'matched' for r in b.values()),
            'wider_error_above_10px_count': sum(b[k]['annotation_error_px'] > 10 for k in keys)}


def read_run(directory, report, registration):
    reg = json.loads(registration.read_text())
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed source binding: ' + name)
    audit = json.loads(report.read_text())
    if file_digest(registration) != audit['registration_sha256']:
        raise ValueError('Report registration mismatch')
    completion = json.loads((directory/'completion.json').read_text())
    if completion['report_sha256'] != file_digest(report):
        raise ValueError('Report completion mismatch')
    for name in ('local_controls', 'local_pairs'):
        if completion[name+'_sha256'] != file_digest(directory/(name+'.json')):
            raise ValueError('Local row artifact changed')
    return reg, json.loads((directory/'local_controls.json').read_text()), json.loads((directory/'local_pairs.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    public = ROOT/'outputs/publication_readiness_2026_09'
    data = ROOT/'data/stage_cvpr2027_experiments'
    r0, controls, pairs0 = read_run(data/'past_motion_correspondence',
        public/'past_motion_correspondence/audit.json', ROOT/'configs/m3w_past_motion_correspondence.json')
    r1, controls1, pairs1 = read_run(data/'past_motion_search_support',
        public/'past_motion_search_support/audit.json', ROOT/'configs/m3w_past_motion_search_support.json')
    if controls != controls1:
        raise ValueError('Selected controls changed')
    comparison_fields = ('parent_protocol_sha256', 'recordings', 'max_agents_per_recording',
                         'min_past_net_motion_px', 'inspection_crop_size', 'template_sizes')
    if any(r0[k] != r1[k] for k in comparison_fields):
        raise ValueError('Change beyond registered search radius')
    lookup = {(r['recording_id'], r['agent_id']): r for r in controls}
    result = {}
    for scene in r0['recordings']:
        result[scene] = {}
        for size in r0['template_sizes']:
            a = [r for r in pairs0 if r['recording_id'] == scene and r['template_size'] == size]
            b = [r for r in pairs1 if r['recording_id'] == scene and r['template_size'] == size]
            inside, outside = [], []
            for row in a:
                c, j = lookup[(scene, row['agent_id'])], row['pair']
                shift = np.rint(c['image_xy'][j]) - np.rint(c['image_xy'][j-1])
                (inside if np.max(np.abs(shift)) <= r0['search_radius'] else outside).append(row)
            result[scene][str(size)] = {'paired': paired_errors(a,b), 'initial_search_support': {
                name: {'count': len(rows), 'matched': sum(r['status'] == 'matched' for r in rows),
                       'mean_error_px': float(np.mean([r['annotation_error_px'] for r in rows if r['status']=='matched']))
                       if any(r['status']=='matched' for r in rows) else None}
                for name, rows in (('inside', inside), ('outside', outside))}}
    output = args.report_dir.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('New workspace-local report directory required')
    output.mkdir(parents=True)
    report = {'result_source': 'fresh_analysis_of_cached_verified_correspondence_runs',
        'selected_agents': len(controls), 'sites': len(r0['recordings']),
        'input_run_hashes': {str(p.relative_to(ROOT)): file_digest(p) for p in (
            public/'past_motion_correspondence/audit.json', public/'past_motion_search_support/audit.json')},
        'comparisons': result, 'same_past_controls': True, 'future_targets_opened': False,
        'new_training': False, 'adaptive_followup_independent_confirmation': False}
    atomic_json(output/'comparison.json', report)
    lines = ['# Search Support and Paired Image Correspondence', '',
        'Same 24 moving fit controls; fresh analysis of hash-verified runs. No future forecasting.', '',
        '| Source | Template | Outside original search / 84 | Joint support | Radius24 error px | Radius64 error px | Radius64 missing / 84 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for scene, values in result.items():
        for size, v in values.items():
            p = v['paired']
            lines.append(f'| {scene} | {size} | {v["initial_search_support"]["outside"]["count"]} | '
                f'{p["jointly_supported"]} | {p["initial_mean_error_px"]:.3f} | '
                f'{p["wider_mean_error_px"]:.3f} | {p["wider_missing"]} |')
    lines += ['', 'Wider search repairs capacity for Hotel but also permits distractors and loses border support.',
              'All rows remain counted; missing matches are not zero error. Repeated pairs/templates are not independent samples.',
              'No radius selected for future forecasting; no temporal offset fit or physical clock certification.', '']
    (output/'results.md').write_text('\n'.join(lines))
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
