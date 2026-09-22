"""Separate raw CSV recount; never imports the TRAF audit parser/window counter."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def verify_recording(path, reported):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != reported['sha256']:
        raise ValueError('Source hash changed')
    frames_by_agent = defaultdict(list)
    classes, counts, boxes, errors = {}, Counter(), Counter(), 0
    previous = -1
    lines = list(csv.reader(io.StringIO(content.decode('utf-8-sig'))))
    for row in lines:
        row = [s.strip() for s in row]
        # Unexpected format changes are refused, not reproduced by copying
        # the primary parser's error recovery.
        if len(row) < 2 or not all(re.fullmatch('[0-9]+', s) for s in row[:2]):
            raise ValueError('Verifier does not admit malformed frame/count')
        frame, n = int(row[0]), int(row[1])
        if len(row) != 2 + n * 5:
            raise ValueError('Verifier field count differs')
        ids = row[6::5]
        if len(set(ids)) != n or frame <= previous:
            errors += 1
            continue
        previous = frame
        counts['frames'] += 1
        for j, agent in enumerate(ids):
            values = list(map(float, row[2 + 5*j:6 + 5*j]))
            a, b, c, d = values
            boxes['boxes'] += 1
            boxes['incompatible_with_xyxy'] += int(c <= a or d <= b)
            boxes['positive_third_fourth_values'] += int(c > 0 and d > 0)
            boxes['negative_fields'] += int(min(values) < 0)
            frames_by_agent[agent].append(frame)
            match = re.fullmatch(r'([A-Za-z]+)([0-9]+)', agent)
            classes[agent] = match.group(1).lower() if match else 'untyped_id'
    assert len(lines) == reported['input_lines']
    assert counts['frames'] == reported['parsed_frames']
    assert errors == reported['frame_error_count']
    assert dict(boxes) == reported['bbox_checks']
    actual = defaultdict(Counter)
    for agent, frames in frames_by_agent.items():
        c = actual[classes[agent]]
        c['tracks'] += 1
        c['points'] += len(frames)
        # Ending-at-frame recurrence is independent of the run-cut audit.
        for step in (1, 12):
            lengths = {}
            for f in frames:
                lengths[f] = lengths.get(f-step, 0) + 1
                length = lengths[f]
                for k in (8, 16, 32, 64):
                    c[f'{step}_history_{k}'] += int(length >= k)
                    c[f'{step}_future12_{k}'] += int(length >= k+12)
                if step == 1:
                    for h in (10, 25, 50, 100):
                        c[f'raw{h}'] += int(length >= 8+h)
    assert set(actual) == set(reported['by_class_before_recording_quarantine'])
    for kind, c in actual.items():
        ref = reported['by_class_before_recording_quarantine'][kind]
        assert c['tracks'] == ref['track_count'] and c['points'] == ref['point_count']
        for step in (1, 12):
            for k in (8, 16, 32, 64):
                assert c[f'{step}_history_{k}'] == ref[f'stride{step}']['history_only'][str(k)]
                assert c[f'{step}_future12_{k}'] == ref[f'stride{step}']['history_and_12step_future'][str(k)]
        for h in (10, 25, 50, 100):
            assert c[f'raw{h}'] == ref['history8_full_raw_horizon'][str(h)]
    return {'recording': reported['recording'], 'parsed_frames': counts['frames'],
            'rejected_frames': errors, 'points': boxes['boxes'], 'tracks': len(frames_by_agent),
            'class_window_tables_verified': len(actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT /
                        'outputs/publication_readiness_2026_09/traf_intake_v1/analysis.json')
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text())
    source = ROOT / 'external_data/OpenTraj/datasets/TRAF/Annotated Ground Truth Files'
    reports = [verify_recording(source / r['file'], r) for r in result['recordings']]
    receipt = {'result_source': 'fresh_run_separate_csv_and_recurrence_verification',
               'analysis_sha256': hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
               'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'recordings': reports, 'independent_research_replication': False,
               'new_training': False, 'role_assignment': False}
    out = args.analysis.with_name('separate_verification.json')
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Verified {len(reports)} raw recordings; no forecasting performed', flush=True)


if __name__ == '__main__':
    main()
