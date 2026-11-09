"""Separate label-class coverage from tracking losses on frozen VisDrone prefixes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_m3w_visdrone_camera_pilot import FRAMES, PRIVATE, groups, measure, source, write_json


def scoped_labels(rows, categories):
    a = rows[(rows[:, 6] == 1) & np.isin(rows[:, 7], categories)]
    # Keep label identities separate if an integer ID is reused across categories.
    xyxy = np.column_stack((a[:, 0], 2*a[:, 1]+(a[:, 7] == 2), a[:, 2:4], a[:, 2:4]+a[:, 4:6]))
    by_frame = groups(xyxy)
    return [by_frame.get(t, np.empty((0, 6)))[:, 1:] for t in FRAMES]


def run(verify):
    freeze_bytes = (source.OUT/'camera_freeze.json').read_bytes()
    freeze = json.loads(freeze_bytes)
    registration = (source.OUT/'camera_registration.json').read_bytes()
    manifest = (source.DATA/'source_manifest.json').read_bytes()
    reg = json.loads(registration)
    if (source.digest(registration) != freeze['registration_sha256'] or
            source.digest(manifest) != reg['source_manifest_sha256'] or sorted(freeze['prefixes']) != reg['clips']):
        raise ValueError('Source or camera freeze changed')
    metadata = json.loads(manifest)
    strict = json.loads((source.OUT/'camera_support.json').read_bytes())
    if strict['freeze_sha256'] != source.digest(freeze_bytes):
        raise ValueError('Strict readout does not reference the frozen prefixes')
    expected = {row['clip']: row for row in strict['clips']}
    records = []
    for clip, binding in freeze['prefixes'].items():
        prefix = (PRIVATE/f'{clip}_prefix.json').read_bytes()
        payload = (source.DATA/f'{clip}.txt').read_bytes()
        if source.digest(prefix) != binding['prefix_sha256'] or source.digest(payload) != metadata['labels'][clip]:
            raise ValueError('Frozen prefix or reference labels changed')
        bundle = json.loads(prefix)
        a = source.parse(payload, metadata['frames'][clip])
        report = dict(pedestrian=measure(bundle, scoped_labels(a, (1,))),
                      people_including_other=measure(bundle, scoped_labels(a, (1, 2))))
        if report['pedestrian'] != expected[clip]:
            raise ValueError('Scoped helper changed strict counts')
        records.append(report)
    totals = {}
    for name in records[0]:
        count = Counter()
        for row in records:
            count.update({k: v for k, v in row[name].items() if k != 'clip'})
        totals[name] = dict(count)
    result = dict(result_source='fresh_run', prefixes='cached_verified', freeze_sha256=source.digest(freeze_bytes),
                  role='training_source_label_scope_sensitivity', records=records, totals=totals,
                  scientific_evaluation='not_run', predictions_changed=False)
    write_json(source.OUT/'camera_label_scope.json', result, verify)
    print(json.dumps(totals, sort_keys=True), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    run(p.parse_args().verify)
