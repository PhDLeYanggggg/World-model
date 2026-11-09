"""Pair the score-only ablations at the same source resamples."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_reviewer_evidence as archive
from reproducibility.vehicle_recording_replay import verify

OUT = archive.PUBLIC/'opendd_vehicle_fixed_release_v1'
SCALARS = archive.LOCAL/'vehicle_recording_replay.zip'
PARENT = archive.PUBLIC/'vehicle_recording_replay_v1/receipt.json'


def main():
    p = OUT/'readout.json'
    r = json.loads(p.read_text())
    if (r['freeze'] != archive.main.digest((OUT/'freeze.json').read_bytes()) or not r['same_proposals']
            or r['held_tuning'] or not r['original_2pct_unchanged']):
        raise ValueError('Expected the frozen score-only comparison')
    scalar = json.loads(PARENT.read_text())
    if archive.main.digest(SCALARS.read_bytes()) != scalar['archive_sha256']:
        raise ValueError('Changed paired source resamples')
    with ZipFile(SCALARS) as z:
        draws = json.loads(z.read('evidence.json'))['source_resamples']
    result = {}
    for a, b in (('direct', 'archived'), ('nested', 'direct')):
        x, y = (r['intervals'][k] for k in (a, b))
        result[f'{a}_minus_{b}'] = {}
        for k in x:
            rows = {i: [u-v] for i, (u, v) in enumerate(zip(x[k]['site_means'], y[k]['site_means'], strict=True))}
            result[f'{a}_minus_{b}'][k] = verify.interval(rows, draws)
    data = archive.main.as_json(dict(contrasts=result, readout_sha256=archive.main.digest(p.read_bytes()),
                                    original_2pct_unchanged=True, same_proposals=True, held_tuning=False))
    out = OUT/'paired_contrasts.json'
    if out.exists() and out.read_bytes() != data:
        raise ValueError('Recorded contrasts differ')
    if not out.exists():
        with out.open('xb') as f:
            f.write(data)
    print(data.decode(), end='')


if __name__ == '__main__':
    main()
