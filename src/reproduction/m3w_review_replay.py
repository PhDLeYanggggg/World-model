"""Portable reviewer replay from de-identified aggregate sufficient statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.dont_write_bytecode = True

import numpy as np

from src.evaluation.m3w_grouped_recording_readout import grouped_readout
from src.evaluation.m3w_joint_path_audit import bootstrap


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_manifest(root):
    manifest = read(root/'manifest.json')
    expected = set(manifest)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected | {'manifest.json'}:
        raise ValueError('Bundle file roster differs from the manifest')
    for name, digest in manifest.items():
        p = root/name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Bundle checksum mismatch: {name}')


def recompute(packet):
    if packet['schema_version'] != 1:
        raise ValueError('Unsupported statistics schema')
    roster, seeds = packet['roster'], packet['seeds']
    if (len(roster), len(set(roster.values())), len(seeds)) != (34, 14, 3):
        raise ValueError('Recording, source or seed roster changed')
    result = {}
    for name, rows in packet['recording_studies'].items():
        for row in rows:
            for a in row['metrics'].values():
                risk = a['risk']
                values = [risk[k] for k in ('completion_all_risk_upper',
                    'completion_easy_risk_upper', 'known_easy_net_degradation')]
                risk['supported'] = all(v is not None and v <= .02 for v in values)
        result[name] = grouped_readout(rows, roster, seeds)
    p = packet['joint_cells']
    groups = sorted(set(roster.values()))
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    if sorted(p) != groups or any(sorted(p[g]) != sorted(map(str, seeds)) for g in groups):
        raise ValueError('Missing joint source/seed cells')
    example = p[groups[0]][str(seeds[0])]
    result['joint_pair_contrasts'] = {
        pair: {field: bootstrap([[p[g][str(s)][pair][field] for s in seeds] for g in groups], draw)
               for field in fields} for pair, fields in example.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true', help='Print the complete recomputed result')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify_manifest(root)
    result = recompute(read(root/'statistics.json'))
    if result != read(root/'expected.json'):
        raise ValueError('Recomputed values differ from the frozen results')
    if args.show:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        print('Exact replay: two recording-risk/utility tables and ten joint contrasts; '
              '34 recordings, 14 source groups, three seeds.')


if __name__ == '__main__':
    main()
