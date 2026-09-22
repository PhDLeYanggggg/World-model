"""Separate CSV/hash recount of the acquired DroneCrowd metadata snapshot.

Does not import the primary audit/parser, acquire archives or evaluate a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(source, output):
    manifest = json.loads((output / 'source_manifest.json').read_text())
    analysis = json.loads((output / 'analysis.json').read_text())
    if analysis['source_manifest_sha256'] != sha(output / 'source_manifest.json'):
        raise ValueError('Analysis source identity changed')
    files = manifest['files']
    names = {'README.md', 'trainlist.txt', 'testlist.txt', 'xml2mat.m', 'saveGT.m'}
    if len(files) != len(names) or {f['name'] for f in files} != names:
        raise ValueError('Unexpected fixed source file set')
    for entry in files:
        path = source / entry['name']
        if path.is_symlink() or sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError('Frozen metadata bytes mismatch')
        if analysis['source_sha256'][entry['name']] != entry['sha256']:
            raise ValueError('Analysis and manifest disagree')
    counts, splits = {}, {}
    for split in ('train', 'test'):
        rows = list(csv.reader(io.StringIO((source / f'{split}list.txt').read_text())))
        if any(len(row) != 1 or len(row[0]) != 5 or not row[0].isascii()
               or not row[0].isdigit() or int(row[0]) <= 0 for row in rows):
            raise ValueError('Invalid sequence row in separate recount')
        values = [int(row[0]) for row in rows]
        if len(set(values)) != len(values):
            raise ValueError('Duplicate sequence')
        counts[split + '_sequences'] = len(values)
        splits[split] = set(values)
    counts['union_sequences'] = len(splits['train'] | splits['test'])
    overlap = sorted(f'{i:05d}' for i in splits['train'] & splits['test'])
    partition = counts == {'train_sequences': 82, 'test_sequences': 30, 'union_sequences': 112}
    partition = partition and not overlap and splits['train'] | splits['test'] == set(range(1, 113))
    if counts != analysis['counts'] or overlap != analysis['sequence_list_overlap'] or partition != analysis['official_id_partition_pass']:
        raise ValueError('Separate split recount disagrees')
    declarations = {}
    for name, token in (
        ('README.md', 'val set is sampled from the test set'),
        ('README.md', 'only for academic and non-commercial uses'),
        ('saveGT.m', 'idx = anno(:,1)==i-1;'),
        ('saveGT.m', 'anno(idx,2)+1'),
        ('xml2mat.m', 'if(outside~=1 && occluded~=1)'),
        ('xml2mat.m', "if(exist(matfile, 'file'))"),
        ('xml2mat.m', "save(matfile, 'anno', 'countNum', 'label')"),
    ):
        found = [i for i, line in enumerate((source / name).read_text().splitlines(), 1) if token in line]
        if len(found) != 1:
            raise ValueError('Reviewed source locator no longer unique')
        declarations[f'{name}:{found[0]}'] = token
    if analysis['admitted_recordings'] or analysis['scientific_roles_assigned']:
        raise ValueError('Metadata cannot admit data or assign roles')
    return {
        'result_source': 'fresh_run', 'scope': 'separate_metadata_hash_csv_recount',
        'not_independent_research_replication': True,
        'analysis_sha256': sha(output / 'analysis.json'),
        'source_manifest_sha256': sha(output / 'source_manifest.json'),
        'verified_metadata_files': len(files), 'counts': counts,
        'sequence_list_overlap': overlap, 'official_id_partition_pass': partition,
        'reviewed_source_line_locators': declarations,
        'actual_xml_screening': 'not_run', 'model_training': 'not_run',
        'admitted_recordings': 0,
        'verifier_sha256': sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/DroneCrowd_release_metadata')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/dronecrowd_metadata_v1')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.source_root.resolve().is_relative_to(ROOT / 'external_data') or not args.output_dir.resolve().is_relative_to(ROOT / 'outputs'):
        raise SystemExit('Use workspace source and report trees')
    result = check(args.source_root, args.output_dir)
    path = args.output_dir / 'separate_verification.json'
    if args.verify:
        if result != json.loads(path.read_text()):
            raise SystemExit('Separate verification changed')
        print('cached_verified: separate verification exactly reproduced')
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
