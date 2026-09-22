"""Acquire/verify five small official metadata files. Never fetch annotations.zip."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_dronecrowd_intake import (
    FOLDER_URL, METADATA_FILES, MAX_METADATA_BYTES, audit_release, inspect_xml,
    render_release_report, sha, validate_metadata,
)


def fetch_metadata(name):
    if name not in METADATA_FILES:
        raise ValueError('Archives and arbitrary URLs are not permitted')
    url = f'https://drive.google.com/uc?export=download&id={METADATA_FILES[name]}'
    reply = subprocess.run(['/usr/bin/curl', '--fail', '--silent', '--show-error',
        '--location', '--proto', '=https', '--proto-redir', '=https', '--max-redirs', '3',
        '--max-time', '45', '--max-filesize', str(MAX_METADATA_BYTES), url],
        check=True, capture_output=True, timeout=50)
    validate_metadata(name, reply.stdout)
    return reply.stdout


def plain_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlinked source/output paths refused')
    return path.resolve()


def acquire(source, output, fetcher=fetch_metadata):
    """Save source bytes once; a manifest is the local snapshot, not upstream signing."""
    source, output = plain_path(source), plain_path(output)
    manifest_path = output / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    expected = {} if manifest is None else {f['name']: f for f in manifest['files']}
    if manifest is not None and set(expected) != set(METADATA_FILES):
        raise ValueError('Source manifest file set mismatch')
    files = {}
    downloaded = 0
    for name, source_id in METADATA_FILES.items():
        path = plain_path(source / name)
        if path.exists():
            if not 0 < path.stat().st_size <= MAX_METADATA_BYTES:
                raise ValueError('Local metadata size exceeds bound')
            if name not in expected:
                # An interrupted first acquisition must be compared with the source.
                local, data = path.read_bytes(), fetcher(name)
                if local != data:
                    raise ValueError('Unbound local file differs from official response')
            else:
                data = path.read_bytes()
        else:
            data = fetcher(name)
            validate_metadata(name, data)
            if manifest is not None and (expected[name]['sha256'] != sha(data)
                                        or expected[name]['bytes'] != len(data)):
                raise ValueError('Upstream bytes changed from frozen metadata snapshot')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
            downloaded += 1
        validate_metadata(name, data)
        if name in expected and (expected[name]['sha256'] != sha(data)
                or expected[name]['bytes'] != len(data) or expected[name]['file_id'] != source_id):
            raise ValueError('Changed metadata bytes or identity')
        files[name] = data
    if manifest is None:
        manifest = {
            'schema_version': 1, 'status': 'local_snapshot_verified_not_upstream_immutable',
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_folder': FOLDER_URL, 'upstream_immutable_version': None,
            'files': [{'name': n, 'file_id': METADATA_FILES[n], 'bytes': len(b),
                       'sha256': sha(b),
                       'source_url': f'https://drive.google.com/file/d/{METADATA_FILES[n]}/view'}
                      for n, b in sorted(files.items())],
            'annotations_images_videos_downloaded': False, 'third_party_code_executed': False,
        }
        output.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('x') as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return files, downloaded


def verify_local(source, manifest):
    if {f['name'] for f in manifest['files']} != set(METADATA_FILES) or len(manifest['files']) != 5:
        raise ValueError('Fixed five-file manifest required')
    files = {}
    for entry in manifest['files']:
        name = entry['name']
        path = plain_path(source / name)
        if not 0 < path.stat().st_size <= MAX_METADATA_BYTES:
            raise ValueError('Local metadata file size exceeds bound')
        data = path.read_bytes()
        validate_metadata(name, data)
        if sha(data) != entry['sha256'] or len(data) != entry['bytes'] or entry['file_id'] != METADATA_FILES[name]:
            raise ValueError('Frozen source metadata changed')
        files[name] = data
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--download-metadata', action='store_true')
    action.add_argument('--verify', action='store_true', help='Offline exact source+analysis verification')
    action.add_argument('--inspect-xml', type=Path, help='Explicit local diagnostic XML screen, no acquisition')
    parser.add_argument('--sequence-id')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/DroneCrowd_release_metadata')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/dronecrowd_metadata_v1')
    args = parser.parse_args()
    source, output = plain_path(args.source_root), plain_path(args.output_dir)
    if not source.is_relative_to(ROOT / 'external_data') or not output.is_relative_to(ROOT / 'outputs'):
        raise SystemExit('Source stays in ignored external_data; reports stay under outputs')
    if args.inspect_xml:
        path = plain_path(args.inspect_xml)
        if not path.is_relative_to(ROOT / 'external_data') or path.suffix != '.xml':
            raise SystemExit('Only explicit local external_data XML files may be screened')
        if path.stat().st_size > 64 * 1024 * 1024:
            raise SystemExit('XML exceeds single-recording bound')
        print(json.dumps(inspect_xml(path.read_bytes(), args.sequence_id or ''), indent=2))
        return
    if subprocess.run(['git', 'check-ignore', '--quiet', str(source / 'README.md')], cwd=ROOT).returncode:
        raise SystemExit('Raw metadata must be ignored by Git')
    manifest_path, analysis_path = output / 'source_manifest.json', output / 'analysis.json'
    if args.verify:
        files = verify_local(source, json.loads(manifest_path.read_text()))
        downloaded = 0
    else:
        files, downloaded = acquire(source, output)
    result = audit_release(files)
    result['source_manifest_sha256'] = sha(manifest_path.read_bytes())
    result['implementation_sha256'] = {n: sha((ROOT / n).read_bytes()) for n in (
        'src/evaluation/m3w_dronecrowd_intake.py', 'scripts/audit_m3w_dronecrowd_metadata.py',
        'tests/test_m3w_dronecrowd_intake.py')}
    if args.verify or analysis_path.exists():
        if json.loads(analysis_path.read_text()) != result:
            raise SystemExit('Frozen audit differs; use a new versioned output directory')
        if (output / 'release_audit.md').read_text() != render_release_report(result):
            raise SystemExit('Report differs from frozen analysis')
        status = 'cached_verified'
    else:
        with analysis_path.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
        with (output / 'release_audit.md').open('x') as stream:
            stream.write(render_release_report(result))
        status = 'fresh_run'
    print(json.dumps({'result_source': status, 'metadata_files_downloaded': downloaded,
                      'analysis_sha256': sha(analysis_path.read_bytes()), 'counts': result['counts'],
                      'official_id_partition_pass': result['official_id_partition_pass'],
                      'admitted_recordings': 0, 'training_run': False}, indent=2))


if __name__ == '__main__':
    main()
