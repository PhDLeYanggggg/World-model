"""Pin and verify the author's DUT raw annotations for diagnostic intake only.

No images, video, filtered trajectories, credentials or executable third-party
entry points are used. Fetching is not approval of research-use terms or roles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REPO = 'dongfang-steven-yang/vci-dataset-dut'
COMMIT = '80b8c746833664cd1e5244fccad79c7f2a7cbe31'
TREE = 'b4bbe87e9b714cf9d2c571ab45ea4da6862049d1'
CLIPS = tuple([f'intersection_{i:02d}' for i in range(1, 18)] + [f'roundabout_{i:02d}' for i in range(1, 12)])
RAW_PATHS = {f'data/trajectories/{c}_traj_{t}.csv' for c in CLIPS for t in ('ped', 'veh')}
RATIO_PATHS = {f'data/ratios/{c}_ratio_pixel2meter.txt' for c in CLIPS}
AUDIT_PATHS = {'README.md', 'scripts/filter_trajectories.py'}
ALLOWLIST = RAW_PATHS | RATIO_PATHS | AUDIT_PATHS
MAX_BYTES = 30_000_000


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def fetch(url, max_bytes):
    reply = subprocess.run(['/usr/bin/curl', '--fail', '--silent', '--show-error',
        '--proto', '=https', '--max-time', '60', '--max-filesize', str(max_bytes),
        '-H', 'User-Agent: M3W-DUT-diagnostic-source-audit', url],
        check=True, capture_output=True, timeout=65)
    if len(reply.stdout) > max_bytes:
        raise ValueError('Source response exceeds bounded annotation budget')
    return reply.stdout


def select_tree(tree):
    if tree.get('truncated') or tree.get('sha') != TREE:
        raise ValueError('Pinned complete author tree required')
    entries = [e for e in tree['tree'] if e['path'] in ALLOWLIST]
    if len(entries) != len(ALLOWLIST) or {e['path'] for e in entries} != ALLOWLIST:
        raise ValueError('Author raw annotation file set changed')
    for e in entries:
        if (e.get('type') != 'blob' or e.get('mode') != '100644'
                or type(e.get('size')) is not int or not 0 < e['size'] <= MAX_BYTES
                or not re.fullmatch('[0-9a-f]{40}', e.get('sha', ''))):
            raise ValueError('Only bounded regular author blobs are accepted')
    if sum(e['size'] for e in entries) > MAX_BYTES:
        raise ValueError('Total annotation budget exceeded')
    return [{k: e[k] for k in ('path', 'size', 'sha')} for e in sorted(entries, key=lambda x: x['path'])]


def verify_bytes(data, entry):
    if len(data) != entry['size'] or blob_hash(data) != entry['sha']:
        raise ValueError('Downloaded or existing bytes differ from pinned author blob')


def safe_path(root, relative):
    root = Path(root).absolute()
    if relative not in ALLOWLIST:
        raise ValueError('Path not in the raw annotation allowlist')
    path = root / relative
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlinked source destination refused')
    return path


def acquire(root, entries, fetcher=fetch):
    written, reused, records = 0, 0, []
    for e in entries:
        path = safe_path(root, e['path'])
        if path.exists():
            data = path.read_bytes()
            verify_bytes(data, e)
            reused += 1
        else:
            data = fetcher(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{e["path"]}', e['size'])
            verify_bytes(data, e)
            path.parent.mkdir(parents=True, exist_ok=True)
            # A unique temporary file avoids replacing a prior interrupted download.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temp = Path(stream.name)
                stream.write(data)
            os.replace(temp, path)
            written += 1
        records.append({**e, 'sha256': hashlib.sha256(data).hexdigest()})
    actual = {str(p.relative_to(root)) for p in (Path(root) / 'data/trajectories').glob('*.csv')}
    if actual != RAW_PATHS:
        raise ValueError('Unexpected raw files mixed into the pinned source')
    return records, written, reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Explicitly acquire the raw annotation allowlist')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/DUT_author_raw')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/dut_causal_intake/source_manifest.json')
    args = parser.parse_args()
    if not args.source_root.resolve().is_relative_to(ROOT / 'external_data') or not args.report.resolve().is_relative_to(ROOT / 'outputs'):
        raise SystemExit('Keep raw data under ignored external_data and reports under outputs')
    for p in (args.source_root, *args.source_root.parents, args.report, *args.report.parents):
        if p.is_symlink():
            raise SystemExit('Symlinked output path refused')
    ignored = subprocess.run(['git', 'check-ignore', '--quiet', str(args.source_root / 'data/trajectories/intersection_01_traj_ped.csv')], cwd=ROOT)
    if ignored.returncode != 0:
        raise SystemExit('Raw data destination must be ignored by Git')
    commit = json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/{COMMIT}', 1_000_000))
    if commit['sha'] != COMMIT or commit['commit']['tree']['sha'] != TREE:
        raise ValueError('Commit/tree identity mismatch')
    tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{TREE}?recursive=1', 2_000_000))
    entries = select_tree(tree)
    result = {'result_source': 'fresh_run', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'repository': 'https://github.com/' + REPO, 'upstream_commit': COMMIT, 'upstream_tree': TREE,
        'selected_entries': entries, 'allowlisted_bytes': sum(e['size'] for e in entries),
        'license_or_terms_paths': [e['path'] for e in tree['tree'] if any(t in e['path'].lower() for t in ('license', 'terms', 'copying'))],
        'source_use_approval': False, 'official_split_assigned': False, 'training_run': False,
        'videos_images_or_filtered_data_downloaded': False, 'third_party_code_executed': False,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.download:
        files, written, reused = acquire(args.source_root, entries)
        raw = [(e['path'], e['sha']) for e in files if e['path'] in RAW_PATHS]
        result.update(status='verified', files=files, files_downloaded=written, files_reused=reused,
            raw_local_csv_files=len(raw), raw_path_and_git_blob_manifest_sha256=hashlib.sha256(json.dumps(raw).encode()).hexdigest())
        if not written:
            result['result_source'] = 'cached_verified'
    else:
        result['status'] = 'metadata_only_no_download'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, args.report)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files', 'selected_entries')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
