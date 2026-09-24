"""Acquire pinned public European Squares sources without running publisher code."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'external_data/EuropeanSquares_source_audit'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/european_squares_intake_v1'
RECORD = 'https://zenodo.org/api/records/18267205'
COMMIT = 'd7225dd37d6b2bf962ee8533f9ca603593f5c1fc'
REPOSITORY = 'kaktusracing/pedestrian_trajectories'
FILES = {
    '02.Resources.zip': (254168, '40df088c361a9d61912f15e9a3342898'),
    'Place_Overview.xlsb': (24080, '86be3ba9bd17fc4cfcc286517f79cd38'),
    'README.pdf': (1136091, 'd52b8f86d3c1c86ec08c48eaf7a9cca2'),
    'stats_comparative.csv': (31786, '2445829c098e0b771a43ca297b843050'),
    'stats_season.csv': (44182, 'fd619dc9d2591185136ffb6a26f194f3'),
    '01.Trajectories.zip': (9451499010, '1e650c56c070e676253464b37468af97'),
}


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def save(path, obj):
    text = json.dumps(obj, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Immutable receipt differs: ' + str(path))
        return
    with path.open('x') as out:
        out.write(text)


def beat(state, **fields):
    event = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), state=state, **fields)
    tmp = RAW / 'heartbeat.tmp'
    tmp.write_text(json.dumps(event) + '\n')
    tmp.replace(RAW / 'heartbeat.json')
    with (RAW / 'events.jsonl').open('a') as out:
        out.write(json.dumps(event) + '\n')
    print(json.dumps(event), flush=True)


def fetch(url, path, size_limit, *, expected_size=None, md5=None, blob=None):
    if path.is_symlink():
        raise ValueError('Symlink refused')
    if not path.exists():
        partial = path.with_name(path.name + '.part')
        remaining = size_limit - (partial.stat().st_size if partial.exists() else 0)
        if remaining < 0 or shutil.disk_usage(RAW).free < max(remaining, 0) + 15_000_000_000:
            raise ValueError('Insufficient disk for acquisition plus 15 GB reserve')
        command = ['curl', '--location', '--fail', '--silent', '--show-error',
                   '--proto', '=https', '--proto-redir', '=https', '--connect-timeout', '20',
                   '--max-time', '43200', '--max-filesize', str(size_limit),
                   '--continue-at', '-', '--output', str(partial), url]
        start = time.monotonic()
        with (RAW / 'download.stderr.log').open('ab') as log:
            p = subprocess.Popen(command, stderr=log)
            try:
                while True:
                    beat('downloading', name=path.name, child_pid=p.pid,
                         bytes=partial.stat().st_size if partial.exists() else 0,
                         expected_bytes=expected_size, elapsed_seconds=time.monotonic()-start)
                    try:
                        p.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if p.returncode:
                    raise RuntimeError(f'curl exit {p.returncode}; partial retained, use same command to resume')
            finally:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=10)
        check(partial, expected_size, md5, blob)
        partial.replace(path)
    check(path, expected_size, md5, blob)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path))


def check(path, expected_size, md5, blob):
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError('Source length mismatch')
    if md5 and digest(path, 'md5') != md5:
        raise ValueError('Publisher checksum mismatch')
    if blob:
        payload = path.read_bytes()
        actual = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        if actual != blob:
            raise ValueError('Pinned Git blob mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories', action='store_true')
    args = parser.parse_args()
    for directory in (RAW, PUBLIC):
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise ValueError('Symlinked source directory refused')
        directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'check-ignore', '--quiet', str(RAW / '01.Trajectories.zip')], cwd=ROOT, check=True)
    with (RAW / 'acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = [fetch(RECORD, RAW / 'zenodo_record.json', 2_000_000)]
        meta = json.loads((RAW / 'zenodo_record.json').read_text())
        if meta['id'] != 18267205 or meta['metadata']['license']['id'] != 'cc-by-4.0':
            raise ValueError('Different dataset version or license')
        entries = {f['key']: f for f in meta['files']}
        for name, (size, md5) in FILES.items():
            item = entries[name]
            url = f'{RECORD}/files/{name}/content'
            if item['size'] != size or item['checksum'] != 'md5:' + md5 or item['links']['self'] != url:
                raise ValueError('Dataset file identity changed')
            if name == '01.Trajectories.zip':
                continue
            paths.append(fetch(url, RAW / name, size, expected_size=size, md5=md5))
        tree_url = f'https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1'
        paths.append(fetch(tree_url, RAW / 'source_tree.json', 2_000_000))
        tree = json.loads((RAW / 'source_tree.json').read_text())
        if tree['sha'] != COMMIT or tree['truncated']:
            raise ValueError('Incomplete publisher code identity')
        for item in tree['tree']:
            name = item['path']
            if name not in ('README.md', 'Object_detection.py', 'Video_preprocessing.py', 'modify_csv_detections.py') and not (
                    name.startswith('csv_functions_v2/') and name.endswith('.py')):
                continue
            target = RAW / ('source_' + name.replace('/', '_'))
            paths.append(fetch(f'https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{name}', target,
                               200000, expected_size=item['size'], blob=item['sha']))
        source = dict(result_source='fresh_run', dataset_record=RECORD, dataset_license='CC-BY-4.0',
                      publisher_code_commit=COMMIT, role='quarantined_unassigned_source_audit',
                      purpose='metadata_raw_support_provenance_no_forecast_errors',
                      published_site_count=39, site_count_locally_verified=False,
                      predictive_readout=False, fitting=False, third_party_code_executed=False,
                      stage5c_executed=False, smc_enabled=False, private_files=paths)
        save(PUBLIC / 'metadata_manifest.json', source)
        if args.trajectories:
            name = '01.Trajectories.zip'
            size, md5 = FILES[name]
            beat('trajectory_acquisition_started', expected_bytes=size)
            item = fetch(f'{RECORD}/files/{name}/content', RAW / name, size, expected_size=size, md5=md5)
            save(PUBLIC / 'trajectory_manifest.json', dict(result_source='fresh_run',
                 metadata_manifest_sha256=digest(PUBLIC / 'metadata_manifest.json'),
                 publisher_md5=md5, private_file=item, forecast_errors_opened=False))
        beat('complete', trajectories=args.trajectories)


if __name__ == '__main__':
    main()
