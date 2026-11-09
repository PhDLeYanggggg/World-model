"""Fetch and verify Songdo's first-day files without parsing trajectory contents."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import json
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_traffic as audit
from scripts import build_m3w_songdo_prefix_pack as pack

PUBLIC = audit.PUBLIC.parent / 'songdo_first_day_v1'
PRIVATE = pack.CACHE.parent / 'songdo_first_day_v1'
PUBLISHER = 'https://zenodo.org/api/records/17924857'


def inventory(files):
    records = {}
    for f in files:
        if re.fullmatch(r'2022-10-04_[A-Z]\.zip', f['key']):
            filename = f['key']
            url = f'{PUBLISHER}/files/{filename}/content'
            if (f['links']['self'] != url or not re.fullmatch(r'md5:[0-9a-f]{32}', f['checksum'])
                    or not 0 < f['size'] <= 400_000_000 or filename in records):
                raise ValueError(f'Unexpected publisher file metadata: {filename}')
            records[filename] = dict(bytes=f['size'], md5=f['checksum'][4:], url=url)
    if len(records) != 20 or {f'{audit.DATE}_{s}.zip' for s in audit.PUBLISHER} - set(records):
        raise ValueError('Unexpected first-day publisher roster')
    return records


def fetch(entry):
    name, m = entry
    path = audit.LOCAL / name
    p = path if path.exists() else path.with_suffix('.part')
    if p != path and (not p.exists() or p.stat().st_size < m['bytes']):
        subprocess.run(['curl', '-fsS', '--retry', '2', '--connect-timeout', '15',
                        '--max-time', '240', '--max-filesize', str(m['bytes']),
                        '--continue-at', '-', '--output', str(p), m['url']], check=True)
    md5, sha = audit.digest(p)
    if md5 != m['md5'] or p.stat().st_size != m['bytes']:
        raise ValueError(f'Archive failed publisher verification: {p}')
    if p != path:
        p.replace(path)
    print(name, m['bytes'], flush=True)
    return name, dict(**m, sha256=sha)


def main():
    for p in (PUBLIC, PRIVATE, audit.LOCAL):
        p.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'fetch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(subprocess.check_output(['curl', '-fsS', '--max-time', '20', PUBLISHER]))
        files = inventory(record['files'])
        pack.write_json(PUBLIC / 'publisher.json', dict(url=PUBLISHER, date=audit.DATE, files=files))
        with ThreadPoolExecutor(max_workers=2) as executor:
            verified = dict(executor.map(fetch, sorted(files.items())))
        pack.write_json(PUBLIC / 'archives.json', dict(
            publisher_sha256=pack.sha(PUBLIC / 'publisher.json'), files=verified,
            trajectory_csv_parsed=False, split_assigned=False))


if __name__ == '__main__':
    main()
