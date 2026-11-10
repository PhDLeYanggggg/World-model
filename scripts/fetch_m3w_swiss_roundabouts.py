"""Cache the pinned public Swiss roundabout trajectory release."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = '15077435'
SOURCE = f'https://zenodo.org/records/{RECORD}'
ROSTER = '3ee2653c7d9f64dfb5de2cef0a05bf6083d66aba2aaf9e307f468038a31962ed'
LOCAL = ROOT/'external_data/SwissRoundabouts_source_audit'
OUT = ROOT/'outputs/publication_readiness_2026_09/swiss_roundabouts_source_intake_v1'


def write(path, data):
    raw = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Existing receipt changed: {path.name}')
    else:
        with path.open('xb') as f:
            f.write(raw)


def digest(path, algorithm='sha256'):
    with path.open('rb') as f:
        return file_digest(f, algorithm).hexdigest()


def verify_roster(files):
    raw = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if sha256(raw).hexdigest() != ROSTER:
        raise ValueError('Publisher file roster changed')
    if len([f for f in files if re.fullmatch(r'D[12]_[AP]M[1-7]_(F[12]|L1)\.csv', f['key'])]) != 18:
        raise ValueError('Unexpected trajectory roster')


def member(entry):
    name, size, checksum = (entry[k] for k in ('key', 'size', 'checksum'))
    if Path(name).name != name or not checksum.startswith('md5:'):
        raise ValueError('Invalid public member')
    p = LOCAL/name
    q = p.with_suffix(p.suffix+'.part')
    check = lambda x: x.stat().st_size == size and digest(x, 'md5') == checksum[4:]
    if not p.exists():
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2',
                        '--max-time', '120', '--output', str(q), f'{SOURCE}/files/{name}'], check=True)
        if not check(q):
            raise ValueError(f'Publisher checksum failed for {name}')
        q.replace(p)
    if not check(p):
        raise ValueError(f'Cached copy differs from publisher: {name}')
    print(name, 'verified', flush=True)
    return dict(entry, sha256=digest(p))


def run():
    metadata = json.loads(subprocess.check_output(
        ['curl', '-fsSL', '--max-time', '20', f'https://zenodo.org/api/records/{RECORD}']))
    if metadata['doi'] != f'10.5281/zenodo.{RECORD}' or metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Unexpected publisher record or license')
    files = sorted(({k: e[k] for k in ('key', 'size', 'checksum')} for e in metadata['files']), key=lambda e: e['key'])
    verify_roster(files)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as workers:
        checked = list(workers.map(member, files))
    write(OUT/'source.json', dict(doi=metadata['doi'], license='CC BY 4.0',
        source=SOURCE, roster_sha256=ROSTER, members=checked, predictive_roles_assigned=False))


if __name__ == '__main__':
    run()
