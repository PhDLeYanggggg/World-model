"""Acquire the official labels-only HT21 archive for quarantined source audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_dut_annotations import fetch

URL = 'https://motchallenge.net/data/HT21Labels.zip'
EXPECTED_SIZE = 53547421
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/ht21_annotations_v1'
ARCHIVE = ROOT / 'external_data/HT21_annotations/HT21Labels.zip'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inventory(path):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        if len(entries) > 1000 or len({e.filename for e in entries}) != len(entries):
            raise ValueError('Ambiguous or excessive archive member inventory')
        rows = []
        for e in entries:
            p = Path(e.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in e.filename or e.flag_bits & 1:
                raise ValueError('Unsafe or encrypted archive member')
            if (e.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlink archive member')
            if e.file_size > 300_000_000:
                raise ValueError('Oversized annotation member')
            if not e.is_dir() and p.suffix not in ('.txt', '.ini', '.md'):
                raise ValueError('Non-label content in labels-only source')
            rows.append(dict(path=e.filename, size=e.file_size, compressed_size=e.compress_size,
                             crc32=e.CRC, directory=e.is_dir()))
        if sum(e.file_size for e in entries) > 750_000_000:
            raise ValueError('Excessive uncompressed annotation budget')
        return sorted(rows,key=lambda r:r['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args = parser.parse_args()
    if any(p.is_symlink() for p in (ARCHIVE,*ARCHIVE.parents,PUBLIC,*PUBLIC.parents)):
        raise SystemExit('Symlinked destination refused')
    subprocess.run(['git','check-ignore','--quiet',str(ARCHIVE)],cwd=ROOT,check=True)
    manifest = PUBLIC/'source_manifest.json'
    if ARCHIVE.exists():
        data = ARCHIVE.read_bytes()
        if not manifest.exists():
            raise SystemExit('Existing archive without acquisition receipt; audit provenance before reuse')
        old = json.loads(manifest.read_text())
        if sha(data) != old['archive_sha256'] or len(data) != old['archive_bytes']:
            raise SystemExit('Existing archive differs from frozen receipt')
        if inventory(ARCHIVE) != old['members']:
            raise SystemExit('Archive inventory mismatch')
        print(json.dumps(dict(result_source='cached_verified',archive_sha256=sha(data),members=len(old['members']))))
        return
    if not args.download:
        raise SystemExit('Use --download for the explicitly scoped public labels-only acquisition')
    data = fetch(URL,60_000_000)
    if len(data) != EXPECTED_SIZE:
        raise SystemExit('Official archive changed from checked HTTP size; inspect before acquisition')
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARCHIVE.parent,delete=False) as f:
        f.write(data); temp = Path(f.name)
    members = inventory(temp)
    os.replace(temp,ARCHIVE)
    result = dict(result_source='fresh_run',acquired_at_utc=datetime.now(timezone.utc).isoformat(),
        source_url=URL,source_page='https://motchallenge.net/data/Head_Tracking_21/',
        archive_path=str(ARCHIVE.relative_to(ROOT)),archive_bytes=len(data),archive_sha256=sha(data),members=members,
        authorization='delegated_routine_research_source_audit_and_public_data_acquisition',
        scope='quarantined_annotation_audit_only',source_use_approval=False,
        scientific_roles_assigned=False,training=False,predictive_evaluation=False,
        videos_or_images_downloaded=False,third_party_code_executed=False,
        implementation_sha256=sha(Path(__file__).read_bytes()))
    PUBLIC.mkdir(parents=True,exist_ok=True)
    with manifest.open('x') as f: f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='members'},indent=2))
    print(json.dumps(members,indent=2))


if __name__ == '__main__': main()
