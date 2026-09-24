"""Acquire the official IMPTC sample archive; source audit, not forecasting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'external_data/IMPTC_source_audit'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/imptc_intake_v1'
META_URL = 'https://zenodo.org/api/records/14811016'
FILE = 'imptc_samples.tar.gz'
SIZE = 347692691
MD5 = 'd2bf9579b127b74927150bd3666a034f'
DOC_COMMIT = '991f1afd57da10b5ecb865902d2113f0f1248bba'
DOCS = ('README.md', 'data_formats/vru_tracks.md', 'data_formats/vehicle_tracks.md',
        'data_formats/timestamp_master.md')


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    for path in (RAW, PUBLIC):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise SystemExit('Symlinked destination refused')
        path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'check-ignore', '--quiet', str(RAW / FILE)], cwd=ROOT, check=True)
    archive, receipt_path = RAW / FILE, PUBLIC / 'source_manifest.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        for entry in old['private_files']:
            path = ROOT / entry['path']
            if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise SystemExit('Previously acquired source differs from its receipt')
        print(json.dumps(dict(result_source='cached_verified', archive_sha256=digest(archive))))
        return
    if not args.download:
        raise SystemExit('Use --download for this public source-audit acquisition')
    if shutil.disk_usage(RAW).free < 2_000_000_000:
        raise SystemExit('Insufficient free disk for bounded intake')
    for path in RAW.rglob('*'):
        if path.is_symlink():
            raise SystemExit('Symlinked destination refused')
    meta_path = RAW / 'zenodo_record.json'
    if not meta_path.exists():
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https',
            '--max-time', '60', '--max-filesize', '2000000', '-o', str(meta_path), META_URL], check=True)
    metadata = json.loads(meta_path.read_text())
    if metadata['id'] != 14811016 or metadata['metadata']['license']['id'] != 'cc-by-nc-4.0':
        raise SystemExit('Changed source identity or research-use license')
    entries = [e for e in metadata['files'] if e['key'] == FILE]
    if len(entries) != 1 or entries[0]['size'] != SIZE or entries[0]['checksum'] != 'md5:' + MD5:
        raise SystemExit('Official version differs from checked source')
    url = entries[0]['links']['self']
    expected = f'https://zenodo.org/api/records/14811016/files/{FILE}/content'
    if url != expected:
        raise SystemExit('Unexpected archive endpoint')
    started = time.monotonic()
    partial = archive.with_suffix(archive.suffix + '.part')
    if not archive.exists():
        command = ['curl', '-L', '--fail', '--silent', '--show-error', '--proto', '=https',
            '--proto-redir', '=https', '--connect-timeout', '20', '--max-time', '7200',
            '--max-filesize', str(SIZE), '--continue-at', '-', '-o', str(partial), url]
        with (RAW / 'download.stderr.log').open('ab') as log:
            process = subprocess.Popen(command, stderr=log)
            try:
                while True:
                    event = dict(pid=process.pid, state='downloading',
                        bytes=partial.stat().st_size if partial.exists() else 0,
                        expected_bytes=SIZE, elapsed_seconds=time.monotonic() - started,
                        timestamp_utc=datetime.now(timezone.utc).isoformat())
                    (RAW / 'heartbeat.json').write_text(json.dumps(event, indent=2) + '\n')
                    print(json.dumps(event), flush=True)
                    try:
                        process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise SystemExit(f'Download exit {process.returncode}; retained partial and stderr for resume')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        if partial.stat().st_size != SIZE or digest(partial, 'md5') != MD5:
            raise SystemExit('Downloaded bytes fail publisher checksum; retained, not admitted')
        os.replace(partial, archive)
    if archive.stat().st_size != SIZE or digest(archive, 'md5') != MD5:
        raise SystemExit('Archive size/checksum mismatch')
    paths = [archive, meta_path]
    for name in DOCS:
        path = RAW / ('source_' + name.replace('/', '_'))
        if not path.exists():
            subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                '--max-time', '60', '--max-filesize', '100000', '-o', str(path),
                f'https://raw.githubusercontent.com/kav-institute/imptc-dataset/{DOC_COMMIT}/{name}'], check=True)
        paths.append(path)
    receipt = dict(result_source='fresh_run', source_url=url, metadata_url=META_URL,
        acquired_at_utc=datetime.now(timezone.utc).isoformat(), publisher_md5=MD5,
        archive_sha256=digest(archive), archive_bytes=SIZE,
        dataset_license='CC-BY-NC-4.0', use_scope='noncommercial_research_source_audit_only',
        docs_commit=DOC_COMMIT, role='unassigned_quarantined_source_audit',
        physical_sites=1, source_sequences_are_not_independent_sites=True,
        predictive_readout=False, fitting=False, scientific_role_assignment=False,
        media_present_in_bundled_archive=True, media_extracted_or_viewed=False,
        third_party_code_executed=False, stage5c_executed=False, smc_enabled=False,
        private_files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p)) for p in paths])
    with receipt_path.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    (RAW / 'heartbeat.json').write_text(json.dumps(dict(state='complete', seconds=time.monotonic()-started))+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
