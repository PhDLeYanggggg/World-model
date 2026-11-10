"""Read a bounded subset of the publisher archive for structural coverage."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_m3w_songdo_publisher_footprints import curl
from scripts.run_m3w_opendd_location import write_json
from src.data_unification.m3w_public_zip_range import RangeArchive


ROOT = Path(__file__).resolve().parents[1]
RECORD = 'https://zenodo.org/api/records/10491409'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/pneuma_source_intake_v1'
LOCAL = ROOT/'external_data/pNEUMA_source_intake'
CHECKSUM = 'md5:4e842b2200bea2f80d41c6b785a9de25'
SIZE = 15759302461
MEMBER = re.compile(r'(?P<day>2018\d{4})_(?P<drone>[dD](?:10|[1-9]|[xX]))_(?P<period>\d{4}_\d{4})\.csv')


class ZenodoArchive(RangeArchive):
    """Versionless Zenodo responses are checked against the pinned ZIP catalog."""

    def _fetch(self, first, last):
        size = last-first+1
        if self.total+size > self.byte_limit:
            raise ValueError('Bounded publisher range budget exceeded')
        raw = curl(self.url, size, '-H', f'Range: bytes={first}-{last}', '-D', '-')
        header, payload = raw.split(b'\r\n\r\n', 1)
        if (b' 206 ' not in header.splitlines()[0]+b' '
                or f'content-range: bytes {first}-{last}/{self.length}'.encode() not in header.lower().splitlines()
                or len(payload) != size):
            raise ValueError('Publisher did not return exact range')
        self.total += size
        return payload


def record():
    metadata = json.loads(curl(RECORD, 1 << 20))
    item, = (f for f in metadata['files'] if f['key'] == 'pNEUMA_dataset.zip')
    if (item['size'], item['checksum'], metadata['metadata']['license']['id']) != (SIZE, CHECKSUM, 'cc-by-4.0'):
        raise ValueError('Publisher record or licence changed')
    if item['links']['self'] != f'{RECORD}/files/pNEUMA_dataset.zip/content':
        raise ValueError('Unrecognized publisher archive URL')
    return item


def inventory(z):
    entries = {}
    for info in z.infolist():
        name = Path(info.filename).name
        m = MEMBER.fullmatch(name)
        if m is None or info.filename.startswith('__MACOSX/') or not info.file_size:
            continue
        if name in entries:
            raise ValueError('Duplicate archive member name')
        entries[name] = dict(path=info.filename, size=info.file_size, compressed=info.compress_size,
                             crc32=f'{info.CRC:08x}', **m.groupdict())
    if not entries:
        raise ValueError('No recognised vehicle CSV')
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('catalog', 'sample'))
    p.add_argument('members', nargs='*')
    a = p.parse_args()
    if bool(a.members) != (a.phase == 'sample') or len(a.members) > 12 or len(set(a.members)) != len(a.members):
        p.error('Supply up to twelve distinct catalog names for sample only')
    item = record()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    dest = PUBLIC/'catalog.json'
    url = 'https://zenodo.org/records/10491409/files/pNEUMA_dataset.zip'
    with ZenodoArchive(url, SIZE, block_size=2 << 20, byte_limit=(320 << 20 if a.members else 4 << 20)) as remote:
        with ZipFile(remote) as z:
            rows = inventory(z)
            if not a.members:
                write_json(dest, dict(result_source='fresh_publisher_catalog', doi='10.5281/zenodo.10491409',
                    publisher_archive_bytes=item['size'], publisher_checksum=item['checksum'],
                    license_in_record='CC-BY-4.0', full_archive_checksum_verified=False, members=rows,
                    network_bytes=remote.total, member_payloads_read=False, predictive_roles_assigned=False), immutable=True)
                print('Catalog members:', len(rows), 'range bytes:', remote.total, flush=True)
                return
            if rows != json.loads(dest.read_text())['members'] or set(a.members)-rows.keys():
                raise ValueError('Catalog changed or member not declared')
            if sum(rows[m]['size'] for m in a.members) > 768 << 20:
                raise ValueError('Pilot would exceed local size budget')
            LOCAL.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in a.members:
                member = rows[name]
                path = LOCAL/name
                digest = hashlib.sha256()
                if not path.exists():
                    scratch = path.with_suffix('.pending')
                    with z.open(member['path']) as source, scratch.open('wb') as target:
                        for block in iter(lambda: source.read(1 << 20), b''):
                            target.write(block)
                            digest.update(block)
                    if scratch.stat().st_size != member['size']:
                        raise ValueError('Member length changed')
                    scratch.replace(path)
                else:
                    crc = 0
                    with path.open('rb') as source:
                        for block in iter(lambda: source.read(1 << 20), b''):
                            crc = zlib.crc32(block, crc)
                            digest.update(block)
                    if path.stat().st_size != member['size'] or f'{crc:08x}' != member['crc32']:
                        raise ValueError('Local pilot member differs from archive')
                files[name] = dict(**member, sha256=digest.hexdigest(), zip_crc_verified=True)
                print('Verified:', name, member['size'], flush=True)
    write_json(PUBLIC/'samples.json', dict(result_source='fresh_verified_pilot_members', members=files,
        full_archive_checksum_verified=False, predictive_roles_assigned=False, forecasts_scored=False), immutable=True)


if __name__ == '__main__':
    main()
