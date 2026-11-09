"""Read publisher geometry metadata without opening Songdo trajectory archives."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_location as paths

ROOT = 'https://zenodo.org/api/records/17924857'
OUT = paths.PUBLIC.parent / 'songdo_publisher_footprints_v1'
BLOCK, LIMIT = 65536, 2*1024*1024


def curl(url, limit, *options):
    return subprocess.check_output(['curl', '-fsS', '--max-time', '20', '--max-filesize', str(limit),
                                    *options, url])


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.cache, self.fetched = {}, 0

    def seek(self, offset, whence=io.SEEK_SET):
        p = (0, self.pos, self.size)[whence]+offset
        if p < 0:
            raise ValueError('Seek before archive start')
        self.pos = p
        return p

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos+n)
        if end-self.pos > BLOCK:
            raise ValueError('Unexpected archive read larger than one block')
        chunks = []
        while self.pos < end:
            lo = self.pos//BLOCK*BLOCK
            hi = min(lo+BLOCK, self.size)
            if lo not in self.cache:
                if self.fetched+hi-lo > LIMIT:
                    raise ValueError('Metadata range-read budget exceeded')
                raw = curl(self.url, BLOCK, '-H', f'Range: bytes={lo}-{hi-1}', '-D', '-')
                header, data = raw.split(b'\r\n\r\n', 1)
                if (b' 206 ' not in header.splitlines()[0]+b' '
                        or f'content-range: bytes {lo}-{hi-1}/{self.size}'.encode() not in header.lower()
                        or len(data) != hi-lo):
                    raise ValueError('Publisher did not return the exact requested range')
                self.cache[lo] = data
                self.fetched += len(data)
            until = min(hi, end)
            chunks.append(self.cache[lo][self.pos-lo:until-lo])
            self.pos = until
        return b''.join(chunks)


def footprints(centers):
    names = sorted(centers)
    edges = []
    groups = {a: {a} for a in names}
    for a, b in combinations(names, 2):
        dx, dy = (abs(x-y) for x, y in zip(centers[a], centers[b], strict=True))
        if dx <= 15000 and dy <= 15000:
            edges.append([a, b])
            merged = groups[a] | groups[b]
            groups.update({x: merged for x in merged})
    return edges, [list(g) for g in sorted({tuple(sorted(v)) for v in groups.values()})]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    record = json.loads(curl(ROOT, 1024*1024))
    by_day = defaultdict(dict)
    for f in record['files']:
        m = re.fullmatch(r'(2022-10-0[4-7])_([A-Z])\.zip', f['key'])
        if m:
            by_day[m[1]][m[2]] = f['size']
    files = {f['key']: f for f in record['files']}
    ortho = files['orthophotos.zip']
    remote = RemoteZip(ortho['links']['self'], ortho['size'])
    centers, hashes = {}, {}
    with ZipFile(remote) as z:
        for info in z.infolist():
            m = re.fullmatch(r'([A-Z])_center\.txt', Path(info.filename).name)
            if m:
                if info.file_size > 256 or m[1] in centers:
                    raise ValueError('Unexpected center metadata schema')
                payload = z.read(info)
                center = [float(x) for x in payload.split()]
                if len(center) != 2 or not all(map(math.isfinite, center)):
                    raise ValueError('Expected two finite center coordinates')
                centers[m[1]] = center
                hashes[m[1]] = hashlib.sha256(payload).hexdigest()
    if len(by_day) != 4 or len(centers) != 20 or any(set(v) != set(centers) for v in by_day.values()):
        raise ValueError('Publisher file inventory or center roster changed')
    edges, groups = footprints(centers)
    OUT.mkdir(parents=True, exist_ok=True)
    paths.write_json(OUT / 'readout.json', dict(source='https://zenodo.org/records/17924857',
        result_source='fresh_run', per_date={k: dict(archives=len(v), bytes=sum(v.values())) for k, v in by_day.items()},
        base_cutout_width_px=15000, published_archive_checksum=ortho['checksum'],
        full_archive_checksum_verified=False, zip_member_crc_verified=True,
        metadata_member_sha256=hashes, network_payload_bytes=remote.fetched,
        map_cutout_overlap_edges=edges, map_cutout_components=groups,
        overlap_screen_only_not_source_independence=True, track_rows_read=0, outcomes_read=False,
        split_assigned=False), immutable=True)
    print(f'Metadata bytes: {remote.fetched}; cutout groups: {groups}')


if __name__ == '__main__':
    main()
