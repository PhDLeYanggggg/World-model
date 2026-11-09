"""Inspect VisDrone-MOT train structure using bounded reads from two pinned mirrors."""

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image

from scripts.audit_m3w_ht21_detection_support import write_json
from src.evaluation.m3w_remote_zip import RangeFile

DATA = ROOT/'external_data/VisDrone_MOT_train_source_audit'
OUT = ROOT/'outputs/publication_readiness_2026_09/visdrone_mot_train_intake_v1'
ARCHIVE = 'VisDrone2019-MOT-train.zip'
PREFIX = 'VisDrone2019-MOT-train'
SIZE = 8_080_572_990
LFS_OID = '566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014'
MIRRORS = (('vanthanh/VisDrone2019-MOT', '1ec3d15d4d5a5b4db3f00f29f93ca8979fbebc91'),
           ('myn0908/visdrone-dataset', '186fa9d034c72fb27e724e5bab0965fb91308a70'))
STRIDES = (1, 4)
OBS, FUTURE = 8, 12
CLIP = r'(uav\d{7})_(\d{5})_v'
FIRST = 'uav0000013_00000_v'


def digest(b):
    return sha256(b).hexdigest()


def inventory(z):
    seen, labels, frame_files = set(), {}, defaultdict(list)
    entries = []
    for info in z.infolist():
        name = info.filename
        if name in seen or not name.startswith(PREFIX+'/') or '..' in name.split('/') or '\\' in name:
            raise ValueError('Unexpected or duplicate ZIP path')
        if info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Encrypted or symlinked entry')
        seen.add(name)
        entries.append((name, info.header_offset, info.CRC, info.file_size, info.compress_size))
        if info.is_dir():
            continue
        annot = re.fullmatch(f'{PREFIX}/annotations/({CLIP})\\.txt', name)
        frame = re.fullmatch(f'{PREFIX}/sequences/({CLIP})/(\\d{{7}})\\.jpg', name)
        if annot:
            if not 0 < info.file_size <= 8_000_000:
                raise ValueError('Invalid annotation length')
            labels[annot[1]] = info
        elif frame:
            frame_files[frame[1]].append(int(frame[4]))
        else:
            raise ValueError(f'Unexpected publisher-training member type: {name}')
    if len(labels) != 56 or set(labels) != set(frame_files):
        raise ValueError('Expected the 56 training sequences in the original archive layout')
    frames = {k: sorted(v) for k, v in frame_files.items()}
    return labels, frames, digest(json.dumps(entries, separators=(',', ':')).encode())


def obtain():
    ids = [{'repo': repo, 'revision': rev} for repo, rev in MIRRORS]
    out = []
    for asset in ids:
        url = f"https://huggingface.co/datasets/{asset['repo']}/resolve/{asset['revision']}/{ARCHIVE}?download=true"
        file = RangeFile(url, SIZE)
        with ZipFile(file) as z:
            labels, frames, root = inventory(z)
            if not out:
                a = min(i.header_offset for i in labels.values())
                b = max(i.header_offset+i.compress_size for i in labels.values())
                if b-a > 7_500_000:
                    raise ValueError('Annotation payload no longer a bounded contiguous span')
                file.span(a, b-a+1024)
                raw = {key: z.read(info) for key, info in sorted(labels.items())}
                jpeg_info = z.getinfo(f'{PREFIX}/sequences/{FIRST}/0000001.jpg')
                if jpeg_info.file_size > 2_000_000:
                    raise ValueError('Oversized sample JPEG')
                jpeg = z.read(jpeg_info)
                with Image.open(BytesIO(jpeg)) as im:
                    if im.format != 'JPEG':
                        raise ValueError('Expected JPEG')
                    dimensions = list(im.size)
                    im.verify()
                image_sample = dict(clip=FIRST, frame=1, sha256=digest(jpeg), shape_width_height=dimensions,
                                    read_for_decoding_only=True)
            else:
                if root != out[0]['inventory_sha256']:
                    raise ValueError('Pinned mirror ZIP directories disagree')
                for k in sorted(labels)[::19]:
                    if z.read(labels[k]) != raw[k]:
                        raise ValueError('Pinned mirrors disagree on sampled labels')
        out.append({**asset, 'inventory_sha256': root, 'bytes_fetched': file.transferred, 'requests': file.requests})
    return raw, dict(mirrors=out, declared_archive_size=SIZE, mirror_declared_lfs_oid=LFS_OID,
                     full_archive_sha256_verified=False, frames=frames, image_sample=image_sample,
                     validation_payload_read=False, test_payload_read=False)


def parse(payload, frames):
    a = np.loadtxt(BytesIO(payload), delimiter=',', ndmin=2)
    if (a.ndim != 2 or a.shape[1] != 10 or not np.isfinite(a).all()
            or not np.array_equal(a, np.floor(a))):
        raise ValueError('Expected ten finite integer columns')
    if not np.isin(a[:, 0], frames).all() or not np.isin(a[:, 6], [0, 1]).all():
        raise ValueError('Annotation frame absent from image roster, or unexpected inclusion flag')
    good = a[:, 6] == 1
    if (a[good, 4:6] <= 0).any():
        raise ValueError('Invalid included box extent')
    for cat in (1, 2):
        ids = a[good & (a[:, 7] == cat), :2]
        if len(np.unique(ids, axis=0)) != len(ids) or (ids[:, 1] < 0).any():
            raise ValueError('Invalid or duplicate included person identity')
    return a.astype(np.int64)


def support(a, frames, stride):
    labels = defaultdict(set)
    clear = defaultdict(set)
    for frame, person, _, _, _, _, included, category, trunc, occlusion in a:
        if included == 1 and category == 1:
            labels[frame].add(person)
            if trunc == 0 and occlusion == 0:
                clear[frame].add(person)
    result = dict(windows=0, history_instances=0, complete=0, partial=0, no_future=0,
                  multi_complete_windows=0, complete_untruncated_unoccluded=0)
    available = set(frames)
    for start in range(min(frames), max(frames)-(OBS+FUTURE-1)*stride+1, (OBS+FUTURE)*stride):
        wanted = [start+i*stride for i in range(OBS+FUTURE)]
        if not set(wanted) <= available:
            continue
        hist = set.intersection(*(labels[t] for t in wanted[:OBS]))
        complete = hist & set.intersection(*(labels[t] for t in wanted[OBS:]))
        partial = (hist & set.union(*(labels[t] for t in wanted[OBS:]))) - complete
        result['windows'] += 1
        result['history_instances'] += len(hist)
        result['complete'] += len(complete)
        result['partial'] += len(partial)
        result['no_future'] += len(hist)-len(complete)-len(partial)
        result['multi_complete_windows'] += len(complete) >= 2
        result['complete_untruncated_unoccluded'] += len(set.intersection(*(clear[t] for t in wanted)))
    return result


def summarize(labels, metadata):
    results, parent = {}, defaultdict(list)
    for k, frames in sorted(metadata['frames'].items()):
        a = parse(labels[k], frames)
        family, start = re.fullmatch(CLIP, k).groups()
        # Filename grouping is conservative; suffix offsets are not a validated clock.
        parent[family].append(dict(clip=k, suffix_start=int(start), release_frames=len(frames)))
        category, n = np.unique(a[:, 7], return_counts=True)
        results[k] = dict(release_frames=len(frames), image_number_gaps=sum(b != c+1 for c, b in zip(frames, frames[1:])),
                          label_rows=len(a), included_pedestrian_rows=int(((a[:, 6] == 1) & (a[:, 7] == 1)).sum()),
                          category_rows={str(k): int(v) for k, v in zip(category, n, strict=True)},
                          support={str(s): support(a, frames, s) for s in STRIDES})
    totals = {str(s): {k: sum(r['support'][str(s)][k] for r in results.values())
                       for k in next(iter(results.values()))['support'][str(s)]} for s in STRIDES}
    return dict(clips=results, totals=totals, candidate_recording_groups=parent,
                frame_files=sum(len(v) for v in metadata['frames'].values()),
                labels_only_support=True, online_roster_evaluated=False, source_roles_assigned=False,
                detector_training_disjointness_verified=False, scientific_evaluation='not_run',
                coordinate_unit='image_pixel', effective_seconds='not_audited', independent_confirmation=False)


def run(download, verify):
    for p in (DATA, *DATA.parents):
        if p.is_symlink():
            raise ValueError('Refuse symlinked source directory')
    manifest = DATA/'source_manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if set(record['labels']) != set(record['frames']) or len(record['labels']) != 56:
            raise ValueError('Unexpected cached training roster')
        labels = {}
        for k, sha in record['labels'].items():
            if not re.fullmatch(CLIP, k):
                raise ValueError('Unexpected cached annotation name')
            labels[k] = (DATA/f'{k}.txt').read_bytes()
            if digest(labels[k]) != sha:
                raise ValueError('Cached annotation differs from range receipt')
    else:
        if not download or verify:
            raise FileNotFoundError('Use --download to acquire the pinned training annotation ranges')
        subprocess.run(['git', 'check-ignore', '--quiet', str(manifest)], cwd=ROOT, check=True)
        labels, record = obtain()
        record['labels'] = {k: digest(b) for k, b in labels.items()}
        DATA.mkdir(parents=True, exist_ok=True)
        for k, b in labels.items():
            p = DATA/f'{k}.txt'
            if p.exists() and p.read_bytes() != b:
                raise ValueError('Existing source member differs')
            if not p.exists():
                with p.open('xb') as f:
                    f.write(b)
        write_json(manifest, record, False)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {k: v for k, v in record.items() if k not in ('frames', 'labels')}
    write_json(OUT/'source_receipt.json', dict(receipt, train_members=len(record['labels']),
                                              source_manifest_sha256=digest(manifest.read_bytes())), verify)
    stats = summarize(labels, record)
    write_json(OUT/'readout.json', stats, verify)
    print(json.dumps({k: stats[k] for k in ('totals', 'frame_files', 'scientific_evaluation')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    run(args.download, args.verify)
