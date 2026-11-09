"""Measure KIT AIS publisher-train timing and detector-roster support only."""

import argparse
from datetime import datetime
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_m3w_ht21_detection_support import support, write_json
from scripts.fetch_m3w_dut_annotations import fetch
from src.world_model.m3w_streaming_boxes import StreamingTracks

DATA = ROOT/'external_data/KIT_AIS_source_audit'
OUT = ROOT/'outputs/publication_readiness_2026_09/kit_ais_source_audit_v1'
URL = 'https://www2.ipf.kit.edu/Personen/schmidt/'
FILES = {
    'kit_ais_train.7z': ('KIT_AIS_People_Training_Data_Set.7z', 19883890,
                         '5301a7e6dfe13f84372df570f03a66c78323f93a40ffccca4caf38a0f36193b9'),
    'kit_ais_detections.7z': ('KIT_AIS_People_IPF_Detections.7z', 2220844,
                              '7d9013ecf13954ac1c7ef231a855a4c3f9e7929c796605d603692931639c85a0'),
}
CLIPS = ('Munich01', 'AA_Crossing_01', 'AA_Easy_01', 'AA_Easy_02', 'AA_Easy_Entrance', 'AA_Walking_01',
         'RaR_Snack_Zone_01')
THRESHOLDS = (0., .6, .75)
OBS, FUTURE = 8, 12


def digest(raw):
    return sha256(raw).hexdigest()


def archive(path, expected, acquire):
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Refuse symlinked source location')
    source, size, sha = expected
    if not path.exists():
        if not acquire:
            raise FileNotFoundError('Use --download to acquire the pinned publisher archive')
        subprocess.run(['git', 'check-ignore', '--quiet', str(path)], cwd=ROOT, check=True)
        raw = fetch(URL+source, size)
        if len(raw) != size or digest(raw) != sha:
            raise ValueError('Unexpected publisher archive bytes')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.pending')
        with tmp.open('xb') as f:
            f.write(raw)
        tmp.replace(path)
    raw = path.read_bytes()
    if len(raw) != size or digest(raw) != sha:
        raise ValueError('Cached archive differs from official pinned bytes')
    return dict(url=URL+source, bytes=size, sha256=sha)


def tar(arguments):
    return subprocess.run(['/usr/bin/bsdtar', *map(str, arguments)], check=True, capture_output=True,
                          timeout=15, env={**os.environ, 'LC_ALL': 'C'}).stdout


def members(train, detection):
    allow = set(tar(['-tf', train]).decode().splitlines())
    det = set(tar(['-tf', detection]).decode().splitlines())
    for c in CLIPS:
        if f'Training Data Sets/{c}/{c}.xml' not in allow or f'IPF Detections/{c}-D.xml' not in det:
            raise ValueError('Publisher-train roster changed')
    return allow


def parse(payload, name, *, detector):
    if len(payload) > 6_000_000 or b'<!DOCTYPE' in payload.upper() or b'<!ENTITY' in payload.upper():
        raise ValueError('Unexpected XML declarations or size')
    root = ET.fromstring(payload)
    if root.tag != 'dataset' or root.get('name') != name:
        raise ValueError('Incorrect dataset name')
    frames, rows = [], []
    for frame in root.findall('frame'):
        filename = PurePosixPath(frame.attrib['file'].replace('\\', '/')).name
        stamp = datetime.strptime(frame.attrib['utc'], '%Y-%b-%d %H:%M:%S.%f')
        n = int(frame.attrib['number'])
        frames.append((n, filename, stamp))
        r, ids = [], set()
        for obj in frame.findall('objectlist/object'):
            box = obj.find('box')
            x, y, w, h = [float(box.attrib[k]) for k in ('xc', 'yc', 'w', 'h')]
            if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
                raise ValueError('Invalid box')
            coords = (x-w/2., y-h/2., x+w/2., y+h/2.)
            if detector:
                # Published detector IDs are not online identities.
                score = float(obj.find('detection_score').attrib['value'])
                if not 0. <= score <= 1.:
                    raise ValueError('Invalid confidence score')
                r.append((*coords, score, 1.))
            else:
                person = int(obj.attrib['id'])
                if person in ids:
                    raise ValueError('Duplicate person in reference frame')
                ids.add(person)
                r.append((person, *coords))
        rows.append(np.asarray(r, dtype=float).reshape(-1, 6 if detector else 5))
    if (not frames or len({f[1] for f in frames}) != len(frames)
            or any(b[0] != a[0]+1 or b[2] <= a[2] for a, b in zip(frames, frames[1:]))):
        raise ValueError('Expected unique files and monotonically ordered release clock')
    return frames, rows


def prefix(records, threshold):
    tracker = StreamingTracks()
    return [tracker.step(i, a[a[:, 4] >= threshold])[0][:, :5] for i, a in enumerate(records[:OBS])]


def summarize_timing(frames):
    gaps = [(b[2]-a[2]).total_seconds() for a, b in zip(frames, frames[1:])]
    mid = float(np.median(gaps))
    cuts = [i+1 for i, gap in enumerate(gaps) if gap > 1.5*mid]
    boundaries = [0, *cuts, len(frames)]
    return dict(min=min(gaps), median=mid, max=max(gaps), gaps_over_1p5_median=len(cuts),
                run_lengths_between_large_gaps=[b-a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def run(acquire, verify):
    sources = {name: archive(DATA/name, entry, acquire) for name, entry in FILES.items()}
    train, detections = (DATA/k for k in FILES)
    inventory = members(train, detections)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/'registration.json', dict(sources=sources, publisher='KIT AIS', license='CC BY 3.0',
        recordings=CLIPS, observations=OBS, future_label_positions=FUTURE, confidence_arms=THRESHOLDS,
        query='first 20 release positions only if timeline fits', evaluation_members_read=False,
        scope='publisher-train source support only; no risk or forecast scoring',
        code={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in (Path(__file__).resolve(),
            ROOT/'src/world_model/m3w_streaming_boxes.py', ROOT/'scripts/audit_m3w_ht21_detection_support.py')}), verify)
    all_inputs, receipt = {}, {}
    for c in CLIPS:
        raw = tar(['-xOf', detections, f'IPF Detections/{c}-D.xml'])
        timeline, rows = parse(raw, c, detector=True)
        if any(f'Training Data Sets/{c}/{file}' not in inventory for _, file, _ in timeline):
            raise ValueError('Detector XML refers to missing original publisher images')
        bank = {str(t): prefix(rows, t) for t in THRESHOLDS} if len(rows) >= OBS+FUTURE else {}
        all_inputs[c] = timeline, bank
        receipt[c] = dict(detector_xml_sha256=digest(raw), frame_count=len(rows), detections=sum(map(len, rows)),
                          prefixes={t: digest(b''.join(np.asarray(a.shape, dtype='<i8').tobytes()+
                                                     a.astype('<f8', copy=False).tobytes() for a in p))
                                    for t, p in bank.items()})
    write_json(OUT/'prefix_freeze.json', dict(clips=receipt, evaluation_members_read=False), verify)
    result, clocks = {}, {}
    for c, (timeline, bank) in all_inputs.items():
        raw = tar(['-xOf', train, f'Training Data Sets/{c}/{c}.xml'])
        ref_clock, labels = parse(raw, c, detector=False)
        if timeline != ref_clock:
            raise ValueError('Detection and label clocks differ')
        clocks[c] = {x[2] for x in timeline}
        counts = {t: dict(support(p, labels[:OBS+FUTURE])) for t, p in bank.items()}
        result[c] = dict(release_frames=len(timeline), timing_seconds=summarize_timing(timeline),
                         label_rows=sum(map(len, labels)), label_xml_sha256=digest(raw), matched_support=counts)
    overlaps = [dict(a=a, b=b, shared_utc_frames=len(clocks[a] & clocks[b]))
                for a, b in combinations(CLIPS, 2) if clocks[a] & clocks[b]]
    data = dict(result_source='fresh_run', clips=result, shared_timestamps=overlaps,
                complete_8_to_12_training_clips=sum(bool(r['matched_support']) for r in result.values()),
                archive_inventory_matched=True, clip_clock_join=True, metric_status='not_verified',
                cadence_selected=False, forecast_evaluation='not_run', independent_confirmation=False,
                stage5c=False, smc=False)
    write_json(OUT/'readout.json', data, verify)
    print(json.dumps(data, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run(args.download, args.verify)
