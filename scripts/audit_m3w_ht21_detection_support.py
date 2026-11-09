"""Check label support for prefix-tracked public detections on publisher train only."""

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.fetch_m3w_ht21_annotations import ARCHIVE, PUBLIC as SOURCE, sha
from src.evaluation.m3w_ht21_admission import require_role
from src.evaluation.m3w_ht21_intake import human_observation_mask, read_ground_truth
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap

TAG = 'ht21_detection_support_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / TAG
PUBLIC = SOURCE.parent / TAG
OBS, FUTURE = 8, 12
STRIDES = (1, 5, 10)
FIELDS = ('windows', 'multi_online', 'all_complete_multi_online', 'two_complete', 'online',
          'stable', 'complete', 'partial', 'no_future', 'unmatched', 'reference_complete')


def starts(frames, stride):
    span = OBS + FUTURE
    return range(1, frames - (span - 1)*stride + 1, span*stride)


def groups(rows):
    order = np.argsort(rows[:, 0], kind='stable')
    sorted_rows = rows[order]
    frames, bounds = np.unique(sorted_rows[:, 0].astype(int), return_index=True)
    return dict(zip(frames, np.split(sorted_rows, bounds[1:]), strict=True)) if len(frames) else {}


def detections(payload, frames):
    rows = np.loadtxt(io.BytesIO(payload), delimiter=',', ndmin=2)
    f = rows[:, 0]
    if (rows.shape[1] != 10 or not np.isfinite(rows).all() or not np.all(f == np.floor(f))
            or np.any((f < 1) | (f > frames)) or not np.all(rows[:, 1] == -1)
            or np.any(rows[:, 4:6] <= 0) or np.any((rows[:, 6] < 0) | (rows[:, 6] > 100))):
        raise ValueError('Invalid public detection frame, anonymous identity, box or score')
    return rows


def prefix(by_frame, start, stride):
    tracker = StreamingTracks()
    records = []
    for j in range(OBS):
        r = by_frame.get(start + j*stride, np.empty((0, 10)))
        boxes = np.column_stack((r[:, 2:4], r[:, 2:4]+r[:, 4:6], r[:, 6]/100., np.ones(len(r))))
        online, _ = tracker.step(j, boxes)
        records.append(online[:, :5])
    return records


def associate(online, label):
    if len(online) == 0 or len(label) == 0:
        return set()
    a, b = online[:, 1:5], label[:, 1:5]
    iou = overlap(a, b)
    distance = .5*np.linalg.norm((a[:, None, :2]+a[:, None, 2:]) - (b[:, :2]+b[:, 2:]), axis=-1)
    relative = distance / (.5*np.linalg.norm(b[:, 2:]-b[:, :2], axis=1)).clip(1.)
    valid = (iou >= .1) & (relative <= 1.)
    i, j = linear_sum_assignment(np.where(valid, 1-iou + .25*relative, 1e6))
    return {(int(online[x, 0]), int(label[y, 0])) for x, y in zip(i, j, strict=True) if valid[x, y]}


def support(past, label):
    online = set.intersection(*(set(a[:, 0].astype(int)) for a in past))
    stable = set.intersection(*(associate(a, b) for a, b in zip(past, label[:OBS], strict=True)))
    if not {i for i, _ in stable} <= online:
        raise ValueError('Stable match without a full online history')
    future_ids = Counter(int(i) for b in label[OBS:] for i in b[:, 0])
    count = Counter(windows=1, multi_online=int(len(online) >= 2), online=len(online),
                    stable=len(stable), unmatched=len(online)-len(stable))
    label_histories = set.intersection(*(set(a[:, 0].astype(int)) for a in label[:OBS]))
    count['reference_complete'] = sum(future_ids[i] == FUTURE for i in label_histories)
    for _, i in stable:
        n = future_ids[i]
        count['complete' if n == FUTURE else 'partial' if n else 'no_future'] += 1
    count['two_complete'] = int(count['complete'] >= 2)
    count['all_complete_multi_online'] = int(len(online) >= 2 and count['complete'] == len(online))
    return count


def write_json(path, payload, verify):
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if verify or path.exists():
        if path.read_text() != data:
            raise ValueError(f'Frozen artifact mismatch: {path.name}')
        return
    tmp = path.with_suffix('.pending')
    tmp.write_text(data)
    tmp.replace(path)


def box_member(z, name, stem):
    if name not in {f'HT21-0{i}' for i in range(1, 5)} or stem not in ('det', 'gt'):
        raise ValueError('Only the four publisher-train recordings are permitted')
    return z.read(f'HT21Labels/train/{name}/{stem}/{stem}.txt')


def prefix_bundle(rows, frames, stride):
    by_frame = groups(rows)
    digest = hashlib.sha256()
    windows = []
    for start in starts(frames, stride):
        past = prefix(by_frame, start, stride)
        for frame in past:
            digest.update(np.asarray([start, len(frame)], dtype='<i8').tobytes())
            digest.update(frame.astype('<f8', copy=False).tobytes())
        windows.append(past)
    return windows, digest.hexdigest()


def labels_for(payload, metadata):
    rows = read_ground_truth(payload, metadata)
    rows = rows[human_observation_mask(rows)]
    xyxy = np.column_stack((rows[:, :4], rows[:, 2:4]+rows[:, 4:6]))
    return {f: boxes[:, 1:] for f, boxes in groups(xyxy).items()}


def run(verify):
    audit_bytes = (SOURCE / 'analysis.json').read_bytes()
    audit = json.loads(audit_bytes)
    archive_hash = sha(ARCHIVE.read_bytes())
    if archive_hash != audit['archive_sha256']:
        raise ValueError('Archive differs from the audited source')
    records = [r for r in audit['recordings'] if r['supplied_split'] == 'train']
    if sorted(r['sequence'] for r in records) != [f'HT21-0{i}' for i in range(1, 5)]:
        raise ValueError('Unexpected publisher-train inventory')
    for record in records:
        require_role(record, 'source_audit')
    code = (Path(__file__), ROOT / 'src/evaluation/m3w_ht21_intake.py',
            ROOT / 'src/evaluation/m3w_ht21_admission.py', ROOT / 'src/world_model/m3w_streaming_boxes.py')
    registration = dict(source_analysis_sha256=sha(audit_bytes), archive_sha256=archive_hash,
                        scientific_role='source_audit_only', observations=OBS, future_label_steps=FUTURE,
                        strides=STRIDES, start=1, window_step='20 * raw stride',
                        detection_filter='none; use all released boxes',
                        association='one-to-one IoU >= .10 and center within half label diagonal',
                        code_sha256={p.name: sha(p.read_bytes()) for p in code},
                        prediction_evaluation=False, reserved_outcomes_opened=False)
    write_json(PUBLIC / 'registration.json', registration, verify)
    bound = sha((PUBLIC / 'registration.json').read_bytes())
    counts = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for record in records:
            name, metadata = record['sequence'], record['metadata']
            member = box_member(z, name, 'det')
            rows = detections(member, metadata['frames'])
            entries, frozen = [], {}
            for stride in STRIDES:
                windows, digest = prefix_bundle(rows, metadata['frames'], stride)
                frozen[stride] = windows
                entries.append(dict(stride=stride, windows=len(windows), prefix_sha256=digest))
            freeze = dict(sequence=name, registration_sha256=bound, detection_sha256=sha(member),
                          released_detections=len(rows), scores=[float(rows[:, 6].min()), float(rows[:, 6].max())],
                          entries=entries)
            write_json(PUBLIC / f'{name}_prefix.json', freeze, verify)
            member = box_member(z, name, 'gt')
            if sha(member) != record['ground_truth_sha256']:
                raise ValueError('Ground truth differs from the audited member')
            labels = labels_for(member, metadata)
            for stride, windows in frozen.items():
                count = Counter()
                for s, past in zip(starts(metadata['frames'], stride), windows, strict=True):
                    label = [labels.get(s+j*stride, np.empty((0, 5))) for j in range(OBS+FUTURE)]
                    count.update(support(past, label))
                counts.append(dict(sequence=name, stride=stride, camera_motion=metadata['camera_motion'],
                                   **{k: count[k] for k in FIELDS}))
            heartbeat = PRIVATE / 'heartbeat.pending'
            heartbeat.write_text(json.dumps(dict(pid=os.getpid(), completed=len(counts)//len(STRIDES),
                                                 total=len(records), last=name)) + '\n')
            heartbeat.replace(PRIVATE / 'heartbeat.json')
            print(name, flush=True)
    totals = [dict(stride=s, **{k: sum(row[k] for row in counts if row['stride'] == s) for k in FIELDS})
              for s in STRIDES]
    result = dict(result_source='fresh_run', archive_source='cached_verified', registration_sha256=bound,
                  recordings=counts, totals=totals, predictive_evaluation='not_run', scientific_admission=False,
                  independent_confirmation=False, stage5c=False, smc=False)
    write_json(PUBLIC / 'readout.json', result, verify)
    print(json.dumps(totals, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Replay and require byte-identical artifacts')
    args = parser.parse_args()
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE / 'heartbeat.json')], cwd=ROOT, check=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.verify)
    if args.verify:
        print('cached_verified: all prefixes and support counts match exactly')


if __name__ == '__main__':
    main()
