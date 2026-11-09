"""Test coarse-priority continuation using frozen VisDrone camera detections."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts import audit_m3w_visdrone_detector_tiles as parent
from scripts.audit_m3w_ht21_detection_support import associate
from scripts.audit_m3w_visdrone_camera_pilot import FRAMES, OBS, label_frames, measure, source, write_json
from src.world_model.m3w_continuation_boxes import ContinuationBoxes

OUT = parent.OUT.parent/'visdrone_continuation_boxes_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
ARMS = ('coarse', 'single_fine_bridge', 'continued_fine_bridge')


def make_prefix(clip, frames, motions):
    if [row['binding']['frame'] for row in frames] != list(FRAMES[:OBS]) or len(motions) != OBS-1:
        raise ValueError('Frozen frame roster mismatch')
    arm = {k: dict(clip=clip, past=[]) for k in ARMS}
    state = {k: ContinuationBoxes(k != ARMS[2]) for k in ARMS}
    for i, frame in enumerate(frames):
        coarse, fine = (parent.person_boxes(frame[k]) for k in ('coarse', 'fine'))
        for k in ARMS:
            primary, bridge = state[k].step(i, coarse, np.empty((0, 6)) if k == 'coarse' else fine,
                                            motions[i-1] if i else None)
            output = np.concatenate((primary, bridge))
            arm[k]['past'].append(dict(online=output[:, :5].tolist(), fine_ids=bridge[:, 0].astype(int).tolist(),
                detections=np.column_stack((np.arange(len(output)), output[:, 1:5])).tolist()))
    return arm


def count_support(bundle, labels):
    data = measure(bundle, labels)
    stable = set.intersection(*(associate(np.asarray(r['online']).reshape(-1, 5), a)
                                for r, a in zip(bundle['past'], labels[:OBS], strict=True)))
    complete = set.intersection(*(set(a[:, 0].astype(int)) for a in labels[OBS:]))
    bridge_ids = {i for row in bundle['past'] for i in row['fine_ids']}
    bridged = sum(i in bridge_ids and j in complete for i, j in stable)
    return dict(**data, fine_associations=sum(len(row['fine_ids']) for row in bundle['past']),
                complete_matches_using_fine=bridged)


def run(verify):
    reg, manifest, pilot, camera, _ = parent.evidence()
    before, freeze, readout = ((parent.OUT/f).read_bytes() for f in ('registration.json', 'freeze.json', 'readout.json'))
    tile, locked, old = map(json.loads, (before, freeze, readout))
    if (source.digest(before) != locked['registration_sha256'] or source.digest(freeze) != old['freeze_sha256']
            or reg['camera_freeze_sha256'] != tile['camera_freeze_sha256'] or reg['clips'] != tile['clips']):
        raise ValueError('Detector tile freeze changed')
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE/'prefixes.json')], cwd=ROOT, check=True)
    if any(p.is_symlink() for p in (PRIVATE, *PRIVATE.parents)):
        raise ValueError('Refuse symlinked cache path')
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(parent.__file__), ROOT/'src/world_model/m3w_continuation_boxes.py',
             ROOT/'src/world_model/m3w_camera_association.py', ROOT/'src/world_model/m3w_streaming_boxes.py',
             ROOT/'scripts/audit_m3w_visdrone_camera_pilot.py', ROOT/'scripts/audit_m3w_ht21_detection_support.py']
    registration = dict(parent_detector_freeze_sha256=source.digest(freeze), parent_readout_sha256=source.digest(readout),
                        camera_freeze_sha256=reg['camera_freeze_sha256'], clips=reg['clips'], arms=ARMS,
                        frames=FRAMES, rule='coarse matching first; fine may only extend unmatched old tracks',
                        cross_scale_duplicate_iou=.5, role='reused_training_windows_only',
                        implementation={p.name: source.digest(p.read_bytes()) for p in paths})
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(OUT/'registration.json', registration, verify)
        bound = source.digest((OUT/'registration.json').read_bytes())
        predictions = {}
        for clip in reg['clips']:
            frames = []
            for f, jpeg in zip(FRAMES[:OBS], pilot['prefixes'][clip]['images'], strict=True):
                path = parent.PRIVATE/f'{clip}_{f:07}.json'
                raw = path.read_bytes()
                if source.digest(raw) != locked['detections'][path.name]:
                    raise ValueError('Detection checkpoint differs from freeze')
                data = json.loads(raw)
                if data['binding'] != dict(frame=f, jpeg_sha256=jpeg, registration_sha256=locked['registration_sha256']):
                    raise ValueError('Detection checkpoint has the wrong frame binding')
                frames.append(data)
            motions = camera['camera_steps'][clip]
            if [m['frame'] for m in motions] != list(FRAMES[1:OBS]):
                raise ValueError('Camera step indices changed')
            predictions[clip] = make_prefix(clip, frames, [np.asarray(m['matrix']) for m in motions])
        payload = PRIVATE/'prefixes.json'
        write_json(payload, predictions, verify)
        write_json(OUT/'freeze.json', dict(registration_sha256=bound, prefix_sha256=source.digest(payload.read_bytes()),
                                          reference_payload_opened_this_run=False), verify)
        expected = {r['clip']: r['arms']['coarse'] for r in old['records']}
        records, totals = [], {k: Counter() for k in ARMS}
        for clip, arms in json.loads(payload.read_text()).items():
            raw = (source.DATA/f'{clip}.txt').read_bytes()
            if source.digest(raw) != manifest['labels'][clip]:
                raise ValueError('Training annotation changed')
            labels = label_frames(raw, manifest['frames'][clip])
            stats = {k: count_support(bundle, labels) for k, bundle in arms.items()}
            if any(stats['coarse'][k] != v for k, v in expected[clip].items()):
                raise ValueError('Frozen coarse counts changed')
            for k, row in stats.items():
                totals[k].update({f: v for f, v in row.items() if f != 'clip'})
            records.append(dict(clip=clip, arms=stats))
        result = dict(result_source='fresh_run', detector_inputs='cached_verified', registration_sha256=bound,
                      freeze_sha256=source.digest((OUT/'freeze.json').read_bytes()), totals=totals, records=records,
                      pooled_parent=old['totals']['pooled'], validation_or_test_opened=False,
                      scientific_evaluation='not_run', independent_confirmation=False, deployment_changed=False)
        write_json(OUT/'readout.json', result, verify)
        print(json.dumps(totals, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args().verify)
