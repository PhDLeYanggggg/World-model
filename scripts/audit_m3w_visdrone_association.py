"""Re-associate identical training-pilot boxes with and without camera compensation."""

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from scripts.audit_m3w_visdrone_camera_pilot import FRAMES, OBS, PRIVATE as PILOT_CACHE, PastImages
from scripts.audit_m3w_visdrone_camera_pilot import label_frames, measure, source, write_json
from src.world_model.m3w_camera_association import MotionAssociation, camera_step
from src.world_model.m3w_streaming_boxes import StreamingTracks, overlap

OUT = source.OUT.parent/'visdrone_association_v1'
PRIVATE = PILOT_CACHE.parent/'visdrone_association_v1'
ARMS = ('last_box', 'velocity', 'camera', 'camera_velocity')


def detection(row):
    a = np.asarray(row['detections'], dtype=float).reshape(-1, 5)
    if not np.array_equal(a[:, 0], np.arange(len(a))):
        raise ValueError('Frozen detection order changed')
    # All frozen detections are persons; association does not read confidence.
    return np.column_stack((a[:, 1:], np.ones((len(a), 2))))


def prefixes(bundle, images):
    variants = {k: [] for k in ARMS}
    tracker = {ARMS[0]: StreamingTracks(), ARMS[1]: MotionAssociation(True),
               ARMS[2]: MotionAssociation(False), ARMS[3]: MotionAssociation(True)}
    if [row['frame'] for row in bundle['past']] != list(FRAMES[:OBS]):
        raise ValueError('Unexpected past-frame roster')
    old_image, old_boxes, moves = None, None, []
    for i, row in enumerate(bundle['past']):
        d = detection(row)
        raw = images.read(bundle['clip'], row['frame'])
        if source.digest(raw) != row['jpeg_sha256']:
            raise ValueError('Past image differs from frozen pilot input')
        gray = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError('Cannot decode cached pilot image')
        motion = None
        if old_image is not None:
            motion, diagnostic = camera_step(old_image, gray, old_boxes, d)
            moves.append(dict(frame=row['frame'], matrix=motion.tolist(), **diagnostic))
        original, _ = tracker[ARMS[0]].step(i, d)
        if not np.array_equal(original[:, :5], np.asarray(row['online']).reshape(-1, 5)):
            raise ValueError('The original online tracker no longer reproduces the frozen row')
        variants[ARMS[0]].append(row)
        for k in ARMS[1:]:
            tracked = tracker[k].step(i, d, motion if k.startswith('camera') else None)
            if not np.array_equal(tracked[:, 1:5], d[:, :4]):
                raise ValueError('An association arm modified the frozen detection boxes')
            variants[k].append({**row, 'online': tracked[:, :5].tolist()})
        old_image, old_boxes = gray, d
    return {k: dict(clip=bundle['clip'], past=v) for k, v in variants.items()}, moves


def possible_reference_support(prefix, labels):
    """Reference-only upper bound, ignoring conflicts and identity assignment."""
    complete = set.intersection(*(set(a[:, 0].astype(int)) for a in labels))
    for row, target in zip(prefix['past'], labels[:OBS], strict=True):
        a, b = detection(row)[:, :4], target[:, 1:5]
        distance = .5*np.linalg.norm(a[:, None, :2]+a[:, None, 2:] - (b[:, :2]+b[:, 2:]), axis=-1)
        radius = (.5*np.linalg.norm(b[:, 2:]-b[:, :2], axis=-1)).clip(1.)
        eligible = ((overlap(a, b) >= .1) & (distance <= radius)).any(axis=0)
        complete.intersection_update(target[eligible, 0].astype(int))
    return len(complete)


def run(verify):
    base = source.OUT
    freeze = (base/'camera_freeze.json').read_bytes()
    prior = json.loads(freeze)
    base_reg = (base/'camera_registration.json').read_bytes()
    base_counts = (base/'camera_support.json').read_bytes()
    manifest = (source.DATA/'source_manifest.json').read_bytes()
    m, r, counts = map(json.loads, (manifest, base_reg, base_counts))
    if (source.digest(base_reg) != prior['registration_sha256']
            or source.digest(manifest) != r['source_manifest_sha256']
            or source.digest(freeze) != counts['freeze_sha256']
            or sorted(prior['prefixes']) != r['clips']):
        raise ValueError('The pilot freeze no longer binds the input inventory')
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE/'prefixes.json')], cwd=ROOT, check=True)
    if any(p.is_symlink() for p in (PRIVATE, *PRIVATE.parents)):
        raise ValueError('Refuse symlinked private workspace')
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT/'src/world_model/m3w_camera_association.py',
             ROOT/'src/world_model/m3w_streaming_boxes.py', ROOT/'scripts/audit_m3w_visdrone_camera_pilot.py',
             ROOT/'scripts/audit_m3w_ht21_detection_support.py', ROOT/'scripts/audit_m3w_visdrone_mot_train.py']
    reg = dict(parent_freeze_sha256=source.digest(freeze), parent_counts_sha256=source.digest(base_counts),
               clips=r['clips'], arms=ARMS, past_frames=r['past_frames'], references=r['reference_frames'],
               motion='background-masked two-frame LK, forward/back check, robust similarity, identity on weak fit',
               velocity='last two assigned detections in current coordinates, divided by sampled-frame gap',
               unchanged='same detections, order, assignment cost and gate, two-frame track age, and scorer',
               role='reused_training_camera_windows_only', scientific_evaluation='not_run',
               cv2=cv2.__version__, cv_threads=1, rng_seed=27,
               implementation={p.name: source.digest(p.read_bytes()) for p in paths})
    cv2.setNumThreads(1)
    cv2.setRNGSeed(reg['rng_seed'])
    with (PRIVATE/'lock').open('a') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(OUT/'registration.json', reg, verify)
        images = PastImages(m, stack, verify=True)
        registered = source.digest((OUT/'registration.json').read_bytes())
        registered_prefixes, cameras = {}, {}
        for clip, parent in prior['prefixes'].items():
            raw = (PILOT_CACHE/f'{clip}_prefix.json').read_bytes()
            if source.digest(raw) != parent['prefix_sha256']:
                raise ValueError('Prior camera prefix differs from freeze')
            registered_prefixes[clip], cameras[clip] = prefixes(json.loads(raw), images)
        payload_path = PRIVATE/'prefixes.json'
        write_json(payload_path, dict(registration_sha256=registered, clips=registered_prefixes), verify)
        write_json(OUT/'freeze.json', dict(registration_sha256=registered,
                   new_prefixes_sha256=source.digest(payload_path.read_bytes()), camera_steps=cameras,
                   reference_payload_opened_this_run=False), verify)
        expected = {x['clip']: x for x in counts['clips']}
        records = []
        for clip, arms in json.loads(payload_path.read_text())['clips'].items():
            raw = (source.DATA/f'{clip}.txt').read_bytes()
            if source.digest(raw) != m['labels'][clip]:
                raise ValueError('Training reference differs from intake manifest')
            labels = label_frames(raw, m['frames'][clip])
            rows = {k: measure(bundle, labels) for k, bundle in arms.items()}
            if rows[ARMS[0]] != expected[clip]:
                raise ValueError('The original track support counts do not replay')
            records.append(dict(clip=clip, arms=rows, eligible_reference_upper_bound=
                                possible_reference_support(arms[ARMS[0]], labels)))
        totals = {k: Counter() for k in ARMS}
        for record in records:
            for k, row in record['arms'].items():
                totals[k].update({f: n for f, n in row.items() if f != 'clip'})
        result = dict(result_source='fresh_run', images_and_boxes='cached_verified', records=records, totals=totals,
                      registration_sha256=registered, freeze_sha256=source.digest((OUT/'freeze.json').read_bytes()),
                      eligible_reference_upper_bound=sum(x['eligible_reference_upper_bound'] for x in records),
                      accepted_camera_steps=sum(step['accepted'] for xs in cameras.values() for step in xs),
                      camera_steps=sum(map(len, cameras.values())), scientific_evaluation='not_run',
                      validation_or_test_opened=False, independent_confirmation=False, deployment_changed=False)
        write_json(OUT/'readout.json', result, verify)
        print(json.dumps({k: result[k] for k in ('totals', 'eligible_reference_upper_bound',
                         'accepted_camera_steps', 'camera_steps')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args().verify)
